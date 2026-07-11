use std::{
    borrow::Cow,
    collections::BTreeMap,
    convert::TryInto,
    time::{Duration, Instant},
};

use ark_ff::{AdditiveGroup, BigInteger, Field, PrimeField};
use clap::Parser;
use rayon::prelude::*;
use whir::{
    algebra::{
        embedding::Identity,
        fields::Field192,
        linear_form::{Evaluate, LinearForm, MultilinearExtension, SparseCovector},
        sumcheck::compute_sumcheck_polynomial,
        MultilinearPoint,
    },
    bits::Bits,
    hash::{BLAKE3, HASH_COUNTER},
    lilac_merkle::{
        combine_equal_subtrees, parent, prefix_root, zero_roots, Digest, MerkleAccumulator,
    },
    parameters::ProtocolParameters,
    transcript::{codecs::Empty, DomainSeparator, Proof, ProverState, VerifierState},
    utils::workload_size,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = PRODUCTION_FIELDS)]
    fields: usize,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long, default_value_t = false)]
    semantic: bool,
    #[arg(long, default_value_t = 64)]
    batch_lanes: usize,
    /// Commit the real final Phi vector directly with WHIR and prove its
    /// terminal multilinear and sparse linear forms.
    #[arg(long, default_value_t = false)]
    direct_whir_tail: bool,
    /// Bind the combined sumcheck's terminal evaluations to two actual WHIR
    /// commitments of the packed relation vectors. This closes the terminal
    /// oracle-opening layer; field-Merkle root compatibility is measured
    /// separately.
    #[arg(long, default_value_t = false)]
    relation_whir: bool,
    /// Execute the actual pre-Carry batch and one production systematic-QA
    /// CarryOpen transition, including its selected-row front and WHIR tail.
    #[arg(long, default_value_t = false)]
    carryopen: bool,
    #[arg(long, default_value_t = 10)]
    whir_verifier_repetitions: usize,
    /// Run only the canonical three-stage strong-terminal end-to-end path,
    /// omitting communication ablations and diagnostic terminal proofs.
    #[arg(long, default_value_t = false)]
    final_only: bool,
}

const PRODUCTION_FIELDS: usize = 60_443_724;
const PRODUCTION_VARIABLES: usize = 26;
const TRANSCRIPT_BYTES: usize = 1_526;
const SUMCHECK_MAGIC: &[u8; 8] = b"LILSC001";

#[derive(Clone, Debug, PartialEq, Eq)]
struct PackedSumcheckProof {
    fields: usize,
    roots: Vec<Digest>,
    claimed_sum: Field192,
    pairs: Vec<(Field192, Field192)>,
    terminal_left: Field192,
    terminal_right: Field192,
}

impl PackedSumcheckProof {
    fn variables(&self) -> usize {
        self.pairs.len()
    }

    fn challenges(&self) -> Option<Vec<Field192>> {
        if self.fields <= 1
            || self.variables() != self.fields.next_power_of_two().trailing_zeros() as usize
            || self.roots.is_empty()
        {
            return None;
        }
        let mut claim = self.claimed_sum;
        let mut challenges = Vec::with_capacity(self.variables());
        for prefix in 1..=self.pairs.len() {
            let (constant, quadratic) = self.pairs[prefix - 1];
            let challenge = transcript_challenge(
                self.fields,
                self.variables(),
                &self.roots,
                self.claimed_sum,
                &self.pairs[..prefix],
            );
            let linear = claim - constant.double() - quadratic;
            claim = (quadratic * challenge + linear) * challenge + constant;
            challenges.push(challenge);
        }
        (claim == self.terminal_left * self.terminal_right).then_some(challenges)
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.challenges().is_some());
        let mut output = Vec::with_capacity(
            8 + 8 + 4 + 4 + self.roots.len() * 32 + (2 * self.pairs.len() + 3) * 24,
        );
        output.extend_from_slice(SUMCHECK_MAGIC);
        output.extend_from_slice(&(self.fields as u64).to_le_bytes());
        output.extend_from_slice(&(self.roots.len() as u32).to_le_bytes());
        output.extend_from_slice(&(self.pairs.len() as u32).to_le_bytes());
        for root in &self.roots {
            output.extend_from_slice(root);
        }
        output.extend_from_slice(&canonical_field_bytes(self.claimed_sum));
        for (constant, quadratic) in &self.pairs {
            output.extend_from_slice(&canonical_field_bytes(*constant));
            output.extend_from_slice(&canonical_field_bytes(*quadratic));
        }
        output.extend_from_slice(&canonical_field_bytes(self.terminal_left));
        output.extend_from_slice(&canonical_field_bytes(self.terminal_right));
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 8 + 4 + 4;
        if payload.len() < HEADER || &payload[..8] != SUMCHECK_MAGIC {
            return None;
        }
        let fields = u64::from_le_bytes(payload[8..16].try_into().ok()?) as usize;
        let root_count = u32::from_le_bytes(payload[16..20].try_into().ok()?) as usize;
        let variables = u32::from_le_bytes(payload[20..24].try_into().ok()?) as usize;
        let expected = HEADER + root_count * 32 + (2 * variables + 3) * 24;
        if root_count == 0 || payload.len() != expected {
            return None;
        }
        let mut position = HEADER;
        let roots = (0..root_count)
            .map(|_| {
                let root: Digest = payload[position..position + 32].try_into().ok()?;
                position += 32;
                Some(root)
            })
            .collect::<Option<Vec<_>>>()?;
        let mut take_field = || {
            let bytes = payload.get(position..position + 24)?;
            let value = Field192::from_le_bytes_mod_order(bytes);
            if canonical_field_bytes(value).as_slice() != bytes {
                return None;
            }
            position += 24;
            Some(value)
        };
        let claimed_sum = take_field()?;
        let pairs = (0..variables)
            .map(|_| Some((take_field()?, take_field()?)))
            .collect::<Option<Vec<_>>>()?;
        let terminal_left = take_field()?;
        let terminal_right = take_field()?;
        let proof = Self {
            fields,
            roots,
            claimed_sum,
            pairs,
            terminal_left,
            terminal_right,
        };
        proof.challenges()?;
        Some(proof)
    }
}

fn prove_local_product_relation(
    left: Vec<Field192>,
    right: Vec<Field192>,
    roots: Vec<Digest>,
) -> PackedSumcheckProof {
    prove_local_product_relation_with_claim(left, right, roots, Field192::ZERO)
}

fn prove_local_product_relation_with_claim(
    mut left: Vec<Field192>,
    mut right: Vec<Field192>,
    roots: Vec<Digest>,
    claimed_sum: Field192,
) -> PackedSumcheckProof {
    assert!(left.len() > 1 && left.len() == right.len());
    assert_eq!(dot(&left, &right), claimed_sum);
    let fields = left.len();
    let variables = fields.next_power_of_two().trailing_zeros() as usize;
    let mut claim = claimed_sum;
    let mut active = fields;
    let mut pairs = Vec::with_capacity(variables);
    for _ in 0..variables {
        let (constant, quadratic) = compute_sumcheck_polynomial(&left[..active], &right[..active]);
        pairs.push((constant, quadratic));
        let challenge = transcript_challenge(fields, variables, &roots, claimed_sum, &pairs);
        let linear = claim - constant.double() - quadratic;
        let next_left = fold_active(&mut left, active, challenge);
        let next_right = fold_active(&mut right, active, challenge);
        assert_eq!(next_left, next_right);
        active = next_left;
        claim = (quadratic * challenge + linear) * challenge + constant;
    }
    assert_eq!(active, 1);
    let proof = PackedSumcheckProof {
        fields,
        roots,
        claimed_sum,
        pairs,
        terminal_left: left[0],
        terminal_right: right[0],
    };
    assert!(proof.challenges().is_some());
    proof
}

#[derive(Debug)]
struct TensorProductProof {
    proof: PackedSumcheckProof,
    left_ood_row: Vec<Field192>,
    right_ood_row: Vec<Field192>,
}

fn tensor_row_round_polynomial(
    left: &[Field192],
    right: &[Field192],
    active_rows: usize,
    width: usize,
) -> (Field192, Field192) {
    let half = active_rows.next_power_of_two() >> 1;
    let high_rows = active_rows.saturating_sub(half);
    (0..half)
        .into_par_iter()
        .map(|row| {
            let mut constant = Field192::ZERO;
            let mut quadratic = Field192::ZERO;
            let low_start = row * width;
            let high_start = (half + row) * width;
            for lane in 0..width {
                let left_low = left[low_start + lane];
                let right_low = right[low_start + lane];
                let (left_high, right_high) = if row < high_rows {
                    (left[high_start + lane], right[high_start + lane])
                } else {
                    (Field192::ZERO, Field192::ZERO)
                };
                constant += left_low * right_low;
                quadratic += (left_high - left_low) * (right_high - right_low);
            }
            (constant, quadratic)
        })
        .reduce(
            || (Field192::ZERO, Field192::ZERO),
            |(ca, qa), (cb, qb)| (ca + cb, qa + qb),
        )
}

fn fold_tensor_rows(
    values: &mut [Field192],
    active_rows: usize,
    width: usize,
    challenge: Field192,
) -> usize {
    let half = active_rows.next_power_of_two() >> 1;
    let high_rows = active_rows.saturating_sub(half);
    let (low, high_and_tail) = values.split_at_mut(half * width);
    let (high, _) = high_and_tail.split_at_mut(high_rows * width);
    let (paired_low, zero_low) = low.split_at_mut(high_rows * width);
    paired_low
        .par_iter_mut()
        .zip(high.par_iter())
        .for_each(|(low, high)| *low += (*high - *low) * challenge);
    let zero_weight = Field192::ONE - challenge;
    zero_low
        .par_iter_mut()
        .for_each(|value| *value *= zero_weight);
    half
}

fn prove_tensor_product_relation(
    mut left: Vec<Field192>,
    mut right: Vec<Field192>,
    rows: usize,
    width: usize,
    roots: Vec<Digest>,
) -> TensorProductProof {
    assert!(rows > 1 && width > 1 && left.len() == rows * width && right.len() == left.len());
    let claimed_sum = dot(&left, &right);
    let row_domain = rows.next_power_of_two();
    let lane_domain = width.next_power_of_two();
    let fields = row_domain * lane_domain;
    let variables = fields.trailing_zeros() as usize;
    let row_variables = row_domain.trailing_zeros() as usize;
    let mut claim = claimed_sum;
    let mut pairs = Vec::with_capacity(variables);
    let mut active_rows = rows;
    for _ in 0..row_variables {
        let (constant, quadratic) = tensor_row_round_polynomial(&left, &right, active_rows, width);
        pairs.push((constant, quadratic));
        let challenge = transcript_challenge(fields, variables, &roots, claimed_sum, &pairs);
        let linear = claim - constant.double() - quadratic;
        let next_left = fold_tensor_rows(&mut left, active_rows, width, challenge);
        let next_right = fold_tensor_rows(&mut right, active_rows, width, challenge);
        assert_eq!(next_left, next_right);
        active_rows = next_left;
        claim = (quadratic * challenge + linear) * challenge + constant;
    }
    assert_eq!(active_rows, 1);
    let left_ood_row = left[..width].to_vec();
    let right_ood_row = right[..width].to_vec();
    let mut active_lanes = width;
    for _ in row_variables..variables {
        let (constant, quadratic) =
            compute_sumcheck_polynomial(&left[..active_lanes], &right[..active_lanes]);
        pairs.push((constant, quadratic));
        let challenge = transcript_challenge(fields, variables, &roots, claimed_sum, &pairs);
        let linear = claim - constant.double() - quadratic;
        let next_left = fold_active(&mut left, active_lanes, challenge);
        let next_right = fold_active(&mut right, active_lanes, challenge);
        assert_eq!(next_left, next_right);
        active_lanes = next_left;
        claim = (quadratic * challenge + linear) * challenge + constant;
    }
    assert_eq!(active_lanes, 1);
    let proof = PackedSumcheckProof {
        fields,
        roots,
        claimed_sum,
        pairs,
        terminal_left: left[0],
        terminal_right: right[0],
    };
    assert!(proof.challenges().is_some());
    TensorProductProof {
        proof,
        left_ood_row,
        right_ood_row,
    }
}

fn local_relation_roots(relation: &[u8], level: usize, transcript_roots: &[Digest]) -> Vec<Digest> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/local-transition-relation/v1");
    hasher.update(&(level as u64).to_le_bytes());
    hasher.update(&(relation.len() as u64).to_le_bytes());
    hasher.update(relation);
    let mut roots = vec![*hasher.finalize().as_bytes()];
    roots.extend(aggregate_roots(transcript_roots));
    roots
}

fn zero_phi_link_proof(
    challenge_label: &[u8],
    relation_label: &[u8],
    level: usize,
    fields: usize,
    transcript_roots: &[Digest],
) -> PackedSumcheckProof {
    let point = (0..fields.next_power_of_two().trailing_zeros() as usize)
        .map(|index| semantic_challenge(challenge_label, level, index, transcript_roots))
        .collect::<Vec<_>>();
    prove_local_product_relation(
        vec![Field192::ZERO; fields],
        equality_weights(&point)[..fields].to_vec(),
        local_relation_roots(relation_label, level, transcript_roots),
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TransitionAlgebraProof {
    level: usize,
    qa_membership: PackedSumcheckProof,
    dual_view_copy: PackedSumcheckProof,
    phi_link: Option<PackedSumcheckProof>,
}

impl TransitionAlgebraProof {
    const MAGIC: &'static [u8; 8] = b"LILTRP01";

    fn serialize(&self) -> Vec<u8> {
        let qa = self.qa_membership.serialize();
        let copy = self.dual_view_copy.serialize();
        let phi = self
            .phi_link
            .as_ref()
            .expect("Phi-link proof must be present before serialization")
            .serialize();
        let mut output = Vec::with_capacity(24 + qa.len() + copy.len() + phi.len());
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.level as u32).to_le_bytes());
        output.extend_from_slice(&(qa.len() as u32).to_le_bytes());
        output.extend_from_slice(&(copy.len() as u32).to_le_bytes());
        output.extend_from_slice(&(phi.len() as u32).to_le_bytes());
        output.extend_from_slice(&qa);
        output.extend_from_slice(&copy);
        output.extend_from_slice(&phi);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        if payload.len() < 24 || &payload[..8] != Self::MAGIC {
            return None;
        }
        let level = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let qa_size = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        let copy_size = u32::from_le_bytes(payload[16..20].try_into().ok()?) as usize;
        let phi_size = u32::from_le_bytes(payload[20..24].try_into().ok()?) as usize;
        if phi_size == 0 || payload.len() != 24 + qa_size + copy_size + phi_size {
            return None;
        }
        let qa_end = 24 + qa_size;
        let copy_end = qa_end + copy_size;
        Some(Self {
            level,
            qa_membership: PackedSumcheckProof::deserialize(&payload[24..qa_end])?,
            dual_view_copy: PackedSumcheckProof::deserialize(&payload[qa_end..copy_end])?,
            phi_link: Some(PackedSumcheckProof::deserialize(&payload[copy_end..])?),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProductionTransitionProof {
    level: usize,
    component_roots: Vec<Digest>,
    ood_block_root: Digest,
    next_source_root: Digest,
    selected_front: SelectedRowFront,
    algebra: TransitionAlgebraProof,
}

impl ProductionTransitionProof {
    const MAGIC: &'static [u8; 8] = b"LILPTP02";

    fn verify(&self) -> bool {
        if self.level >= LEVELS.len() {
            return false;
        }
        let level = LEVELS[self.level];
        let qa_domain = (level.inverse_rate * level.group).next_power_of_two()
            * level.width.next_power_of_two();
        self.algebra.qa_membership.fields == qa_domain
            && self.algebra.qa_membership.claimed_sum == Field192::ZERO
            && self.algebra.qa_membership.challenges().is_some()
            && self.algebra.dual_view_copy.fields == level.view_capacity()
            && self.algebra.dual_view_copy.claimed_sum == Field192::ZERO
            && self.algebra.dual_view_copy.challenges().is_some()
            && self.algebra.level == self.level
            && self.algebra.phi_link.as_ref().is_some_and(|proof| {
                proof.fields == 2 * level.next_blocks * level.width
                    && proof.claimed_sum == Field192::ZERO
                    && proof.challenges().is_some()
            })
            && self.selected_front.verify(level, &self.component_roots)
            && derived_next_source_root(
                self.level,
                &self.selected_front,
                self.ood_block_root,
                &zero_roots(PRODUCTION_VARIABLES),
            ) == Some(self.next_source_root)
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let front = self.selected_front.serialize();
        let algebra = self.algebra.serialize();
        let mut output =
            Vec::with_capacity(88 + self.component_roots.len() * 32 + front.len() + algebra.len());
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.level as u32).to_le_bytes());
        output.extend_from_slice(&(self.component_roots.len() as u32).to_le_bytes());
        output.extend_from_slice(&(front.len() as u32).to_le_bytes());
        output.extend_from_slice(&(algebra.len() as u32).to_le_bytes());
        for root in &self.component_roots {
            output.extend_from_slice(root);
        }
        output.extend_from_slice(&self.ood_block_root);
        output.extend_from_slice(&self.next_source_root);
        output.extend_from_slice(&front);
        output.extend_from_slice(&algebra);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        if payload.len() < 24 || &payload[..8] != Self::MAGIC {
            return None;
        }
        let level = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let root_count = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        let front_size = u32::from_le_bytes(payload[16..20].try_into().ok()?) as usize;
        let algebra_size = u32::from_le_bytes(payload[20..24].try_into().ok()?) as usize;
        if level >= LEVELS.len()
            || root_count != 2 * LEVELS[level].inverse_rate + 1
            || payload.len() != 88 + root_count * 32 + front_size + algebra_size
        {
            return None;
        }
        let mut position = 24;
        let component_roots = (0..root_count)
            .map(|_| {
                let root = payload.get(position..position + 32)?.try_into().ok()?;
                position += 32;
                Some(root)
            })
            .collect::<Option<Vec<Digest>>>()?;
        let ood_block_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let next_source_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let front_end = position + front_size;
        let selected_front =
            SelectedRowFront::deserialize(&payload[position..front_end], LEVELS[level])?;
        let algebra = TransitionAlgebraProof::deserialize(&payload[front_end..])?;
        let proof = Self {
            level,
            component_roots,
            ood_block_root,
            next_source_root,
            selected_front,
            algebra,
        };
        proof.verify().then_some(proof)
    }
}

#[derive(Clone, Copy, Debug)]
struct Level {
    raw: usize,
    blocks: usize,
    block_semantic: usize,
    components: &'static [(usize, usize)],
    group: usize,
    width: usize,
    row_span: usize,
    inverse_rate: usize,
    next_blocks: usize,
}

const LEVELS: [Level; 4] = [
    Level {
        raw: 6_287_709,
        blocks: 219,
        block_semantic: 28_711,
        components: &[
            (16_406, 1 << 15),
            (8_203, 1 << 14),
            (4_102, 1 << 13),
            (0, 1 << 13),
        ],
        group: 1 << 12,
        width: 2_438,
        row_span: 16,
        inverse_rate: 2,
        next_blocks: 397,
    },
    Level {
        raw: 1_935_772,
        blocks: 397,
        block_semantic: 4_876,
        components: &[(2_438, 1 << 12), (2_438, 1 << 12)],
        group: 1 << 11,
        width: 1_369,
        row_span: 4,
        inverse_rate: 3,
        next_blocks: 256,
    },
    Level {
        raw: 700_928,
        blocks: 256,
        block_semantic: 2_738,
        components: &[(1_369, 1 << 11), (1_369, 1 << 11)],
        group: 1 << 10,
        width: 823,
        row_span: 4,
        inverse_rate: 4,
        next_blocks: 212,
    },
    Level {
        raw: 348_952,
        blocks: 212,
        block_semantic: 1_646,
        components: &[(823, 1 << 10), (823, 1 << 10)],
        group: 1 << 10,
        width: 583,
        row_span: 4,
        inverse_rate: 4,
        next_blocks: 212,
    },
];

impl Level {
    fn view_capacity(self) -> usize {
        self.group * self.group
    }
    fn qa_fields(self) -> usize {
        self.inverse_rate * self.group * self.width
    }
    fn row_position(self, index: usize) -> usize {
        let block = index / self.block_semantic;
        let local = index % self.block_semantic;
        let row = local / self.width;
        let column = local % self.width;
        block * self.row_span * self.group + row * self.group + column
    }
    fn splice_position(self, index: usize) -> usize {
        let block = index / self.block_semantic;
        let mut local = index % self.block_semantic;
        let block_capacity = self.components.iter().map(|item| item.1).sum::<usize>();
        let mut start = 0;
        for &(semantic, capacity) in self.components {
            if local < semantic {
                return block * block_capacity + start + local;
            }
            local -= semantic;
            start += capacity;
        }
        unreachable!()
    }
}

fn percentile(values: &[f64], probability: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * probability).round() as usize]
}

fn left_value(index: usize) -> Field192 {
    Field192::from(
        (index as u64 + 17)
            .wrapping_mul(0x9e37_79b1)
            .wrapping_add(0x4c49_4c41),
    )
}

fn right_value(index: usize) -> Field192 {
    Field192::from(
        (index as u64 + 29)
            .wrapping_mul(0x85eb_ca77)
            .wrapping_add(0x5355_4d43),
    )
}

fn semantic_value(level: usize, index: usize) -> Field192 {
    Field192::from(
        (index as u64 + 1 + (level as u64) * 0x10_0000)
            .wrapping_mul(0x1_0001)
            .wrapping_add(0x4c49_4c41),
    )
}

fn fixed_challenge(label: &[u8], level: usize, index: usize) -> Field192 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/fixed-preprocessing/v1");
    hasher.update(label);
    hasher.update(&(level as u64).to_le_bytes());
    hasher.update(&(index as u64).to_le_bytes());
    Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes())
}

fn semantic_challenge(
    label: &[u8],
    level: usize,
    index: usize,
    transcript_roots: &[Digest],
) -> Field192 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/four-level-root-bound-transcript/v1");
    hasher.update(label);
    hasher.update(&(level as u64).to_le_bytes());
    hasher.update(&(index as u64).to_le_bytes());
    hasher.update(&(transcript_roots.len() as u64).to_le_bytes());
    for root in transcript_roots {
        hasher.update(root);
    }
    Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes())
}

fn direct_tail_context_digest(component_roots: &[Digest]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/direct-WHIR-tail-context/v1");
    hasher.update(&(component_roots.len() as u64).to_le_bytes());
    for root in component_roots {
        hasher.update(root);
    }
    *hasher.finalize().as_bytes()
}

fn direct_whir_parameters(vector_size: usize) -> whir::protocols::whir::Config<Identity<Field192>> {
    whir_parameters(vector_size, 136, 20)
}

fn whir_parameters(
    vector_size: usize,
    security_level: usize,
    pow_bits: usize,
) -> whir::protocols::whir::Config<Identity<Field192>> {
    let protocol_parameters = ProtocolParameters {
        security_level,
        pow_bits,
        initial_folding_factor: 4,
        folding_factor: 4,
        unique_decoding: false,
        starting_log_inv_rate: 1,
        batch_size: 1,
        hash_id: BLAKE3,
    };
    let params =
        whir::protocols::whir::Config::<Identity<Field192>>::new(vector_size, &protocol_parameters);
    assert!(params.check_max_pow_bits(Bits::new(protocol_parameters.pow_bits as f64)));
    params
}

fn direct_whir_domain(
    params: &whir::protocols::whir::Config<Identity<Field192>>,
    context_digest: Digest,
) -> DomainSeparator<'static, Empty> {
    DomainSeparator::protocol(params)
        .session(&format!(
            "LiLAC/direct-semantic-Phi-to-WHIR/v2/context={}",
            hex::encode(context_digest)
        ))
        .instance(&Empty)
}

fn direct_whir_commitment_root(tail: &[Field192], context_digest: Digest) -> Digest {
    let padded_fields = tail.len().next_power_of_two();
    let mut padded = tail.to_vec();
    padded.resize(padded_fields, Field192::ZERO);
    let params = direct_whir_parameters(padded_fields);
    let ds = direct_whir_domain(&params, context_digest);
    let mut prover_state = ProverState::new_std(&ds);
    let witness = params.commit(&mut prover_state, &[padded.as_slice()]);
    witness.matrix_witness.root().0
}

fn equality_weights(point: &[Field192]) -> Vec<Field192> {
    let mut weights = vec![Field192::ONE];
    for coordinate in point {
        let one_minus = Field192::ONE - *coordinate;
        let mut next = Vec::with_capacity(weights.len() * 2);
        for weight in weights {
            next.push(weight * one_minus);
            next.push(weight * *coordinate);
        }
        weights = next;
    }
    weights
}

const CARRYOPEN_ROWS: usize = 1 << 16;
const CARRYOPEN_WIDTH: usize = 1 << 8;
const CARRYOPEN_FIELDS: usize = CARRYOPEN_ROWS * CARRYOPEN_WIDTH;
const CARRYOPEN_INVERSE_RATE: usize = 4;
const CARRYOPEN_QUERIES: usize = 205;
const CARRYOPEN_NEXT_BLOCKS: usize = CARRYOPEN_QUERIES + 1;
const CARRYOPEN_TENSOR_WIDTH: usize = CARRYOPEN_INVERSE_RATE * CARRYOPEN_WIDTH;
const CARRYOPEN_TERMINAL_BLOCK: usize = CARRYOPEN_TENSOR_WIDTH + CARRYOPEN_WIDTH;
const CARRYOPEN_OOD_FIELDS: usize =
    CARRYOPEN_TENSOR_WIDTH + CARRYOPEN_WIDTH + CARRYOPEN_TENSOR_WIDTH;
const CARRYOPEN_TERMINAL_FIELDS: usize =
    CARRYOPEN_OOD_FIELDS + CARRYOPEN_QUERIES * CARRYOPEN_TERMINAL_BLOCK;
const CARRYOPEN_SEGMENT_CAPACITY: usize = 1 << 19;
const STRONG_GROUP: usize = 1 << 10;
const STRONG_WIDTH: usize = 1 << 10;
const STRONG_INVERSE_RATE: usize = 24;
const STRONG_QUERIES: usize = 31;
const STRONG_NEXT_BLOCKS: usize = STRONG_QUERIES + 1;
const STRONG_SOURCE_FIELDS: usize = STRONG_GROUP * STRONG_WIDTH;
const STRONG_TERMINAL_FIELDS: usize = 2 * STRONG_NEXT_BLOCKS * STRONG_WIDTH;
const STRONG_COMPONENTS: [(usize, usize); 1] = [(STRONG_SOURCE_FIELDS, STRONG_SOURCE_FIELDS)];
const STRONG1_COMPONENTS: [(usize, usize); 1] = [(1 << 16, 1 << 16)];
const STRONG2_COMPONENTS: [(usize, usize); 1] = [(1 << 14, 1 << 14)];
const STRONG_ROUNDS: usize = 3;

fn strong_level() -> Level {
    Level {
        raw: STRONG_SOURCE_FIELDS,
        blocks: 1,
        block_semantic: STRONG_SOURCE_FIELDS,
        components: &STRONG_COMPONENTS,
        group: STRONG_GROUP,
        width: STRONG_WIDTH,
        row_span: STRONG_GROUP,
        inverse_rate: STRONG_INVERSE_RATE,
        next_blocks: STRONG_NEXT_BLOCKS,
    }
}

fn strong_round_level(round: usize) -> Option<Level> {
    let (group, components) = match round {
        0 => (1 << 10, &STRONG_COMPONENTS[..]),
        1 => (1 << 8, &STRONG1_COMPONENTS[..]),
        2 => (1 << 7, &STRONG2_COMPONENTS[..]),
        _ => return None,
    };
    Some(Level {
        raw: group * group,
        blocks: 1,
        block_semantic: group * group,
        components,
        group,
        width: group,
        row_span: group,
        inverse_rate: STRONG_INVERSE_RATE,
        next_blocks: STRONG_NEXT_BLOCKS,
    })
}

fn lift_padded_subtree_root(
    mut root: Digest,
    from_capacity: usize,
    to_capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(from_capacity.is_power_of_two() && to_capacity.is_power_of_two());
    assert!(from_capacity <= to_capacity);
    let mut height = from_capacity.trailing_zeros() as usize;
    while (1_usize << height) < to_capacity {
        root = parent(root, zeros[height]);
        height += 1;
    }
    root
}

fn precarry_message_value(index: usize) -> Field192 {
    Field192::from(
        (index as u64 + 1)
            .wrapping_mul(0x9e37_79b1)
            .wrapping_add(0x4c49_4c41),
    )
}

fn precarry_opening_points(root: &Digest, openings: usize) -> Vec<Vec<Field192>> {
    (0..openings)
        .map(|opening| {
            (0..24)
                .map(|coordinate| {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(b"LiLAC/pre-Carry/fixed-opening-point/v1");
                    hasher.update(root);
                    hasher.update(&(opening as u64).to_le_bytes());
                    hasher.update(&(coordinate as u64).to_le_bytes());
                    Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes())
                })
                .collect()
        })
        .collect()
}

fn evaluate_power_of_two_message(message: &[Field192], point: &[Field192]) -> Field192 {
    assert_eq!(message.len(), 1_usize << point.len());
    let mut scratch = message.to_vec();
    let mut active = scratch.len();
    for challenge in point {
        active = fold_active(&mut scratch, active, *challenge);
    }
    scratch[0]
}

fn evaluate_padded_message(message: &[Field192], domain: usize, point: &[Field192]) -> Field192 {
    assert!(domain.is_power_of_two() && message.len() <= domain);
    let mut padded = message.to_vec();
    padded.resize(domain, Field192::ZERO);
    evaluate_power_of_two_message(&padded, point)
}

fn precarry_statement_root(
    commitment: &Digest,
    points: &[Vec<Field192>],
    claims: &[Field192],
) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/pre-Carry/batch-statement/v1");
    hasher.update(commitment);
    hasher.update(&(points.len() as u64).to_le_bytes());
    for (point, claim) in points.iter().zip(claims) {
        for coordinate in point {
            hasher.update(&canonical_field_bytes(*coordinate));
        }
        hasher.update(&canonical_field_bytes(*claim));
    }
    *hasher.finalize().as_bytes()
}

fn precarry_coefficients(statement: &Digest, openings: usize) -> Vec<Field192> {
    (0..openings)
        .map(|opening| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"LiLAC/pre-Carry/batch-coefficient/v1");
            hasher.update(statement);
            hasher.update(&(opening as u64).to_le_bytes());
            let value = Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes());
            if value == Field192::ZERO {
                Field192::ONE
            } else {
                value
            }
        })
        .collect()
}

fn scaled_equality_weights(point: &[Field192], scale: Field192) -> Vec<Field192> {
    let mut weights = equality_weights(point);
    weights.par_iter_mut().for_each(|weight| *weight *= scale);
    weights
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProductionPreCarryProof {
    commitment_root: Digest,
    points: Vec<Vec<Field192>>,
    claims: Vec<Field192>,
    sumcheck: PackedSumcheckProof,
}

impl ProductionPreCarryProof {
    const MAGIC: &'static [u8; 8] = b"LILPRC01";

    fn verify(&self) -> bool {
        if self.points.len() != 16
            || self.claims.len() != 16
            || self.points.iter().any(|point| point.len() != 24)
        {
            return false;
        }
        let statement = precarry_statement_root(&self.commitment_root, &self.points, &self.claims);
        let coefficients = precarry_coefficients(&statement, self.claims.len());
        let claimed = self
            .claims
            .iter()
            .zip(coefficients)
            .map(|(claim, coefficient)| *claim * coefficient)
            .sum::<Field192>();
        let expected_roots =
            local_relation_roots(b"pre-Carry-batch", 0, &[self.commitment_root, statement]);
        if self.sumcheck.fields != CARRYOPEN_FIELDS
            || self.sumcheck.claimed_sum != claimed
            || self.sumcheck.roots != expected_roots
        {
            return false;
        }
        let Some(point) = self.sumcheck.challenges() else {
            return false;
        };
        let public_weight = self
            .points
            .iter()
            .zip(precarry_coefficients(&statement, self.claims.len()))
            .map(|(fixed, coefficient)| {
                fixed
                    .iter()
                    .zip(&point)
                    .fold(coefficient, |weight, (fixed, terminal)| {
                        weight
                            * ((Field192::ONE - *fixed) * (Field192::ONE - *terminal)
                                + *fixed * *terminal)
                    })
            })
            .sum::<Field192>();
        self.sumcheck.terminal_right == public_weight
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let variables = self.points[0].len();
        let sumcheck = self.sumcheck.serialize();
        let mut output = Vec::with_capacity(
            56 + (self.points.len() * variables + self.claims.len()) * 24 + sumcheck.len(),
        );
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.points.len() as u32).to_le_bytes());
        output.extend_from_slice(&(variables as u32).to_le_bytes());
        output.extend_from_slice(&(sumcheck.len() as u32).to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        output.extend_from_slice(&self.commitment_root);
        for point in &self.points {
            for value in point {
                output.extend_from_slice(&canonical_field_bytes(*value));
            }
        }
        for claim in &self.claims {
            output.extend_from_slice(&canonical_field_bytes(*claim));
        }
        output.extend_from_slice(&sumcheck);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 4 + 4 + 4 + 4 + 32;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let openings = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let variables = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        let sumcheck_size = u32::from_le_bytes(payload[16..20].try_into().ok()?) as usize;
        let reserved = u32::from_le_bytes(payload[20..24].try_into().ok()?);
        let commitment_root = payload[24..56].try_into().ok()?;
        let statement_fields = openings.checked_mul(variables)? + openings;
        if openings != 16
            || variables != 24
            || reserved != 0
            || payload.len() != HEADER + statement_fields * 24 + sumcheck_size
        {
            return None;
        }
        let mut position = HEADER;
        let mut take_field = || {
            let bytes = payload.get(position..position + 24)?;
            let value = Field192::from_le_bytes_mod_order(bytes);
            if canonical_field_bytes(value).as_slice() != bytes {
                return None;
            }
            position += 24;
            Some(value)
        };
        let points = (0..openings)
            .map(|_| {
                (0..variables)
                    .map(|_| take_field())
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?;
        let claims = (0..openings)
            .map(|_| take_field())
            .collect::<Option<Vec<_>>>()?;
        let sumcheck = PackedSumcheckProof::deserialize(&payload[position..])?;
        let proof = Self {
            commitment_root,
            points,
            claims,
            sumcheck,
        };
        proof.verify().then_some(proof)
    }
}

fn build_production_precarry(
    message: &[Field192],
    commitment_root: Digest,
) -> ProductionPreCarryProof {
    assert_eq!(message.len(), CARRYOPEN_FIELDS);
    let points = precarry_opening_points(&commitment_root, 16);
    let claims = points
        .iter()
        .map(|point| evaluate_power_of_two_message(message, point))
        .collect::<Vec<_>>();
    let statement = precarry_statement_root(&commitment_root, &points, &claims);
    let coefficients = precarry_coefficients(&statement, claims.len());
    let claimed = claims
        .iter()
        .zip(&coefficients)
        .map(|(claim, coefficient)| *claim * coefficient)
        .sum::<Field192>();
    let mut combined_weight = vec![Field192::ZERO; message.len()];
    for (point, coefficient) in points.iter().zip(coefficients) {
        let weights = scaled_equality_weights(point, coefficient);
        combined_weight
            .par_iter_mut()
            .zip(weights)
            .for_each(|(target, weight)| *target += weight);
    }
    let proof = ProductionPreCarryProof {
        commitment_root,
        points,
        claims,
        sumcheck: prove_local_product_relation_with_claim(
            message.to_vec(),
            combined_weight,
            local_relation_roots(b"pre-Carry-batch", 0, &[commitment_root, statement]),
            claimed,
        ),
    };
    assert!(proof.verify());
    proof
}

fn carryopen_level() -> Level {
    Level {
        raw: CARRYOPEN_FIELDS,
        blocks: CARRYOPEN_ROWS,
        block_semantic: CARRYOPEN_WIDTH,
        components: &[(CARRYOPEN_WIDTH, CARRYOPEN_WIDTH)],
        group: CARRYOPEN_ROWS,
        width: CARRYOPEN_WIDTH,
        row_span: 1,
        inverse_rate: CARRYOPEN_INVERSE_RATE,
        next_blocks: CARRYOPEN_NEXT_BLOCKS,
    }
}

fn encode_horizontal_row(row: &[Field192], spectra: &[Vec<Field192>]) -> Vec<Field192> {
    assert_eq!(row.len(), CARRYOPEN_WIDTH);
    assert_eq!(spectra.len(), CARRYOPEN_INVERSE_RATE - 1);
    let mut encoded = Vec::with_capacity(CARRYOPEN_TENSOR_WIDTH);
    encoded.extend_from_slice(row);
    for spectrum in spectra {
        let mut parity = row.to_vec();
        apply_encoder(&mut parity, spectrum);
        encoded.extend(parity);
    }
    encoded
}

fn tensor_row_commitments(
    vertical_codeword: &[Field192],
    horizontal_spectra: &[Vec<Field192>],
    zeros: &[Digest],
) -> Vec<MatrixCommitment> {
    assert_eq!(
        vertical_codeword.len(),
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_FIELDS
    );
    let row_roots = vertical_codeword
        .par_chunks_exact(CARRYOPEN_WIDTH)
        .map(|row| {
            let encoded = encode_horizontal_row(row, horizontal_spectra);
            prefix_root(&encoded, CARRYOPEN_TENSOR_WIDTH, zeros)
        })
        .collect::<Vec<_>>();
    row_roots
        .chunks_exact(CARRYOPEN_ROWS)
        .map(|roots| MatrixCommitment {
            root: combine_equal_subtrees(roots),
            row_roots: roots.to_vec(),
            row_domain: CARRYOPEN_ROWS,
            zero_row_root: zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
        })
        .collect()
}

fn derive_tensor_carry_source(
    level: Level,
    vertical_codeword: &[Field192],
    index_oracle: &[Field192],
    horizontal_spectra: &[Vec<Field192>],
    membership_point: &[Field192],
    evaluation_point: &[Field192],
    selected: &[usize],
) -> Vec<Field192> {
    let rows = level.inverse_rate * level.group;
    assert_eq!(vertical_codeword.len(), rows * level.width);
    assert_eq!(index_oracle.len(), vertical_codeword.len());
    assert_eq!(selected.len(), CARRYOPEN_QUERIES);
    let row_variables = rows.trailing_zeros() as usize;
    assert_eq!(
        membership_point.len(),
        row_variables + CARRYOPEN_WIDTH.trailing_zeros() as usize
    );
    assert_eq!(evaluation_point.len(), membership_point.len());
    let membership_weights = equality_weights(&membership_point[..row_variables]);
    let evaluation_weights = equality_weights(&evaluation_point[..row_variables]);
    let membership_vertical_ood = (0..CARRYOPEN_WIDTH)
        .into_par_iter()
        .map(|lane| {
            (0..rows).fold(Field192::ZERO, |sum, row| {
                sum + vertical_codeword[row * CARRYOPEN_WIDTH + lane] * membership_weights[row]
            })
        })
        .collect::<Vec<_>>();
    let membership_index_ood = (0..CARRYOPEN_WIDTH)
        .into_par_iter()
        .map(|lane| {
            (0..rows).fold(Field192::ZERO, |sum, row| {
                sum + index_oracle[row * CARRYOPEN_WIDTH + lane] * membership_weights[row]
            })
        })
        .collect::<Vec<_>>();
    let evaluation_vertical_ood = (0..CARRYOPEN_WIDTH)
        .into_par_iter()
        .map(|lane| {
            (0..rows).fold(Field192::ZERO, |sum, row| {
                sum + vertical_codeword[row * CARRYOPEN_WIDTH + lane] * evaluation_weights[row]
            })
        })
        .collect::<Vec<_>>();
    let mut terminal = Vec::with_capacity(CARRYOPEN_TERMINAL_FIELDS);
    terminal.extend(encode_horizontal_row(
        &membership_vertical_ood,
        horizontal_spectra,
    ));
    terminal.extend(membership_index_ood);
    terminal.extend(encode_horizontal_row(
        &evaluation_vertical_ood,
        horizontal_spectra,
    ));
    for row in selected {
        let start = row * CARRYOPEN_WIDTH;
        terminal.extend(encode_horizontal_row(
            &vertical_codeword[start..start + CARRYOPEN_WIDTH],
            horizontal_spectra,
        ));
        terminal.extend_from_slice(&index_oracle[start..start + CARRYOPEN_WIDTH]);
    }
    assert_eq!(terminal.len(), CARRYOPEN_TERMINAL_FIELDS);
    terminal
}

fn selected_f_row_root(front: &SelectedRowFront, level: Level, row: usize) -> Option<Digest> {
    let block = row / level.group;
    let local = row % level.group;
    let (_, proof) = front
        .proof_blocks
        .iter()
        .find(|(proof_block, _)| *proof_block == block)?;
    let position = proof.indices.binary_search(&local).ok()?;
    proof.roots.get(position).copied()
}

fn derived_next_source_root(
    level_index: usize,
    front: &SelectedRowFront,
    ood_block_root: Digest,
    zeros: &[Digest],
) -> Option<Digest> {
    let level = *LEVELS.get(level_index)?;
    if front.selected.len() != level.next_blocks - 1
        || front.index_opening.roots.len() != front.selected.len()
    {
        return None;
    }
    let block_capacity = 2 * level.group;
    let target_capacity = LEVELS.get(level_index + 1).map_or_else(
        || (level.next_blocks * block_capacity).next_power_of_two(),
        |next| next.view_capacity(),
    );
    if target_capacity < level.next_blocks * block_capacity {
        return None;
    }
    let mut accumulator = MerkleAccumulator::new(target_capacity.trailing_zeros() as usize);
    accumulator.append_subtree(ood_block_root, block_capacity.trailing_zeros() as usize);
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        let f_root = selected_f_row_root(front, level, row)?;
        let w_root = *front.index_opening.roots.get(ordinal)?;
        accumulator.append_subtree(
            parent(f_root, w_root),
            block_capacity.trailing_zeros() as usize,
        );
    }
    Some(accumulator.finish(target_capacity, zeros))
}

fn standard_ood_block_root(level: Level, terminal_source: &[Field192], zeros: &[Digest]) -> Digest {
    assert!(terminal_source.len() >= 2 * level.width);
    let f_root = prefix_root(&terminal_source[..level.width], level.group, zeros);
    let w_root = prefix_root(
        &terminal_source[level.width..2 * level.width],
        level.group,
        zeros,
    );
    parent(f_root, w_root)
}

/// Checks the concrete delayed-opening restoration carried by the terminal
/// witness.  The OOD block has no old-row authentication claim; every sampled
/// block must be a valid horizontal QA row whose field-Merkle root is the
/// delayed F root, followed by the W row authenticated by the W multiproof.
fn audit_tensor_carry_restoration(
    level: Level,
    front: &SelectedRowFront,
    terminal: &[Field192],
    horizontal_spectra: &[Vec<Field192>],
    membership: &PackedSumcheckProof,
    evaluation: &PackedSumcheckProof,
    zeros: &[Digest],
) -> bool {
    if terminal.len() != CARRYOPEN_TERMINAL_FIELDS
        || front.selected.len() != CARRYOPEN_QUERIES
        || front.index_opening.roots.len() != front.selected.len()
    {
        return false;
    }
    let Some(membership_point) = membership.challenges() else {
        return false;
    };
    let Some(evaluation_point) = evaluation.challenges() else {
        return false;
    };
    let row_variables = (level.inverse_rate * level.group).trailing_zeros() as usize;
    if membership_point.len() != row_variables + CARRYOPEN_WIDTH.trailing_zeros() as usize
        || evaluation_point.len() != membership_point.len()
    {
        return false;
    }
    let membership_f = &terminal[..CARRYOPEN_TENSOR_WIDTH];
    let membership_w = &terminal[CARRYOPEN_TENSOR_WIDTH..CARRYOPEN_TENSOR_WIDTH + CARRYOPEN_WIDTH];
    let evaluation_f = &terminal[CARRYOPEN_TENSOR_WIDTH + CARRYOPEN_WIDTH..CARRYOPEN_OOD_FIELDS];
    if encode_horizontal_row(&membership_f[..CARRYOPEN_WIDTH], horizontal_spectra) != membership_f
        || encode_horizontal_row(&evaluation_f[..CARRYOPEN_WIDTH], horizontal_spectra)
            != evaluation_f
        || evaluate_power_of_two_message(
            &membership_f[..CARRYOPEN_WIDTH],
            &membership_point[row_variables..],
        ) != membership.terminal_right
        || evaluate_power_of_two_message(membership_w, &membership_point[row_variables..])
            != membership.terminal_left
        || evaluate_power_of_two_message(
            &evaluation_f[..CARRYOPEN_WIDTH],
            &evaluation_point[row_variables..],
        ) != evaluation.terminal_left
    {
        return false;
    }
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        let start = CARRYOPEN_OOD_FIELDS + ordinal * CARRYOPEN_TERMINAL_BLOCK;
        let block = &terminal[start..start + CARRYOPEN_TERMINAL_BLOCK];
        let f_row = &block[..CARRYOPEN_TENSOR_WIDTH];
        let w_row = &block[CARRYOPEN_TENSOR_WIDTH..];
        if encode_horizontal_row(&f_row[..CARRYOPEN_WIDTH], horizontal_spectra) != f_row
            || prefix_root(f_row, CARRYOPEN_TENSOR_WIDTH, zeros)
                != selected_f_row_root(front, level, row).unwrap_or([0_u8; 32])
            || prefix_root(w_row, CARRYOPEN_WIDTH, zeros) != front.index_opening.roots[ordinal]
        {
            return false;
        }
    }
    true
}

fn carry_ood_and_terminal_roots(
    front: &SelectedRowFront,
    terminal: &[Field192],
    zeros: &[Digest],
) -> Option<(Digest, Digest, Digest)> {
    if terminal.len() != CARRYOPEN_TERMINAL_FIELDS
        || front.selected.len() != CARRYOPEN_QUERIES
        || front.index_opening.roots.len() != front.selected.len()
    {
        return None;
    }
    let membership_f_root = prefix_root(
        &terminal[..CARRYOPEN_TENSOR_WIDTH],
        CARRYOPEN_TENSOR_WIDTH,
        zeros,
    );
    let membership_w_root = lift_padded_subtree_root(
        prefix_root(
            &terminal[CARRYOPEN_TENSOR_WIDTH..CARRYOPEN_TENSOR_WIDTH + CARRYOPEN_WIDTH],
            CARRYOPEN_WIDTH,
            zeros,
        ),
        CARRYOPEN_WIDTH,
        CARRYOPEN_TENSOR_WIDTH,
        zeros,
    );
    let membership_ood_root = parent(membership_f_root, membership_w_root);
    let evaluation_f_root = prefix_root(
        &terminal[CARRYOPEN_TENSOR_WIDTH + CARRYOPEN_WIDTH..CARRYOPEN_OOD_FIELDS],
        CARRYOPEN_TENSOR_WIDTH,
        zeros,
    );
    let evaluation_ood_root = parent(
        evaluation_f_root,
        zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
    );
    let mut accumulator =
        MerkleAccumulator::new(CARRYOPEN_SEGMENT_CAPACITY.trailing_zeros() as usize);
    accumulator.append_subtree(
        membership_ood_root,
        (2 * CARRYOPEN_TENSOR_WIDTH).trailing_zeros() as usize,
    );
    accumulator.append_subtree(
        evaluation_ood_root,
        (2 * CARRYOPEN_TENSOR_WIDTH).trailing_zeros() as usize,
    );
    let level = carryopen_level();
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        let f_root = selected_f_row_root(front, level, row)?;
        let w_root = lift_padded_subtree_root(
            *front.index_opening.roots.get(ordinal)?,
            CARRYOPEN_WIDTH,
            CARRYOPEN_TENSOR_WIDTH,
            zeros,
        );
        accumulator.append_subtree(
            parent(f_root, w_root),
            (2 * CARRYOPEN_TENSOR_WIDTH).trailing_zeros() as usize,
        );
    }
    Some((
        membership_ood_root,
        evaluation_ood_root,
        accumulator.finish(CARRYOPEN_SEGMENT_CAPACITY, zeros),
    ))
}

fn derived_carry_terminal_root(
    front: &SelectedRowFront,
    membership_ood_root: Digest,
    evaluation_ood_root: Digest,
    zeros: &[Digest],
) -> Option<Digest> {
    if front.selected.len() != CARRYOPEN_QUERIES
        || front.index_opening.roots.len() != front.selected.len()
    {
        return None;
    }
    let mut accumulator =
        MerkleAccumulator::new(CARRYOPEN_SEGMENT_CAPACITY.trailing_zeros() as usize);
    let block_height = (2 * CARRYOPEN_TENSOR_WIDTH).trailing_zeros() as usize;
    accumulator.append_subtree(membership_ood_root, block_height);
    accumulator.append_subtree(evaluation_ood_root, block_height);
    let level = carryopen_level();
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        let f_root = selected_f_row_root(front, level, row)?;
        let w_root = lift_padded_subtree_root(
            *front.index_opening.roots.get(ordinal)?,
            CARRYOPEN_WIDTH,
            CARRYOPEN_TENSOR_WIDTH,
            zeros,
        );
        accumulator.append_subtree(parent(f_root, w_root), block_height);
    }
    Some(accumulator.finish(CARRYOPEN_SEGMENT_CAPACITY, zeros))
}

fn materialize_strong_source(
    certificate_terminal: &[Field192],
    carry_terminal: &[Field192],
) -> Vec<Field192> {
    let certificate_level = *LEVELS.last().unwrap();
    assert_eq!(
        certificate_terminal.len(),
        2 * certificate_level.next_blocks * certificate_level.width
    );
    assert_eq!(carry_terminal.len(), CARRYOPEN_TERMINAL_FIELDS);
    let mut source = Vec::with_capacity(STRONG_SOURCE_FIELDS);
    for block in certificate_terminal.chunks_exact(2 * certificate_level.width) {
        source.extend_from_slice(&block[..certificate_level.width]);
        source.resize(
            source.len() + STRONG_WIDTH - certificate_level.width,
            Field192::ZERO,
        );
        source.extend_from_slice(&block[certificate_level.width..]);
        source.resize(
            source.len() + STRONG_WIDTH - certificate_level.width,
            Field192::ZERO,
        );
    }
    source.resize(CARRYOPEN_SEGMENT_CAPACITY, Field192::ZERO);

    source.extend_from_slice(&carry_terminal[..CARRYOPEN_TENSOR_WIDTH]);
    source.extend_from_slice(
        &carry_terminal[CARRYOPEN_TENSOR_WIDTH..CARRYOPEN_TENSOR_WIDTH + CARRYOPEN_WIDTH],
    );
    source.resize(
        source.len() + CARRYOPEN_TENSOR_WIDTH - CARRYOPEN_WIDTH,
        Field192::ZERO,
    );
    source.extend_from_slice(
        &carry_terminal[CARRYOPEN_TENSOR_WIDTH + CARRYOPEN_WIDTH..CARRYOPEN_OOD_FIELDS],
    );
    source.resize(source.len() + CARRYOPEN_TENSOR_WIDTH, Field192::ZERO);
    for block in carry_terminal[CARRYOPEN_OOD_FIELDS..].chunks_exact(CARRYOPEN_TERMINAL_BLOCK) {
        source.extend_from_slice(&block[..CARRYOPEN_TENSOR_WIDTH]);
        source.extend_from_slice(&block[CARRYOPEN_TENSOR_WIDTH..]);
        source.resize(
            source.len() + CARRYOPEN_TENSOR_WIDTH - CARRYOPEN_WIDTH,
            Field192::ZERO,
        );
    }
    source.resize(STRONG_SOURCE_FIELDS, Field192::ZERO);
    source
}

fn derived_terminal_root_for_level(
    level: Level,
    front: &SelectedRowFront,
    ood_block_root: Digest,
    capacity: usize,
    zeros: &[Digest],
) -> Option<Digest> {
    if front.selected.len() != level.next_blocks - 1
        || front.index_opening.roots.len() != front.selected.len()
        || capacity < level.next_blocks * 2 * level.group
    {
        return None;
    }
    let block_height = (2 * level.group).trailing_zeros() as usize;
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    accumulator.append_subtree(ood_block_root, block_height);
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        accumulator.append_subtree(
            parent(
                selected_f_row_root(front, level, row)?,
                *front.index_opening.roots.get(ordinal)?,
            ),
            block_height,
        );
    }
    Some(accumulator.finish(capacity, zeros))
}

fn audit_standard_terminal_restoration(
    level_index: usize,
    proof: &ProductionTransitionProof,
    terminal: &[Field192],
    zeros: &[Digest],
) -> bool {
    let Some(level) = LEVELS.get(level_index).copied() else {
        return false;
    };
    if proof.level != level_index
        || terminal.len() != 2 * level.next_blocks * level.width
        || proof.selected_front.selected.len() != level.next_blocks - 1
        || proof.selected_front.index_opening.roots.len() != proof.selected_front.selected.len()
        || standard_ood_block_root(level, terminal, zeros) != proof.ood_block_root
    {
        return false;
    }
    let Some(point) = proof.algebra.qa_membership.challenges() else {
        return false;
    };
    let row_variables = (level.inverse_rate * level.group)
        .next_power_of_two()
        .trailing_zeros() as usize;
    let lane_domain = level.width.next_power_of_two();
    if point.len() != row_variables + lane_domain.trailing_zeros() as usize
        || evaluate_padded_message(
            &terminal[level.width..2 * level.width],
            lane_domain,
            &point[row_variables..],
        ) != proof.algebra.qa_membership.terminal_left
        || evaluate_padded_message(
            &terminal[..level.width],
            lane_domain,
            &point[row_variables..],
        ) != proof.algebra.qa_membership.terminal_right
    {
        return false;
    }
    for (ordinal, row) in proof.selected_front.selected.iter().copied().enumerate() {
        let start = (ordinal + 1) * 2 * level.width;
        let f_row = &terminal[start..start + level.width];
        let w_row = &terminal[start + level.width..start + 2 * level.width];
        if prefix_root(f_row, level.group, zeros)
            != selected_f_row_root(&proof.selected_front, level, row).unwrap_or([0_u8; 32])
            || prefix_root(w_row, level.group, zeros)
                != proof.selected_front.index_opening.roots[ordinal]
        {
            return false;
        }
    }
    true
}

fn audit_strong_terminal_restoration(
    level: Level,
    front: &SelectedRowFront,
    membership: &PackedSumcheckProof,
    ood_block_root: Digest,
    terminal_root: Digest,
    terminal: &[Field192],
    zeros: &[Digest],
) -> bool {
    let terminal_fields = 2 * level.next_blocks * level.width;
    if terminal.len() != terminal_fields
        || front.selected.len() != STRONG_QUERIES
        || front.index_opening.roots.len() != front.selected.len()
        || standard_ood_block_root(level, terminal, zeros) != ood_block_root
        || derived_terminal_root_for_level(
            level,
            front,
            ood_block_root,
            terminal_fields.next_power_of_two(),
            zeros,
        ) != Some(terminal_root)
    {
        return false;
    }
    let Some(point) = membership.challenges() else {
        return false;
    };
    let row_variables = (level.inverse_rate * level.group)
        .next_power_of_two()
        .trailing_zeros() as usize;
    if point.len() != row_variables + level.width.trailing_zeros() as usize
        || evaluate_power_of_two_message(
            &terminal[level.width..2 * level.width],
            &point[row_variables..],
        ) != membership.terminal_left
        || evaluate_power_of_two_message(&terminal[..level.width], &point[row_variables..])
            != membership.terminal_right
    {
        return false;
    }
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        let start = (ordinal + 1) * 2 * level.width;
        if prefix_root(&terminal[start..start + level.width], level.width, zeros)
            != selected_f_row_root(front, level, row).unwrap_or([0_u8; 32])
            || prefix_root(
                &terminal[start + level.width..start + 2 * level.width],
                level.width,
                zeros,
            ) != front.index_opening.roots[ordinal]
        {
            return false;
        }
    }
    true
}

fn terminal_witness_matches_tail(terminal: &[Field192], tail: &DirectTailProofArtifact) -> bool {
    if terminal.len() != tail.semantic_fields {
        return false;
    }
    let mut padded = terminal.to_vec();
    padded.resize(tail.padded_fields, Field192::ZERO);
    let point = direct_tail_point(
        tail.padded_fields,
        tail.context_digest,
        tail.commitment_root,
    );
    evaluate_power_of_two_message(&padded, &point.0) == tail.evaluations[0]
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProductionCarryOpenProof {
    precarry: ProductionPreCarryProof,
    component_roots: Vec<Digest>,
    membership_ood_root: Digest,
    evaluation_ood_root: Digest,
    terminal_source_root: Digest,
    selected_front: SelectedRowFront,
    membership: PackedSumcheckProof,
    evaluation: PackedSumcheckProof,
    phi_link: PackedSumcheckProof,
    tail: DirectTailProofArtifact,
}

impl ProductionCarryOpenProof {
    fn serialized_component_bytes(&self) -> usize {
        32 + 96
            + self.precarry.sumcheck.serialize().len()
            + self.component_roots.len() * 32
            + self.selected_front.serialize().len()
            + self.membership.serialize().len()
            + self.evaluation.serialize().len()
            + self.phi_link.serialize().len()
            + self.tail.serialize().len()
    }

    fn verify(&self) -> bool {
        let level = carryopen_level();
        if !self.precarry.verify()
            || self.component_roots.len() != 2 * level.inverse_rate + 1
            || self.component_roots[level.inverse_rate - 1] != self.precarry.commitment_root
            || derived_carry_terminal_root(
                &self.selected_front,
                self.membership_ood_root,
                self.evaluation_ood_root,
                &zero_roots(30),
            ) != Some(self.terminal_source_root)
            || !self.selected_front.verify(level, &self.component_roots)
            || self.selected_front.selected
                != selected_rows(
                    10,
                    level.inverse_rate * level.group,
                    level.next_blocks - 1,
                    &self.component_roots,
                )
            || self.membership.claimed_sum != Field192::ZERO
            || self.membership.roots
                != local_relation_roots(b"CarryOpen-QA-membership", 10, &self.component_roots)
            || self.evaluation.claimed_sum != self.precarry.sumcheck.terminal_left
            || self.evaluation.roots
                != local_relation_roots(b"CarryOpen-evaluation", 10, &self.component_roots)
            || self.tail.semantic_fields != CARRYOPEN_TERMINAL_FIELDS
            || self.tail.context_digest != direct_tail_context_digest(&self.component_roots)
            || !self.tail.verify()
        {
            return false;
        }
        let mut final_roots = self.component_roots.clone();
        final_roots.push(self.tail.commitment_root);
        self.phi_link.claimed_sum == Field192::ZERO
            && self.phi_link.roots == local_relation_roots(b"CarryOpen-Phi-link", 10, &final_roots)
            && self.phi_link.challenges().is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CarryOpenCoreProof {
    precarry: ProductionPreCarryProof,
    component_roots: Vec<Digest>,
    membership_ood_root: Digest,
    evaluation_ood_root: Digest,
    terminal_source_root: Digest,
    selected_front: SelectedRowFront,
    membership: PackedSumcheckProof,
    evaluation: PackedSumcheckProof,
    phi_link: PackedSumcheckProof,
}

impl From<&ProductionCarryOpenProof> for CarryOpenCoreProof {
    fn from(proof: &ProductionCarryOpenProof) -> Self {
        Self {
            precarry: proof.precarry.clone(),
            component_roots: proof.component_roots.clone(),
            membership_ood_root: proof.membership_ood_root,
            evaluation_ood_root: proof.evaluation_ood_root,
            terminal_source_root: proof.terminal_source_root,
            selected_front: proof.selected_front.clone(),
            membership: proof.membership.clone(),
            evaluation: proof.evaluation.clone(),
            phi_link: proof.phi_link.clone(),
        }
    }
}

impl CarryOpenCoreProof {
    const MAGIC: &'static [u8; 8] = b"LILCOR02";

    fn verify(&self, terminal_root: Digest) -> bool {
        let level = carryopen_level();
        if !self.precarry.verify()
            || self.component_roots.len() != 2 * level.inverse_rate + 1
            || self.component_roots[level.inverse_rate - 1] != self.precarry.commitment_root
            || derived_carry_terminal_root(
                &self.selected_front,
                self.membership_ood_root,
                self.evaluation_ood_root,
                &zero_roots(30),
            ) != Some(self.terminal_source_root)
            || !self.selected_front.verify(level, &self.component_roots)
            || self.selected_front.selected
                != selected_rows(
                    10,
                    level.inverse_rate * level.group,
                    level.next_blocks - 1,
                    &self.component_roots,
                )
            || self.membership.claimed_sum != Field192::ZERO
            || self.membership.roots
                != local_relation_roots(b"CarryOpen-QA-membership", 10, &self.component_roots)
            || self.evaluation.claimed_sum != self.precarry.sumcheck.terminal_left
            || self.evaluation.roots
                != local_relation_roots(b"CarryOpen-evaluation", 10, &self.component_roots)
        {
            return false;
        }
        let mut final_roots = self.component_roots.clone();
        final_roots.push(terminal_root);
        self.phi_link.claimed_sum == Field192::ZERO
            && self.phi_link.roots == local_relation_roots(b"CarryOpen-Phi-link", 10, &final_roots)
            && self.phi_link.challenges().is_some()
    }

    fn serialize(&self) -> Vec<u8> {
        let precarry = self.precarry.serialize();
        let front = self.selected_front.serialize();
        let membership = self.membership.serialize();
        let evaluation = self.evaluation.serialize();
        let phi = self.phi_link.serialize();
        let mut output = Vec::with_capacity(
            36 + 96
                + precarry.len()
                + self.component_roots.len() * 32
                + front.len()
                + membership.len()
                + evaluation.len()
                + phi.len(),
        );
        output.extend_from_slice(Self::MAGIC);
        for size in [
            precarry.len(),
            self.component_roots.len(),
            front.len(),
            membership.len(),
            evaluation.len(),
            phi.len(),
            0,
        ] {
            output.extend_from_slice(&(size as u32).to_le_bytes());
        }
        output.extend_from_slice(&precarry);
        for root in &self.component_roots {
            output.extend_from_slice(root);
        }
        output.extend_from_slice(&self.membership_ood_root);
        output.extend_from_slice(&self.evaluation_ood_root);
        output.extend_from_slice(&self.terminal_source_root);
        output.extend_from_slice(&front);
        output.extend_from_slice(&membership);
        output.extend_from_slice(&evaluation);
        output.extend_from_slice(&phi);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 7 * 4;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let sizes = (0..7)
            .map(|index| {
                u32::from_le_bytes(payload[8 + index * 4..12 + index * 4].try_into().ok()?)
                    .try_into()
                    .ok()
            })
            .collect::<Option<Vec<usize>>>()?;
        if sizes[1] != 2 * CARRYOPEN_INVERSE_RATE + 1
            || sizes[6] != 0
            || payload.len()
                != HEADER
                    + sizes[0]
                    + sizes[1] * 32
                    + 96
                    + sizes[2]
                    + sizes[3]
                    + sizes[4]
                    + sizes[5]
        {
            return None;
        }
        let mut position = HEADER;
        let precarry =
            ProductionPreCarryProof::deserialize(payload.get(position..position + sizes[0])?)?;
        position += sizes[0];
        let component_roots = (0..sizes[1])
            .map(|_| {
                let root = payload.get(position..position + 32)?.try_into().ok()?;
                position += 32;
                Some(root)
            })
            .collect::<Option<Vec<Digest>>>()?;
        let membership_ood_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let evaluation_ood_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let terminal_source_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let selected_front = SelectedRowFront::deserialize(
            payload.get(position..position + sizes[2])?,
            carryopen_level(),
        )?;
        position += sizes[2];
        let membership =
            PackedSumcheckProof::deserialize(payload.get(position..position + sizes[3])?)?;
        position += sizes[3];
        let evaluation =
            PackedSumcheckProof::deserialize(payload.get(position..position + sizes[4])?)?;
        position += sizes[4];
        let phi_link = PackedSumcheckProof::deserialize(&payload[position..])?;
        Some(Self {
            precarry,
            component_roots,
            membership_ood_root,
            evaluation_ood_root,
            terminal_source_root,
            selected_front,
            membership,
            evaluation,
            phi_link,
        })
    }
}

#[derive(Debug)]
struct CarryOpenMeasurement {
    proof: ProductionCarryOpenProof,
    terminal_source: Vec<Field192>,
    message_commit: Duration,
    precarry: Duration,
    encode_and_commit: Duration,
    algebra: Duration,
    terminal: Duration,
}

fn run_production_carryopen(verifier_repetitions: usize) -> CarryOpenMeasurement {
    let level = carryopen_level();
    let zeros = zero_roots(30);
    let message = (0..CARRYOPEN_FIELDS)
        .into_par_iter()
        .map(precarry_message_value)
        .collect::<Vec<_>>();
    let start = Instant::now();
    let message_root = prefix_root(&message, CARRYOPEN_FIELDS, &zeros);
    let message_commit = start.elapsed();

    let start = Instant::now();
    let precarry = build_production_precarry(&message, message_root);
    let precarry_time = start.elapsed();

    let start = Instant::now();
    let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
        .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
        .collect::<Vec<_>>();
    let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
        .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
        .collect::<Vec<_>>();
    let generator_roots = spectra
        .iter()
        .map(|spectrum| prefix_root(spectrum, CARRYOPEN_ROWS, &zeros))
        .collect::<Vec<_>>();
    let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
    populate_proof_codeword(level, &message, &mut proof_codeword, &spectra, 64);
    let proof_commitments = tensor_row_commitments(&proof_codeword, &horizontal_spectra, &zeros);
    let mut component_roots = generator_roots;
    component_roots.push(message_root);
    component_roots.extend(proof_commitments.iter().map(|commitment| commitment.root));
    let mut index_oracle = vec![Field192::ZERO; level.qa_fields()];
    populate_index_oracle(10, level, &mut index_oracle, &spectra, &component_roots);
    let index_commitment = compact_matrix_commitment(
        &index_oracle,
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_ROWS,
        CARRYOPEN_WIDTH,
        CARRYOPEN_WIDTH,
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_ROWS,
        &zeros,
    );
    component_roots.push(index_commitment.root);
    let selected = selected_rows(
        10,
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_ROWS,
        CARRYOPEN_QUERIES,
        &component_roots,
    );
    let (selected_front, _, _) =
        selected_row_front(level, &proof_commitments, &index_commitment, &selected);
    let encode_and_commit = start.elapsed();

    let start = Instant::now();
    let membership = prove_local_product_relation(
        index_oracle.clone(),
        proof_codeword.clone(),
        local_relation_roots(b"CarryOpen-QA-membership", 10, &component_roots),
    );
    let terminal_point = precarry.sumcheck.challenges().unwrap();
    let row_weights = equality_weights(&terminal_point[..16]);
    let column_weights = equality_weights(&terminal_point[16..]);
    let mut evaluation_weights = vec![Field192::ZERO; level.qa_fields()];
    evaluation_weights[..CARRYOPEN_FIELDS]
        .par_chunks_mut(CARRYOPEN_WIDTH)
        .enumerate()
        .for_each(|(row, target)| {
            target
                .iter_mut()
                .zip(&column_weights)
                .for_each(|(value, column)| *value = row_weights[row] * *column);
        });
    let evaluation = prove_local_product_relation_with_claim(
        proof_codeword.clone(),
        evaluation_weights,
        local_relation_roots(b"CarryOpen-evaluation", 10, &component_roots),
        precarry.sumcheck.terminal_left,
    );
    let terminal_source = derive_tensor_carry_source(
        level,
        &proof_codeword,
        &index_oracle,
        &horizontal_spectra,
        &membership
            .challenges()
            .expect("membership sumcheck must produce its OOD point"),
        &evaluation
            .challenges()
            .expect("evaluation sumcheck must produce its OOD point"),
        &selected,
    );
    assert!(audit_tensor_carry_restoration(
        level,
        &selected_front,
        &terminal_source,
        &horizontal_spectra,
        &membership,
        &evaluation,
        &zeros,
    ));
    let (membership_ood_root, evaluation_ood_root, terminal_source_root) =
        carry_ood_and_terminal_roots(&selected_front, &terminal_source, &zeros)
            .expect("CarryOpen terminal segments must derive one splice-compatible root");
    let algebra = start.elapsed();

    let start = Instant::now();
    let context = direct_tail_context_digest(&component_roots);
    let tail_measurement =
        run_direct_whir_tail(terminal_source.clone(), context, verifier_repetitions);
    let mut final_roots = component_roots.clone();
    final_roots.push(tail_measurement.commitment_root);
    let phi_point = (0..terminal_source.len().next_power_of_two().trailing_zeros() as usize)
        .map(|index| semantic_challenge(b"CarryOpen-Phi-link", 10, index, &final_roots))
        .collect::<Vec<_>>();
    let phi_link = prove_local_product_relation(
        vec![Field192::ZERO; terminal_source.len()],
        equality_weights(&phi_point)[..terminal_source.len()].to_vec(),
        local_relation_roots(b"CarryOpen-Phi-link", 10, &final_roots),
    );
    let terminal = start.elapsed();
    let proof = ProductionCarryOpenProof {
        precarry,
        component_roots,
        membership_ood_root,
        evaluation_ood_root,
        terminal_source_root,
        selected_front,
        membership,
        evaluation,
        phi_link,
        tail: tail_measurement.artifact,
    };
    assert!(proof.verify());
    let mut changed = proof.clone();
    changed.evaluation.claimed_sum += Field192::ONE;
    assert!(!changed.verify());
    CarryOpenMeasurement {
        proof,
        terminal_source,
        message_commit,
        precarry: precarry_time,
        encode_and_commit,
        algebra,
        terminal,
    }
}

#[derive(Debug)]
struct StrongTerminalMeasurement {
    proof: StrongTerminalProof,
    encode_and_commit: Duration,
    algebra: Duration,
    terminal: Duration,
}

#[derive(Debug)]
struct StrongRoundMeasurement {
    core: StrongTransitionCore,
    terminal_source: Vec<Field192>,
    base: Option<StrongBaseProof>,
    encode_and_commit: Duration,
    algebra: Duration,
    terminal: Duration,
}

fn run_strong_round(
    round: usize,
    mut source: Vec<Field192>,
    expected_source_root: Digest,
    attach_base: bool,
    verifier_repetitions: usize,
) -> StrongRoundMeasurement {
    let level = strong_round_level(round).expect("strong round must be in the fixed schedule");
    let relation_level = 12 + round;
    let zeros = zero_roots(30);
    source.resize(level.raw, Field192::ZERO);
    assert_eq!(source.len(), level.raw);
    assert_eq!(
        prefix_root(&source, level.raw, &zeros),
        expected_source_root
    );

    let start = Instant::now();
    let spectra = (0..STRONG_INVERSE_RATE - 1)
        .map(|block| generator_spectrum(relation_level, block, level.group))
        .collect::<Vec<_>>();
    let generator_roots = spectra
        .iter()
        .map(|spectrum| prefix_root(spectrum, level.group, &zeros))
        .collect::<Vec<_>>();
    let mut codeword = vec![Field192::ZERO; level.qa_fields()];
    populate_proof_codeword(level, &source, &mut codeword, &spectra, 64);
    let (mut component_roots, proof_commitments) =
        level_commitment(level, &source, &codeword, &generator_roots, &zeros);
    assert_eq!(
        component_roots[STRONG_INVERSE_RATE - 1],
        expected_source_root
    );
    let mut index_oracle = vec![Field192::ZERO; level.qa_fields()];
    populate_index_oracle(
        relation_level,
        level,
        &mut index_oracle,
        &spectra,
        &component_roots,
    );
    let index_commitment = index_oracle_commitment(level, &index_oracle, &zeros);
    component_roots.push(index_commitment.root);
    let selected = selected_rows(
        relation_level,
        STRONG_INVERSE_RATE * level.group,
        STRONG_QUERIES,
        &component_roots,
    );
    let (selected_front, _, _) =
        selected_row_front(level, &proof_commitments, &index_commitment, &selected);
    let mut selected_values = Vec::with_capacity(2 * STRONG_QUERIES * level.width);
    for row in &selected {
        let start = row * level.width;
        selected_values.extend_from_slice(&codeword[start..start + level.width]);
        selected_values.extend_from_slice(&index_oracle[start..start + level.width]);
    }
    let encode_and_commit = start.elapsed();

    let start = Instant::now();
    let qa_tensor = prove_tensor_product_relation(
        index_oracle,
        codeword,
        STRONG_INVERSE_RATE * level.group,
        level.width,
        local_relation_roots(b"strong-QA-membership", relation_level, &component_roots),
    );
    assert_eq!(qa_tensor.proof.claimed_sum, Field192::ZERO);
    let mut terminal_source = Vec::with_capacity(2 * STRONG_NEXT_BLOCKS * level.width);
    terminal_source.extend_from_slice(&qa_tensor.right_ood_row);
    terminal_source.extend_from_slice(&qa_tensor.left_ood_row);
    terminal_source.extend_from_slice(&selected_values);
    let expected_terminal_fields = 2 * STRONG_NEXT_BLOCKS * level.width;
    assert_eq!(terminal_source.len(), expected_terminal_fields);
    let ood_block_root = standard_ood_block_root(level, &terminal_source, &zeros);
    let terminal_source_root = derived_terminal_root_for_level(
        level,
        &selected_front,
        ood_block_root,
        expected_terminal_fields.next_power_of_two(),
        &zeros,
    )
    .expect("strong round must derive its next root from authenticated F/W rows");
    let core = StrongTransitionCore {
        round,
        component_roots,
        ood_block_root,
        terminal_source_root,
        selected_front,
        membership: qa_tensor.proof,
    };
    assert!(core.verify());
    let algebra = start.elapsed();

    let start = Instant::now();
    let base = attach_base.then(|| {
        let mut tail_roots = core.component_roots.clone();
        tail_roots.push(core.terminal_source_root);
        let tail = run_direct_whir_tail(
            terminal_source.clone(),
            direct_tail_context_digest(&tail_roots),
            verifier_repetitions,
        )
        .artifact;
        StrongBaseProof {
            tail,
            terminal_witness: terminal_source.clone(),
        }
    });
    if let Some(base) = &base {
        assert!(base.verify(&core));
    }
    let terminal = start.elapsed();
    StrongRoundMeasurement {
        core,
        terminal_source,
        base,
        encode_and_commit,
        algebra,
        terminal,
    }
}

fn build_recursive_strong_end_to_end(
    carry_measurement: &CarryOpenMeasurement,
    measurement: &KernelMeasurement,
    verifier_repetitions: usize,
) -> RecursiveStrongEndToEndProof {
    let mut certificate = measurement.transition_proofs.clone().unwrap();
    let certificate_root = certificate.last().unwrap().next_source_root;
    let mut carryopen = CarryOpenCoreProof::from(&carry_measurement.proof);
    let mut carry_roots = carryopen.component_roots.clone();
    carry_roots.push(carryopen.terminal_source_root);
    carryopen.phi_link = zero_phi_link_proof(
        b"CarryOpen-Phi-link",
        b"CarryOpen-Phi-link",
        10,
        CARRYOPEN_TERMINAL_FIELDS,
        &carry_roots,
    );
    let mut certificate_roots = certificate
        .iter()
        .flat_map(|proof| proof.component_roots.iter().copied())
        .collect::<Vec<_>>();
    certificate_roots.push(certificate_root);
    for (level_index, transition) in certificate.iter_mut().enumerate() {
        transition.algebra.phi_link = Some(zero_phi_link_proof(
            b"Phi-link",
            b"Phi-link",
            level_index,
            2 * LEVELS[level_index].next_blocks * LEVELS[level_index].width,
            &certificate_roots,
        ));
    }

    let mut source = materialize_strong_source(
        measurement.terminal_tail.as_ref().unwrap(),
        &carry_measurement.terminal_source,
    );
    let mut expected_root = parent(certificate_root, carryopen.terminal_source_root);
    let mut strong = Vec::with_capacity(STRONG_ROUNDS);
    let mut base = None;
    for round in 0..STRONG_ROUNDS {
        let round_measurement = run_strong_round(
            round,
            source,
            expected_root,
            round + 1 == STRONG_ROUNDS,
            verifier_repetitions,
        );
        expected_root = round_measurement.core.terminal_source_root;
        source = round_measurement.terminal_source;
        base = round_measurement.base.or(base);
        strong.push(round_measurement.core);
    }
    let proof = RecursiveStrongEndToEndProof {
        carryopen,
        certificate,
        strong,
        base: base.expect("last strong round must attach the terminal base"),
    };
    assert!(proof.verify());
    proof
}

fn run_strong_terminal_switch(
    certificate_terminal: &[Field192],
    carry_terminal: &[Field192],
    certificate_root: Digest,
    carry_root: Digest,
    verifier_repetitions: usize,
) -> StrongTerminalMeasurement {
    let level = strong_level();
    let zeros = zero_roots(30);
    let source = materialize_strong_source(certificate_terminal, carry_terminal);
    let expected_source_root = parent(certificate_root, carry_root);
    assert_eq!(
        prefix_root(&source, STRONG_SOURCE_FIELDS, &zeros),
        expected_source_root
    );

    let start = Instant::now();
    let spectra = (0..STRONG_INVERSE_RATE - 1)
        .map(|block| generator_spectrum(12, block, STRONG_GROUP))
        .collect::<Vec<_>>();
    let generator_roots = spectra
        .iter()
        .map(|spectrum| prefix_root(spectrum, STRONG_GROUP, &zeros))
        .collect::<Vec<_>>();
    let mut codeword = vec![Field192::ZERO; level.qa_fields()];
    populate_proof_codeword(level, &source, &mut codeword, &spectra, 64);
    let (mut component_roots, proof_commitments) =
        level_commitment(level, &source, &codeword, &generator_roots, &zeros);
    assert_eq!(
        component_roots[STRONG_INVERSE_RATE - 1],
        expected_source_root
    );
    let mut index_oracle = vec![Field192::ZERO; level.qa_fields()];
    populate_index_oracle(12, level, &mut index_oracle, &spectra, &component_roots);
    let index_commitment = index_oracle_commitment(level, &index_oracle, &zeros);
    component_roots.push(index_commitment.root);
    let selected = selected_rows(
        12,
        STRONG_INVERSE_RATE * STRONG_GROUP,
        STRONG_QUERIES,
        &component_roots,
    );
    let (selected_front, _, _) =
        selected_row_front(level, &proof_commitments, &index_commitment, &selected);
    let mut selected_values = Vec::with_capacity(2 * STRONG_QUERIES * STRONG_WIDTH);
    for row in &selected {
        let start = row * STRONG_WIDTH;
        selected_values.extend_from_slice(&codeword[start..start + STRONG_WIDTH]);
        selected_values.extend_from_slice(&index_oracle[start..start + STRONG_WIDTH]);
    }
    let encode_and_commit = start.elapsed();

    let start = Instant::now();
    let qa_tensor = prove_tensor_product_relation(
        index_oracle,
        codeword,
        STRONG_INVERSE_RATE * STRONG_GROUP,
        STRONG_WIDTH,
        local_relation_roots(b"strong-QA-membership", 12, &component_roots),
    );
    assert_eq!(qa_tensor.proof.claimed_sum, Field192::ZERO);
    let mut terminal_witness = Vec::with_capacity(STRONG_TERMINAL_FIELDS);
    terminal_witness.extend_from_slice(&qa_tensor.right_ood_row);
    terminal_witness.extend_from_slice(&qa_tensor.left_ood_row);
    terminal_witness.extend_from_slice(&selected_values);
    assert_eq!(terminal_witness.len(), STRONG_TERMINAL_FIELDS);
    let ood_block_root = standard_ood_block_root(level, &terminal_witness, &zeros);
    let terminal_source_root = derived_terminal_root_for_level(
        level,
        &selected_front,
        ood_block_root,
        STRONG_TERMINAL_FIELDS.next_power_of_two(),
        &zeros,
    )
    .expect("strong terminal roots must derive the compressed terminal source");
    let algebra = start.elapsed();

    let start = Instant::now();
    let mut tail_roots = component_roots.clone();
    tail_roots.push(terminal_source_root);
    let context = direct_tail_context_digest(&tail_roots);
    let tail =
        run_direct_whir_tail(terminal_witness.clone(), context, verifier_repetitions).artifact;
    let terminal = start.elapsed();
    let proof = StrongTerminalProof {
        component_roots,
        ood_block_root,
        terminal_source_root,
        selected_front,
        membership: qa_tensor.proof,
        tail,
        terminal_witness,
    };
    assert!(proof.verify());
    let payload = proof.serialize();
    assert_eq!(
        StrongTerminalProof::deserialize(&payload),
        Some(proof.clone())
    );
    let mut changed = proof.clone();
    changed.terminal_witness[2 * STRONG_WIDTH] += Field192::ONE;
    assert!(!changed.verify());
    StrongTerminalMeasurement {
        proof,
        encode_and_commit,
        algebra,
        terminal,
    }
}

fn wht(values: &mut [Field192]) {
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(2 * half) {
            let (left, right) = block.split_at_mut(half);
            for (a, b) in left.iter_mut().zip(right) {
                let old_a = *a;
                let old_b = *b;
                *a = old_a + old_b;
                *b = old_a - old_b;
            }
        }
        half *= 2;
    }
}

fn generator_spectrum(level_index: usize, parity_block: usize, group: usize) -> Vec<Field192> {
    let inverse = Field192::from(group as u64).inverse().unwrap();
    (0..group)
        .map(|index| {
            fixed_challenge(
                b"fixed-generator-spectrum",
                level_index * 4 + parity_block,
                index,
            ) * inverse
        })
        .collect()
}

fn systematic_lane(level: Level, source: &[Field192], lane: usize) -> Vec<Field192> {
    let mut message = vec![Field192::ZERO; level.group];
    for block in 0..level.blocks {
        for row in 0..level.row_span {
            let local = row * level.width + lane;
            if local < level.block_semantic {
                message[block * level.row_span + row] =
                    source[block * level.block_semantic + local];
            }
        }
    }
    message
}

fn apply_encoder(message: &mut [Field192], spectrum: &[Field192]) {
    wht(message);
    for (value, multiplier) in message.iter_mut().zip(spectrum) {
        *value *= multiplier;
    }
    wht(message);
}

fn dot(left: &[Field192], right: &[Field192]) -> Field192 {
    assert_eq!(left.len(), right.len());
    left.par_iter()
        .zip(right)
        .map(|(left, right)| *left * right)
        .reduce(|| Field192::ZERO, |left, right| left + right)
}

fn populate_proof_codeword(
    level: Level,
    source: &[Field192],
    proof_codeword: &mut [Field192],
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
) {
    assert_eq!(proof_codeword.len(), level.qa_fields());
    assert_eq!(spectra.len(), level.inverse_rate - 1);
    proof_codeword[..level.group * level.width]
        .par_chunks_mut(level.width)
        .enumerate()
        .for_each(|(row, target)| {
            let block = row / level.row_span;
            let local_row = row % level.row_span;
            if block < level.blocks {
                for (lane, value) in target.iter_mut().enumerate() {
                    let local = local_row * level.width + lane;
                    if local < level.block_semantic {
                        *value = source[block * level.block_semantic + local];
                    }
                }
            }
        });

    for lane_start in (0..level.width).step_by(batch_lanes) {
        let lane_end = (lane_start + batch_lanes).min(level.width);
        let encoded = (lane_start..lane_end)
            .into_par_iter()
            .map(|lane| {
                let message = systematic_lane(level, source, lane);
                spectra
                    .iter()
                    .map(|spectrum| {
                        let mut parity = message.clone();
                        apply_encoder(&mut parity, spectrum);
                        parity
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for parity_block in 0..level.inverse_rate - 1 {
            let start = (parity_block + 1) * level.group * level.width;
            proof_codeword[start..start + level.group * level.width]
                .par_chunks_mut(level.width)
                .enumerate()
                .for_each(|(row, target)| {
                    for (offset, lane) in encoded.iter().enumerate() {
                        target[lane_start + offset] = lane[parity_block][row];
                    }
                });
        }
    }
}

fn populate_index_oracle(
    level_index: usize,
    level: Level,
    index_oracle: &mut [Field192],
    spectra: &[Vec<Field192>],
    transcript_roots: &[Digest],
) {
    assert_eq!(index_oracle.len(), level.qa_fields());
    let codeword_rows = level.inverse_rate * level.group;
    let position_domain = codeword_rows.next_power_of_two();
    let beta = equality_weights(
        &(0..position_domain.trailing_zeros() as usize)
            .map(|index| semantic_challenge(b"qa-row", level_index, index, transcript_roots))
            .collect::<Vec<_>>(),
    );
    let alpha = equality_weights(
        &(0..level.group.trailing_zeros() as usize)
            .map(|index| semantic_challenge(b"qa-lane", level_index, index, transcript_roots))
            .collect::<Vec<_>>(),
    );
    let mut message_weights = beta[..level.group].to_vec();
    for (block, spectrum) in spectra.iter().enumerate() {
        let start = (block + 1) * level.group;
        let mut transformed = beta[start..start + level.group].to_vec();
        apply_encoder(&mut transformed, spectrum);
        message_weights
            .iter_mut()
            .zip(transformed)
            .for_each(|(target, value)| *target += value);
    }
    index_oracle
        .par_chunks_mut(level.width)
        .enumerate()
        .for_each(|(row, target)| {
            let coefficient = if row < level.group {
                beta[row] - message_weights[row]
            } else {
                beta[row]
            };
            target
                .iter_mut()
                .zip(&alpha)
                .for_each(|(value, lane_weight)| *value = coefficient * *lane_weight);
        });
}

#[derive(Debug)]
struct MatrixCommitment {
    root: Digest,
    row_roots: Vec<Digest>,
    row_domain: usize,
    zero_row_root: Digest,
}

fn compact_matrix_commitment(
    values: &[Field192],
    rows: usize,
    width: usize,
    row_capacity: usize,
    row_domain: usize,
    zeros: &[Digest],
) -> MatrixCommitment {
    assert_eq!(values.len(), rows * width);
    assert!(rows <= row_domain && row_domain.is_power_of_two());
    assert!(width <= row_capacity && row_capacity.is_power_of_two());
    let row_roots = values
        .par_chunks_exact(width)
        .map(|row| prefix_root(row, row_capacity, zeros))
        .collect::<Vec<_>>();
    let capacity = row_domain * row_capacity;
    let row_height = row_capacity.trailing_zeros() as usize;
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for root in &row_roots {
        accumulator.append_subtree(*root, row_height);
    }
    MatrixCommitment {
        root: accumulator.finish(capacity, zeros),
        row_roots,
        row_domain,
        zero_row_root: zeros[row_height],
    }
}

#[cfg(test)]
fn compact_matrix_root(
    values: &[Field192],
    rows: usize,
    width: usize,
    row_capacity: usize,
    row_domain: usize,
    zeros: &[Digest],
) -> Digest {
    compact_matrix_commitment(values, rows, width, row_capacity, row_domain, zeros).root
}

fn splice_root(level: Level, source: &[Field192], zeros: &[Digest]) -> Digest {
    assert_eq!(source.len(), level.raw);
    let block_capacity = level.components.iter().map(|item| item.1).sum::<usize>();
    assert!(block_capacity.is_power_of_two());
    let block_roots = source
        .par_chunks_exact(level.block_semantic)
        .map(|block| {
            let mut position = 0;
            let mut accumulator = MerkleAccumulator::new(block_capacity.trailing_zeros() as usize);
            for &(semantic, capacity) in level.components {
                let root = prefix_root(&block[position..position + semantic], capacity, zeros);
                position += semantic;
                accumulator.append_subtree(root, capacity.trailing_zeros() as usize);
            }
            assert_eq!(position, level.block_semantic);
            accumulator.finish(block_capacity, zeros)
        })
        .collect::<Vec<_>>();
    let mut accumulator = MerkleAccumulator::new(level.view_capacity().trailing_zeros() as usize);
    for root in block_roots {
        accumulator.append_subtree(root, block_capacity.trailing_zeros() as usize);
    }
    accumulator.finish(level.view_capacity(), zeros)
}

fn level_commitment(
    level: Level,
    source: &[Field192],
    proof_codeword: &[Field192],
    generator_roots: &[Digest],
    zeros: &[Digest],
) -> (Vec<Digest>, Vec<MatrixCommitment>) {
    assert_eq!(generator_roots.len(), level.inverse_rate - 1);
    assert_eq!(proof_codeword.len(), level.qa_fields());
    let mut roots = Vec::with_capacity(level.inverse_rate + 2);
    roots.extend_from_slice(generator_roots);
    roots.push(splice_root(level, source, zeros));
    let mut matrices = Vec::with_capacity(level.inverse_rate);
    for block in 0..level.inverse_rate {
        let start = block * level.group * level.width;
        let commitment = compact_matrix_commitment(
            &proof_codeword[start..start + level.group * level.width],
            level.group,
            level.width,
            level.group,
            level.group,
            zeros,
        );
        roots.push(commitment.root);
        matrices.push(commitment);
    }
    (roots, matrices)
}

#[cfg(test)]
fn level_commitment_roots(
    level: Level,
    source: &[Field192],
    proof_codeword: &[Field192],
    generator_roots: &[Digest],
    zeros: &[Digest],
) -> Vec<Digest> {
    level_commitment(level, source, proof_codeword, generator_roots, zeros).0
}

#[cfg(test)]
fn index_oracle_root(level: Level, index_oracle: &[Field192], zeros: &[Digest]) -> Digest {
    let rows = level.inverse_rate * level.group;
    compact_matrix_root(
        index_oracle,
        rows,
        level.width,
        level.group,
        rows.next_power_of_two(),
        zeros,
    )
}

fn index_oracle_commitment(
    level: Level,
    index_oracle: &[Field192],
    zeros: &[Digest],
) -> MatrixCommitment {
    let rows = level.inverse_rate * level.group;
    compact_matrix_commitment(
        index_oracle,
        rows,
        level.width,
        level.group,
        rows.next_power_of_two(),
        zeros,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RowSubtreeMultiProof {
    indices: Vec<usize>,
    roots: Vec<Digest>,
    frontier: Vec<Digest>,
}

impl RowSubtreeMultiProof {
    const MAGIC: &'static [u8; 8] = b"LILROW01";

    fn serialized_bytes(&self) -> usize {
        // Indices are Fiat--Shamir-derived and belong to the statement.
        16 + 32 * (self.roots.len() + self.frontier.len())
    }

    fn serialize(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.serialized_bytes());
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.roots.len() as u32).to_le_bytes());
        output.extend_from_slice(&(self.frontier.len() as u32).to_le_bytes());
        for root in self.roots.iter().chain(&self.frontier) {
            output.extend_from_slice(root);
        }
        output
    }

    fn deserialize(payload: &[u8], indices: Vec<usize>) -> Option<Self> {
        if payload.len() < 16 || &payload[..8] != Self::MAGIC {
            return None;
        }
        let roots = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let frontier = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        if roots != indices.len() || payload.len() != 16 + 32 * (roots + frontier) {
            return None;
        }
        let mut position = 16;
        let mut take_roots = |count: usize| {
            (0..count)
                .map(|_| {
                    let root: Digest = payload.get(position..position + 32)?.try_into().ok()?;
                    position += 32;
                    Some(root)
                })
                .collect::<Option<Vec<_>>>()
        };
        let roots_data = take_roots(roots)?;
        let frontier_data = take_roots(frontier)?;
        Some(Self {
            indices,
            roots: roots_data,
            frontier: frontier_data,
        })
    }

    fn verify(&self, expected_root: Digest, row_domain: usize) -> bool {
        if self.indices.is_empty()
            || self.indices.len() != self.roots.len()
            || self.indices.windows(2).any(|pair| pair[0] >= pair[1])
            || self
                .indices
                .last()
                .is_some_and(|index| *index >= row_domain)
        {
            return false;
        }
        let mut active = self
            .indices
            .iter()
            .copied()
            .zip(self.roots.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let mut frontier_position = 0;
        let mut width = row_domain;
        while width > 1 {
            let mut parents = BTreeMap::new();
            let mut consumed = BTreeMap::<usize, ()>::new();
            for (&index, &root) in &active {
                if consumed.contains_key(&index) {
                    continue;
                }
                let sibling = index ^ 1;
                let sibling_root = if let Some(value) = active.get(&sibling) {
                    consumed.insert(sibling, ());
                    *value
                } else {
                    let Some(value) = self.frontier.get(frontier_position) else {
                        return false;
                    };
                    frontier_position += 1;
                    *value
                };
                let parent_root = if index & 1 == 0 {
                    parent(root, sibling_root)
                } else {
                    parent(sibling_root, root)
                };
                parents.insert(index >> 1, parent_root);
                consumed.insert(index, ());
            }
            active = parents;
            width >>= 1;
        }
        frontier_position == self.frontier.len()
            && active.len() == 1
            && active.get(&0) == Some(&expected_root)
    }
}

fn open_row_subtrees(commitment: &MatrixCommitment, indices: &[usize]) -> RowSubtreeMultiProof {
    assert!(!indices.is_empty());
    assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(indices.last().unwrap() < &commitment.row_domain);
    let mut level = commitment.row_roots.clone();
    level.resize(commitment.row_domain, commitment.zero_row_root);
    let roots = indices
        .iter()
        .map(|index| level[*index])
        .collect::<Vec<_>>();
    let mut active = indices.iter().copied().collect::<Vec<_>>();
    let mut frontier = Vec::new();
    while level.len() > 1 {
        for index in &active {
            let sibling = *index ^ 1;
            if active.binary_search(&sibling).is_err() {
                frontier.push(level[sibling]);
            }
        }
        active = active
            .into_iter()
            .map(|index| index >> 1)
            .collect::<Vec<_>>();
        active.dedup();
        level = level
            .par_chunks_exact(2)
            .map(|pair| parent(pair[0], pair[1]))
            .collect();
    }
    let proof = RowSubtreeMultiProof {
        indices: indices.to_vec(),
        roots,
        frontier,
    };
    assert!(proof.verify(commitment.root, commitment.row_domain));
    proof
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SelectedRowFront {
    selected: Vec<usize>,
    proof_blocks: Vec<(usize, RowSubtreeMultiProof)>,
    index_opening: RowSubtreeMultiProof,
}

impl SelectedRowFront {
    const MAGIC: &'static [u8; 8] = b"LILSRF01";

    fn verify(&self, level: Level, component_roots: &[Digest]) -> bool {
        if component_roots.len() != 2 * level.inverse_rate + 1
            || self.selected.is_empty()
            || self.selected.windows(2).any(|pair| pair[0] >= pair[1])
            || self
                .selected
                .last()
                .is_some_and(|index| *index >= level.inverse_rate * level.group)
        {
            return false;
        }
        let proof_roots = &component_roots[level.inverse_rate..2 * level.inverse_rate];
        let expected_blocks = (0..level.inverse_rate)
            .filter(|block| {
                let start = block * level.group;
                self.selected
                    .iter()
                    .any(|index| *index >= start && *index < start + level.group)
            })
            .collect::<Vec<_>>();
        if self
            .proof_blocks
            .iter()
            .map(|(block, _)| *block)
            .collect::<Vec<_>>()
            != expected_blocks
        {
            return false;
        }
        for (block, proof) in &self.proof_blocks {
            let start = block * level.group;
            let local = self
                .selected
                .iter()
                .filter_map(|index| {
                    (*index >= start && *index < start + level.group).then_some(*index - start)
                })
                .collect::<Vec<_>>();
            if proof.indices != local || !proof.verify(proof_roots[*block], level.group) {
                return false;
            }
        }
        self.index_opening.indices == self.selected
            && self.index_opening.verify(
                *component_roots.last().unwrap(),
                (level.inverse_rate * level.group).next_power_of_two(),
            )
    }

    fn serialize(&self) -> Vec<u8> {
        let proof_payloads = self
            .proof_blocks
            .iter()
            .map(|(block, proof)| (*block, proof.serialize()))
            .collect::<Vec<_>>();
        let index_payload = self.index_opening.serialize();
        let size = 24
            + 4 * self.selected.len()
            + proof_payloads
                .iter()
                .map(|(_, proof)| 8 + proof.len())
                .sum::<usize>()
            + 4
            + index_payload.len();
        let mut output = Vec::with_capacity(size);
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.selected.len() as u32).to_le_bytes());
        output.extend_from_slice(&(proof_payloads.len() as u32).to_le_bytes());
        output.extend_from_slice(&(index_payload.len() as u32).to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        for index in &self.selected {
            output.extend_from_slice(&(*index as u32).to_le_bytes());
        }
        for (block, payload) in proof_payloads {
            output.extend_from_slice(&(block as u32).to_le_bytes());
            output.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            output.extend_from_slice(&payload);
        }
        output.extend_from_slice(&(index_payload.len() as u32).to_le_bytes());
        output.extend_from_slice(&index_payload);
        output
    }

    fn deserialize(payload: &[u8], level: Level) -> Option<Self> {
        if payload.len() < 24 || &payload[..8] != Self::MAGIC {
            return None;
        }
        let selected_count = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let proof_count = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        let declared_index_size = u32::from_le_bytes(payload[16..20].try_into().ok()?) as usize;
        let reserved = u32::from_le_bytes(payload[20..24].try_into().ok()?);
        if selected_count == 0 || reserved != 0 {
            return None;
        }
        let mut position = 24;
        let mut selected = Vec::with_capacity(selected_count);
        for _ in 0..selected_count {
            let bytes = payload.get(position..position + 4)?;
            selected.push(u32::from_le_bytes(bytes.try_into().ok()?) as usize);
            position += 4;
        }
        let mut proof_blocks = Vec::with_capacity(proof_count);
        for _ in 0..proof_count {
            let block =
                u32::from_le_bytes(payload.get(position..position + 4)?.try_into().ok()?) as usize;
            let size = u32::from_le_bytes(payload.get(position + 4..position + 8)?.try_into().ok()?)
                as usize;
            position += 8;
            let start = block.checked_mul(level.group)?;
            let local = selected
                .iter()
                .filter_map(|index| {
                    (*index >= start && *index < start + level.group).then_some(*index - start)
                })
                .collect::<Vec<_>>();
            let proof =
                RowSubtreeMultiProof::deserialize(payload.get(position..position + size)?, local)?;
            position += size;
            proof_blocks.push((block, proof));
        }
        let index_size =
            u32::from_le_bytes(payload.get(position..position + 4)?.try_into().ok()?) as usize;
        position += 4;
        if index_size != declared_index_size || payload.len() != position + index_size {
            return None;
        }
        let index_opening =
            RowSubtreeMultiProof::deserialize(&payload[position..], selected.clone())?;
        Some(Self {
            selected,
            proof_blocks,
            index_opening,
        })
    }
}

fn selected_row_front(
    level: Level,
    proof_commitments: &[MatrixCommitment],
    index_commitment: &MatrixCommitment,
    selected: &[usize],
) -> (SelectedRowFront, usize, usize) {
    assert_eq!(proof_commitments.len(), level.inverse_rate);
    let mut frontier_hashes = 0;
    let mut proof_blocks = Vec::new();
    for (block, commitment) in proof_commitments.iter().enumerate() {
        let start = block * level.group;
        let local = selected
            .iter()
            .filter_map(|index| {
                (*index >= start && *index < start + level.group).then_some(*index - start)
            })
            .collect::<Vec<_>>();
        if !local.is_empty() {
            let opening = open_row_subtrees(commitment, &local);
            let payload = opening.serialize();
            let parsed = RowSubtreeMultiProof::deserialize(&payload, local)
                .expect("canonical F row-subtree multiproof must parse");
            assert_eq!(parsed, opening);
            assert!(parsed.verify(commitment.root, commitment.row_domain));
            frontier_hashes += opening.frontier.len();
            proof_blocks.push((block, opening));
        }
    }
    let index_opening = open_row_subtrees(index_commitment, selected);
    let payload = index_opening.serialize();
    let parsed = RowSubtreeMultiProof::deserialize(&payload, selected.to_vec())
        .expect("canonical W row-subtree multiproof must parse");
    assert_eq!(parsed, index_opening);
    assert!(parsed.verify(index_commitment.root, index_commitment.row_domain));
    frontier_hashes += index_opening.frontier.len();
    let front = SelectedRowFront {
        selected: selected.to_vec(),
        proof_blocks,
        index_opening,
    };
    let payload = front.serialize();
    let parsed = SelectedRowFront::deserialize(&payload, level)
        .expect("canonical selected-row front must parse");
    let mut component_roots = Vec::with_capacity(2 * level.inverse_rate + 1);
    component_roots.extend((0..level.inverse_rate).map(|_| [0_u8; 32]));
    component_roots.extend(proof_commitments.iter().map(|commitment| commitment.root));
    component_roots.push(index_commitment.root);
    assert!(parsed.verify(level, &component_roots));
    (front, payload.len(), frontier_hashes)
}

fn populate_copy_relation(
    level_index: usize,
    level: Level,
    source: &[Field192],
    left: &mut [Field192],
    right: &mut [Field192],
    transcript_roots: &[Digest],
) {
    assert_eq!(left.len(), level.view_capacity());
    let mut splice = vec![Field192::ZERO; level.view_capacity()];
    let mut row = vec![Field192::ZERO; level.view_capacity()];
    for (index, value) in source.iter().enumerate() {
        splice[level.splice_position(index)] = *value;
        row[level.row_position(index)] = *value;
    }
    let mut permuted = vec![Field192::ZERO; level.view_capacity()];
    for index in 0..level.raw {
        permuted[level.row_position(index)] = splice[level.splice_position(index)];
    }
    left.par_iter_mut()
        .zip(row.par_iter().zip(&permuted))
        .for_each(|(residual, (actual, expected))| *residual = *actual - *expected);
    let point = (0..level.view_capacity().trailing_zeros() as usize)
        .map(|index| semantic_challenge(b"copy", level_index, index, transcript_roots))
        .collect::<Vec<_>>();
    right.copy_from_slice(&equality_weights(&point));
    assert_eq!(dot(left, right), Field192::ZERO);
}

fn selected_rows(
    level_index: usize,
    domain: usize,
    count: usize,
    transcript_roots: &[Digest],
) -> Vec<usize> {
    let mut selected = Vec::with_capacity(count);
    let mut counter = 0;
    while selected.len() < count {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"LiLAC/Phi/selected-row/v1");
        hasher.update(&(level_index as u64).to_le_bytes());
        hasher.update(&(counter as u64).to_le_bytes());
        for root in transcript_roots {
            hasher.update(root);
        }
        let candidate = u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap())
            as usize
            % domain;
        if !selected.contains(&candidate) {
            selected.push(candidate);
        }
        counter += 1;
    }
    selected.sort_unstable();
    selected
}

fn derive_next_source(
    level: Level,
    proof_codeword: &[Field192],
    index_oracle: &[Field192],
    proof_ood: &[Field192],
    index_ood: &[Field192],
    indices: &[usize],
) -> Vec<Field192> {
    let codeword_rows = level.inverse_rate * level.group;
    assert_eq!(proof_codeword.len(), codeword_rows * level.width);
    assert_eq!(index_oracle.len(), proof_codeword.len());
    let query_count = level.next_blocks - 1;
    assert_eq!(indices.len(), query_count);
    assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(indices.last().is_none_or(|index| *index < codeword_rows));
    assert_eq!(proof_ood.len(), level.width);
    assert_eq!(index_ood.len(), level.width);

    let mut next = vec![Field192::ZERO; 2 * level.next_blocks * level.width];
    next[..level.width].copy_from_slice(&proof_ood);
    next[level.width..2 * level.width].copy_from_slice(&index_ood);
    for (block, row_index) in indices.iter().copied().enumerate() {
        let target = (block + 1) * 2 * level.width;
        let source = row_index * level.width;
        next[target..target + level.width]
            .copy_from_slice(&proof_codeword[source..source + level.width]);
        next[target + level.width..target + 2 * level.width]
            .copy_from_slice(&index_oracle[source..source + level.width]);
    }
    next
}

#[cfg(test)]
fn populate_link_relation(
    level_index: usize,
    actual: &[Field192],
    expected: &[Field192],
    left: &mut [Field192],
    right: &mut [Field192],
    transcript_roots: &[Digest],
) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(left.len(), actual.len());
    left.par_iter_mut()
        .zip(actual.par_iter().zip(expected))
        .for_each(|(residual, (actual, expected))| *residual = *actual - *expected);
    let domain = actual.len().next_power_of_two();
    let point = (0..domain.trailing_zeros() as usize)
        .map(|index| semantic_challenge(b"Phi-link", level_index, index, transcript_roots))
        .collect::<Vec<_>>();
    right.copy_from_slice(&equality_weights(&point)[..actual.len()]);
    assert_eq!(dot(left, right), Field192::ZERO);
}

#[derive(Clone, Copy, Debug, Default)]
struct SemanticBreakdown {
    generator_preprocessing: Duration,
    proof_encoding: Duration,
    proof_commitment: Duration,
    index_oracle: Duration,
    index_commitment: Duration,
    source_opening: Duration,
    source_opening_bytes: usize,
    source_frontier_hashes: usize,
    copy_and_phi: Duration,
    transition_sumchecks: Duration,
    transition_proof_bytes: usize,
    terminal_commitment: Duration,
}

fn semantic_vectors(
    batch_lanes: usize,
    direct_whir_tail: bool,
) -> (
    Vec<Field192>,
    Vec<Field192>,
    Vec<Digest>,
    Vec<Field192>,
    Option<Digest>,
    Digest,
    Vec<ProductionTransitionProof>,
    SemanticBreakdown,
) {
    assert!(batch_lanes > 0);
    assert_eq!(
        LEVELS
            .iter()
            .map(|level| {
                level.qa_fields() + level.view_capacity() + 2 * level.next_blocks * level.width
            })
            .sum::<usize>(),
        PRODUCTION_FIELDS
    );
    let mut left = vec![Field192::ZERO; PRODUCTION_FIELDS];
    let mut right = vec![Field192::ZERO; PRODUCTION_FIELDS];
    let zeros = zero_roots(PRODUCTION_VARIABLES);
    let mut transcript_roots = Vec::<Digest>::new();
    let mut link_ranges = Vec::<(usize, usize, usize)>::new();
    let mut transition_proofs = Vec::with_capacity(LEVELS.len());
    let mut breakdown = SemanticBreakdown::default();
    let mut offset = 0;
    let mut source = (0..LEVELS[0].raw)
        .into_par_iter()
        .map(|index| semantic_value(0, index))
        .collect::<Vec<_>>();
    for (level_index, level) in LEVELS.into_iter().enumerate() {
        assert_eq!(level.raw, level.blocks * level.block_semantic);
        assert_eq!(source.len(), level.raw);
        let qa_start = offset;
        let qa_end = offset + level.qa_fields();

        let start = Instant::now();
        let spectra = (0..level.inverse_rate - 1)
            .map(|block| generator_spectrum(level_index, block, level.group))
            .collect::<Vec<_>>();
        let generator_roots = spectra
            .iter()
            .map(|spectrum| prefix_root(spectrum, level.group, &zeros))
            .collect::<Vec<_>>();
        breakdown.generator_preprocessing += start.elapsed();

        let start = Instant::now();
        populate_proof_codeword(
            level,
            &source,
            &mut right[offset..qa_end],
            &spectra,
            batch_lanes,
        );
        breakdown.proof_encoding += start.elapsed();

        let start = Instant::now();
        let (level_roots, proof_commitments) = level_commitment(
            level,
            &source,
            &right[qa_start..qa_end],
            &generator_roots,
            &zeros,
        );
        let mut current_component_roots = level_roots.clone();
        transcript_roots.extend(level_roots);
        breakdown.proof_commitment += start.elapsed();

        let start = Instant::now();
        populate_index_oracle(
            level_index,
            level,
            &mut left[qa_start..qa_end],
            &spectra,
            &transcript_roots,
        );
        breakdown.index_oracle += start.elapsed();
        assert_eq!(
            dot(&left[qa_start..qa_end], &right[qa_start..qa_end]),
            Field192::ZERO
        );

        let start = Instant::now();
        let index_commitment = index_oracle_commitment(level, &left[qa_start..qa_end], &zeros);
        transcript_roots.push(index_commitment.root);
        current_component_roots.push(index_commitment.root);
        breakdown.index_commitment += start.elapsed();

        let start = Instant::now();
        let qa_tensor = prove_tensor_product_relation(
            left[qa_start..qa_end].to_vec(),
            right[qa_start..qa_end].to_vec(),
            level.inverse_rate * level.group,
            level.width,
            local_relation_roots(b"QA-membership", level_index, &transcript_roots),
        );
        assert_eq!(qa_tensor.proof.claimed_sum, Field192::ZERO);
        let qa_membership = qa_tensor.proof.clone();
        let qa_point = qa_membership
            .challenges()
            .expect("tensor QA sumcheck must produce row/lane OOD challenges");
        let row_variables = (level.inverse_rate * level.group)
            .next_power_of_two()
            .trailing_zeros() as usize;
        assert_eq!(
            evaluate_padded_message(
                &qa_tensor.left_ood_row,
                level.width.next_power_of_two(),
                &qa_point[row_variables..],
            ),
            qa_membership.terminal_left
        );
        assert_eq!(
            evaluate_padded_message(
                &qa_tensor.right_ood_row,
                level.width.next_power_of_two(),
                &qa_point[row_variables..],
            ),
            qa_membership.terminal_right
        );
        breakdown.transition_sumchecks += start.elapsed();

        let selected = selected_rows(
            level_index,
            level.inverse_rate * level.group,
            level.next_blocks - 1,
            &transcript_roots,
        );
        let start = Instant::now();
        let (selected_front, opening_bytes, frontier_hashes) =
            selected_row_front(level, &proof_commitments, &index_commitment, &selected);
        breakdown.source_opening += start.elapsed();
        breakdown.source_opening_bytes += opening_bytes;
        breakdown.source_frontier_hashes += frontier_hashes;

        offset = qa_end;
        let copy_end = offset + level.view_capacity();
        let start = Instant::now();
        populate_copy_relation(
            level_index,
            level,
            &source,
            &mut left[offset..copy_end],
            &mut right[offset..copy_end],
            &transcript_roots,
        );
        let start_sumcheck = Instant::now();
        let dual_view_copy = prove_local_product_relation(
            left[offset..copy_end].to_vec(),
            right[offset..copy_end].to_vec(),
            local_relation_roots(b"dual-view-copy", level_index, &transcript_roots),
        );
        breakdown.transition_sumchecks += start_sumcheck.elapsed();
        let algebra = TransitionAlgebraProof {
            level: level_index,
            qa_membership,
            dual_view_copy,
            phi_link: None,
        };
        transition_proofs.push(ProductionTransitionProof {
            level: level_index,
            component_roots: current_component_roots,
            ood_block_root: [0_u8; 32],
            next_source_root: [0_u8; 32],
            selected_front,
            algebra,
        });
        offset = copy_end;
        let expected_next = derive_next_source(
            level,
            &right[qa_start..qa_end],
            &left[qa_start..qa_end],
            &qa_tensor.right_ood_row,
            &qa_tensor.left_ood_row,
            &selected,
        );
        let ood_block_root = standard_ood_block_root(level, &expected_next, &zeros);
        let next_source_root = derived_next_source_root(
            level_index,
            &transition_proofs[level_index].selected_front,
            ood_block_root,
            &zeros,
        )
        .expect("selected F/W roots must derive the next splice-compatible root");
        if let Some(next_level) = LEVELS.get(level_index + 1) {
            assert_eq!(
                next_source_root,
                splice_root(*next_level, &expected_next, &zeros)
            );
        }
        transition_proofs[level_index].ood_block_root = ood_block_root;
        transition_proofs[level_index].next_source_root = next_source_root;
        let link_end = offset + expected_next.len();
        link_ranges.push((level_index, offset, link_end));
        offset = link_end;
        source = expected_next;
        breakdown.copy_and_phi += start.elapsed();
    }
    let start = Instant::now();
    let direct_context = direct_whir_tail.then(|| direct_tail_context_digest(&transcript_roots));
    let terminal_root = if let Some(context) = direct_context {
        direct_whir_commitment_root(&source, context)
    } else {
        prefix_root(&source, source.len().next_power_of_two(), &zeros)
    };
    transcript_roots.push(terminal_root);
    breakdown.terminal_commitment += start.elapsed();
    for (level_index, start, end) in link_ranges {
        let point = (0..(end - start).next_power_of_two().trailing_zeros() as usize)
            .map(|index| semantic_challenge(b"Phi-link", level_index, index, &transcript_roots))
            .collect::<Vec<_>>();
        right[start..end].copy_from_slice(&equality_weights(&point)[..end - start]);
        let start_sumcheck = Instant::now();
        transition_proofs[level_index].algebra.phi_link = Some(prove_local_product_relation(
            left[start..end].to_vec(),
            right[start..end].to_vec(),
            local_relation_roots(b"Phi-link", level_index, &transcript_roots),
        ));
        breakdown.transition_sumchecks += start_sumcheck.elapsed();
    }
    for proof in &transition_proofs {
        let payload = proof.serialize();
        assert_eq!(
            ProductionTransitionProof::deserialize(&payload),
            Some(proof.clone())
        );
        breakdown.transition_proof_bytes += payload.len();
    }
    assert_eq!(offset, PRODUCTION_FIELDS);
    assert_eq!(dot(&left, &right), Field192::ZERO);
    (
        left,
        right,
        transcript_roots,
        source,
        direct_context,
        terminal_root,
        transition_proofs,
        breakdown,
    )
}

fn field_bytes(value: Field192) -> Vec<u8> {
    value.into_bigint().to_bytes_le()
}

fn canonical_field_bytes(value: Field192) -> [u8; 24] {
    field_bytes(value)
        .try_into()
        .expect("Field192 canonical encoding must be 24 bytes")
}

fn aggregate_roots(component_roots: &[Digest]) -> Vec<Digest> {
    (0_u64..3)
        .map(|slot| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"LiLAC/combined-sumcheck-root-aggregate/v1");
            hasher.update(&slot.to_le_bytes());
            hasher.update(&(component_roots.len() as u64).to_le_bytes());
            for root in component_roots {
                hasher.update(root);
            }
            *hasher.finalize().as_bytes()
        })
        .collect()
}

fn transcript_challenge(
    fields: usize,
    variables: usize,
    roots: &[Digest],
    claimed_sum: Field192,
    pairs: &[(Field192, Field192)],
) -> Field192 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/combined-copy-qa-sumcheck/v1");
    hasher.update(&(fields as u64).to_le_bytes());
    hasher.update(&(variables as u64).to_le_bytes());
    hasher.update(&(roots.len() as u64).to_le_bytes());
    for root in roots {
        hasher.update(root);
    }
    hasher.update(&field_bytes(claimed_sum));
    for (constant, quadratic) in pairs {
        hasher.update(&field_bytes(*constant));
        hasher.update(&field_bytes(*quadratic));
    }
    Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes())
}

fn fold_active(values: &mut [Field192], active: usize, weight: Field192) -> usize {
    assert!(active > 1 && active <= values.len());
    let half = active.next_power_of_two() >> 1;
    let high_len = active.saturating_sub(half);
    let (low, high_and_tail) = values.split_at_mut(half);
    let (high, _) = high_and_tail.split_at_mut(high_len);
    let (paired_low, zero_tail) = low.split_at_mut(high_len);
    paired_low
        .par_iter_mut()
        .zip(high.par_iter())
        .for_each(|(low, high)| *low += (*high - *low) * weight);
    let zero_weight = Field192::ONE - weight;
    if zero_tail.len() > workload_size::<Field192>() {
        zero_tail
            .par_iter_mut()
            .for_each(|value| *value *= zero_weight);
    } else {
        zero_tail.iter_mut().for_each(|value| *value *= zero_weight);
    }
    half
}

#[derive(Debug)]
struct KernelMeasurement {
    setup: Duration,
    polynomial: Duration,
    folding: Duration,
    total: Duration,
    terminal_left: Field192,
    terminal_right: Field192,
    terminal_claim: Field192,
    pairs: Vec<(Field192, Field192)>,
    roots: Vec<Digest>,
    fields: usize,
    component_root_count: usize,
    transcript_root_count: usize,
    semantic_breakdown: Option<SemanticBreakdown>,
    transition_proofs: Option<Vec<ProductionTransitionProof>>,
    terminal_tail: Option<Vec<Field192>>,
    terminal_context: Option<Digest>,
    terminal_binding_root: Option<Digest>,
}

impl KernelMeasurement {
    fn sumcheck_proof(&self) -> PackedSumcheckProof {
        PackedSumcheckProof {
            fields: self.fields,
            roots: self.roots.clone(),
            claimed_sum: Field192::ZERO,
            pairs: self.pairs.clone(),
            terminal_left: self.terminal_left,
            terminal_right: self.terminal_right,
        }
    }
}

fn prove_vectors(
    mut left: Vec<Field192>,
    mut right: Vec<Field192>,
    setup: Duration,
    roots: Vec<Digest>,
    component_root_count: usize,
    semantic_breakdown: Option<SemanticBreakdown>,
    transition_proofs: Option<Vec<ProductionTransitionProof>>,
    terminal_tail: Option<Vec<Field192>>,
    terminal_context: Option<Digest>,
    terminal_binding_root: Option<Digest>,
) -> KernelMeasurement {
    let fields = left.len();
    assert!(fields > 1 && right.len() == fields);
    let variables = fields.next_power_of_two().trailing_zeros() as usize;
    let mut claim = Field192::ZERO;
    let total_start = Instant::now();
    let mut polynomial = Duration::ZERO;
    let mut folding = Duration::ZERO;
    let mut active = fields;
    let mut pairs = Vec::with_capacity(variables);
    for _ in 0..variables {
        let start = Instant::now();
        let (constant, quadratic) = compute_sumcheck_polynomial(&left[..active], &right[..active]);
        polynomial += start.elapsed();
        pairs.push((constant, quadratic));
        let challenge = transcript_challenge(fields, variables, &roots, Field192::ZERO, &pairs);
        let linear = claim - constant.double() - quadratic;

        let start = Instant::now();
        let next_left = fold_active(&mut left, active, challenge);
        let next_right = fold_active(&mut right, active, challenge);
        folding += start.elapsed();
        assert_eq!(next_left, next_right);
        active = next_left;
        claim = (quadratic * challenge + linear) * challenge + constant;
    }
    let total = total_start.elapsed();
    assert_eq!(active, 1);
    assert_eq!(claim, left[0] * right[0]);
    KernelMeasurement {
        setup,
        polynomial,
        folding,
        total,
        terminal_left: left[0],
        terminal_right: right[0],
        terminal_claim: claim,
        pairs,
        component_root_count,
        transcript_root_count: roots.len(),
        roots,
        fields,
        semantic_breakdown,
        transition_proofs,
        terminal_tail,
        terminal_context,
        terminal_binding_root,
    }
}

fn run_kernel(fields: usize) -> KernelMeasurement {
    assert!(fields > 1);
    let start = Instant::now();
    let left = (0..fields)
        .into_par_iter()
        .map(left_value)
        .collect::<Vec<_>>();
    let mut right = (0..fields)
        .into_par_iter()
        .map(right_value)
        .collect::<Vec<_>>();
    let initial_claim = dot(&left, &right);
    right[0] -= initial_claim * left[0].inverse().unwrap();
    assert_eq!(dot(&left, &right), Field192::ZERO);
    let setup = start.elapsed();
    let component_roots = (0_u64..4)
        .map(|index| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"LiLAC/synthetic-kernel-root/v1");
            hasher.update(&index.to_le_bytes());
            *hasher.finalize().as_bytes()
        })
        .collect::<Vec<_>>();
    let component_root_count = component_roots.len();
    prove_vectors(
        left,
        right,
        setup,
        aggregate_roots(&component_roots),
        component_root_count,
        None,
        None,
        None,
        None,
        None,
    )
}

fn run_semantic_kernel(batch_lanes: usize, direct_whir_tail: bool) -> KernelMeasurement {
    let start = Instant::now();
    let (
        left,
        right,
        roots,
        terminal_tail,
        terminal_context,
        terminal_binding_root,
        transition_proofs,
        breakdown,
    ) = semantic_vectors(batch_lanes, direct_whir_tail);
    let setup = start.elapsed();
    let component_root_count = roots.len();
    prove_vectors(
        left,
        right,
        setup,
        aggregate_roots(&roots),
        component_root_count,
        Some(breakdown),
        Some(transition_proofs),
        Some(terminal_tail),
        terminal_context,
        Some(terminal_binding_root),
    )
}

#[derive(Debug)]
struct WhirTailMeasurement {
    semantic_fields: usize,
    padded_fields: usize,
    commit: Duration,
    prove: Duration,
    verify: Duration,
    proof_bytes: usize,
    verifier_hashes: usize,
    commitment_root: Digest,
    artifact: DirectTailProofArtifact,
}

fn direct_tail_point(
    padded_fields: usize,
    context_digest: Digest,
    commitment_root: Digest,
) -> MultilinearPoint<Field192> {
    let variables = padded_fields.trailing_zeros() as usize;
    let transcript_roots = [context_digest, commitment_root];
    MultilinearPoint(
        (0..variables)
            .map(|index| {
                semantic_challenge(
                    b"direct-tail-MLE-after-commitment",
                    4,
                    index,
                    &transcript_roots,
                )
            })
            .collect(),
    )
}

fn direct_tail_sparse_entries(label: &[u8], semantic_fields: usize) -> Vec<(usize, Field192)> {
    (0..19)
        .map(|ordinal| {
            let digest = blake3::hash(&[label, &(ordinal as u64).to_le_bytes()].concat());
            let mut index_bytes = [0_u8; 8];
            index_bytes.copy_from_slice(&digest.as_bytes()[..8]);
            (
                u64::from_le_bytes(index_bytes) as usize % semantic_fields,
                Field192::from_le_bytes_mod_order(&digest.as_bytes()[8..]),
            )
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectTailProofArtifact {
    semantic_fields: usize,
    padded_fields: usize,
    context_digest: Digest,
    commitment_root: Digest,
    evaluations: [Field192; 3],
    proof_payload: Vec<u8>,
}

impl DirectTailProofArtifact {
    const MAGIC: &'static [u8; 8] = b"LILTAIL1";

    fn serialize(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(120 + 72 + self.proof_payload.len());
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.semantic_fields as u64).to_le_bytes());
        output.extend_from_slice(&(self.padded_fields as u64).to_le_bytes());
        output.extend_from_slice(&(self.proof_payload.len() as u64).to_le_bytes());
        output.extend_from_slice(&self.context_digest);
        output.extend_from_slice(&self.commitment_root);
        for value in self.evaluations {
            output.extend_from_slice(&canonical_field_bytes(value));
        }
        output.extend_from_slice(&self.proof_payload);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 8 + 8 + 8 + 32 + 32 + 72;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let semantic_fields = u64::from_le_bytes(payload[8..16].try_into().ok()?) as usize;
        let padded_fields = u64::from_le_bytes(payload[16..24].try_into().ok()?) as usize;
        let proof_size = u64::from_le_bytes(payload[24..32].try_into().ok()?) as usize;
        if semantic_fields == 0
            || padded_fields != semantic_fields.next_power_of_two()
            || payload.len() != HEADER + proof_size
        {
            return None;
        }
        let context_digest = payload[32..64].try_into().ok()?;
        let commitment_root = payload[64..96].try_into().ok()?;
        let mut evaluations = [Field192::ZERO; 3];
        for (index, value) in evaluations.iter_mut().enumerate() {
            let start = 96 + index * 24;
            let bytes = &payload[start..start + 24];
            *value = Field192::from_le_bytes_mod_order(bytes);
            if canonical_field_bytes(*value).as_slice() != bytes {
                return None;
            }
        }
        let artifact = Self {
            semantic_fields,
            padded_fields,
            context_digest,
            commitment_root,
            evaluations,
            proof_payload: payload[HEADER..].to_vec(),
        };
        artifact.verify().then_some(artifact)
    }

    fn verify(&self) -> bool {
        if self.semantic_fields == 0
            || self.padded_fields != self.semantic_fields.next_power_of_two()
        {
            return false;
        }
        let params = direct_whir_parameters(self.padded_fields);
        let ds = direct_whir_domain(&params, self.context_digest);
        let Some(proof) = parse_canonical_proof(&self.proof_payload) else {
            return false;
        };
        let mle = Box::new(MultilinearExtension::new(
            direct_tail_point(
                self.padded_fields,
                self.context_digest,
                self.commitment_root,
            )
            .0,
        ));
        let sparse_left = Box::new(SparseCovector::new(
            self.padded_fields,
            direct_tail_sparse_entries(b"LiLAC/direct-tail-left/v1", self.semantic_fields),
        ));
        let sparse_right = Box::new(SparseCovector::new(
            self.padded_fields,
            direct_tail_sparse_entries(b"LiLAC/direct-tail-right/v1", self.semantic_fields),
        ));
        let forms: Vec<Box<dyn Evaluate<Identity<Field192>>>> =
            vec![mle, sparse_left, sparse_right];
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let Ok(commitment) = params.receive_commitment(&mut verifier_state) else {
            return false;
        };
        if commitment.matrix_root().0 != self.commitment_root {
            return false;
        }
        let Ok(final_claim) = params.verify(&mut verifier_state, &[&commitment], &self.evaluations)
        else {
            return false;
        };
        final_claim
            .verify(
                forms
                    .iter()
                    .map(|form| form.as_ref() as &dyn LinearForm<Field192>),
            )
            .is_ok()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StrongTerminalProof {
    component_roots: Vec<Digest>,
    ood_block_root: Digest,
    terminal_source_root: Digest,
    selected_front: SelectedRowFront,
    membership: PackedSumcheckProof,
    tail: DirectTailProofArtifact,
    terminal_witness: Vec<Field192>,
}

impl StrongTerminalProof {
    const MAGIC: &'static [u8; 8] = b"LILSTR01";

    fn source_root(&self) -> Option<Digest> {
        self.component_roots.get(STRONG_INVERSE_RATE - 1).copied()
    }

    fn verify(&self) -> bool {
        let level = strong_level();
        let zeros = zero_roots(30);
        let mut tail_roots = self.component_roots.clone();
        tail_roots.push(self.terminal_source_root);
        self.component_roots.len() == 2 * STRONG_INVERSE_RATE + 1
            && self.selected_front.verify(level, &self.component_roots)
            && self.selected_front.selected
                == selected_rows(
                    12,
                    STRONG_INVERSE_RATE * STRONG_GROUP,
                    STRONG_QUERIES,
                    &self.component_roots,
                )
            && self.membership.fields
                == (STRONG_INVERSE_RATE * STRONG_GROUP).next_power_of_two() * STRONG_WIDTH
            && self.membership.claimed_sum == Field192::ZERO
            && self.membership.roots
                == local_relation_roots(b"strong-QA-membership", 12, &self.component_roots)
            && self.membership.challenges().is_some()
            && audit_strong_terminal_restoration(
                level,
                &self.selected_front,
                &self.membership,
                self.ood_block_root,
                self.terminal_source_root,
                &self.terminal_witness,
                &zeros,
            )
            && self.tail.semantic_fields == STRONG_TERMINAL_FIELDS
            && self.tail.context_digest == direct_tail_context_digest(&tail_roots)
            && self.tail.verify()
            && terminal_witness_matches_tail(&self.terminal_witness, &self.tail)
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let front = self.selected_front.serialize();
        let membership = self.membership.serialize();
        let tail = self.tail.serialize();
        let mut output = Vec::with_capacity(
            36 + self.component_roots.len() * 32
                + 64
                + front.len()
                + membership.len()
                + tail.len()
                + self.terminal_witness.len() * 24,
        );
        output.extend_from_slice(Self::MAGIC);
        for value in [
            self.component_roots.len(),
            front.len(),
            membership.len(),
            tail.len(),
            self.terminal_witness.len(),
            0,
            0,
        ] {
            output.extend_from_slice(&(value as u32).to_le_bytes());
        }
        for root in &self.component_roots {
            output.extend_from_slice(root);
        }
        output.extend_from_slice(&self.ood_block_root);
        output.extend_from_slice(&self.terminal_source_root);
        output.extend_from_slice(&front);
        output.extend_from_slice(&membership);
        output.extend_from_slice(&tail);
        for value in &self.terminal_witness {
            output.extend_from_slice(&canonical_field_bytes(*value));
        }
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 7 * 4;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let sizes = (0..7)
            .map(|index| {
                Some(
                    u32::from_le_bytes(payload[8 + index * 4..12 + index * 4].try_into().ok()?)
                        as usize,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        if sizes[0] != 2 * STRONG_INVERSE_RATE + 1
            || sizes[4] != STRONG_TERMINAL_FIELDS
            || sizes[5] != 0
            || sizes[6] != 0
            || payload.len()
                != HEADER + sizes[0] * 32 + 64 + sizes[1] + sizes[2] + sizes[3] + sizes[4] * 24
        {
            return None;
        }
        let mut position = HEADER;
        let component_roots = (0..sizes[0])
            .map(|_| {
                let root = payload.get(position..position + 32)?.try_into().ok()?;
                position += 32;
                Some(root)
            })
            .collect::<Option<Vec<Digest>>>()?;
        let ood_block_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let terminal_source_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let selected_front = SelectedRowFront::deserialize(
            payload.get(position..position + sizes[1])?,
            strong_level(),
        )?;
        position += sizes[1];
        let membership =
            PackedSumcheckProof::deserialize(payload.get(position..position + sizes[2])?)?;
        position += sizes[2];
        let tail =
            DirectTailProofArtifact::deserialize(payload.get(position..position + sizes[3])?)?;
        position += sizes[3];
        let mut terminal_witness = Vec::with_capacity(sizes[4]);
        for _ in 0..sizes[4] {
            let bytes = payload.get(position..position + 24)?;
            let value = Field192::from_le_bytes_mod_order(bytes);
            if canonical_field_bytes(value).as_slice() != bytes {
                return None;
            }
            terminal_witness.push(value);
            position += 24;
        }
        let proof = Self {
            component_roots,
            ood_block_root,
            terminal_source_root,
            selected_front,
            membership,
            tail,
            terminal_witness,
        };
        proof.verify().then_some(proof)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StrongTransitionCore {
    round: usize,
    component_roots: Vec<Digest>,
    ood_block_root: Digest,
    terminal_source_root: Digest,
    selected_front: SelectedRowFront,
    membership: PackedSumcheckProof,
}

impl StrongTransitionCore {
    const MAGIC: &'static [u8; 8] = b"LILSTC01";

    fn level(&self) -> Option<Level> {
        strong_round_level(self.round)
    }

    fn source_root(&self) -> Option<Digest> {
        self.component_roots.get(STRONG_INVERSE_RATE - 1).copied()
    }

    fn terminal_fields(&self) -> Option<usize> {
        self.level()
            .map(|level| 2 * level.next_blocks * level.width)
    }

    fn verify(&self) -> bool {
        let Some(level) = self.level() else {
            return false;
        };
        let relation_level = 12 + self.round;
        self.component_roots.len() == 2 * STRONG_INVERSE_RATE + 1
            && self.selected_front.verify(level, &self.component_roots)
            && self.selected_front.selected
                == selected_rows(
                    relation_level,
                    STRONG_INVERSE_RATE * level.group,
                    STRONG_QUERIES,
                    &self.component_roots,
                )
            && self.membership.fields
                == (STRONG_INVERSE_RATE * level.group).next_power_of_two()
                    * level.width.next_power_of_two()
            && self.membership.claimed_sum == Field192::ZERO
            && self.membership.roots
                == local_relation_roots(
                    b"strong-QA-membership",
                    relation_level,
                    &self.component_roots,
                )
            && self.membership.challenges().is_some()
            && derived_terminal_root_for_level(
                level,
                &self.selected_front,
                self.ood_block_root,
                self.terminal_fields().unwrap().next_power_of_two(),
                &zero_roots(30),
            ) == Some(self.terminal_source_root)
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let front = self.selected_front.serialize();
        let membership = self.membership.serialize();
        let mut output = Vec::with_capacity(
            32 + self.component_roots.len() * 32 + 64 + front.len() + membership.len(),
        );
        output.extend_from_slice(Self::MAGIC);
        for value in [
            self.round,
            self.component_roots.len(),
            front.len(),
            membership.len(),
            0,
            0,
        ] {
            output.extend_from_slice(&(value as u32).to_le_bytes());
        }
        for root in &self.component_roots {
            output.extend_from_slice(root);
        }
        output.extend_from_slice(&self.ood_block_root);
        output.extend_from_slice(&self.terminal_source_root);
        output.extend_from_slice(&front);
        output.extend_from_slice(&membership);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 6 * 4;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let values = (0..6)
            .map(|index| {
                Some(
                    u32::from_le_bytes(payload[8 + index * 4..12 + index * 4].try_into().ok()?)
                        as usize,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        let round = values[0];
        let level = strong_round_level(round)?;
        if values[1] != 2 * STRONG_INVERSE_RATE + 1
            || values[4] != 0
            || values[5] != 0
            || payload.len() != HEADER + values[1] * 32 + 64 + values[2] + values[3]
        {
            return None;
        }
        let mut position = HEADER;
        let component_roots = (0..values[1])
            .map(|_| {
                let root = payload.get(position..position + 32)?.try_into().ok()?;
                position += 32;
                Some(root)
            })
            .collect::<Option<Vec<Digest>>>()?;
        let ood_block_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let terminal_source_root = payload.get(position..position + 32)?.try_into().ok()?;
        position += 32;
        let selected_front =
            SelectedRowFront::deserialize(payload.get(position..position + values[2])?, level)?;
        position += values[2];
        let membership = PackedSumcheckProof::deserialize(&payload[position..])?;
        let proof = Self {
            round,
            component_roots,
            ood_block_root,
            terminal_source_root,
            selected_front,
            membership,
        };
        proof.verify().then_some(proof)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StrongBaseProof {
    tail: DirectTailProofArtifact,
    terminal_witness: Vec<Field192>,
}

impl StrongBaseProof {
    const MAGIC: &'static [u8; 8] = b"LILSBP01";

    fn verify(&self, core: &StrongTransitionCore) -> bool {
        let Some(level) = core.level() else {
            return false;
        };
        let mut tail_roots = core.component_roots.clone();
        tail_roots.push(core.terminal_source_root);
        self.terminal_witness.len() == core.terminal_fields().unwrap()
            && audit_strong_terminal_restoration(
                level,
                &core.selected_front,
                &core.membership,
                core.ood_block_root,
                core.terminal_source_root,
                &self.terminal_witness,
                &zero_roots(30),
            )
            && self.tail.semantic_fields == self.terminal_witness.len()
            && self.tail.context_digest == direct_tail_context_digest(&tail_roots)
            && self.tail.verify()
            && terminal_witness_matches_tail(&self.terminal_witness, &self.tail)
    }

    fn serialize(&self, core: &StrongTransitionCore) -> Vec<u8> {
        assert!(self.verify(core));
        let tail = self.tail.serialize();
        let mut output = Vec::with_capacity(24 + tail.len() + self.terminal_witness.len() * 24);
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(tail.len() as u32).to_le_bytes());
        output.extend_from_slice(&(self.terminal_witness.len() as u32).to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        output.extend_from_slice(&tail);
        for value in &self.terminal_witness {
            output.extend_from_slice(&canonical_field_bytes(*value));
        }
        output
    }

    fn deserialize(payload: &[u8], core: &StrongTransitionCore) -> Option<Self> {
        if payload.len() < 24 || &payload[..8] != Self::MAGIC {
            return None;
        }
        let tail_size = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let fields = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        let reserved0 = u32::from_le_bytes(payload[16..20].try_into().ok()?);
        let reserved1 = u32::from_le_bytes(payload[20..24].try_into().ok()?);
        if fields != core.terminal_fields()?
            || reserved0 != 0
            || reserved1 != 0
            || payload.len() != 24 + tail_size + fields * 24
        {
            return None;
        }
        let tail = DirectTailProofArtifact::deserialize(payload.get(24..24 + tail_size)?)?;
        let mut position = 24 + tail_size;
        let mut terminal_witness = Vec::with_capacity(fields);
        for _ in 0..fields {
            let bytes = payload.get(position..position + 24)?;
            let value = Field192::from_le_bytes_mod_order(bytes);
            if canonical_field_bytes(value).as_slice() != bytes {
                return None;
            }
            terminal_witness.push(value);
            position += 24;
        }
        let proof = Self {
            tail,
            terminal_witness,
        };
        proof.verify(core).then_some(proof)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProductionCertificateProof {
    transitions: Vec<ProductionTransitionProof>,
    tail: DirectTailProofArtifact,
}

impl ProductionCertificateProof {
    const MAGIC: &'static [u8; 8] = b"LILCERT1";

    fn verify(&self) -> bool {
        if self.transitions.len() != LEVELS.len()
            || self.tail.semantic_fields
                != 2 * LEVELS.last().unwrap().next_blocks * LEVELS.last().unwrap().width
            || !self.tail.verify()
        {
            return false;
        }
        let mut transcript_roots = Vec::new();
        for (level_index, transition) in self.transitions.iter().enumerate() {
            if transition.level != level_index || !transition.verify() {
                return false;
            }
            if level_index > 0
                && self.transitions[level_index - 1].next_source_root
                    != transition.component_roots[LEVELS[level_index].inverse_rate - 1]
            {
                return false;
            }
            transcript_roots.extend_from_slice(&transition.component_roots);
            let level = LEVELS[level_index];
            let expected_selected = selected_rows(
                level_index,
                level.inverse_rate * level.group,
                level.next_blocks - 1,
                &transcript_roots,
            );
            if transition.selected_front.selected != expected_selected
                || transition.algebra.qa_membership.roots
                    != local_relation_roots(b"QA-membership", level_index, &transcript_roots)
                || transition.algebra.dual_view_copy.roots
                    != local_relation_roots(b"dual-view-copy", level_index, &transcript_roots)
            {
                return false;
            }
        }
        if self.tail.context_digest != direct_tail_context_digest(&transcript_roots) {
            return false;
        }
        transcript_roots.push(self.tail.commitment_root);
        self.transitions
            .iter()
            .enumerate()
            .all(|(level, transition)| {
                transition.algebra.phi_link.as_ref().is_some_and(|proof| {
                    proof.roots == local_relation_roots(b"Phi-link", level, &transcript_roots)
                })
            })
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let transitions = self
            .transitions
            .iter()
            .map(ProductionTransitionProof::serialize)
            .collect::<Vec<_>>();
        let tail = self.tail.serialize();
        let mut output = Vec::with_capacity(
            16 + transitions
                .iter()
                .map(|proof| 4 + proof.len())
                .sum::<usize>()
                + tail.len(),
        );
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(transitions.len() as u32).to_le_bytes());
        output.extend_from_slice(&(tail.len() as u32).to_le_bytes());
        for proof in transitions {
            output.extend_from_slice(&(proof.len() as u32).to_le_bytes());
            output.extend_from_slice(&proof);
        }
        output.extend_from_slice(&tail);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        if payload.len() < 16 || &payload[..8] != Self::MAGIC {
            return None;
        }
        let count = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let tail_size = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        if count != LEVELS.len() {
            return None;
        }
        let mut position = 16;
        let mut transitions = Vec::with_capacity(count);
        for _ in 0..count {
            let size =
                u32::from_le_bytes(payload.get(position..position + 4)?.try_into().ok()?) as usize;
            position += 4;
            transitions.push(ProductionTransitionProof::deserialize(
                payload.get(position..position + size)?,
            )?);
            position += size;
        }
        if payload.len() != position + tail_size {
            return None;
        }
        let tail = DirectTailProofArtifact::deserialize(&payload[position..])?;
        let proof = Self { transitions, tail };
        proof.verify().then_some(proof)
    }
}

fn verify_certificate_core(
    transitions: &[ProductionTransitionProof],
    terminal_root: Digest,
) -> bool {
    if transitions.len() != LEVELS.len() {
        return false;
    }
    let mut transcript_roots = Vec::new();
    for (level_index, transition) in transitions.iter().enumerate() {
        if transition.level != level_index || !transition.verify() {
            return false;
        }
        if level_index > 0
            && transitions[level_index - 1].next_source_root
                != transition.component_roots[LEVELS[level_index].inverse_rate - 1]
        {
            return false;
        }
        transcript_roots.extend_from_slice(&transition.component_roots);
        let level = LEVELS[level_index];
        if transition.selected_front.selected
            != selected_rows(
                level_index,
                level.inverse_rate * level.group,
                level.next_blocks - 1,
                &transcript_roots,
            )
            || transition.algebra.qa_membership.roots
                != local_relation_roots(b"QA-membership", level_index, &transcript_roots)
            || transition.algebra.dual_view_copy.roots
                != local_relation_roots(b"dual-view-copy", level_index, &transcript_roots)
        {
            return false;
        }
    }
    transcript_roots.push(terminal_root);
    transitions.iter().enumerate().all(|(level, transition)| {
        transition.algebra.phi_link.as_ref().is_some_and(|proof| {
            proof.roots == local_relation_roots(b"Phi-link", level, &transcript_roots)
        })
    })
}

fn joint_tail_context(carry_roots: &[Digest], certificate_roots: &[Digest]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/joint-CarryOpen-certificate-tail/v1");
    hasher.update(&(carry_roots.len() as u64).to_le_bytes());
    for root in carry_roots {
        hasher.update(root);
    }
    hasher.update(&(certificate_roots.len() as u64).to_le_bytes());
    for root in certificate_roots {
        hasher.update(root);
    }
    *hasher.finalize().as_bytes()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProductionEndToEndProof {
    carryopen: CarryOpenCoreProof,
    certificate: Vec<ProductionTransitionProof>,
    joint_tail: DirectTailProofArtifact,
    restoration_witness: Vec<Field192>,
}

impl ProductionEndToEndProof {
    const MAGIC: &'static [u8; 8] = b"LILE2E02";

    fn certificate_roots(&self) -> Vec<Digest> {
        self.certificate
            .iter()
            .flat_map(|proof| proof.component_roots.iter().copied())
            .collect()
    }

    fn verify(&self) -> bool {
        let certificate_roots = self.certificate_roots();
        let certificate_fields =
            2 * LEVELS.last().unwrap().next_blocks * LEVELS.last().unwrap().width;
        let expected_fields = certificate_fields + CARRYOPEN_TERMINAL_FIELDS;
        let zeros = zero_roots(30);
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        self.joint_tail.semantic_fields == expected_fields
            && self.restoration_witness.len() == expected_fields
            && self.joint_tail.context_digest
                == joint_tail_context(&self.carryopen.component_roots, &certificate_roots)
            && self.joint_tail.verify()
            && terminal_witness_matches_tail(&self.restoration_witness, &self.joint_tail)
            && self.certificate.last().is_some_and(|proof| {
                audit_standard_terminal_restoration(
                    LEVELS.len() - 1,
                    proof,
                    &self.restoration_witness[..certificate_fields],
                    &zeros,
                )
            })
            && audit_tensor_carry_restoration(
                carryopen_level(),
                &self.carryopen.selected_front,
                &self.restoration_witness[certificate_fields..],
                &horizontal_spectra,
                &self.carryopen.membership,
                &self.carryopen.evaluation,
                &zeros,
            )
            && self.carryopen.verify(self.joint_tail.commitment_root)
            && verify_certificate_core(&self.certificate, self.joint_tail.commitment_root)
    }

    fn serialized_component_bytes(&self) -> usize {
        self.serialize().len()
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let carryopen = self.carryopen.serialize();
        let certificate = self
            .certificate
            .iter()
            .map(ProductionTransitionProof::serialize)
            .collect::<Vec<_>>();
        let tail = self.joint_tail.serialize();
        let mut output = Vec::with_capacity(
            24 + carryopen.len()
                + certificate
                    .iter()
                    .map(|proof| 4 + proof.len())
                    .sum::<usize>()
                + tail.len()
                + self.restoration_witness.len() * 24,
        );
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(carryopen.len() as u32).to_le_bytes());
        output.extend_from_slice(&(certificate.len() as u32).to_le_bytes());
        output.extend_from_slice(&(tail.len() as u32).to_le_bytes());
        output.extend_from_slice(&(self.restoration_witness.len() as u32).to_le_bytes());
        output.extend_from_slice(&carryopen);
        for proof in certificate {
            output.extend_from_slice(&(proof.len() as u32).to_le_bytes());
            output.extend_from_slice(&proof);
        }
        output.extend_from_slice(&tail);
        for value in &self.restoration_witness {
            output.extend_from_slice(&canonical_field_bytes(*value));
        }
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 4 * 4;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let carry_size = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let certificate_count = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        let tail_size = u32::from_le_bytes(payload[16..20].try_into().ok()?) as usize;
        let witness_fields = u32::from_le_bytes(payload[20..24].try_into().ok()?) as usize;
        let expected_fields =
            2 * LEVELS.last()?.next_blocks * LEVELS.last()?.width + CARRYOPEN_TERMINAL_FIELDS;
        if certificate_count != LEVELS.len() || witness_fields != expected_fields {
            return None;
        }
        let mut position = HEADER;
        let carryopen =
            CarryOpenCoreProof::deserialize(payload.get(position..position + carry_size)?)?;
        position += carry_size;
        let mut certificate = Vec::with_capacity(certificate_count);
        for _ in 0..certificate_count {
            let size =
                u32::from_le_bytes(payload.get(position..position + 4)?.try_into().ok()?) as usize;
            position += 4;
            certificate.push(ProductionTransitionProof::deserialize(
                payload.get(position..position + size)?,
            )?);
            position += size;
        }
        if payload.len() != position + tail_size + witness_fields * 24 {
            return None;
        }
        let joint_tail =
            DirectTailProofArtifact::deserialize(payload.get(position..position + tail_size)?)?;
        position += tail_size;
        let mut restoration_witness = Vec::with_capacity(witness_fields);
        for _ in 0..witness_fields {
            let bytes = payload.get(position..position + 24)?;
            let value = Field192::from_le_bytes_mod_order(bytes);
            if canonical_field_bytes(value).as_slice() != bytes {
                return None;
            }
            restoration_witness.push(value);
            position += 24;
        }
        let proof = Self {
            carryopen,
            certificate,
            joint_tail,
            restoration_witness,
        };
        proof.verify().then_some(proof)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StrongEndToEndProof {
    carryopen: CarryOpenCoreProof,
    certificate: Vec<ProductionTransitionProof>,
    strong: StrongTerminalProof,
}

impl StrongEndToEndProof {
    const MAGIC: &'static [u8; 8] = b"LILSE201";

    fn verify(&self) -> bool {
        let Some(certificate_root) = self.certificate.last().map(|proof| proof.next_source_root)
        else {
            return false;
        };
        self.strong.verify()
            && self.strong.source_root()
                == Some(parent(
                    certificate_root,
                    self.carryopen.terminal_source_root,
                ))
            && self.carryopen.verify(self.carryopen.terminal_source_root)
            && verify_certificate_core(&self.certificate, certificate_root)
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let carry = self.carryopen.serialize();
        let certificate = self
            .certificate
            .iter()
            .map(ProductionTransitionProof::serialize)
            .collect::<Vec<_>>();
        let strong = self.strong.serialize();
        let mut output = Vec::with_capacity(
            24 + carry.len()
                + certificate
                    .iter()
                    .map(|proof| proof.len() + 4)
                    .sum::<usize>()
                + strong.len(),
        );
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(carry.len() as u32).to_le_bytes());
        output.extend_from_slice(&(certificate.len() as u32).to_le_bytes());
        output.extend_from_slice(&(strong.len() as u32).to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        output.extend_from_slice(&carry);
        for proof in certificate {
            output.extend_from_slice(&(proof.len() as u32).to_le_bytes());
            output.extend_from_slice(&proof);
        }
        output.extend_from_slice(&strong);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 24;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let carry_size = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let certificate_count = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        let strong_size = u32::from_le_bytes(payload[16..20].try_into().ok()?) as usize;
        let reserved = u32::from_le_bytes(payload[20..24].try_into().ok()?);
        if certificate_count != LEVELS.len() || reserved != 0 {
            return None;
        }
        let mut position = HEADER;
        let carryopen =
            CarryOpenCoreProof::deserialize(payload.get(position..position + carry_size)?)?;
        position += carry_size;
        let mut certificate = Vec::with_capacity(certificate_count);
        for _ in 0..certificate_count {
            let size =
                u32::from_le_bytes(payload.get(position..position + 4)?.try_into().ok()?) as usize;
            position += 4;
            certificate.push(ProductionTransitionProof::deserialize(
                payload.get(position..position + size)?,
            )?);
            position += size;
        }
        if payload.len() != position + strong_size {
            return None;
        }
        let strong = StrongTerminalProof::deserialize(&payload[position..])?;
        let proof = Self {
            carryopen,
            certificate,
            strong,
        };
        proof.verify().then_some(proof)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RecursiveStrongEndToEndProof {
    carryopen: CarryOpenCoreProof,
    certificate: Vec<ProductionTransitionProof>,
    strong: Vec<StrongTransitionCore>,
    base: StrongBaseProof,
}

impl RecursiveStrongEndToEndProof {
    const MAGIC: &'static [u8; 8] = b"LILRS301";

    fn verify(&self) -> bool {
        if self.strong.len() != STRONG_ROUNDS || self.certificate.len() != LEVELS.len() {
            return false;
        }
        let certificate_root = self.certificate.last().unwrap().next_source_root;
        if self.strong[0].source_root()
            != Some(parent(
                certificate_root,
                self.carryopen.terminal_source_root,
            ))
        {
            return false;
        }
        for round in 0..self.strong.len() {
            if self.strong[round].round != round || !self.strong[round].verify() {
                return false;
            }
            if round > 0
                && self.strong[round].source_root()
                    != Some(self.strong[round - 1].terminal_source_root)
            {
                return false;
            }
        }
        self.base.verify(self.strong.last().unwrap())
            && self.carryopen.verify(self.carryopen.terminal_source_root)
            && verify_certificate_core(&self.certificate, certificate_root)
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let carry = self.carryopen.serialize();
        let certificate = self
            .certificate
            .iter()
            .map(ProductionTransitionProof::serialize)
            .collect::<Vec<_>>();
        let strong = self
            .strong
            .iter()
            .map(StrongTransitionCore::serialize)
            .collect::<Vec<_>>();
        let base = self.base.serialize(self.strong.last().unwrap());
        let mut output = Vec::with_capacity(
            28 + carry.len()
                + certificate
                    .iter()
                    .map(|proof| proof.len() + 4)
                    .sum::<usize>()
                + strong.iter().map(|proof| proof.len() + 4).sum::<usize>()
                + base.len(),
        );
        output.extend_from_slice(Self::MAGIC);
        for value in [carry.len(), certificate.len(), strong.len(), base.len(), 0] {
            output.extend_from_slice(&(value as u32).to_le_bytes());
        }
        output.extend_from_slice(&carry);
        for proof in certificate {
            output.extend_from_slice(&(proof.len() as u32).to_le_bytes());
            output.extend_from_slice(&proof);
        }
        for proof in strong {
            output.extend_from_slice(&(proof.len() as u32).to_le_bytes());
            output.extend_from_slice(&proof);
        }
        output.extend_from_slice(&base);
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 5 * 4;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let values = (0..5)
            .map(|index| {
                Some(
                    u32::from_le_bytes(payload[8 + index * 4..12 + index * 4].try_into().ok()?)
                        as usize,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        if values[1] != LEVELS.len() || values[2] != STRONG_ROUNDS || values[4] != 0 {
            return None;
        }
        let mut position = HEADER;
        let carryopen =
            CarryOpenCoreProof::deserialize(payload.get(position..position + values[0])?)?;
        position += values[0];
        let mut certificate = Vec::with_capacity(values[1]);
        for _ in 0..values[1] {
            let size =
                u32::from_le_bytes(payload.get(position..position + 4)?.try_into().ok()?) as usize;
            position += 4;
            certificate.push(ProductionTransitionProof::deserialize(
                payload.get(position..position + size)?,
            )?);
            position += size;
        }
        let mut strong = Vec::with_capacity(values[2]);
        for _ in 0..values[2] {
            let size =
                u32::from_le_bytes(payload.get(position..position + 4)?.try_into().ok()?) as usize;
            position += 4;
            strong.push(StrongTransitionCore::deserialize(
                payload.get(position..position + size)?,
            )?);
            position += size;
        }
        if payload.len() != position + values[3] {
            return None;
        }
        let base = StrongBaseProof::deserialize(&payload[position..], strong.last()?)?;
        let proof = Self {
            carryopen,
            certificate,
            strong,
            base,
        };
        proof.verify().then_some(proof)
    }
}

#[derive(Debug)]
struct RelationWhirMeasurement {
    commit: Duration,
    witness_regeneration: Duration,
    prove: Duration,
    verify: Duration,
    proof_bytes: usize,
    canonical_proof_bytes: usize,
    commitment_roots: [Digest; 2],
}

fn relation_precommit_context(component_roots: &[Digest]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/combined-relation-WHIR-precommit/v1");
    hasher.update(&(component_roots.len() as u64).to_le_bytes());
    for root in component_roots {
        hasher.update(root);
    }
    *hasher.finalize().as_bytes()
}

fn run_semantic_kernel_with_relation_whir(
    batch_lanes: usize,
    direct_whir_tail: bool,
    verifier_repetitions: usize,
) -> (KernelMeasurement, RelationWhirMeasurement) {
    assert!(batch_lanes > 0 && verifier_repetitions > 0);
    let semantic_start = Instant::now();
    let (
        mut left,
        mut right,
        component_roots,
        terminal_tail,
        terminal_context,
        terminal_binding_root,
        transition_proofs,
        breakdown,
    ) = semantic_vectors(batch_lanes, direct_whir_tail);
    let setup = semantic_start.elapsed();
    let fields = left.len();
    let padded_fields = fields.next_power_of_two();
    left.resize(padded_fields, Field192::ZERO);
    right.resize(padded_fields, Field192::ZERO);

    let params = whir_parameters(padded_fields, 128, 20);
    let context = relation_precommit_context(&component_roots);
    let ds = DomainSeparator::protocol(&params)
        .session(&format!(
            "LiLAC/combined-relation-WHIR/v2/context={}",
            hex::encode(context)
        ))
        .instance(&Empty);
    let mut prover_state = ProverState::new_std(&ds);
    let commit_start = Instant::now();
    let left_witness = params.commit(&mut prover_state, &[left.as_slice()]);
    let right_witness = params.commit(&mut prover_state, &[right.as_slice()]);
    let commitment_roots = [
        left_witness.matrix_witness.root().0,
        right_witness.matrix_witness.root().0,
    ];
    let commit = commit_start.elapsed();

    let mut sumcheck_roots = aggregate_roots(&component_roots);
    sumcheck_roots.extend(commitment_roots);
    left.truncate(fields);
    right.truncate(fields);
    let measurement = prove_vectors(
        left,
        right,
        setup,
        sumcheck_roots,
        component_roots.len() + commitment_roots.len(),
        Some(breakdown),
        Some(transition_proofs),
        Some(terminal_tail),
        terminal_context,
        Some(terminal_binding_root),
    );

    let regeneration_start = Instant::now();
    let (mut left, mut right, regenerated_roots, _, _, _, _, _) =
        semantic_vectors(batch_lanes, direct_whir_tail);
    let witness_regeneration = regeneration_start.elapsed();
    assert_eq!(regenerated_roots, component_roots);
    left.resize(padded_fields, Field192::ZERO);
    right.resize(padded_fields, Field192::ZERO);
    let sumcheck = measurement.sumcheck_proof();
    assert_eq!(sumcheck.roots.last().copied(), Some(commitment_roots[1]));
    assert_eq!(
        sumcheck.roots.get(sumcheck.roots.len() - 2).copied(),
        Some(commitment_roots[0])
    );
    let point = sumcheck
        .challenges()
        .expect("precommitted relation sumcheck must verify");
    let mle = Box::new(MultilinearExtension::new(point));
    let embedding = Identity::<Field192>::new();
    let evaluations = vec![
        mle.evaluate(&embedding, &left),
        mle.evaluate(&embedding, &right),
    ];
    assert_eq!(evaluations[0], sumcheck.terminal_left);
    assert_eq!(evaluations[1], sumcheck.terminal_right);

    let prove_start = Instant::now();
    let _ = params.prove(
        &mut prover_state,
        vec![
            Cow::Borrowed(left.as_slice()),
            Cow::Borrowed(right.as_slice()),
        ],
        vec![Cow::Owned(left_witness), Cow::Owned(right_witness)],
        vec![mle.clone()],
        Cow::Borrowed(evaluations.as_slice()),
    );
    let prove = prove_start.elapsed();
    let proof = prover_state.proof();
    let proof_bytes = proof.narg_string.len() + proof.hints.len();
    let canonical_payload = canonical_proof_payload(&proof);
    let parsed_proof = parse_canonical_proof(&canonical_payload)
        .expect("canonical precommitted relation-WHIR proof must parse");

    let verify_start = Instant::now();
    for _ in 0..verifier_repetitions {
        assert!(verify_relation_whir_proof(
            &params,
            &ds,
            &parsed_proof,
            &evaluations,
            mle.as_ref(),
            commitment_roots,
        ));
    }
    let verify = verify_start.elapsed() / verifier_repetitions as u32;
    let mut wrong_evaluations = evaluations;
    wrong_evaluations[0] += Field192::ONE;
    assert!(!verify_relation_whir_proof(
        &params,
        &ds,
        &parsed_proof,
        &wrong_evaluations,
        mle.as_ref(),
        commitment_roots,
    ));

    (
        measurement,
        RelationWhirMeasurement {
            commit,
            witness_regeneration,
            prove,
            verify,
            proof_bytes,
            canonical_proof_bytes: canonical_payload.len(),
            commitment_roots,
        },
    )
}

#[cfg(debug_assertions)]
fn canonical_proof_payload(proof: &Proof) -> Vec<u8> {
    let mut payload = Vec::new();
    ciborium::into_writer(proof, &mut payload).expect("WHIR proof serialization must succeed");
    payload
}

#[cfg(debug_assertions)]
fn parse_canonical_proof(payload: &[u8]) -> Option<Proof> {
    ciborium::from_reader(payload).ok()
}

#[cfg(not(debug_assertions))]
fn canonical_proof_payload(proof: &Proof) -> Vec<u8> {
    let mut payload = Vec::with_capacity(24 + proof.narg_string.len() + proof.hints.len());
    payload.extend_from_slice(b"LILWHP01");
    payload.extend_from_slice(&(proof.narg_string.len() as u64).to_le_bytes());
    payload.extend_from_slice(&(proof.hints.len() as u64).to_le_bytes());
    payload.extend_from_slice(&proof.narg_string);
    payload.extend_from_slice(&proof.hints);
    payload
}

#[cfg(not(debug_assertions))]
fn parse_canonical_proof(payload: &[u8]) -> Option<Proof> {
    if payload.len() < 24 || &payload[..8] != b"LILWHP01" {
        return None;
    }
    let narg_size = u64::from_le_bytes(payload[8..16].try_into().ok()?) as usize;
    let hint_size = u64::from_le_bytes(payload[16..24].try_into().ok()?) as usize;
    if payload.len() != 24 + narg_size + hint_size {
        return None;
    }
    Some(Proof {
        narg_string: payload[24..24 + narg_size].to_vec(),
        hints: payload[24 + narg_size..].to_vec(),
    })
}

fn verify_relation_whir_proof(
    params: &whir::protocols::whir::Config<Identity<Field192>>,
    ds: &DomainSeparator<'static, Empty>,
    proof: &Proof,
    evaluations: &[Field192],
    mle: &MultilinearExtension<Field192>,
    expected_roots: [Digest; 2],
) -> bool {
    let mut verifier_state = VerifierState::new_std(ds, proof);
    let Ok(left_commitment) = params.receive_commitment(&mut verifier_state) else {
        return false;
    };
    let Ok(right_commitment) = params.receive_commitment(&mut verifier_state) else {
        return false;
    };
    if left_commitment.matrix_root().0 != expected_roots[0]
        || right_commitment.matrix_root().0 != expected_roots[1]
    {
        return false;
    }
    let Ok(final_claim) = params.verify(
        &mut verifier_state,
        &[&left_commitment, &right_commitment],
        evaluations,
    ) else {
        return false;
    };
    final_claim
        .verify([mle as &dyn LinearForm<Field192>])
        .is_ok()
}

#[cfg(test)]
fn relation_whir_context(sumcheck: &PackedSumcheckProof) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/combined-relation-WHIR-context/v1");
    hasher.update(&sumcheck.serialize());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
fn run_relation_whir(
    mut left: Vec<Field192>,
    mut right: Vec<Field192>,
    sumcheck: &PackedSumcheckProof,
    verifier_repetitions: usize,
    security_level: usize,
    pow_bits: usize,
) -> RelationWhirMeasurement {
    assert!(verifier_repetitions > 0);
    assert_eq!(left.len(), sumcheck.fields);
    assert_eq!(right.len(), sumcheck.fields);
    let padded_fields = sumcheck.fields.next_power_of_two();
    left.resize(padded_fields, Field192::ZERO);
    right.resize(padded_fields, Field192::ZERO);
    let params = whir_parameters(padded_fields, security_level, pow_bits);
    let context = relation_whir_context(sumcheck);
    let ds = DomainSeparator::protocol(&params)
        .session(&format!(
            "LiLAC/combined-relation-WHIR/v1/context={}",
            hex::encode(context)
        ))
        .instance(&Empty);
    let mut prover_state = ProverState::new_std(&ds);

    let commit_start = Instant::now();
    let left_witness = params.commit(&mut prover_state, &[left.as_slice()]);
    let right_witness = params.commit(&mut prover_state, &[right.as_slice()]);
    let commitment_roots = [
        left_witness.matrix_witness.root().0,
        right_witness.matrix_witness.root().0,
    ];
    let commit = commit_start.elapsed();

    let point = sumcheck
        .challenges()
        .expect("sumcheck must verify before its terminal PCS opening");
    let mle = Box::new(MultilinearExtension::new(point));
    let embedding = Identity::<Field192>::new();
    let evaluations = vec![
        mle.evaluate(&embedding, &left),
        mle.evaluate(&embedding, &right),
    ];
    assert_eq!(evaluations[0], sumcheck.terminal_left);
    assert_eq!(evaluations[1], sumcheck.terminal_right);

    let prove_start = Instant::now();
    let _ = params.prove(
        &mut prover_state,
        vec![
            Cow::Borrowed(left.as_slice()),
            Cow::Borrowed(right.as_slice()),
        ],
        vec![Cow::Owned(left_witness), Cow::Owned(right_witness)],
        vec![mle.clone()],
        Cow::Borrowed(evaluations.as_slice()),
    );
    let prove = prove_start.elapsed();
    let proof = prover_state.proof();
    let proof_bytes = proof.narg_string.len() + proof.hints.len();
    let canonical_payload = canonical_proof_payload(&proof);
    let parsed_proof = parse_canonical_proof(&canonical_payload)
        .expect("canonical relation-WHIR proof must parse");

    HASH_COUNTER.reset();
    let verify_start = Instant::now();
    for _ in 0..verifier_repetitions {
        assert!(verify_relation_whir_proof(
            &params,
            &ds,
            &parsed_proof,
            &evaluations,
            mle.as_ref(),
            commitment_roots,
        ));
    }
    let verify = verify_start.elapsed() / verifier_repetitions as u32;
    let mut wrong_evaluations = evaluations.clone();
    wrong_evaluations[0] += Field192::ONE;
    assert!(!verify_relation_whir_proof(
        &params,
        &ds,
        &parsed_proof,
        &wrong_evaluations,
        mle.as_ref(),
        commitment_roots,
    ));
    RelationWhirMeasurement {
        commit,
        witness_regeneration: Duration::ZERO,
        prove,
        verify,
        proof_bytes,
        canonical_proof_bytes: canonical_payload.len(),
        commitment_roots,
    }
}

fn run_direct_whir_tail(
    mut tail: Vec<Field192>,
    context_digest: Digest,
    verifier_repetitions: usize,
) -> WhirTailMeasurement {
    assert!(verifier_repetitions > 0);
    let semantic_fields = tail.len();
    let padded_fields = semantic_fields.next_power_of_two();
    tail.resize(padded_fields, Field192::ZERO);
    let params = direct_whir_parameters(padded_fields);
    let ds = direct_whir_domain(&params, context_digest);
    let mut prover_state = ProverState::new_std(&ds);

    let commit_start = Instant::now();
    let witness = params.commit(&mut prover_state, &[tail.as_slice()]);
    let commitment_root = witness.matrix_witness.root().0;
    let commit = commit_start.elapsed();

    let point = direct_tail_point(padded_fields, context_digest, commitment_root);
    let mle = Box::new(MultilinearExtension::new(point.0.clone()));
    let sparse_left = Box::new(SparseCovector::new(
        padded_fields,
        direct_tail_sparse_entries(b"LiLAC/direct-tail-left/v1", semantic_fields),
    ));
    let sparse_right = Box::new(SparseCovector::new(
        padded_fields,
        direct_tail_sparse_entries(b"LiLAC/direct-tail-right/v1", semantic_fields),
    ));
    let embedding = Identity::<Field192>::new();
    let evaluations = vec![
        mle.evaluate(&embedding, &tail),
        sparse_left.evaluate(&embedding, &tail),
        sparse_right.evaluate(&embedding, &tail),
    ];
    let verifier_forms: Vec<Box<dyn Evaluate<Identity<Field192>>>> =
        vec![mle.clone(), sparse_left.clone(), sparse_right.clone()];
    let prover_forms: Vec<Box<dyn LinearForm<Field192>>> = vec![mle, sparse_left, sparse_right];

    let prove_start = Instant::now();
    let _ = params.prove(
        &mut prover_state,
        vec![Cow::Borrowed(tail.as_slice())],
        vec![Cow::Owned(witness)],
        prover_forms,
        Cow::Borrowed(evaluations.as_slice()),
    );
    let prove = prove_start.elapsed();
    let proof = prover_state.proof();
    let proof_bytes = proof.narg_string.len() + proof.hints.len();
    let artifact = DirectTailProofArtifact {
        semantic_fields,
        padded_fields,
        context_digest,
        commitment_root,
        evaluations: evaluations.clone().try_into().unwrap(),
        proof_payload: canonical_proof_payload(&proof),
    };
    assert!(artifact.verify());
    let serialized_artifact = artifact.serialize();
    assert_eq!(
        DirectTailProofArtifact::deserialize(&serialized_artifact),
        Some(artifact.clone())
    );

    HASH_COUNTER.reset();
    let verify_start = Instant::now();
    for _ in 0..verifier_repetitions {
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params.receive_commitment(&mut verifier_state).unwrap();
        let final_claim = params
            .verify(&mut verifier_state, &[&commitment], &evaluations)
            .unwrap();
        final_claim
            .verify(
                verifier_forms
                    .iter()
                    .map(|form| form.as_ref() as &dyn LinearForm<Field192>),
            )
            .unwrap();
    }
    let verify = verify_start.elapsed() / verifier_repetitions as u32;
    let verifier_hashes = HASH_COUNTER.get() / verifier_repetitions;
    WhirTailMeasurement {
        semantic_fields,
        padded_fields,
        commit,
        prove,
        verify,
        proof_bytes,
        verifier_hashes,
        commitment_root,
        artifact,
    }
}

fn main() {
    let args = Args::parse();
    assert!(args.fields > 1 && args.iterations > 0 && args.batch_lanes > 0);
    if args.semantic {
        assert_eq!(args.fields, PRODUCTION_FIELDS);
    }
    assert!(!args.direct_whir_tail || args.semantic);
    assert!(!args.relation_whir || args.semantic);
    assert!(!args.final_only || (args.semantic && args.carryopen && args.direct_whir_tail));
    let variables = args.fields.next_power_of_two().trailing_zeros() as usize;
    if args.fields == PRODUCTION_FIELDS {
        assert_eq!(variables, PRODUCTION_VARIABLES);
    }
    let mut setup_ms = Vec::with_capacity(args.iterations);
    let mut polynomial_ms = Vec::with_capacity(args.iterations);
    let mut folding_ms = Vec::with_capacity(args.iterations);
    let mut total_ms = Vec::with_capacity(args.iterations);
    let mut generator_ms = Vec::with_capacity(args.iterations);
    let mut proof_encoding_ms = Vec::with_capacity(args.iterations);
    let mut proof_commitment_ms = Vec::with_capacity(args.iterations);
    let mut index_oracle_ms = Vec::with_capacity(args.iterations);
    let mut index_commitment_ms = Vec::with_capacity(args.iterations);
    let mut source_opening_ms = Vec::with_capacity(args.iterations);
    let mut copy_phi_ms = Vec::with_capacity(args.iterations);
    let mut transition_sumcheck_ms = Vec::with_capacity(args.iterations);
    let mut terminal_commitment_ms = Vec::with_capacity(args.iterations);
    let mut component_root_count = 0;
    let mut transcript_root_count = 0;
    let mut source_opening_bytes = 0;
    let mut source_frontier_hashes = 0;
    let mut transition_proof_bytes = 0;
    let mut transition_proof_count = 0;
    let mut checksum = Field192::ZERO;
    let mut serialized_sumcheck_bytes = 0;
    let mut whir_tail_measurements = Vec::with_capacity(args.iterations);
    let mut relation_whir_measurements = Vec::with_capacity(args.iterations);
    let mut certificate_verify_ms = Vec::with_capacity(args.iterations);
    let mut certificate_proof_bytes = 0;
    let mut carryopen_measurements = Vec::with_capacity(args.iterations);
    let joint_mode = args.carryopen && args.semantic && args.direct_whir_tail;
    let certificate_direct_tail = args.direct_whir_tail && !joint_mode;
    let mut joint_tail_measurements = Vec::with_capacity(args.iterations);
    let mut end_to_end_verify_ms = Vec::with_capacity(args.iterations);
    let mut end_to_end_proof_bytes = 0;
    let mut strong_end_to_end_verify_ms = Vec::with_capacity(args.iterations);
    let mut strong_end_to_end_proof_bytes = 0;
    let mut strong_encode_ms = Vec::with_capacity(args.iterations);
    let mut strong_algebra_ms = Vec::with_capacity(args.iterations);
    let mut strong_terminal_ms = Vec::with_capacity(args.iterations);
    let mut recursive_strong_proof_bytes = 0;
    let mut recursive_strong_verify_ms = Vec::with_capacity(args.iterations);
    let mut recursive_strong_prover_ms = Vec::with_capacity(args.iterations);
    let mut final_total_prover_ms = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let iteration_start = Instant::now();
        if args.carryopen {
            carryopen_measurements.push(run_production_carryopen(args.whir_verifier_repetitions));
        }
        let measurement = if args.semantic && args.relation_whir {
            let (measurement, relation) = run_semantic_kernel_with_relation_whir(
                args.batch_lanes,
                certificate_direct_tail,
                args.whir_verifier_repetitions,
            );
            relation_whir_measurements.push(relation);
            measurement
        } else if args.semantic {
            run_semantic_kernel(args.batch_lanes, certificate_direct_tail)
        } else {
            run_kernel(args.fields)
        };
        let sumcheck_payload = measurement.sumcheck_proof().serialize();
        let parsed_sumcheck = PackedSumcheckProof::deserialize(&sumcheck_payload)
            .expect("serialized combined sumcheck must verify");
        assert_eq!(parsed_sumcheck, measurement.sumcheck_proof());
        serialized_sumcheck_bytes = sumcheck_payload.len();
        if let Some(proofs) = &measurement.transition_proofs {
            transition_proof_count = proofs.len();
            transition_proof_bytes = proofs.iter().map(|proof| proof.serialize().len()).sum();
            assert!(proofs.iter().all(|proof| {
                let payload = proof.serialize();
                ProductionTransitionProof::deserialize(&payload).as_ref() == Some(proof)
            }));
        }
        assert_eq!(measurement.pairs.len(), variables);
        setup_ms.push(measurement.setup.as_secs_f64() * 1_000.0);
        polynomial_ms.push(measurement.polynomial.as_secs_f64() * 1_000.0);
        folding_ms.push(measurement.folding.as_secs_f64() * 1_000.0);
        total_ms.push(measurement.total.as_secs_f64() * 1_000.0);
        component_root_count = measurement.component_root_count;
        transcript_root_count = measurement.transcript_root_count;
        if let Some(breakdown) = measurement.semantic_breakdown {
            generator_ms.push(breakdown.generator_preprocessing.as_secs_f64() * 1_000.0);
            proof_encoding_ms.push(breakdown.proof_encoding.as_secs_f64() * 1_000.0);
            proof_commitment_ms.push(breakdown.proof_commitment.as_secs_f64() * 1_000.0);
            index_oracle_ms.push(breakdown.index_oracle.as_secs_f64() * 1_000.0);
            index_commitment_ms.push(breakdown.index_commitment.as_secs_f64() * 1_000.0);
            source_opening_ms.push(breakdown.source_opening.as_secs_f64() * 1_000.0);
            if source_opening_bytes != 0 {
                assert_eq!(source_opening_bytes, breakdown.source_opening_bytes);
                assert_eq!(source_frontier_hashes, breakdown.source_frontier_hashes);
            }
            source_opening_bytes = breakdown.source_opening_bytes;
            source_frontier_hashes = breakdown.source_frontier_hashes;
            copy_phi_ms.push(breakdown.copy_and_phi.as_secs_f64() * 1_000.0);
            transition_sumcheck_ms.push(breakdown.transition_sumchecks.as_secs_f64() * 1_000.0);
            assert_eq!(transition_proof_bytes, breakdown.transition_proof_bytes);
            terminal_commitment_ms.push(breakdown.terminal_commitment.as_secs_f64() * 1_000.0);
        }
        if joint_mode {
            let carry_measurement = carryopen_measurements.last().unwrap();
            if args.final_only {
                let prover_start = Instant::now();
                let proof = build_recursive_strong_end_to_end(
                    carry_measurement,
                    &measurement,
                    args.whir_verifier_repetitions,
                );
                let payload = proof.serialize();
                recursive_strong_proof_bytes = payload.len();
                recursive_strong_prover_ms.push(prover_start.elapsed().as_secs_f64() * 1_000.0);
                final_total_prover_ms.push(iteration_start.elapsed().as_secs_f64() * 1_000.0);
                let verify_start = Instant::now();
                let parsed = RecursiveStrongEndToEndProof::deserialize(&payload)
                    .expect("final-only recursive strong proof must verify");
                recursive_strong_verify_ms.push(verify_start.elapsed().as_secs_f64() * 1_000.0);
                assert_eq!(parsed, proof);
                let mut changed = payload;
                *changed.last_mut().unwrap() ^= 1;
                assert!(RecursiveStrongEndToEndProof::deserialize(&changed).is_none());
            } else {
                let mut joint_terminal = measurement.terminal_tail.clone().unwrap();
                joint_terminal.extend_from_slice(&carry_measurement.terminal_source);
                let mut certificate = measurement.transition_proofs.clone().unwrap();
                let certificate_roots = certificate
                    .iter()
                    .flat_map(|proof| proof.component_roots.iter().copied())
                    .collect::<Vec<_>>();
                let context = joint_tail_context(
                    &carry_measurement.proof.component_roots,
                    &certificate_roots,
                );
                let restoration_witness = joint_terminal.clone();
                joint_tail_measurements.push(run_direct_whir_tail(
                    joint_terminal,
                    context,
                    args.whir_verifier_repetitions,
                ));
                let joint_tail = joint_tail_measurements.last().unwrap().artifact.clone();
                let mut carryopen = CarryOpenCoreProof::from(&carry_measurement.proof);
                let mut carry_roots = carryopen.component_roots.clone();
                carry_roots.push(joint_tail.commitment_root);
                carryopen.phi_link = zero_phi_link_proof(
                    b"CarryOpen-Phi-link",
                    b"CarryOpen-Phi-link",
                    10,
                    CARRYOPEN_TERMINAL_FIELDS,
                    &carry_roots,
                );
                let mut certificate_final_roots = certificate_roots;
                certificate_final_roots.push(joint_tail.commitment_root);
                for (level_index, proof) in certificate.iter_mut().enumerate() {
                    proof.algebra.phi_link = Some(zero_phi_link_proof(
                        b"Phi-link",
                        b"Phi-link",
                        level_index,
                        2 * LEVELS[level_index].next_blocks * LEVELS[level_index].width,
                        &certificate_final_roots,
                    ));
                }
                let proof = ProductionEndToEndProof {
                    carryopen,
                    certificate,
                    joint_tail,
                    restoration_witness,
                };
                assert!(proof.verify());
                let mut changed_terminal = proof.clone();
                let certificate_fields =
                    2 * LEVELS.last().unwrap().next_blocks * LEVELS.last().unwrap().width;
                changed_terminal.restoration_witness[2 * LEVELS.last().unwrap().width] +=
                    Field192::ONE;
                assert!(!changed_terminal.verify());
                let mut changed_carry_parity = proof.clone();
                changed_carry_parity.restoration_witness
                    [certificate_fields + CARRYOPEN_OOD_FIELDS + CARRYOPEN_WIDTH] += Field192::ONE;
                assert!(!changed_carry_parity.verify());
                let mut changed_link = proof.clone();
                changed_link.certificate[0].next_source_root[0] ^= 1;
                assert!(!changed_link.verify());
                let mut changed_successor = proof.clone();
                let source_root_index = LEVELS[1].inverse_rate - 1;
                changed_successor.certificate[1].component_roots[source_root_index][0] ^= 1;
                assert!(!changed_successor.verify());
                let payload = proof.serialize();
                end_to_end_proof_bytes = payload.len();
                assert_eq!(end_to_end_proof_bytes, proof.serialized_component_bytes());
                let verify_start = Instant::now();
                let parsed = ProductionEndToEndProof::deserialize(&payload)
                    .expect("serialized end-to-end proof must verify");
                end_to_end_verify_ms.push(verify_start.elapsed().as_secs_f64() * 1_000.0);
                assert_eq!(parsed, proof);
                let mut changed = payload;
                changed[96] ^= 1;
                assert!(ProductionEndToEndProof::deserialize(&changed).is_none());

                let mut strong_certificate = measurement.transition_proofs.clone().unwrap();
                let certificate_root = strong_certificate.last().unwrap().next_source_root;
                let strong_measurement = run_strong_terminal_switch(
                    measurement.terminal_tail.as_ref().unwrap(),
                    &carry_measurement.terminal_source,
                    certificate_root,
                    carry_measurement.proof.terminal_source_root,
                    args.whir_verifier_repetitions,
                );
                strong_encode_ms.push(strong_measurement.encode_and_commit.as_secs_f64() * 1_000.0);
                strong_algebra_ms.push(strong_measurement.algebra.as_secs_f64() * 1_000.0);
                strong_terminal_ms.push(strong_measurement.terminal.as_secs_f64() * 1_000.0);
                let mut strong_carry = CarryOpenCoreProof::from(&carry_measurement.proof);
                let mut strong_carry_roots = strong_carry.component_roots.clone();
                strong_carry_roots.push(strong_carry.terminal_source_root);
                strong_carry.phi_link = zero_phi_link_proof(
                    b"CarryOpen-Phi-link",
                    b"CarryOpen-Phi-link",
                    10,
                    CARRYOPEN_TERMINAL_FIELDS,
                    &strong_carry_roots,
                );
                let mut strong_certificate_roots = strong_certificate
                    .iter()
                    .flat_map(|proof| proof.component_roots.iter().copied())
                    .collect::<Vec<_>>();
                strong_certificate_roots.push(certificate_root);
                for (level_index, transition) in strong_certificate.iter_mut().enumerate() {
                    transition.algebra.phi_link = Some(zero_phi_link_proof(
                        b"Phi-link",
                        b"Phi-link",
                        level_index,
                        2 * LEVELS[level_index].next_blocks * LEVELS[level_index].width,
                        &strong_certificate_roots,
                    ));
                }
                let strong_proof = StrongEndToEndProof {
                    carryopen: strong_carry,
                    certificate: strong_certificate,
                    strong: strong_measurement.proof,
                };
                assert!(strong_proof.verify());
                let strong_payload = strong_proof.serialize();
                strong_end_to_end_proof_bytes = strong_payload.len();
                let verify_start = Instant::now();
                let parsed = StrongEndToEndProof::deserialize(&strong_payload)
                    .expect("strong-terminal end-to-end proof must verify");
                strong_end_to_end_verify_ms.push(verify_start.elapsed().as_secs_f64() * 1_000.0);
                assert_eq!(parsed, strong_proof);
                let mut changed = strong_payload;
                *changed.last_mut().unwrap() ^= 1;
                assert!(StrongEndToEndProof::deserialize(&changed).is_none());

                let recursive_start = Instant::now();
                let mut recursive_source = materialize_strong_source(
                    measurement.terminal_tail.as_ref().unwrap(),
                    &carry_measurement.terminal_source,
                );
                let mut recursive_expected_root = parent(
                    certificate_root,
                    carry_measurement.proof.terminal_source_root,
                );
                let mut recursive_cores = Vec::with_capacity(STRONG_ROUNDS);
                let mut recursive_base = None;
                for round in 0..STRONG_ROUNDS {
                    let round_measurement = run_strong_round(
                        round,
                        recursive_source,
                        recursive_expected_root,
                        round + 1 == STRONG_ROUNDS,
                        args.whir_verifier_repetitions,
                    );
                    let _accounted_round_time = round_measurement.encode_and_commit
                        + round_measurement.algebra
                        + round_measurement.terminal;
                    recursive_expected_root = round_measurement.core.terminal_source_root;
                    recursive_source = round_measurement.terminal_source;
                    recursive_base = round_measurement.base.or(recursive_base);
                    recursive_cores.push(round_measurement.core);
                }
                let recursive_proof = RecursiveStrongEndToEndProof {
                    carryopen: strong_proof.carryopen.clone(),
                    certificate: strong_proof.certificate.clone(),
                    strong: recursive_cores,
                    base: recursive_base.expect("last strong round must attach the terminal base"),
                };
                assert!(recursive_proof.verify());
                let recursive_payload = recursive_proof.serialize();
                recursive_strong_proof_bytes = recursive_payload.len();
                recursive_strong_prover_ms.push(recursive_start.elapsed().as_secs_f64() * 1_000.0);
                let verify_start = Instant::now();
                let parsed = RecursiveStrongEndToEndProof::deserialize(&recursive_payload)
                    .expect("recursive strong-terminal proof must verify");
                recursive_strong_verify_ms.push(verify_start.elapsed().as_secs_f64() * 1_000.0);
                assert_eq!(parsed, recursive_proof);
                let mut changed = recursive_payload;
                *changed.last_mut().unwrap() ^= 1;
                assert!(RecursiveStrongEndToEndProof::deserialize(&changed).is_none());
            }
        } else if args.direct_whir_tail {
            whir_tail_measurements.push(run_direct_whir_tail(
                measurement.terminal_tail.clone().unwrap(),
                measurement.terminal_context.unwrap(),
                args.whir_verifier_repetitions,
            ));
            assert_eq!(
                whir_tail_measurements.last().unwrap().commitment_root,
                measurement.terminal_binding_root.unwrap()
            );
            let certificate = ProductionCertificateProof {
                transitions: measurement.transition_proofs.clone().unwrap(),
                tail: whir_tail_measurements.last().unwrap().artifact.clone(),
            };
            assert!(certificate.verify());
            let payload = certificate.serialize();
            certificate_proof_bytes = payload.len();
            let verify_start = Instant::now();
            let parsed = ProductionCertificateProof::deserialize(&payload)
                .expect("canonical production certificate proof must verify");
            certificate_verify_ms.push(verify_start.elapsed().as_secs_f64() * 1_000.0);
            assert_eq!(parsed, certificate);
            let mut changed = payload;
            changed[64] ^= 1;
            assert!(ProductionCertificateProof::deserialize(&changed).is_none());
        }
        checksum +=
            measurement.terminal_left + measurement.terminal_right + measurement.terminal_claim;
    }
    println!("LiLAC packed combined product-sumcheck kernel");
    println!(
        "- vector mode: {}",
        if args.semantic {
            "four production QA/copy relations + cross-level Phi links"
        } else {
            "dense synthetic product pairs"
        }
    );
    println!(
        "- semantic fields / implicit capacity: {}/{}",
        args.fields,
        args.fields.next_power_of_two()
    );
    println!("- variables: {variables}");
    println!("- transcript-bound component roots: {component_root_count}");
    println!("- serialized algebra root aggregates: {transcript_root_count}");
    println!(
        "- two live Field192 vectors: {:.3} GiB",
        2.0 * args.fields as f64 * 24.0 / (1_u64 << 30) as f64
    );
    println!(
        "- witness-vector setup median/p95: {:.3}/{:.3} ms",
        percentile(&setup_ms, 0.5),
        percentile(&setup_ms, 0.95)
    );
    println!(
        "- round-polynomial generation median/p95: {:.3}/{:.3} ms",
        percentile(&polynomial_ms, 0.5),
        percentile(&polynomial_ms, 0.95)
    );
    println!(
        "- in-place prefix folding median/p95: {:.3}/{:.3} ms",
        percentile(&folding_ms, 0.5),
        percentile(&folding_ms, 0.95)
    );
    println!(
        "- 26-round kernel median/p95: {:.3}/{:.3} ms",
        percentile(&total_ms, 0.5),
        percentile(&total_ms, 0.95)
    );
    println!("- legacy compressed-transcript model: {TRANSCRIPT_BYTES} B");
    println!("- canonical serialized sumcheck proof: {serialized_sumcheck_bytes} B");
    println!(
        "- terminal checksum nonzero: {}",
        checksum != Field192::ZERO
    );
    if args.carryopen {
        let message_ms = carryopen_measurements
            .iter()
            .map(|item| item.message_commit.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let precarry_ms = carryopen_measurements
            .iter()
            .map(|item| item.precarry.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let encode_ms = carryopen_measurements
            .iter()
            .map(|item| item.encode_and_commit.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let algebra_ms = carryopen_measurements
            .iter()
            .map(|item| item.algebra.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let terminal_ms = carryopen_measurements
            .iter()
            .map(|item| item.terminal.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let first = &carryopen_measurements[0].proof;
        println!("- production CarryOpen code: vertical and horizontal systematic QA rate 1/4, 2^16 x 2^8 message, 1024-field tensor rows, q=205");
        println!(
            "- CarryOpen M-root/pre-Carry/encode+commit/algebra/tail medians: {:.3}/{:.3}/{:.3}/{:.3}/{:.3} ms",
            percentile(&message_ms, 0.5),
            percentile(&precarry_ms, 0.5),
            percentile(&encode_ms, 0.5),
            percentile(&algebra_ms, 0.5),
            percentile(&terminal_ms, 0.5)
        );
        println!(
            "- CarryOpen terminal fields/padded: {}/{}; serialized component proof: {} B",
            first.tail.semantic_fields,
            first.tail.padded_fields,
            first.serialized_component_bytes()
        );
        println!("- CarryOpen verifier checks actual M root, pre-Carry claim, QA membership/evaluation, F/W subtree front, Phi link, and terminal WHIR; claim mutation rejected");
    }
    if args.semantic {
        println!(
            "- fixed-G preprocessing median/p95: {:.3}/{:.3} ms",
            percentile(&generator_ms, 0.5),
            percentile(&generator_ms, 0.95)
        );
        println!(
            "- four-level F encoding median/p95: {:.3}/{:.3} ms",
            percentile(&proof_encoding_ms, 0.5),
            percentile(&proof_encoding_ms, 0.95)
        );
        println!(
            "- splice/F field-Merkle commitments median/p95: {:.3}/{:.3} ms",
            percentile(&proof_commitment_ms, 0.5),
            percentile(&proof_commitment_ms, 0.95)
        );
        println!(
            "- W construction median/p95: {:.3}/{:.3} ms",
            percentile(&index_oracle_ms, 0.5),
            percentile(&index_oracle_ms, 0.95)
        );
        println!(
            "- W field-Merkle commitments median/p95: {:.3}/{:.3} ms",
            percentile(&index_commitment_ms, 0.5),
            percentile(&index_commitment_ms, 0.95)
        );
        println!(
            "- authenticated F/W selected-row fronts median/p95: {:.3}/{:.3} ms",
            percentile(&source_opening_ms, 0.5),
            percentile(&source_opening_ms, 0.95)
        );
        println!(
            "- selected-row front communication/frontier: {} B / {} hashes",
            source_opening_bytes, source_frontier_hashes
        );
        println!(
            "- copy/Phi witness work median/p95: {:.3}/{:.3} ms",
            percentile(&copy_phi_ms, 0.5),
            percentile(&copy_phi_ms, 0.95)
        );
        println!(
            "- {} local QA/copy transition proofs: {} B; prover median/p95 {:.3}/{:.3} ms",
            transition_proof_count,
            transition_proof_bytes,
            percentile(&transition_sumcheck_ms, 0.5),
            percentile(&transition_sumcheck_ms, 0.95)
        );
        println!(
            "- terminal Phi commitment median/p95: {:.3}/{:.3} ms",
            percentile(&terminal_commitment_ms, 0.5),
            percentile(&terminal_commitment_ms, 0.95)
        );
        println!("- all F/W/Phi challenges, row fronts, and local transition transcripts are bound to concrete roots");
        if args.relation_whir {
            let commit_ms = relation_whir_measurements
                .iter()
                .map(|measurement| measurement.commit.as_secs_f64() * 1_000.0)
                .collect::<Vec<_>>();
            let prove_ms = relation_whir_measurements
                .iter()
                .map(|measurement| measurement.prove.as_secs_f64() * 1_000.0)
                .collect::<Vec<_>>();
            let regeneration_ms = relation_whir_measurements
                .iter()
                .map(|measurement| measurement.witness_regeneration.as_secs_f64() * 1_000.0)
                .collect::<Vec<_>>();
            let verify_ms = relation_whir_measurements
                .iter()
                .map(|measurement| measurement.verify.as_secs_f64() * 1_000.0)
                .collect::<Vec<_>>();
            let first = &relation_whir_measurements[0];
            assert!(relation_whir_measurements.iter().all(|measurement| {
                measurement.proof_bytes == first.proof_bytes
                    && measurement.canonical_proof_bytes == first.canonical_proof_bytes
                    && measurement.commitment_roots == first.commitment_roots
            }));
            println!(
                "- relation-vector WHIR commit median/p95: {:.3}/{:.3} ms",
                percentile(&commit_ms, 0.5),
                percentile(&commit_ms, 0.95)
            );
            println!(
                "- relation-vector WHIR prove median/p95: {:.3}/{:.3} ms",
                percentile(&prove_ms, 0.5),
                percentile(&prove_ms, 0.95)
            );
            println!(
                "- memory-bounded witness regeneration median/p95: {:.3}/{:.3} ms",
                percentile(&regeneration_ms, 0.5),
                percentile(&regeneration_ms, 0.95)
            );
            println!(
                "- relation-vector WHIR verify median/p95: {:.3}/{:.3} ms",
                percentile(&verify_ms, 0.5),
                percentile(&verify_ms, 0.95)
            );
            println!(
                "- relation-vector WHIR proof raw/canonical: {}/{} B (two commitments, one shared MLE form)",
                first.proof_bytes,
                first.canonical_proof_bytes,
            );
            println!(
                "- combined sumcheck terminal values are PCS-opened at its Fiat--Shamir point"
            );
        }
        if args.direct_whir_tail && !joint_mode {
            let commit_ms = whir_tail_measurements
                .iter()
                .map(|measurement| measurement.commit.as_secs_f64() * 1_000.0)
                .collect::<Vec<_>>();
            let prove_ms = whir_tail_measurements
                .iter()
                .map(|measurement| measurement.prove.as_secs_f64() * 1_000.0)
                .collect::<Vec<_>>();
            let verify_ms = whir_tail_measurements
                .iter()
                .map(|measurement| measurement.verify.as_secs_f64() * 1_000.0)
                .collect::<Vec<_>>();
            let first = &whir_tail_measurements[0];
            assert!(whir_tail_measurements.iter().all(|measurement| {
                measurement.semantic_fields == first.semantic_fields
                    && measurement.padded_fields == first.padded_fields
                    && measurement.proof_bytes == first.proof_bytes
            }));
            println!(
                "- direct WHIR tail semantic/padded fields: {}/{}",
                first.semantic_fields, first.padded_fields
            );
            println!(
                "- direct WHIR tail commit median/p95: {:.3}/{:.3} ms",
                percentile(&commit_ms, 0.5),
                percentile(&commit_ms, 0.95)
            );
            println!(
                "- direct WHIR tail prove median/p95: {:.3}/{:.3} ms",
                percentile(&prove_ms, 0.5),
                percentile(&prove_ms, 0.95)
            );
            println!(
                "- direct WHIR tail verify median/p95: {:.3}/{:.3} ms",
                percentile(&verify_ms, 0.5),
                percentile(&verify_ms, 0.95)
            );
            println!(
                "- direct WHIR tail proof / verifier hashes: {} B / {}",
                first.proof_bytes, first.verifier_hashes
            );
            println!("- direct tail forms: one full terminal MLE and two width-19 sparse forms over the same WHIR message");
            println!(
                "- canonical four-transition certificate proof: {} B; parse+verify median/p95 {:.3}/{:.3} ms",
                certificate_proof_bytes,
                percentile(&certificate_verify_ms, 0.5),
                percentile(&certificate_verify_ms, 0.95)
            );
            println!("- certificate verifier checks transition order, FS-derived rows, F/W subtree proofs, QA/copy/Phi transcripts, and the terminal WHIR root");
        }
        if joint_mode {
            if args.final_only {
                println!(
                    "- final-only recursive strong schedule: 2^20 -> 2^16 -> 2^14 -> {} terminal fields",
                    2 * STRONG_NEXT_BLOCKS * (1 << 7)
                );
                println!(
                    "- final-only canonical proof: {} B; total prover median/p95 {:.3}/{:.3} ms; strong-chain median/p95 {:.3}/{:.3} ms; canonical verifier median/p95 {:.3}/{:.3} ms",
                    recursive_strong_proof_bytes,
                    percentile(&final_total_prover_ms, 0.5),
                    percentile(&final_total_prover_ms, 0.95),
                    percentile(&recursive_strong_prover_ms, 0.5),
                    percentile(&recursive_strong_prover_ms, 0.95),
                    percentile(&recursive_strong_verify_ms, 0.5),
                    percentile(&recursive_strong_verify_ms, 0.95)
                );
            } else {
                let commit_ms = joint_tail_measurements
                    .iter()
                    .map(|measurement| measurement.commit.as_secs_f64() * 1_000.0)
                    .collect::<Vec<_>>();
                let prove_ms = joint_tail_measurements
                    .iter()
                    .map(|measurement| measurement.prove.as_secs_f64() * 1_000.0)
                    .collect::<Vec<_>>();
                let verify_ms = joint_tail_measurements
                    .iter()
                    .map(|measurement| measurement.verify.as_secs_f64() * 1_000.0)
                    .collect::<Vec<_>>();
                let first = &joint_tail_measurements[0];
                println!(
                    "- joint certificate/CarryOpen WHIR terminal semantic/padded fields: {}/{}",
                    first.semantic_fields, first.padded_fields
                );
                println!(
                    "- joint WHIR commit/prove/verify median: {:.3}/{:.3}/{:.3} ms; raw proof {} B",
                    percentile(&commit_ms, 0.5),
                    percentile(&prove_ms, 0.5),
                    percentile(&verify_ms, 0.5),
                    first.proof_bytes
                );
                println!(
                    "- single end-to-end component proof: {} B; verifier median/p95 {:.3}/{:.3} ms",
                    end_to_end_proof_bytes,
                    percentile(&end_to_end_verify_ms, 0.5),
                    percentile(&end_to_end_verify_ms, 0.95)
                );
                println!(
                "- one joint root binds the 247,192-field certificate Phi state and {}-field tensor CarryOpen terminal; root mutation rejected",
                CARRYOPEN_TERMINAL_FIELDS
            );
                println!(
                "- strong late switch: systematic QA rate 1/{}, finite-length distance target 0.95, q={}, source/terminal fields {}/{}",
                STRONG_INVERSE_RATE,
                STRONG_QUERIES,
                STRONG_SOURCE_FIELDS,
                STRONG_TERMINAL_FIELDS
            );
                println!(
                    "- strong switch encode+commit/algebra/terminal median: {:.3}/{:.3}/{:.3} ms",
                    percentile(&strong_encode_ms, 0.5),
                    percentile(&strong_algebra_ms, 0.5),
                    percentile(&strong_terminal_ms, 0.5)
                );
                println!(
                "- strong-terminal canonical end-to-end proof: {} B; verifier median/p95 {:.3}/{:.3} ms",
                strong_end_to_end_proof_bytes,
                percentile(&strong_end_to_end_verify_ms, 0.5),
                percentile(&strong_end_to_end_verify_ms, 0.95)
            );
                println!(
                    "- recursive strong schedule: 2^20 -> 2^16 -> 2^14 -> {} terminal fields",
                    2 * STRONG_NEXT_BLOCKS * (1 << 7)
                );
                println!(
                "- recursive-strong canonical end-to-end proof: {} B; added strong-chain prover/verifier median {:.3}/{:.3} ms",
                recursive_strong_proof_bytes,
                percentile(&recursive_strong_prover_ms, 0.5),
                percentile(&recursive_strong_verify_ms, 0.5)
            );
            }
        }
    } else {
        println!("- benchmark uses dense synthetic product vectors at the exact production length; semantic witness generation is excluded");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_non_power_of_two_kernel_reaches_valid_terminal_claim() {
        let measurement = run_kernel(37);
        assert_eq!(measurement.pairs.len(), 6);
        assert_eq!(
            measurement.terminal_claim,
            measurement.terminal_left * measurement.terminal_right
        );
        let proof = measurement.sumcheck_proof();
        let payload = proof.serialize();
        assert_eq!(PackedSumcheckProof::deserialize(&payload), Some(proof));
        let mut changed = payload;
        *changed.last_mut().unwrap() ^= 1;
        assert!(PackedSumcheckProof::deserialize(&changed).is_none());
    }

    #[test]
    fn whir_opens_the_serialized_sumcheck_terminal_values() {
        let fields = 37;
        let left = (0..fields).map(left_value).collect::<Vec<_>>();
        let mut right = (0..fields).map(right_value).collect::<Vec<_>>();
        let initial_claim = dot(&left, &right);
        right[0] -= initial_claim * left[0].inverse().unwrap();
        let component_roots = (0_u64..4)
            .map(|index| *blake3::hash(&index.to_le_bytes()).as_bytes())
            .collect::<Vec<_>>();
        let measurement = prove_vectors(
            left.clone(),
            right.clone(),
            Duration::ZERO,
            aggregate_roots(&component_roots),
            component_roots.len(),
            None,
            None,
            None,
            None,
            None,
        );
        let opened = run_relation_whir(left, right, &measurement.sumcheck_proof(), 1, 32, 0);
        assert!(opened.proof_bytes > 0);
        assert_ne!(opened.commitment_roots[0], opened.commitment_roots[1]);
    }

    #[test]
    fn local_transition_algebra_proof_roundtrips_and_rejects_mutation() {
        let fields = 37;
        let left = (0..fields).map(left_value).collect::<Vec<_>>();
        let mut right = (0..fields).map(right_value).collect::<Vec<_>>();
        let claim = dot(&left, &right);
        right[0] -= claim * left[0].inverse().unwrap();
        let qa = prove_local_product_relation(
            left.clone(),
            right.clone(),
            local_relation_roots(b"QA-membership", 0, &[[7_u8; 32]]),
        );
        let copy = prove_local_product_relation(
            left,
            right,
            local_relation_roots(b"dual-view-copy", 0, &[[11_u8; 32]]),
        );
        let proof = TransitionAlgebraProof {
            level: 0,
            qa_membership: qa.clone(),
            dual_view_copy: copy,
            phi_link: Some(qa),
        };
        let payload = proof.serialize();
        assert_eq!(TransitionAlgebraProof::deserialize(&payload), Some(proof));
        let mut changed = payload;
        *changed.last_mut().unwrap() ^= 1;
        assert!(TransitionAlgebraProof::deserialize(&changed).is_none());
    }

    #[test]
    fn active_fold_matches_explicit_zero_padding() {
        let weight = Field192::from(7_u64);
        let mut packed = (0..13).map(left_value).collect::<Vec<_>>();
        let mut padded = packed.clone();
        padded.resize(16, Field192::ZERO);
        let active = packed.len();
        packed.resize(16, Field192::ZERO);
        let packed_len = fold_active(&mut packed, active, weight);
        let padded_len = fold_active(&mut padded, 16, weight);
        assert_eq!(packed_len, padded_len);
        assert_eq!(&packed[..packed_len], &padded[..padded_len]);
    }

    #[test]
    fn local_semantic_qa_and_full_view_copy_claims_are_zero() {
        let level = Level {
            raw: 10,
            blocks: 2,
            block_semantic: 5,
            components: &[(5, 32)],
            group: 8,
            width: 3,
            row_span: 4,
            inverse_rate: 2,
            next_blocks: 2,
        };
        let source = (0..level.raw)
            .map(|index| semantic_value(0, index))
            .collect::<Vec<_>>();
        let mut qa_left = vec![Field192::ZERO; level.qa_fields()];
        let mut qa_right = vec![Field192::ZERO; level.qa_fields()];
        let zeros = zero_roots(10);
        let spectra = vec![generator_spectrum(0, 0, level.group)];
        populate_proof_codeword(level, &source, &mut qa_right, &spectra, 2);
        let generator_roots = spectra
            .iter()
            .map(|spectrum| prefix_root(spectrum, level.group, &zeros))
            .collect::<Vec<_>>();
        let mut roots = level_commitment_roots(level, &source, &qa_right, &generator_roots, &zeros);
        populate_index_oracle(0, level, &mut qa_left, &spectra, &roots);
        roots.push(index_oracle_root(level, &qa_left, &zeros));
        assert_eq!(dot(&qa_left, &qa_right), Field192::ZERO);

        let mut copy_left = vec![Field192::ZERO; level.view_capacity()];
        let mut copy_right = vec![Field192::ZERO; level.view_capacity()];
        populate_copy_relation(0, level, &source, &mut copy_left, &mut copy_right, &roots);
        assert_eq!(dot(&copy_left, &copy_right), Field192::ZERO);
        copy_left[7] += Field192::ONE;
        assert_ne!(dot(&copy_left, &copy_right), Field192::ZERO);

        let indices = selected_rows(0, level.inverse_rate * level.group, 1, &roots);
        let qa_tensor = prove_tensor_product_relation(
            qa_left.clone(),
            qa_right.clone(),
            level.inverse_rate * level.group,
            level.width,
            local_relation_roots(b"test-QA", 0, &roots),
        );
        let expected_next = derive_next_source(
            level,
            &qa_right,
            &qa_left,
            &qa_tensor.right_ood_row,
            &qa_tensor.left_ood_row,
            &indices,
        );
        assert_eq!(expected_next.len(), 2 * level.next_blocks * level.width);
        roots.push(prefix_root(
            &expected_next,
            expected_next.len().next_power_of_two(),
            &zeros,
        ));
        let mut link_left = vec![Field192::ZERO; expected_next.len()];
        let mut link_right = vec![Field192::ZERO; expected_next.len()];
        populate_link_relation(
            0,
            &expected_next,
            &expected_next,
            &mut link_left,
            &mut link_right,
            &roots,
        );
        assert_eq!(dot(&link_left, &link_right), Field192::ZERO);
        link_left[0] += Field192::ONE;
        assert_ne!(dot(&link_left, &link_right), Field192::ZERO);
    }

    #[test]
    fn committed_witness_mutations_change_roots_and_challenges() {
        let level = Level {
            raw: 10,
            blocks: 2,
            block_semantic: 5,
            components: &[(5, 32)],
            group: 8,
            width: 3,
            row_span: 4,
            inverse_rate: 2,
            next_blocks: 2,
        };
        let zeros = zero_roots(10);
        let source = (0..level.raw)
            .map(|index| semantic_value(0, index))
            .collect::<Vec<_>>();
        let spectra = vec![generator_spectrum(0, 0, level.group)];
        let generator_roots = spectra
            .iter()
            .map(|spectrum| prefix_root(spectrum, level.group, &zeros))
            .collect::<Vec<_>>();
        let mut proof = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &source, &mut proof, &spectra, 2);
        let roots = level_commitment_roots(level, &source, &proof, &generator_roots, &zeros);
        let challenge = semantic_challenge(b"qa-row", 0, 0, &roots);

        let mut mutated_proof = proof.clone();
        mutated_proof[0] += Field192::ONE;
        let mutated_roots =
            level_commitment_roots(level, &source, &mutated_proof, &generator_roots, &zeros);
        assert_ne!(roots, mutated_roots);
        assert_ne!(
            challenge,
            semantic_challenge(b"qa-row", 0, 0, &mutated_roots)
        );

        let mut index_oracle = vec![Field192::ZERO; level.qa_fields()];
        populate_index_oracle(0, level, &mut index_oracle, &spectra, &roots);
        let root = index_oracle_root(level, &index_oracle, &zeros);
        index_oracle[0] += Field192::ONE;
        assert_ne!(root, index_oracle_root(level, &index_oracle, &zeros));
    }

    #[test]
    fn row_subtree_multiproof_matches_field_root_and_rejects_mutation() {
        let zeros = zero_roots(10);
        let values = (0..18).map(semantic_value_for_test).collect::<Vec<_>>();
        let commitment = compact_matrix_commitment(&values, 6, 3, 8, 8, &zeros);
        let proof = open_row_subtrees(&commitment, &[1, 4, 5]);
        assert!(proof.verify(commitment.root, commitment.row_domain));
        assert!(!proof.frontier.is_empty());
        let payload = proof.serialize();
        assert_eq!(
            RowSubtreeMultiProof::deserialize(&payload, vec![1, 4, 5]),
            Some(proof.clone())
        );
        let mut changed_payload = payload;
        *changed_payload.last_mut().unwrap() ^= 1;
        let parsed = RowSubtreeMultiProof::deserialize(&changed_payload, vec![1, 4, 5])
            .expect("mutated proof remains syntactically parseable");
        assert!(!parsed.verify(commitment.root, commitment.row_domain));
        let mut changed = proof;
        changed.frontier[0][0] ^= 1;
        assert!(!changed.verify(commitment.root, commitment.row_domain));
    }

    fn semantic_value_for_test(index: usize) -> Field192 {
        semantic_value(0, index)
    }

    #[test]
    fn tensor_carry_restoration_rejects_parity_and_w_mutations() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let membership = PackedSumcheckProof {
            fields: CARRYOPEN_INVERSE_RATE * CARRYOPEN_FIELDS,
            roots: vec![[7_u8; 32]],
            claimed_sum: Field192::ZERO,
            pairs: vec![(Field192::ZERO, Field192::ZERO); 26],
            terminal_left: Field192::ZERO,
            terminal_right: Field192::ONE,
        };
        let evaluation = PackedSumcheckProof {
            fields: CARRYOPEN_INVERSE_RATE * CARRYOPEN_FIELDS,
            roots: vec![[9_u8; 32]],
            claimed_sum: Field192::ZERO,
            pairs: vec![(Field192::ZERO, Field192::ZERO); 26],
            terminal_left: Field192::ZERO,
            terminal_right: Field192::ONE,
        };
        assert!(membership.challenges().is_some());
        assert!(evaluation.challenges().is_some());
        let selected = (0..CARRYOPEN_QUERIES).collect::<Vec<_>>();
        let mut terminal = Vec::with_capacity(CARRYOPEN_TERMINAL_FIELDS);
        terminal.extend(encode_horizontal_row(
            &vec![Field192::ONE; CARRYOPEN_WIDTH],
            &horizontal_spectra,
        ));
        terminal.extend(vec![Field192::ZERO; CARRYOPEN_WIDTH]);
        terminal.extend(encode_horizontal_row(
            &vec![Field192::ZERO; CARRYOPEN_WIDTH],
            &horizontal_spectra,
        ));
        let mut f_roots = Vec::with_capacity(selected.len());
        let mut w_roots = Vec::with_capacity(selected.len());
        for ordinal in 0..selected.len() {
            let f = (0..CARRYOPEN_WIDTH)
                .map(|lane| Field192::from((ordinal * CARRYOPEN_WIDTH + lane + 17) as u64))
                .collect::<Vec<_>>();
            let encoded = encode_horizontal_row(&f, &horizontal_spectra);
            f_roots.push(prefix_root(&encoded, CARRYOPEN_TENSOR_WIDTH, &zeros));
            terminal.extend(encoded);
            let w = (0..CARRYOPEN_WIDTH)
                .map(|lane| Field192::from((ordinal * CARRYOPEN_WIDTH + lane + 29) as u64))
                .collect::<Vec<_>>();
            w_roots.push(prefix_root(&w, CARRYOPEN_WIDTH, &zeros));
            terminal.extend(w);
        }
        let front = SelectedRowFront {
            selected: selected.clone(),
            proof_blocks: vec![(
                0,
                RowSubtreeMultiProof {
                    indices: selected,
                    roots: f_roots,
                    frontier: Vec::new(),
                },
            )],
            index_opening: RowSubtreeMultiProof {
                indices: (0..CARRYOPEN_QUERIES).collect(),
                roots: w_roots,
                frontier: Vec::new(),
            },
        };
        assert!(audit_tensor_carry_restoration(
            level,
            &front,
            &terminal,
            &horizontal_spectra,
            &membership,
            &evaluation,
            &zeros,
        ));

        let mut parity_mutation = terminal.clone();
        parity_mutation[CARRYOPEN_OOD_FIELDS + CARRYOPEN_WIDTH] += Field192::ONE;
        assert!(!audit_tensor_carry_restoration(
            level,
            &front,
            &parity_mutation,
            &horizontal_spectra,
            &membership,
            &evaluation,
            &zeros,
        ));

        let mut w_mutation = terminal;
        w_mutation[CARRYOPEN_OOD_FIELDS + CARRYOPEN_TENSOR_WIDTH] += Field192::ONE;
        assert!(!audit_tensor_carry_restoration(
            level,
            &front,
            &w_mutation,
            &horizontal_spectra,
            &membership,
            &evaluation,
            &zeros,
        ));
    }
}
