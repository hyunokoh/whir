use std::{
    borrow::Cow,
    collections::BTreeMap,
    convert::TryInto,
    mem::MaybeUninit,
    sync::{Arc, Mutex, OnceLock},
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
        combine_equal_subtrees, exact_prefix_root_with_scratch, parent, parent_pairs_batched,
        prefix_root, scaled_prefix_root, zero_roots, Digest, MerkleAccumulator,
    },
    parameters::ProtocolParameters,
    transcript::{codecs::Empty, DomainSeparator, Proof, ProverState, VerifierState},
    utils::workload_size,
};

#[cfg(test)]
use whir::lilac_merkle::{
    combine_equal_subtrees_one_block_nodes_v2_for_benchmark,
    combine_equal_subtrees_parallel_for_benchmark,
    combine_equal_subtrees_two_block_nodes_v1_for_benchmark,
    prefix_root_copied_parents_for_benchmark, prefix_root_fused_field_level2_v2_for_benchmark,
    prefix_root_materialized_blocks_for_benchmark, prefix_root_materialized_cv_for_benchmark,
    prefix_root_materialized_leaves_for_benchmark, prefix_root_materialized_parents_for_benchmark,
    prefix_root_one_block_nodes_v2_for_benchmark, prefix_root_scalar_canonical_for_benchmark,
    prefix_root_scalar_leaf_messages_for_benchmark,
    prefix_root_scalar_parent_messages_for_benchmark, prefix_root_scatter_for_benchmark,
    prefix_root_two_block_nodes_v1_for_benchmark, prefix_root_unfused_for_benchmark,
    prefix_root_unfused_leaf_parent_io_for_benchmark,
    prefix_root_unfused_parent_levels_for_benchmark, prefix_root_word_major_upper_v2_for_benchmark,
    scaled_prefix_root_sequential_for_benchmark, zero_roots_one_block_nodes_v2_for_benchmark,
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
    /// Run the diagnostic WHIR terminal backend outside canonical final-only
    /// mode. Canonical final-only proving uses the recursive explicit base.
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
    /// ending in the explicit private-F base and omitting diagnostic WHIR
    /// terminal proofs.
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
    prove_local_product_relation_with_claim_reusing(&mut left, &mut right, roots, claimed_sum)
}

fn prove_local_product_relation_with_claim_reusing(
    left: &mut Vec<Field192>,
    right: &mut Vec<Field192>,
    roots: Vec<Digest>,
    claimed_sum: Field192,
) -> PackedSumcheckProof {
    prove_local_product_relation_with_claim_slices(left, right, roots, claimed_sum)
}

fn prove_local_product_relation_with_claim_slices(
    left: &mut [Field192],
    right: &mut [Field192],
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
        let next_left = fold_active(left, active, challenge);
        let next_right = fold_active(right, active, challenge);
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
    prove_tensor_product_relation_reusing(&mut left, &mut right, rows, width, roots)
}

fn prove_tensor_product_relation_reusing(
    left: &mut Vec<Field192>,
    right: &mut Vec<Field192>,
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
        let next_left = fold_tensor_rows(left, active_rows, width, challenge);
        let next_right = fold_tensor_rows(right, active_rows, width, challenge);
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
        let next_left = fold_active(left, active_lanes, challenge);
        let next_right = fold_active(right, active_lanes, challenge);
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

fn rank_one_left_row_round_polynomial(
    coefficients: &[Field192],
    alpha: &[Field192],
    right: &[Field192],
    active_rows: usize,
    width: usize,
) -> (Field192, Field192) {
    assert!(active_rows.is_power_of_two());
    let half = active_rows / 2;
    (0..half)
        .into_par_iter()
        .map(|row| {
            let low_start = row * width;
            let high_start = (half + row) * width;
            let mut low_inner = Field192::ZERO;
            let mut difference_inner = Field192::ZERO;
            for lane in 0..width {
                low_inner += alpha[lane] * right[low_start + lane];
                difference_inner +=
                    alpha[lane] * (right[high_start + lane] - right[low_start + lane]);
            }
            (
                coefficients[row] * low_inner,
                (coefficients[half + row] - coefficients[row]) * difference_inner,
            )
        })
        .reduce(
            || (Field192::ZERO, Field192::ZERO),
            |(ca, qa), (cb, qb)| (ca + cb, qa + qb),
        )
}

fn fold_rank_one_coefficients(
    coefficients: &mut [Field192],
    active_rows: usize,
    challenge: Field192,
) -> usize {
    assert!(active_rows.is_power_of_two());
    let half = active_rows / 2;
    let (low, high) = coefficients[..active_rows].split_at_mut(half);
    low.par_iter_mut()
        .zip(high.par_iter())
        .for_each(|(low, high)| *low += (*high - *low) * challenge);
    half
}

fn finish_rank_one_tensor_product_relation(
    coefficient: Field192,
    mut alpha: Vec<Field192>,
    mut right: Vec<Field192>,
    width: usize,
    roots: Vec<Digest>,
    claimed_sum: Field192,
    mut pairs: Vec<(Field192, Field192)>,
    mut claim: Field192,
    row_variables: usize,
    variables: usize,
    fields: usize,
) -> TensorProductProof {
    alpha.par_iter_mut().for_each(|value| *value *= coefficient);
    let left_ood_row = alpha;
    let right_ood_row = right[..width].to_vec();
    let mut left = left_ood_row.clone();
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

/// Tensor product sumcheck for a public rank-one left matrix
/// `left[row,lane] = coefficients[row] * alpha[lane]`.  It emits exactly the
/// same transcript as materializing `left` and calling
/// [`prove_tensor_product_relation`], while storing and folding only the row
/// coefficients and the final OOD row.
#[cfg(test)]
fn prove_rank_one_left_tensor_product_relation(
    mut coefficients: Vec<Field192>,
    alpha: Vec<Field192>,
    mut right: Vec<Field192>,
    rows: usize,
    width: usize,
    roots: Vec<Digest>,
) -> TensorProductProof {
    assert!(
        rows > 1
            && width > 1
            && rows.is_power_of_two()
            && width.is_power_of_two()
            && coefficients.len() == rows
            && alpha.len() == width
            && right.len() == rows * width
    );
    let claimed_sum = right
        .par_chunks_exact(width)
        .zip(coefficients.par_iter())
        .map(|(row, coefficient)| {
            let inner = alpha
                .iter()
                .zip(row)
                .fold(Field192::ZERO, |sum, (left, right)| sum + *left * *right);
            *coefficient * inner
        })
        .reduce(|| Field192::ZERO, |left, right| left + right);
    let fields = rows * width;
    let variables = fields.trailing_zeros() as usize;
    let row_variables = rows.trailing_zeros() as usize;
    let mut claim = claimed_sum;
    let mut pairs = Vec::with_capacity(variables);
    let mut active_rows = rows;
    for _ in 0..row_variables {
        let (constant, quadratic) =
            rank_one_left_row_round_polynomial(&coefficients, &alpha, &right, active_rows, width);
        pairs.push((constant, quadratic));
        let challenge = transcript_challenge(fields, variables, &roots, claimed_sum, &pairs);
        let linear = claim - constant.double() - quadratic;
        let next_left = fold_rank_one_coefficients(&mut coefficients, active_rows, challenge);
        let next_right = fold_tensor_rows(&mut right, active_rows, width, challenge);
        assert_eq!(next_left, next_right);
        active_rows = next_left;
        claim = (quadratic * challenge + linear) * challenge + constant;
    }
    assert_eq!(active_rows, 1);
    finish_rank_one_tensor_product_relation(
        coefficients[0],
        alpha,
        right,
        width,
        roots,
        claimed_sum,
        pairs,
        claim,
        row_variables,
        variables,
        fields,
    )
}

#[cfg(test)]
fn prove_rank_one_right_tensor_product_relation(
    left: Vec<Field192>,
    coefficients: Vec<Field192>,
    alpha: Vec<Field192>,
    rows: usize,
    width: usize,
    roots: Vec<Digest>,
) -> TensorProductProof {
    let mut result =
        prove_rank_one_left_tensor_product_relation(coefficients, alpha, left, rows, width, roots);
    std::mem::swap(
        &mut result.proof.terminal_left,
        &mut result.proof.terminal_right,
    );
    std::mem::swap(&mut result.left_ood_row, &mut result.right_ood_row);
    assert!(result.proof.challenges().is_some());
    result
}

#[cfg(test)]
fn shared_rank_one_claims(
    right: &[Field192],
    membership_coefficients: &[Field192],
    membership_alpha: &[Field192],
    evaluation_coefficients: &[Field192],
    evaluation_alpha: &[Field192],
    width: usize,
) -> (Field192, Field192) {
    right
        .par_chunks_exact(width)
        .enumerate()
        .map(|(row_index, row)| {
            let mut membership_inner = Field192::ZERO;
            let mut evaluation_inner = Field192::ZERO;
            for lane in 0..width {
                membership_inner += membership_alpha[lane] * row[lane];
                evaluation_inner += evaluation_alpha[lane] * row[lane];
            }
            (
                membership_coefficients[row_index] * membership_inner,
                evaluation_coefficients[row_index] * evaluation_inner,
            )
        })
        .reduce(
            || (Field192::ZERO, Field192::ZERO),
            |(membership_a, evaluation_a), (membership_b, evaluation_b)| {
                (membership_a + membership_b, evaluation_a + evaluation_b)
            },
        )
}

fn shared_rank_one_row_round_polynomials(
    membership_coefficients: &[Field192],
    membership_alpha: &[Field192],
    evaluation_coefficients: &[Field192],
    evaluation_alpha: &[Field192],
    right: &[Field192],
    active_rows: usize,
    width: usize,
) -> ((Field192, Field192), (Field192, Field192)) {
    assert!(active_rows.is_power_of_two());
    let half = active_rows / 2;
    (0..half)
        .into_par_iter()
        .map(|row| {
            let low_start = row * width;
            let high_start = (half + row) * width;
            let mut membership_low = Field192::ZERO;
            let mut membership_difference = Field192::ZERO;
            let mut evaluation_low = Field192::ZERO;
            let mut evaluation_difference = Field192::ZERO;
            for lane in 0..width {
                let low = right[low_start + lane];
                let difference = right[high_start + lane] - low;
                membership_low += membership_alpha[lane] * low;
                membership_difference += membership_alpha[lane] * difference;
                evaluation_low += evaluation_alpha[lane] * low;
                evaluation_difference += evaluation_alpha[lane] * difference;
            }
            (
                (
                    membership_coefficients[row] * membership_low,
                    (membership_coefficients[half + row] - membership_coefficients[row])
                        * membership_difference,
                ),
                (
                    evaluation_coefficients[row] * evaluation_low,
                    (evaluation_coefficients[half + row] - evaluation_coefficients[row])
                        * evaluation_difference,
                ),
            )
        })
        .reduce(
            || {
                (
                    (Field192::ZERO, Field192::ZERO),
                    (Field192::ZERO, Field192::ZERO),
                )
            },
            |((mc_a, mq_a), (ec_a, eq_a)), ((mc_b, mq_b), (ec_b, eq_b))| {
                ((mc_a + mc_b, mq_a + mq_b), (ec_a + ec_b, eq_a + eq_b))
            },
        )
}

fn fold_tensor_rows_dual_in_place(
    values: &mut [Field192],
    active_rows: usize,
    width: usize,
    first_challenge: Field192,
    second_challenge: Field192,
) -> usize {
    assert!(active_rows.is_power_of_two());
    assert!(values.len() >= active_rows * width);
    let half_fields = active_rows / 2 * width;
    let (low, high) = values[..2 * half_fields].split_at_mut(half_fields);
    low.par_iter_mut()
        .zip(high.par_iter_mut())
        .for_each(|(low, high)| {
            let low_value = *low;
            let difference = *high - low_value;
            *low = low_value + difference * first_challenge;
            *high = low_value + difference * second_challenge;
        });
    active_rows / 2
}

/// Produce the membership and evaluation sumchecks from one shared codeword.
/// The first row round scans the common matrix once. After both transcript
/// challenges are known, the two half-size states occupy the lower and upper
/// halves of the original allocation. This avoids a separate half-codeword
/// evaluation buffer.
fn prove_dual_rank_one_tensor_product_relations(
    mut membership_coefficients: Vec<Field192>,
    membership_alpha: Vec<Field192>,
    mut right: Vec<Field192>,
    mut evaluation_coefficients: Vec<Field192>,
    evaluation_alpha: Vec<Field192>,
    rows: usize,
    width: usize,
    membership_roots: Vec<Digest>,
    evaluation_roots: Vec<Digest>,
    membership_claimed_sum: Field192,
    evaluation_claimed_sum: Field192,
) -> (TensorProductProof, TensorProductProof) {
    assert!(
        rows > 1
            && width > 1
            && rows.is_power_of_two()
            && width.is_power_of_two()
            && membership_coefficients.len() == rows
            && membership_alpha.len() == width
            && evaluation_coefficients.len() == rows
            && evaluation_alpha.len() == width
            && right.len() == rows * width
    );
    #[cfg(test)]
    assert_eq!(
        shared_rank_one_claims(
            &right,
            &membership_coefficients,
            &membership_alpha,
            &evaluation_coefficients,
            &evaluation_alpha,
            width,
        ),
        (membership_claimed_sum, evaluation_claimed_sum)
    );
    let fields = rows * width;
    let variables = fields.trailing_zeros() as usize;
    let row_variables = rows.trailing_zeros() as usize;
    let mut membership_claim = membership_claimed_sum;
    let mut evaluation_claim = evaluation_claimed_sum;
    let mut membership_pairs = Vec::with_capacity(variables);
    let mut evaluation_pairs = Vec::with_capacity(variables);

    let ((membership_constant, membership_quadratic), (evaluation_constant, evaluation_quadratic)) =
        shared_rank_one_row_round_polynomials(
            &membership_coefficients,
            &membership_alpha,
            &evaluation_coefficients,
            &evaluation_alpha,
            &right,
            rows,
            width,
        );
    membership_pairs.push((membership_constant, membership_quadratic));
    evaluation_pairs.push((evaluation_constant, evaluation_quadratic));
    let membership_challenge = transcript_challenge(
        fields,
        variables,
        &membership_roots,
        membership_claimed_sum,
        &membership_pairs,
    );
    let evaluation_challenge = transcript_challenge(
        fields,
        variables,
        &evaluation_roots,
        evaluation_claimed_sum,
        &evaluation_pairs,
    );
    let membership_linear = membership_claim - membership_constant.double() - membership_quadratic;
    let evaluation_linear = evaluation_claim - evaluation_constant.double() - evaluation_quadratic;
    let membership_rows =
        fold_rank_one_coefficients(&mut membership_coefficients, rows, membership_challenge);
    let evaluation_rows =
        fold_rank_one_coefficients(&mut evaluation_coefficients, rows, evaluation_challenge);
    assert_eq!(membership_rows, evaluation_rows);
    assert_eq!(
        fold_tensor_rows_dual_in_place(
            &mut right,
            rows,
            width,
            membership_challenge,
            evaluation_challenge,
        ),
        membership_rows
    );
    let evaluation_offset = rows / 2 * width;
    membership_claim = (membership_quadratic * membership_challenge + membership_linear)
        * membership_challenge
        + membership_constant;
    evaluation_claim = (evaluation_quadratic * evaluation_challenge + evaluation_linear)
        * evaluation_challenge
        + evaluation_constant;

    let mut active_rows = membership_rows;
    for _ in 1..row_variables {
        let (membership_right, evaluation_storage) = right.split_at(evaluation_offset);
        let evaluation_right = &evaluation_storage[..evaluation_offset];
        let (membership_constant, membership_quadratic) = rank_one_left_row_round_polynomial(
            &membership_coefficients,
            &membership_alpha,
            membership_right,
            active_rows,
            width,
        );
        let (evaluation_constant, evaluation_quadratic) = rank_one_left_row_round_polynomial(
            &evaluation_coefficients,
            &evaluation_alpha,
            evaluation_right,
            active_rows,
            width,
        );
        membership_pairs.push((membership_constant, membership_quadratic));
        evaluation_pairs.push((evaluation_constant, evaluation_quadratic));
        let membership_challenge = transcript_challenge(
            fields,
            variables,
            &membership_roots,
            membership_claimed_sum,
            &membership_pairs,
        );
        let evaluation_challenge = transcript_challenge(
            fields,
            variables,
            &evaluation_roots,
            evaluation_claimed_sum,
            &evaluation_pairs,
        );
        let membership_linear =
            membership_claim - membership_constant.double() - membership_quadratic;
        let evaluation_linear =
            evaluation_claim - evaluation_constant.double() - evaluation_quadratic;
        let next_membership_rows = fold_rank_one_coefficients(
            &mut membership_coefficients,
            active_rows,
            membership_challenge,
        );
        let next_evaluation_rows = fold_rank_one_coefficients(
            &mut evaluation_coefficients,
            active_rows,
            evaluation_challenge,
        );
        assert_eq!(next_membership_rows, next_evaluation_rows);
        let (membership_right, evaluation_storage) = right.split_at_mut(evaluation_offset);
        let evaluation_right = &mut evaluation_storage[..evaluation_offset];
        assert_eq!(
            fold_tensor_rows(membership_right, active_rows, width, membership_challenge),
            next_membership_rows
        );
        assert_eq!(
            fold_tensor_rows(evaluation_right, active_rows, width, evaluation_challenge),
            next_evaluation_rows
        );
        active_rows = next_membership_rows;
        membership_claim = (membership_quadratic * membership_challenge + membership_linear)
            * membership_challenge
            + membership_constant;
        evaluation_claim = (evaluation_quadratic * evaluation_challenge + evaluation_linear)
            * evaluation_challenge
            + evaluation_constant;
    }
    assert_eq!(active_rows, 1);

    let evaluation_right = right[evaluation_offset..evaluation_offset + width].to_vec();
    let membership = finish_rank_one_tensor_product_relation(
        membership_coefficients[0],
        membership_alpha,
        right,
        width,
        membership_roots,
        membership_claimed_sum,
        membership_pairs,
        membership_claim,
        row_variables,
        variables,
        fields,
    );
    let mut evaluation = finish_rank_one_tensor_product_relation(
        evaluation_coefficients[0],
        evaluation_alpha,
        evaluation_right,
        width,
        evaluation_roots,
        evaluation_claimed_sum,
        evaluation_pairs,
        evaluation_claim,
        row_variables,
        variables,
        fields,
    );
    std::mem::swap(
        &mut evaluation.proof.terminal_left,
        &mut evaluation.proof.terminal_right,
    );
    std::mem::swap(&mut evaluation.left_ood_row, &mut evaluation.right_ood_row);
    assert!(membership.proof.challenges().is_some());
    assert!(evaluation.proof.challenges().is_some());
    (membership, evaluation)
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

fn zero_weighted_relation_proof(
    challenge_label: &[u8],
    relation_label: &[u8],
    level: usize,
    fields: usize,
    transcript_roots: &[Digest],
) -> PackedSumcheckProof {
    assert!(fields > 1);
    let variables = fields.next_power_of_two().trailing_zeros() as usize;
    let point = (0..variables)
        .map(|index| semantic_challenge(challenge_label, level, index, transcript_roots))
        .collect::<Vec<_>>();
    let roots = local_relation_roots(relation_label, level, transcript_roots);
    let mut pairs = Vec::with_capacity(variables);
    let mut challenges = Vec::with_capacity(variables);
    for _ in 0..variables {
        pairs.push((Field192::ZERO, Field192::ZERO));
        challenges.push(transcript_challenge(
            fields,
            variables,
            &roots,
            Field192::ZERO,
            &pairs,
        ));
    }
    let proof = PackedSumcheckProof {
        fields,
        roots,
        claimed_sum: Field192::ZERO,
        pairs,
        terminal_left: Field192::ZERO,
        terminal_right: equality_prefix_inner_product(&point, &challenges, fields),
    };
    assert_eq!(proof.challenges().as_deref(), Some(challenges.as_slice()));
    proof
}

fn zero_phi_link_proof(
    challenge_label: &[u8],
    relation_label: &[u8],
    level: usize,
    fields: usize,
    transcript_roots: &[Digest],
) -> PackedSumcheckProof {
    zero_weighted_relation_proof(
        challenge_label,
        relation_label,
        level,
        fields,
        transcript_roots,
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
            && self.selected_front.verify(
                self.level,
                level,
                &self.component_roots,
                level.group,
                &zero_roots(PRODUCTION_VARIABLES),
            )
            && derived_next_source_root(
                self.level,
                &self.component_roots,
                &self.selected_front,
                self.ood_block_root,
                &zero_roots(PRODUCTION_VARIABLES),
            ) == Some(self.next_source_root)
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        self.serialize_unchecked()
    }

    /// Serialize bytes after a verified aggregate has accepted this component.
    fn serialize_unchecked(&self) -> Vec<u8> {
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
        let proof = Self::deserialize_unchecked(payload)?;
        proof.verify().then_some(proof)
    }

    /// Structural parser for a containing proof that performs the semantic
    /// verification pass before returning to its caller.
    fn deserialize_unchecked(payload: &[u8]) -> Option<Self> {
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
        Some(proof)
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

/// Return the unnormalized Walsh--Hadamard spectrum of a scaled Boolean
/// equality table without first materializing and transforming that table.
/// For chi_r(x), each frequency bit contributes either 1 or 1 - 2 r_i.
fn scaled_equality_walsh_spectrum(point: &[Field192], scale: Field192) -> Vec<Field192> {
    let mut spectrum = Vec::with_capacity(1_usize << point.len());
    spectrum.push(scale);
    for coordinate in point.iter().rev() {
        let factor = Field192::ONE - (*coordinate + *coordinate);
        let active = spectrum.len();
        for index in 0..active {
            spectrum.push(spectrum[index] * factor);
        }
    }
    spectrum
}

/// Return the canonical leading equality-table entries without expanding the
/// unused suffix. The recursive order is identical to `equality_weights`:
/// each coordinate's zero branch precedes its one branch.
fn equality_weights_prefix(point: &[Field192], length: usize) -> Vec<Field192> {
    let domain = 1usize << point.len();
    assert!(length <= domain);
    fn append_prefix(
        point: &[Field192],
        weight: Field192,
        length: usize,
        output: &mut Vec<Field192>,
    ) {
        if length == 0 {
            return;
        }
        if point.is_empty() {
            output.push(weight);
            return;
        }
        let half = 1usize << (point.len() - 1);
        let low = length.min(half);
        append_prefix(
            &point[1..],
            weight * (Field192::ONE - point[0]),
            low,
            output,
        );
        if length > half {
            append_prefix(&point[1..], weight * point[0], length - half, output);
        }
    }
    let mut output = Vec::with_capacity(length);
    append_prefix(point, Field192::ONE, length, &mut output);
    assert_eq!(output.len(), length);
    output
}

/// Evaluate the inner product of two equality tables over their canonical
/// leading `length` entries without materializing either table.
fn equality_prefix_inner_product(
    left_point: &[Field192],
    right_point: &[Field192],
    length: usize,
) -> Field192 {
    assert_eq!(left_point.len(), right_point.len());
    let domain = 1usize << left_point.len();
    assert!(length <= domain);
    fn recurse(left: &[Field192], right: &[Field192], length: usize) -> Field192 {
        if length == 0 {
            return Field192::ZERO;
        }
        if left.is_empty() {
            assert_eq!(length, 1);
            return Field192::ONE;
        }
        if length == 1usize << left.len() {
            return left
                .iter()
                .zip(right)
                .fold(Field192::ONE, |product, (a, b)| {
                    product * ((Field192::ONE - *a) * (Field192::ONE - *b) + *a * *b)
                });
        }
        let half = 1usize << (left.len() - 1);
        let low_length = length.min(half);
        let low = (Field192::ONE - left[0])
            * (Field192::ONE - right[0])
            * recurse(&left[1..], &right[1..], low_length);
        if length <= half {
            low
        } else {
            low + left[0] * right[0] * recurse(&left[1..], &right[1..], length - half)
        }
    }
    recurse(left_point, right_point, length)
}

const CARRYOPEN_ROWS: usize = 1 << 16;
const CARRYOPEN_WIDTH: usize = 1 << 8;
const CARRYOPEN_FIELDS: usize = CARRYOPEN_ROWS * CARRYOPEN_WIDTH;
const CARRYOPEN_INVERSE_RATE: usize = 4;
const CARRYOPEN_QUERIES: usize = 78;
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
const STRONG3_COMPONENTS: [(usize, usize); 1] = [(1 << 13, 1 << 13)];
const STRONG4_COMPONENTS: [(usize, usize); 2 * STRONG_NEXT_BLOCKS] =
    [(1 << 6, 1 << 7); 2 * STRONG_NEXT_BLOCKS];
const STRONG5_COMPONENTS: [(usize, usize); 2 * STRONG_NEXT_BLOCKS] =
    [(1 << 5, 1 << 7); 2 * STRONG_NEXT_BLOCKS];
const STRONG_ROUNDS: usize = 6;

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
    let (group, width, components) = match round {
        0 => (1 << 10, 1 << 10, &STRONG_COMPONENTS[..]),
        1 => (1 << 8, 1 << 8, &STRONG1_COMPONENTS[..]),
        2 => (1 << 7, 1 << 7, &STRONG2_COMPONENTS[..]),
        3 => (1 << 7, 1 << 6, &STRONG3_COMPONENTS[..]),
        4 => (1 << 7, 1 << 5, &STRONG4_COMPONENTS[..]),
        5 => (1 << 7, 1 << 4, &STRONG5_COMPONENTS[..]),
        _ => return None,
    };
    Some(Level {
        raw: group * width,
        blocks: 1,
        block_semantic: group * width,
        components,
        group,
        width,
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
    fold_message_at_point(&mut scratch, point)
}

#[cfg(test)]
fn evaluate_power_of_two_message_with_scratch(
    message: &[Field192],
    point: &[Field192],
    scratch: &mut [Field192],
) -> Field192 {
    assert_eq!(message.len(), scratch.len());
    assert_eq!(message.len(), 1_usize << point.len());
    scratch.copy_from_slice(message);
    fold_message_at_point(scratch, point)
}

#[cfg(test)]
fn evaluate_power_of_two_message_direct_first_round(
    message: &[Field192],
    point: &[Field192],
    scratch: &mut [Field192],
) -> Field192 {
    assert!(!point.is_empty());
    assert_eq!(message.len(), scratch.len());
    assert_eq!(message.len(), 1_usize << point.len());
    let half = message.len() / 2;
    let (low, high) = message.split_at(half);
    scratch[..half]
        .par_iter_mut()
        .zip(low.par_iter().zip(high.par_iter()))
        .for_each(|(output, (low, high))| *output = *low + (*high - *low) * point[0]);
    fold_message_at_point(&mut scratch[..half], &point[1..])
}

fn evaluate_power_of_two_message_direct_first_round_in_spare(
    message: &[Field192],
    point: &[Field192],
    scratch: &mut [MaybeUninit<Field192>],
) -> Field192 {
    assert!(!point.is_empty());
    assert_eq!(message.len(), scratch.len());
    assert_eq!(message.len(), 1_usize << point.len());
    let half = message.len() / 2;
    let (low, high) = message.split_at(half);
    scratch[..half]
        .par_iter_mut()
        .zip(low.par_iter().zip(high.par_iter()))
        .for_each(|(output, (low, high))| {
            output.write(*low + (*high - *low) * point[0]);
        });
    // SAFETY: the parallel loop above initialized every cell in the lower
    // half. The upper half remains outside this slice and outside Vec length.
    let initialized =
        unsafe { std::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<Field192>(), half) };
    fold_message_at_point(initialized, &point[1..])
}

fn fold_message_at_point(scratch: &mut [Field192], point: &[Field192]) -> Field192 {
    assert_eq!(scratch.len(), 1_usize << point.len());
    let mut active = scratch.len();
    for challenge in point {
        active = fold_active(scratch, active, *challenge);
    }
    assert_eq!(active, 1);
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

#[cfg(test)]
fn fill_scaled_equality_weights(point: &[Field192], scale: Field192, target: &mut [Field192]) {
    assert_eq!(target.len(), 1_usize << point.len());
    target[0] = scale;
    let mut active = 1;
    for coordinate in point.iter().rev() {
        let (low, high_and_tail) = target.split_at_mut(active);
        let high = &mut high_and_tail[..active];
        high.copy_from_slice(low);
        high.par_iter_mut().for_each(|value| *value *= *coordinate);
        low.par_iter_mut()
            .zip(high.par_iter())
            .for_each(|(low, high)| *low -= *high);
        active *= 2;
    }
}

#[cfg(test)]
fn accumulate_scaled_equality_weights_one_round(
    point: &[Field192],
    scale: Field192,
    scratch: &mut [Field192],
    target: &mut [Field192],
) {
    assert!(!point.is_empty());
    assert_eq!(scratch.len(), 1_usize << point.len());
    assert_eq!(target.len(), scratch.len());
    scratch[0] = scale;
    let mut active = 1;
    for coordinate in point[1..].iter().rev() {
        let (low, high_and_tail) = scratch.split_at_mut(active);
        let high = &mut high_and_tail[..active];
        high.copy_from_slice(low);
        high.par_iter_mut().for_each(|value| *value *= *coordinate);
        low.par_iter_mut()
            .zip(high.par_iter())
            .for_each(|(low, high)| *low -= *high);
        active *= 2;
    }
    assert_eq!(2 * active, scratch.len());
    let source = &scratch[..active];
    let (target_low, target_high) = target.split_at_mut(active);
    let coordinate = point[0];
    target_low
        .par_iter_mut()
        .zip(target_high.par_iter_mut())
        .zip(source.par_iter())
        .for_each(|((target_low, target_high), source)| {
            let high = *source * coordinate;
            *target_low += *source - high;
            *target_high += high;
        });
}

/// Expand the final two equality coordinates directly into the four target
/// quarters. This performs the same three multiplications per source value as
/// two ordinary rounds while avoiding both intermediate scratch levels.
#[cfg(test)]
fn accumulate_scaled_equality_weights(
    point: &[Field192],
    scale: Field192,
    scratch: &mut [Field192],
    target: &mut [Field192],
) {
    assert!(point.len() >= 2);
    assert_eq!(scratch.len(), 1_usize << point.len());
    assert_eq!(target.len(), scratch.len());
    scratch[0] = scale;
    let mut active = 1;
    for coordinate in point[2..].iter().rev() {
        let (low, high_and_tail) = scratch.split_at_mut(active);
        let high = &mut high_and_tail[..active];
        high.copy_from_slice(low);
        high.par_iter_mut().for_each(|value| *value *= *coordinate);
        low.par_iter_mut()
            .zip(high.par_iter())
            .for_each(|(low, high)| *low -= *high);
        active *= 2;
    }
    assert_eq!(4 * active, scratch.len());
    let source = &scratch[..active];
    let (target_0, tail) = target.split_at_mut(active);
    let (target_1, tail) = tail.split_at_mut(active);
    let (target_2, target_3) = tail.split_at_mut(active);
    let coordinate_1 = point[1];
    let coordinate_0 = point[0];
    target_0
        .par_iter_mut()
        .zip(target_1.par_iter_mut())
        .zip(target_2.par_iter_mut())
        .zip(target_3.par_iter_mut())
        .zip(source.par_iter())
        .for_each(|((((target_0, target_1), target_2), target_3), source)| {
            let high_1 = *source * coordinate_1;
            let low_1 = *source - high_1;
            let low_1_high_0 = low_1 * coordinate_0;
            let high_1_high_0 = high_1 * coordinate_0;
            *target_0 += low_1 - low_1_high_0;
            *target_1 += high_1 - high_1_high_0;
            *target_2 += low_1_high_0;
            *target_3 += high_1_high_0;
        });
}

fn fill_scaled_equality_suffix_sequential(
    point: &[Field192],
    scale: Field192,
    target: &mut [Field192],
) {
    assert_eq!(target.len(), 1usize << point.len());
    target[0] = scale;
    let mut active = 1;
    for coordinate in point.iter().rev() {
        for index in 0..active {
            let value = target[index];
            let high = value * *coordinate;
            target[index] = value - high;
            target[active + index] = high;
        }
        active *= 2;
    }
}

/// Accumulate a small batch of equality tables block by block.  Each target
/// block remains cache-local while all points contribute, so the full target
/// is written once instead of streamed once per point.
fn accumulate_equality_weights_block_local(
    points: &[Vec<Field192>],
    coefficients: &[Field192],
    target: &mut [Field192],
) {
    const BLOCK_FIELDS: usize = 1 << 10;
    assert!(!points.is_empty() && points.len() == coefficients.len());
    assert!(target.len().is_power_of_two() && target.len() >= BLOCK_FIELDS);
    let variables = target.len().trailing_zeros() as usize;
    assert!(points.iter().all(|point| point.len() == variables));
    let suffix_variables = BLOCK_FIELDS.trailing_zeros() as usize;
    let prefix_variables = variables - suffix_variables;
    target
        .par_chunks_mut(BLOCK_FIELDS)
        .enumerate()
        .for_each_init(
            || vec![Field192::ZERO; BLOCK_FIELDS],
            |scratch, (block_index, output)| {
                for (point, coefficient) in points.iter().zip(coefficients) {
                    let scale = point[..prefix_variables].iter().enumerate().fold(
                        *coefficient,
                        |weight, (coordinate, value)| {
                            let bit = (block_index >> (prefix_variables - 1 - coordinate)) & 1;
                            weight
                                * if bit == 0 {
                                    Field192::ONE - *value
                                } else {
                                    *value
                                }
                        },
                    );
                    fill_scaled_equality_suffix_sequential(
                        &point[prefix_variables..],
                        scale,
                        scratch,
                    );
                    output
                        .iter_mut()
                        .zip(scratch.iter())
                        .for_each(|(output, value)| *output += *value);
                }
            },
        );
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

    #[cfg(test)]
    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        self.serialize_unchecked()
    }

    /// Serialize bytes after a verified CarryOpen aggregate has accepted this statement.
    fn serialize_unchecked(&self) -> Vec<u8> {
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

    /// Structural parser used only inside a verified CarryOpen aggregate.
    fn deserialize_unchecked(payload: &[u8]) -> Option<Self> {
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
        Some(proof)
    }
}

#[cfg(test)]
fn build_production_precarry(
    message: &[Field192],
    commitment_root: Digest,
) -> ProductionPreCarryProof {
    assert_eq!(message.len(), CARRYOPEN_FIELDS);
    let points = precarry_opening_points(&commitment_root, 16);
    let mut scratch = vec![Field192::ZERO; message.len()];
    let claims = points
        .iter()
        .map(|point| evaluate_power_of_two_message_with_scratch(message, point, &mut scratch))
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
        accumulate_scaled_equality_weights(point, coefficient, &mut scratch, &mut combined_weight);
    }
    drop(scratch);
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

/// Build the pre-Carry batch inside the allocation already reserved for the
/// four-quarter vertical codeword. The message occupies the first quarter;
/// only two spare quarters are initialized for evaluation/combined weights
/// and the product-sumcheck copy. Keeping the original first quarter intact
/// avoids regenerating it before vertical encoding, while leaving the fourth
/// quarter untouched avoids one full-message copy. Once the sumcheck finishes,
/// truncating back to the message retains the allocation for vertical parity.
fn build_production_precarry_in_codeword(
    codeword: &mut Vec<Field192>,
    commitment_root: Digest,
) -> ProductionPreCarryProof {
    build_production_precarry_in_codeword_compact(codeword, commitment_root)
}

fn build_production_precarry_in_codeword_compact(
    codeword: &mut Vec<Field192>,
    commitment_root: Digest,
) -> ProductionPreCarryProof {
    assert_eq!(codeword.len(), CARRYOPEN_FIELDS);
    assert!(codeword.capacity() >= 4 * CARRYOPEN_FIELDS);
    let points = precarry_opening_points(&commitment_root, 16);
    let message_pointer = codeword.as_ptr();
    let (claims, sumcheck) = {
        let spare = &mut codeword.spare_capacity_mut()[..2 * CARRYOPEN_FIELDS];
        let (combined_spare, sumcheck_spare) = spare.split_at_mut(CARRYOPEN_FIELDS);
        // SAFETY: the immutable message occupies the initialized Vec prefix;
        // both mutable spare slices begin after that prefix and are disjoint.
        let message = unsafe { std::slice::from_raw_parts(message_pointer, CARRYOPEN_FIELDS) };
        let claims = points
            .iter()
            .map(|point| {
                evaluate_power_of_two_message_direct_first_round_in_spare(
                    message,
                    point,
                    combined_spare,
                )
            })
            .collect::<Vec<_>>();
        let statement = precarry_statement_root(&commitment_root, &points, &claims);
        let coefficients = precarry_coefficients(&statement, claims.len());
        let claimed = claims
            .iter()
            .zip(&coefficients)
            .map(|(claim, coefficient)| *claim * coefficient)
            .sum::<Field192>();

        combined_spare.par_iter_mut().for_each(|value| {
            value.write(Field192::ZERO);
        });
        // SAFETY: every combined-weight cell was initialized immediately
        // above. A panic before this point leaves it outside Vec length.
        let combined_weight = unsafe {
            std::slice::from_raw_parts_mut(
                combined_spare.as_mut_ptr().cast::<Field192>(),
                CARRYOPEN_FIELDS,
            )
        };
        accumulate_equality_weights_block_local(&points, &coefficients, combined_weight);

        sumcheck_spare
            .par_iter_mut()
            .zip(message.par_iter())
            .for_each(|(target, value)| {
                target.write(*value);
            });
        // SAFETY: the disjoint parallel copy initialized the complete second
        // spare quarter. Both workspaces now contain valid Field192 values.
        let sumcheck_message = unsafe {
            std::slice::from_raw_parts_mut(
                sumcheck_spare.as_mut_ptr().cast::<Field192>(),
                CARRYOPEN_FIELDS,
            )
        };
        let sumcheck = prove_local_product_relation_with_claim_slices(
            sumcheck_message,
            combined_weight,
            local_relation_roots(b"pre-Carry-batch", 0, &[commitment_root, statement]),
            claimed,
        );
        (claims, sumcheck)
    };
    // SAFETY: both spare quarters are fully initialized. Extending only now
    // keeps an unwinding path from ever exposing partially initialized cells.
    unsafe { codeword.set_len(3 * CARRYOPEN_FIELDS) };
    codeword.truncate(CARRYOPEN_FIELDS);
    let proof = ProductionPreCarryProof {
        commitment_root,
        points,
        claims,
        sumcheck,
    };
    assert!(proof.verify());
    proof
}

#[cfg(test)]
fn build_production_precarry_in_codeword_with_modes(
    codeword: &mut Vec<Field192>,
    commitment_root: Digest,
    direct_first_round: bool,
    block_local_weights: bool,
) -> ProductionPreCarryProof {
    assert_eq!(codeword.len(), CARRYOPEN_FIELDS);
    assert!(codeword.capacity() >= 4 * CARRYOPEN_FIELDS);
    let points = precarry_opening_points(&commitment_root, 16);
    // Seed the scratch with the first evaluation message, zero only the
    // accumulator quarter, and append the sumcheck copy directly. This avoids
    // zero-filling two quarters that would immediately be overwritten.
    codeword.extend_from_within(..CARRYOPEN_FIELDS);
    codeword.resize(3 * CARRYOPEN_FIELDS, Field192::ZERO);
    codeword.extend_from_within(..CARRYOPEN_FIELDS);
    let (claims, sumcheck) = {
        let (message, workspace) = codeword.split_at_mut(CARRYOPEN_FIELDS);
        let (scratch, workspace) = workspace.split_at_mut(CARRYOPEN_FIELDS);
        let (combined_weight, sumcheck_message) = workspace.split_at_mut(CARRYOPEN_FIELDS);
        assert_eq!(sumcheck_message.len(), CARRYOPEN_FIELDS);
        let mut point_iter = points.iter();
        let first_point = point_iter.next().expect("pre-Carry has fixed openings");
        let mut claims = Vec::with_capacity(points.len());
        claims.push(if direct_first_round {
            evaluate_power_of_two_message_direct_first_round(message, first_point, scratch)
        } else {
            fold_message_at_point(scratch, first_point)
        });
        claims.extend(point_iter.map(|point| {
            if direct_first_round {
                evaluate_power_of_two_message_direct_first_round(message, point, scratch)
            } else {
                evaluate_power_of_two_message_with_scratch(message, point, scratch)
            }
        }));
        let statement = precarry_statement_root(&commitment_root, &points, &claims);
        let coefficients = precarry_coefficients(&statement, claims.len());
        let claimed = claims
            .iter()
            .zip(&coefficients)
            .map(|(claim, coefficient)| *claim * coefficient)
            .sum::<Field192>();
        if block_local_weights {
            accumulate_equality_weights_block_local(&points, &coefficients, combined_weight);
        } else {
            for (point, coefficient) in points.iter().zip(coefficients) {
                accumulate_scaled_equality_weights(point, coefficient, scratch, combined_weight);
            }
        }
        sumcheck_message.copy_from_slice(message);
        let sumcheck = prove_local_product_relation_with_claim_slices(
            sumcheck_message,
            combined_weight,
            local_relation_roots(b"pre-Carry-batch", 0, &[commitment_root, statement]),
            claimed,
        );
        (claims, sumcheck)
    };
    codeword.truncate(CARRYOPEN_FIELDS);
    let proof = ProductionPreCarryProof {
        commitment_root,
        points,
        claims,
        sumcheck,
    };
    assert!(proof.verify());
    proof
}

#[cfg(test)]
fn build_production_precarry_one_round(
    message: &[Field192],
    commitment_root: Digest,
) -> ProductionPreCarryProof {
    assert_eq!(message.len(), CARRYOPEN_FIELDS);
    let points = precarry_opening_points(&commitment_root, 16);
    let mut scratch = vec![Field192::ZERO; message.len()];
    let claims = points
        .iter()
        .map(|point| evaluate_power_of_two_message_with_scratch(message, point, &mut scratch))
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
        accumulate_scaled_equality_weights_one_round(
            point,
            coefficient,
            &mut scratch,
            &mut combined_weight,
        );
    }

    drop(scratch);
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

#[cfg(test)]
fn build_production_precarry_separate_pass(
    message: &[Field192],
    commitment_root: Digest,
) -> ProductionPreCarryProof {
    assert_eq!(message.len(), CARRYOPEN_FIELDS);
    let points = precarry_opening_points(&commitment_root, 16);
    let mut scratch = vec![Field192::ZERO; message.len()];
    let claims = points
        .iter()
        .map(|point| evaluate_power_of_two_message_with_scratch(message, point, &mut scratch))
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
        fill_scaled_equality_weights(point, coefficient, &mut scratch);
        combined_weight
            .par_iter_mut()
            .zip(scratch.par_iter())
            .for_each(|(target, weight)| *target += *weight);
    }
    drop(scratch);
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
    let mut transformed = row.to_vec();
    wht(&mut transformed);
    for spectrum in spectra {
        let mut parity = transformed.clone();
        parity
            .iter_mut()
            .zip(spectrum)
            .for_each(|(value, multiplier)| *value *= multiplier);
        wht(&mut parity);
        encoded.extend(parity);
    }
    encoded
}

fn horizontal_encoded_row_root_with_scratch(
    row: &[Field192],
    spectra: &[Vec<Field192>],
    transformed: &mut [Field192],
    encoded: &mut [Field192],
    digest_scratch: &mut Vec<Digest>,
) -> Digest {
    assert_eq!(row.len(), CARRYOPEN_WIDTH);
    assert_eq!(spectra.len(), CARRYOPEN_INVERSE_RATE - 1);
    assert_eq!(transformed.len(), CARRYOPEN_WIDTH);
    assert_eq!(encoded.len(), CARRYOPEN_TENSOR_WIDTH);
    encoded[..CARRYOPEN_WIDTH].copy_from_slice(row);
    transformed.copy_from_slice(row);
    wht(transformed);
    for (block, spectrum) in spectra.iter().enumerate() {
        let start = (block + 1) * CARRYOPEN_WIDTH;
        let parity = &mut encoded[start..start + CARRYOPEN_WIDTH];
        parity.copy_from_slice(transformed);
        parity
            .iter_mut()
            .zip(spectrum)
            .for_each(|(value, multiplier)| *value *= multiplier);
        wht(parity);
    }
    exact_prefix_root_with_scratch(encoded, digest_scratch)
}

/// Return the ordinary flat Merkle root together with its exact `row_width`
/// subtree roots.  Keeping these roots lets the tensor commitment reuse the
/// already-hashed systematic quarter of its first matrix.
#[cfg(test)]
fn exact_root_with_row_subtrees(
    values: &[Field192],
    row_width: usize,
    zeros: &[Digest],
) -> (Digest, Vec<Digest>) {
    assert!(
        !values.is_empty()
            && values.len().is_power_of_two()
            && row_width.is_power_of_two()
            && row_width <= values.len()
            && values.len() % row_width == 0
    );
    let row_roots = values
        .par_chunks_exact(row_width)
        .map(|row| prefix_root(row, row_width, zeros))
        .collect::<Vec<_>>();
    (combine_equal_subtrees(&row_roots), row_roots)
}

#[cfg(test)]
fn horizontal_row_root_with_systematic_subtree(
    row: &[Field192],
    systematic_root: Digest,
    spectra: &[Vec<Field192>],
    zeros: &[Digest],
) -> Digest {
    assert_eq!(row.len(), CARRYOPEN_WIDTH);
    assert_eq!(spectra.len(), CARRYOPEN_INVERSE_RATE - 1);
    let mut transformed = row.to_vec();
    wht(&mut transformed);
    let mut roots = Vec::with_capacity(CARRYOPEN_INVERSE_RATE);
    roots.push(systematic_root);
    for spectrum in spectra {
        let mut parity = transformed.clone();
        parity
            .iter_mut()
            .zip(spectrum)
            .for_each(|(value, multiplier)| *value *= multiplier);
        wht(&mut parity);
        roots.push(prefix_root(&parity, CARRYOPEN_WIDTH, zeros));
    }
    combine_equal_subtrees(&roots)
}

fn horizontal_row_root_with_systematic_subtree_scratch(
    row: &[Field192],
    systematic_root: Digest,
    spectra: &[Vec<Field192>],
    transformed: &mut [Field192],
    parity: &mut [Field192],
    digest_scratch: &mut Vec<Digest>,
) -> Digest {
    assert_eq!(row.len(), CARRYOPEN_WIDTH);
    assert_eq!(spectra.len(), CARRYOPEN_INVERSE_RATE - 1);
    assert_eq!(transformed.len(), CARRYOPEN_WIDTH);
    assert!(parity.len() >= CARRYOPEN_WIDTH);
    transformed.copy_from_slice(row);
    wht(transformed);
    let mut roots = [systematic_root; CARRYOPEN_INVERSE_RATE];
    for (root, spectrum) in roots[1..].iter_mut().zip(spectra) {
        let parity = &mut parity[..CARRYOPEN_WIDTH];
        parity.copy_from_slice(transformed);
        parity
            .iter_mut()
            .zip(spectrum)
            .for_each(|(value, multiplier)| *value *= multiplier);
        wht(parity);
        *root = exact_prefix_root_with_scratch(parity, digest_scratch);
    }
    combine_equal_subtrees(&roots)
}

fn message_and_systematic_tensor_commitment(
    message: &[Field192],
    horizontal_spectra: &[Vec<Field192>],
    zeros: &[Digest],
) -> (Digest, Vec<Digest>, MatrixCommitment) {
    assert_eq!(message.len(), CARRYOPEN_FIELDS);
    assert_eq!(horizontal_spectra.len(), CARRYOPEN_INVERSE_RATE - 1);
    let row_roots = message
        .par_chunks_exact(CARRYOPEN_WIDTH)
        .map_init(
            || {
                (
                    vec![Field192::ZERO; 2 * CARRYOPEN_WIDTH],
                    Vec::with_capacity(CARRYOPEN_WIDTH / 4),
                )
            },
            |(field_scratch, digest_scratch), row| {
                let systematic_root = exact_prefix_root_with_scratch(row, digest_scratch);
                let (transformed, parity) = field_scratch.split_at_mut(CARRYOPEN_WIDTH);
                let tensor_root = horizontal_row_root_with_systematic_subtree_scratch(
                    row,
                    systematic_root,
                    horizontal_spectra,
                    transformed,
                    parity,
                    digest_scratch,
                );
                (systematic_root, tensor_root)
            },
        )
        .collect::<Vec<_>>();
    let mut message_row_roots = Vec::with_capacity(CARRYOPEN_ROWS);
    let mut tensor_row_roots = Vec::with_capacity(CARRYOPEN_ROWS);
    for (message_root, tensor_root) in row_roots {
        message_row_roots.push(message_root);
        tensor_row_roots.push(tensor_root);
    }
    let message_root = combine_equal_subtrees(&message_row_roots);
    let commitment = MatrixCommitment {
        root: combine_equal_subtrees(&tensor_row_roots),
        row_roots: tensor_row_roots,
        row_domain: CARRYOPEN_ROWS,
        zero_row_root: zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
    };
    (message_root, message_row_roots, commitment)
}

#[cfg(test)]
fn tensor_row_commitments(
    vertical_codeword: &[Field192],
    message_row_roots: &[Digest],
    horizontal_spectra: &[Vec<Field192>],
    zeros: &[Digest],
) -> Vec<MatrixCommitment> {
    assert_eq!(
        vertical_codeword.len(),
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_FIELDS
    );
    assert_eq!(message_row_roots.len(), CARRYOPEN_ROWS);
    vertical_codeword
        .chunks_exact(CARRYOPEN_FIELDS)
        .enumerate()
        .map(|(matrix, matrix_values)| {
            // One allocation serves both field workspaces.  The systematic
            // matrix hashes only 256-field parity subtrees, while the three
            // vertical-parity matrices retain a complete 1,024-field row.
            let encoded_fields = if matrix == 0 {
                CARRYOPEN_WIDTH
            } else {
                CARRYOPEN_TENSOR_WIDTH
            };
            let row_roots = matrix_values
                .par_chunks_exact(CARRYOPEN_WIDTH)
                .enumerate()
                .map_init(
                    || {
                        (
                            vec![Field192::ZERO; CARRYOPEN_WIDTH + encoded_fields],
                            Vec::with_capacity(encoded_fields / 4),
                        )
                    },
                    |(field_scratch, digest_scratch), (row_index, row)| {
                        let (transformed, encoded) = field_scratch.split_at_mut(CARRYOPEN_WIDTH);
                        if matrix == 0 {
                            horizontal_row_root_with_systematic_subtree_scratch(
                                row,
                                message_row_roots[row_index],
                                horizontal_spectra,
                                transformed,
                                encoded,
                                digest_scratch,
                            )
                        } else {
                            horizontal_encoded_row_root_with_scratch(
                                row,
                                horizontal_spectra,
                                transformed,
                                encoded,
                                digest_scratch,
                            )
                        }
                    },
                )
                .collect::<Vec<_>>();
            MatrixCommitment {
                root: combine_equal_subtrees(&row_roots),
                row_roots,
                row_domain: CARRYOPEN_ROWS,
                zero_row_root: zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
            }
        })
        .collect()
}

#[cfg(test)]
fn systematic_tensor_row_commitment(
    systematic_values: &[Field192],
    message_row_roots: &[Digest],
    horizontal_spectra: &[Vec<Field192>],
    zeros: &[Digest],
) -> MatrixCommitment {
    assert_eq!(systematic_values.len(), CARRYOPEN_FIELDS);
    assert_eq!(message_row_roots.len(), CARRYOPEN_ROWS);
    let row_roots = systematic_values
        .par_chunks_exact(CARRYOPEN_WIDTH)
        .enumerate()
        .map_init(
            || {
                (
                    vec![Field192::ZERO; 2 * CARRYOPEN_WIDTH],
                    Vec::with_capacity(CARRYOPEN_WIDTH / 4),
                )
            },
            |(field_scratch, digest_scratch), (row_index, row)| {
                let (transformed, encoded) = field_scratch.split_at_mut(CARRYOPEN_WIDTH);
                horizontal_row_root_with_systematic_subtree_scratch(
                    row,
                    message_row_roots[row_index],
                    horizontal_spectra,
                    transformed,
                    encoded,
                    digest_scratch,
                )
            },
        )
        .collect::<Vec<_>>();
    MatrixCommitment {
        root: combine_equal_subtrees(&row_roots),
        row_roots,
        row_domain: CARRYOPEN_ROWS,
        zero_row_root: zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
    }
}

#[cfg(test)]
fn tensor_row_commitments_flat_staging(
    vertical_codeword: &[Field192],
    message_row_roots: &[Digest],
    horizontal_spectra: &[Vec<Field192>],
    zeros: &[Digest],
) -> Vec<MatrixCommitment> {
    assert_eq!(
        vertical_codeword.len(),
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_FIELDS
    );
    assert_eq!(message_row_roots.len(), CARRYOPEN_ROWS);
    let row_roots = vertical_codeword
        .par_chunks_exact(CARRYOPEN_WIDTH)
        .enumerate()
        .map_init(
            || {
                (
                    vec![Field192::ZERO; CARRYOPEN_WIDTH],
                    vec![Field192::ZERO; CARRYOPEN_TENSOR_WIDTH],
                    Vec::with_capacity(CARRYOPEN_TENSOR_WIDTH / 2),
                )
            },
            |(transformed, encoded, digest_scratch), (index, row)| {
                if index < CARRYOPEN_ROWS {
                    horizontal_row_root_with_systematic_subtree_scratch(
                        row,
                        message_row_roots[index],
                        horizontal_spectra,
                        transformed,
                        encoded,
                        digest_scratch,
                    )
                } else {
                    horizontal_encoded_row_root_with_scratch(
                        row,
                        horizontal_spectra,
                        transformed,
                        encoded,
                        digest_scratch,
                    )
                }
            },
        )
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

#[cfg(test)]
fn tensor_row_commitments_separate_scratch_for_benchmark(
    vertical_codeword: &[Field192],
    message_row_roots: &[Digest],
    horizontal_spectra: &[Vec<Field192>],
    zeros: &[Digest],
) -> Vec<MatrixCommitment> {
    assert_eq!(
        vertical_codeword.len(),
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_FIELDS
    );
    assert_eq!(message_row_roots.len(), CARRYOPEN_ROWS);
    vertical_codeword
        .chunks_exact(CARRYOPEN_FIELDS)
        .enumerate()
        .map(|(matrix, matrix_values)| {
            let row_roots = matrix_values
                .par_chunks_exact(CARRYOPEN_WIDTH)
                .enumerate()
                .map_init(
                    || {
                        (
                            vec![Field192::ZERO; CARRYOPEN_WIDTH],
                            vec![Field192::ZERO; CARRYOPEN_TENSOR_WIDTH],
                            Vec::with_capacity(CARRYOPEN_TENSOR_WIDTH / 2),
                        )
                    },
                    |(transformed, encoded, digest_scratch), (row_index, row)| {
                        if matrix == 0 {
                            horizontal_row_root_with_systematic_subtree_scratch(
                                row,
                                message_row_roots[row_index],
                                horizontal_spectra,
                                transformed,
                                encoded,
                                digest_scratch,
                            )
                        } else {
                            horizontal_encoded_row_root_with_scratch(
                                row,
                                horizontal_spectra,
                                transformed,
                                encoded,
                                digest_scratch,
                            )
                        }
                    },
                )
                .collect::<Vec<_>>();
            MatrixCommitment {
                root: combine_equal_subtrees(&row_roots),
                row_roots,
                row_domain: CARRYOPEN_ROWS,
                zero_row_root: zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
            }
        })
        .collect()
}

#[cfg(test)]
fn tensor_row_commitments_materialized(
    vertical_codeword: &[Field192],
    message_row_roots: &[Digest],
    horizontal_spectra: &[Vec<Field192>],
    zeros: &[Digest],
) -> Vec<MatrixCommitment> {
    let row_roots = vertical_codeword
        .par_chunks_exact(CARRYOPEN_WIDTH)
        .enumerate()
        .map(|(index, row)| {
            if index < CARRYOPEN_ROWS {
                horizontal_row_root_with_systematic_subtree(
                    row,
                    message_row_roots[index],
                    horizontal_spectra,
                    zeros,
                )
            } else {
                let encoded = encode_horizontal_row(row, horizontal_spectra);
                prefix_root(&encoded, CARRYOPEN_TENSOR_WIDTH, zeros)
            }
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

#[cfg(test)]
fn tensor_row_commitments_with_root(
    vertical_codeword: &[Field192],
    message_row_roots: &[Digest],
    horizontal_spectra: &[Vec<Field192>],
    zeros: &[Digest],
    row_root: fn(&[Field192], usize, &[Digest]) -> Digest,
    combine_roots: fn(&[Digest]) -> Digest,
) -> Vec<MatrixCommitment> {
    let row_roots = vertical_codeword
        .par_chunks_exact(CARRYOPEN_WIDTH)
        .enumerate()
        .map_init(
            || {
                (
                    vec![Field192::ZERO; CARRYOPEN_WIDTH],
                    vec![Field192::ZERO; CARRYOPEN_TENSOR_WIDTH],
                )
            },
            |(transformed, encoded), (index, row)| {
                transformed.copy_from_slice(row);
                wht(transformed);
                if index < CARRYOPEN_ROWS {
                    let mut roots = [message_row_roots[index]; CARRYOPEN_INVERSE_RATE];
                    for (root, spectrum) in roots[1..].iter_mut().zip(horizontal_spectra) {
                        let parity = &mut encoded[..CARRYOPEN_WIDTH];
                        parity.copy_from_slice(transformed);
                        parity
                            .iter_mut()
                            .zip(spectrum)
                            .for_each(|(value, multiplier)| *value *= multiplier);
                        wht(parity);
                        *root = row_root(parity, CARRYOPEN_WIDTH, zeros);
                    }
                    combine_roots(&roots)
                } else {
                    encoded[..CARRYOPEN_WIDTH].copy_from_slice(row);
                    for (block, spectrum) in horizontal_spectra.iter().enumerate() {
                        let start = (block + 1) * CARRYOPEN_WIDTH;
                        let parity = &mut encoded[start..start + CARRYOPEN_WIDTH];
                        parity.copy_from_slice(transformed);
                        parity
                            .iter_mut()
                            .zip(spectrum)
                            .for_each(|(value, multiplier)| *value *= multiplier);
                        wht(parity);
                    }
                    row_root(encoded, CARRYOPEN_TENSOR_WIDTH, zeros)
                }
            },
        )
        .collect::<Vec<_>>();
    row_roots
        .chunks_exact(CARRYOPEN_ROWS)
        .map(|roots| MatrixCommitment {
            root: combine_roots(roots),
            row_roots: roots.to_vec(),
            row_domain: CARRYOPEN_ROWS,
            zero_row_root: zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
        })
        .collect()
}

fn selected_tensor_carry_rows(
    vertical_codeword: &[Field192],
    index_coefficients: &[Field192],
    index_alpha: &[Field192],
    horizontal_spectra: &[Vec<Field192>],
    selected: &[usize],
) -> Vec<Field192> {
    assert_eq!(index_alpha.len(), CARRYOPEN_WIDTH);
    assert_eq!(
        vertical_codeword.len(),
        index_coefficients.len() * CARRYOPEN_WIDTH
    );
    assert_eq!(selected.len(), CARRYOPEN_QUERIES);
    let mut terminal = Vec::with_capacity(CARRYOPEN_QUERIES * CARRYOPEN_TERMINAL_BLOCK);
    for row in selected {
        let start = row * CARRYOPEN_WIDTH;
        terminal.extend(encode_horizontal_row(
            &vertical_codeword[start..start + CARRYOPEN_WIDTH],
            horizontal_spectra,
        ));
        terminal.extend(
            index_alpha
                .iter()
                .map(|value| index_coefficients[*row] * *value),
        );
    }
    assert_eq!(terminal.len(), CARRYOPEN_QUERIES * CARRYOPEN_TERMINAL_BLOCK);
    terminal
}

fn derive_tensor_carry_source_from_folds(
    membership: &TensorProductProof,
    evaluation: &TensorProductProof,
    horizontal_spectra: &[Vec<Field192>],
    selected_rows: Vec<Field192>,
) -> Vec<Field192> {
    assert_eq!(membership.left_ood_row.len(), CARRYOPEN_WIDTH);
    assert_eq!(membership.right_ood_row.len(), CARRYOPEN_WIDTH);
    assert_eq!(evaluation.left_ood_row.len(), CARRYOPEN_WIDTH);
    assert_eq!(
        selected_rows.len(),
        CARRYOPEN_QUERIES * CARRYOPEN_TERMINAL_BLOCK
    );
    let mut terminal = Vec::with_capacity(CARRYOPEN_TERMINAL_FIELDS);
    terminal.extend(encode_horizontal_row(
        &membership.right_ood_row,
        horizontal_spectra,
    ));
    terminal.extend_from_slice(&membership.left_ood_row);
    terminal.extend(encode_horizontal_row(
        &evaluation.left_ood_row,
        horizontal_spectra,
    ));
    terminal.extend(selected_rows);
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
    component_roots: &[Digest],
    front: &SelectedRowFront,
    ood_block_root: Digest,
    zeros: &[Digest],
) -> Option<Digest> {
    let level = *LEVELS.get(level_index)?;
    if front.selected.len() != level.next_blocks - 1 {
        return None;
    }
    let w_roots = virtual_index_selected_roots(
        level_index,
        level,
        component_roots,
        &front.selected,
        level.group,
        zeros,
    )?;
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
        let w_root = *w_roots.get(ordinal)?;
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
/// delayed F root, followed by the public rank-one W row. State-root
/// reconstruction still hashes that public row once; this audit compares the
/// carried values directly with `c[row] * alpha` instead of hashing them again.
fn audit_tensor_carry_restoration(
    level: Level,
    component_roots: &[Digest],
    front: &SelectedRowFront,
    terminal: &[Field192],
    horizontal_spectra: &[Vec<Field192>],
    membership: &PackedSumcheckProof,
    evaluation: &PackedSumcheckProof,
    zeros: &[Digest],
) -> bool {
    if terminal.len() != CARRYOPEN_TERMINAL_FIELDS || front.selected.len() != CARRYOPEN_QUERIES {
        return false;
    }
    let Some(membership_point) = membership.challenges() else {
        return false;
    };
    let Some(evaluation_point) = evaluation.challenges() else {
        return false;
    };
    let Some(public_w_factors) = validated_virtual_index_factors(10, level, component_roots) else {
        return false;
    };
    let Some(public_membership_w) =
        virtual_index_ood_row(10, level, component_roots, &membership_point)
    else {
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
    if membership_w != public_membership_w.as_slice()
        || encode_horizontal_row(&membership_f[..CARRYOPEN_WIDTH], horizontal_spectra)
            != membership_f
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
    front
        .selected
        .par_iter()
        .copied()
        .enumerate()
        .all(|(ordinal, row)| {
            let start = CARRYOPEN_OOD_FIELDS + ordinal * CARRYOPEN_TERMINAL_BLOCK;
            let block = &terminal[start..start + CARRYOPEN_TERMINAL_BLOCK];
            let f_row = &block[..CARRYOPEN_TENSOR_WIDTH];
            let w_row = &block[CARRYOPEN_TENSOR_WIDTH..];
            encode_horizontal_row(&f_row[..CARRYOPEN_WIDTH], horizontal_spectra) == f_row
                && prefix_root(f_row, CARRYOPEN_TENSOR_WIDTH, zeros)
                    == selected_f_row_root(front, level, row).unwrap_or([0_u8; 32])
                && virtual_index_row_matches(&public_w_factors, row, w_row)
        })
}

fn carry_ood_and_terminal_roots(
    component_roots: &[Digest],
    front: &SelectedRowFront,
    terminal: &[Field192],
    zeros: &[Digest],
) -> Option<(Digest, Digest, Digest)> {
    if terminal.len() != CARRYOPEN_TERMINAL_FIELDS || front.selected.len() != CARRYOPEN_QUERIES {
        return None;
    }
    let level = carryopen_level();
    let w_roots = virtual_index_selected_roots(
        10,
        level,
        component_roots,
        &front.selected,
        CARRYOPEN_WIDTH,
        zeros,
    )?;
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
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        let f_root = selected_f_row_root(front, level, row)?;
        let w_root = lift_padded_subtree_root(
            *w_roots.get(ordinal)?,
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
    component_roots: &[Digest],
    front: &SelectedRowFront,
    membership_ood_root: Digest,
    evaluation_ood_root: Digest,
    zeros: &[Digest],
) -> Option<Digest> {
    if front.selected.len() != CARRYOPEN_QUERIES {
        return None;
    }
    let level = carryopen_level();
    let w_roots = virtual_index_selected_roots(
        10,
        level,
        component_roots,
        &front.selected,
        CARRYOPEN_WIDTH,
        zeros,
    )?;
    let mut accumulator =
        MerkleAccumulator::new(CARRYOPEN_SEGMENT_CAPACITY.trailing_zeros() as usize);
    let block_height = (2 * CARRYOPEN_TENSOR_WIDTH).trailing_zeros() as usize;
    accumulator.append_subtree(membership_ood_root, block_height);
    accumulator.append_subtree(evaluation_ood_root, block_height);
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        let f_root = selected_f_row_root(front, level, row)?;
        let w_root = lift_padded_subtree_root(
            *w_roots.get(ordinal)?,
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
    level_index: usize,
    level: Level,
    component_roots: &[Digest],
    front: &SelectedRowFront,
    ood_block_root: Digest,
    capacity: usize,
    zeros: &[Digest],
) -> Option<Digest> {
    if front.selected.len() != level.next_blocks - 1
        || capacity < level.next_blocks * 2 * level.group
    {
        return None;
    }
    let w_roots = virtual_index_selected_roots(
        level_index,
        level,
        component_roots,
        &front.selected,
        level.group,
        zeros,
    )?;
    let block_height = (2 * level.group).trailing_zeros() as usize;
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    accumulator.append_subtree(ood_block_root, block_height);
    for (ordinal, row) in front.selected.iter().copied().enumerate() {
        accumulator.append_subtree(
            parent(
                selected_f_row_root(front, level, row)?,
                *w_roots.get(ordinal)?,
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
        || standard_ood_block_root(level, terminal, zeros) != proof.ood_block_root
    {
        return false;
    }
    let Some(point) = proof.algebra.qa_membership.challenges() else {
        return false;
    };
    let Some(public_w_factors) =
        validated_virtual_index_factors(level_index, level, &proof.component_roots)
    else {
        return false;
    };
    let Some(public_ood_w) =
        virtual_index_ood_row(level_index, level, &proof.component_roots, &point)
    else {
        return false;
    };
    let row_variables = (level.inverse_rate * level.group)
        .next_power_of_two()
        .trailing_zeros() as usize;
    let lane_domain = level.width.next_power_of_two();
    if terminal[level.width..2 * level.width] != public_ood_w
        || point.len() != row_variables + lane_domain.trailing_zeros() as usize
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
    proof
        .selected_front
        .selected
        .par_iter()
        .copied()
        .enumerate()
        .all(|(ordinal, row)| {
            let start = (ordinal + 1) * 2 * level.width;
            let f_row = &terminal[start..start + level.width];
            let w_row = &terminal[start + level.width..start + 2 * level.width];
            prefix_root(f_row, level.group, zeros)
                == selected_f_row_root(&proof.selected_front, level, row).unwrap_or([0_u8; 32])
                && virtual_index_row_matches(&public_w_factors, row, w_row)
        })
}

fn audit_strong_terminal_restoration(
    level_index: usize,
    level: Level,
    component_roots: &[Digest],
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
        || standard_ood_block_root(level, terminal, zeros) != ood_block_root
        || derived_terminal_root_for_level(
            level_index,
            level,
            component_roots,
            front,
            ood_block_root,
            (2 * level.next_blocks * level.group).next_power_of_two(),
            zeros,
        ) != Some(terminal_root)
    {
        return false;
    }
    let Some(point) = membership.challenges() else {
        return false;
    };
    let Some(public_w_factors) =
        validated_virtual_index_factors(level_index, level, component_roots)
    else {
        return false;
    };
    let Some(public_ood_w) = virtual_index_ood_row(level_index, level, component_roots, &point)
    else {
        return false;
    };
    let row_variables = (level.inverse_rate * level.group)
        .next_power_of_two()
        .trailing_zeros() as usize;
    if terminal[level.width..2 * level.width] != public_ood_w
        || point.len() != row_variables + level.width.trailing_zeros() as usize
        || evaluate_power_of_two_message(
            &terminal[level.width..2 * level.width],
            &point[row_variables..],
        ) != membership.terminal_left
        || evaluate_power_of_two_message(&terminal[..level.width], &point[row_variables..])
            != membership.terminal_right
    {
        return false;
    }
    front
        .selected
        .par_iter()
        .copied()
        .enumerate()
        .all(|(ordinal, row)| {
            let start = (ordinal + 1) * 2 * level.width;
            prefix_root(&terminal[start..start + level.width], level.group, zeros)
                == selected_f_row_root(front, level, row).unwrap_or([0_u8; 32])
                && virtual_index_row_matches(
                    &public_w_factors,
                    row,
                    &terminal[start + level.width..start + 2 * level.width],
                )
        })
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
    tail: Option<DirectTailProofArtifact>,
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
            + self.tail.as_ref().map_or(0, |tail| tail.serialize().len())
    }

    fn verify(&self) -> bool {
        let level = carryopen_level();
        if !self.precarry.verify()
            || self.component_roots.len() != 2 * level.inverse_rate + 1
            || self.component_roots[level.inverse_rate - 1] != self.precarry.commitment_root
            || derived_carry_terminal_root(
                &self.component_roots,
                &self.selected_front,
                self.membership_ood_root,
                self.evaluation_ood_root,
                &zero_roots(30),
            ) != Some(self.terminal_source_root)
            || !self.selected_front.verify(
                10,
                level,
                &self.component_roots,
                CARRYOPEN_WIDTH,
                &zero_roots(30),
            )
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
        let Some(tail) = &self.tail else {
            return false;
        };
        if tail.semantic_fields != CARRYOPEN_TERMINAL_FIELDS
            || tail.context_digest != direct_tail_context_digest(&self.component_roots)
            || !tail.verify()
        {
            return false;
        }
        let mut final_roots = self.component_roots.clone();
        final_roots.push(tail.commitment_root);
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
                &self.component_roots,
                &self.selected_front,
                self.membership_ood_root,
                self.evaluation_ood_root,
                &zero_roots(30),
            ) != Some(self.terminal_source_root)
            || !self.selected_front.verify(
                10,
                level,
                &self.component_roots,
                CARRYOPEN_WIDTH,
                &zero_roots(30),
            )
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

    #[cfg(test)]
    fn serialize(&self) -> Vec<u8> {
        let precarry = self.precarry.serialize();
        self.serialize_with_precarry(precarry)
    }

    /// Serialize bytes after the containing proof has verified CarryOpen.
    fn serialize_unchecked(&self) -> Vec<u8> {
        let precarry = self.precarry.serialize_unchecked();
        self.serialize_with_precarry(precarry)
    }

    fn serialize_with_precarry(&self, precarry: Vec<u8>) -> Vec<u8> {
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
        let precarry = ProductionPreCarryProof::deserialize_unchecked(
            payload.get(position..position + sizes[0])?,
        )?;
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
    message_and_systematic_commitment: Duration,
    precarry: Duration,
    encode_and_commit: Duration,
    generator_setup: Duration,
    vertical_and_parity_commitment: Duration,
    front_and_selected_rows: Duration,
    algebra: Duration,
    terminal: Duration,
}

fn run_production_carryopen(
    verifier_repetitions: usize,
    include_standalone_tail: bool,
) -> CarryOpenMeasurement {
    let level = carryopen_level();
    let zeros = zero_roots(30);
    // Reserve the final codeword once and initially expose only its systematic
    // prefix as the message. Extending within this capacity later avoids a
    // separate 384-MiB message allocation and copy.
    let mut proof_codeword = Vec::with_capacity(level.qa_fields());
    proof_codeword.resize(CARRYOPEN_FIELDS, Field192::ZERO);
    proof_codeword
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| *value = precarry_message_value(index));
    let horizontal_setup_start = Instant::now();
    let horizontal_spectra =
        fixed_generator_spectra_for_shape(11, CARRYOPEN_INVERSE_RATE, CARRYOPEN_WIDTH);
    let horizontal_setup = horizontal_setup_start.elapsed();
    let start = Instant::now();
    let (message_root, message_row_roots, systematic_matrix_commitment) =
        message_and_systematic_tensor_commitment(
            &proof_codeword,
            horizontal_spectra.as_ref(),
            &zeros,
        );
    let message_and_systematic_commitment = start.elapsed();

    let start = Instant::now();
    let precarry = build_production_precarry_in_codeword(&mut proof_codeword, message_root);
    let precarry_time = start.elapsed();

    let start = Instant::now();
    let spectra = fixed_generator_spectra(10, level);
    let generator_roots = fixed_generator_roots(10, level);
    let generator_setup = horizontal_setup + start.elapsed();
    let vertical_start = Instant::now();
    assert!(proof_codeword.capacity() >= level.qa_fields());
    let parity_commitments = populate_parity_and_commit_in_spare_capacity(
        level,
        &mut proof_codeword,
        spectra.as_ref(),
        horizontal_spectra.as_ref(),
        64,
        &zeros,
    );
    let vertical_and_parity_commitment = vertical_start.elapsed();
    let proof_commitments = std::iter::once(systematic_matrix_commitment)
        .chain(parity_commitments)
        .collect::<Vec<_>>();
    // The final commitments own the row roots needed by the selected front.
    drop(message_row_roots);
    let front_start = Instant::now();
    let mut component_roots = generator_roots.as_ref().clone();
    component_roots.push(message_root);
    component_roots.extend(proof_commitments.iter().map(|commitment| commitment.root));
    let (index_coefficients, index_alpha) =
        index_oracle_factors(10, level, spectra.as_ref(), &component_roots);
    drop(spectra);
    let index_descriptor = virtual_index_descriptor(10, level, &component_roots);
    component_roots.push(index_descriptor);
    let selected = selected_rows(
        10,
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_ROWS,
        CARRYOPEN_QUERIES,
        &component_roots,
    );
    let (selected_front, _, _) = selected_row_front(level, &proof_commitments, &selected);
    // All commitment roots are already transcript-bound and the selected
    // authentication front is self-contained.
    drop(proof_commitments);
    let selected_rows = selected_tensor_carry_rows(
        &proof_codeword,
        &index_coefficients,
        &index_alpha,
        horizontal_spectra.as_ref(),
        &selected,
    );
    let front_and_selected_rows = front_start.elapsed();
    let encode_and_commit = start.elapsed();

    let start = Instant::now();
    let terminal_point = precarry.sumcheck.challenges().unwrap();
    let row_weights = equality_weights(&terminal_point[..16]);
    let column_weights = equality_weights(&terminal_point[16..]);
    let mut evaluation_coefficients = vec![Field192::ZERO; CARRYOPEN_INVERSE_RATE * CARRYOPEN_ROWS];
    evaluation_coefficients[..CARRYOPEN_ROWS].copy_from_slice(&row_weights);
    let (membership_result, evaluation_result) = prove_dual_rank_one_tensor_product_relations(
        index_coefficients,
        index_alpha,
        proof_codeword,
        evaluation_coefficients,
        column_weights,
        CARRYOPEN_INVERSE_RATE * CARRYOPEN_ROWS,
        CARRYOPEN_WIDTH,
        local_relation_roots(b"CarryOpen-QA-membership", 10, &component_roots),
        local_relation_roots(b"CarryOpen-evaluation", 10, &component_roots),
        Field192::ZERO,
        precarry.sumcheck.terminal_left,
    );
    assert_eq!(
        evaluation_result.proof.claimed_sum,
        precarry.sumcheck.terminal_left
    );
    let terminal_source = derive_tensor_carry_source_from_folds(
        &membership_result,
        &evaluation_result,
        horizontal_spectra.as_ref(),
        selected_rows,
    );
    let membership = membership_result.proof;
    let evaluation = evaluation_result.proof;
    assert!(audit_tensor_carry_restoration(
        level,
        &component_roots,
        &selected_front,
        &terminal_source,
        horizontal_spectra.as_ref(),
        &membership,
        &evaluation,
        &zeros,
    ));
    let (membership_ood_root, evaluation_ood_root, terminal_source_root) =
        carry_ood_and_terminal_roots(&component_roots, &selected_front, &terminal_source, &zeros)
            .expect("CarryOpen terminal segments must derive one splice-compatible root");
    let algebra = start.elapsed();

    let start = Instant::now();
    let tail = include_standalone_tail.then(|| {
        let context = direct_tail_context_digest(&component_roots);
        run_direct_whir_tail(terminal_source.clone(), context, verifier_repetitions).artifact
    });
    let mut final_roots = component_roots.clone();
    final_roots.push(
        tail.as_ref()
            .map_or(terminal_source_root, |tail| tail.commitment_root),
    );
    let phi_link = zero_weighted_relation_proof(
        b"CarryOpen-Phi-link",
        b"CarryOpen-Phi-link",
        10,
        terminal_source.len(),
        &final_roots,
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
        tail,
    };
    if include_standalone_tail {
        assert!(proof.verify());
    } else {
        assert!(CarryOpenCoreProof::from(&proof).verify(terminal_source_root));
    }
    let mut changed = proof.clone();
    changed.evaluation.claimed_sum += Field192::ONE;
    if include_standalone_tail {
        assert!(!changed.verify());
    } else {
        assert!(!CarryOpenCoreProof::from(&changed).verify(terminal_source_root));
    }
    CarryOpenMeasurement {
        proof,
        terminal_source,
        message_and_systematic_commitment,
        precarry: precarry_time,
        encode_and_commit,
        generator_setup,
        vertical_and_parity_commitment,
        front_and_selected_rows,
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
    _verifier_repetitions: usize,
) -> StrongRoundMeasurement {
    let level = strong_round_level(round).expect("strong round must be in the fixed schedule");
    let relation_level = 12 + round;
    let zeros = zero_roots(30);
    source.resize(level.raw, Field192::ZERO);
    assert_eq!(source.len(), level.raw);
    assert_eq!(splice_root(level, &source, &zeros), expected_source_root);

    let start = Instant::now();
    let spectra = fixed_generator_spectra(relation_level, level);
    let generator_roots = fixed_generator_roots(relation_level, level);
    let codeword = encode_full_systematic_codeword(level, &source, spectra.as_ref(), 64);
    let (mut component_roots, proof_commitments) =
        level_commitment(level, &source, &codeword, generator_roots.as_ref(), &zeros);
    assert_eq!(
        component_roots[STRONG_INVERSE_RATE - 1],
        expected_source_root
    );
    let index_oracle =
        materialize_index_oracle(relation_level, level, spectra.as_ref(), &component_roots);
    let index_descriptor = virtual_index_descriptor(relation_level, level, &component_roots);
    component_roots.push(index_descriptor);
    let selected = selected_rows(
        relation_level,
        STRONG_INVERSE_RATE * level.group,
        STRONG_QUERIES,
        &component_roots,
    );
    let (selected_front, _, _) = selected_row_front(level, &proof_commitments, &selected);
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
    let terminal_capacity = strong_round_level(round + 1).map_or_else(
        || (2 * STRONG_NEXT_BLOCKS * level.group).next_power_of_two(),
        Level::view_capacity,
    );
    let terminal_source_root = derived_terminal_root_for_level(
        relation_level,
        level,
        &component_roots,
        &selected_front,
        ood_block_root,
        terminal_capacity,
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
    let base = attach_base.then(|| StrongBaseProof {
        private_f_rows: terminal_source
            .chunks_exact(2 * level.width)
            .flat_map(|pair| pair[..level.width].iter().copied())
            .collect(),
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
    let codeword = encode_full_systematic_codeword(level, &source, &spectra, 64);
    let (mut component_roots, proof_commitments) =
        level_commitment(level, &source, &codeword, &generator_roots, &zeros);
    assert_eq!(
        component_roots[STRONG_INVERSE_RATE - 1],
        expected_source_root
    );
    let index_oracle = materialize_index_oracle(12, level, &spectra, &component_roots);
    let index_descriptor = virtual_index_descriptor(12, level, &component_roots);
    component_roots.push(index_descriptor);
    let selected = selected_rows(
        12,
        STRONG_INVERSE_RATE * STRONG_GROUP,
        STRONG_QUERIES,
        &component_roots,
    );
    let (selected_front, _, _) = selected_row_front(level, &proof_commitments, &selected);
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
        12,
        level,
        &component_roots,
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

#[inline(always)]
fn wht_butterfly(left: &mut [Field192], right: &mut [Field192], index: usize) {
    let old_left = left[index];
    let old_right = right[index];
    left[index] = old_left + old_right;
    right[index] = old_left - old_right;
}

fn wht(values: &mut [Field192]) {
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(2 * half) {
            let (left, right) = block.split_at_mut(half);
            let mut index = 0;
            while index + 4 <= half {
                wht_butterfly(left, right, index);
                wht_butterfly(left, right, index + 1);
                wht_butterfly(left, right, index + 2);
                wht_butterfly(left, right, index + 3);
                index += 4;
            }
            while index < half {
                wht_butterfly(left, right, index);
                index += 1;
            }
        }
        half *= 2;
    }
}

fn wht_parallel_for_factors(values: &mut [Field192]) {
    let mut half = 1;
    while half < values.len() {
        values.par_chunks_mut(2 * half).for_each(|block| {
            let (left, right) = block.split_at_mut(half);
            let mut index = 0;
            while index + 4 <= half {
                wht_butterfly(left, right, index);
                wht_butterfly(left, right, index + 1);
                wht_butterfly(left, right, index + 2);
                wht_butterfly(left, right, index + 3);
                index += 4;
            }
            while index < half {
                wht_butterfly(left, right, index);
                index += 1;
            }
        });
        half *= 2;
    }
}

fn apply_encoder_parallel_for_factors(message: &mut [Field192], spectrum: &[Field192]) {
    wht_parallel_for_factors(message);
    message
        .par_iter_mut()
        .zip(spectrum.par_iter())
        .for_each(|(value, multiplier)| *value *= multiplier);
    wht_parallel_for_factors(message);
}

#[cfg(test)]
fn wht_iterator(values: &mut [Field192]) {
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(2 * half) {
            let (left, right) = block.split_at_mut(half);
            for (left, right) in left.iter_mut().zip(right) {
                let old_left = *left;
                let old_right = *right;
                *left = old_left + old_right;
                *right = old_left - old_right;
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

#[cfg(test)]
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

fn encode_parity_blocks(
    mut message: Vec<Field192>,
    spectra: &[Vec<Field192>],
) -> Vec<Vec<Field192>> {
    wht(&mut message);
    let Some((last, prefix)) = spectra.split_last() else {
        return Vec::new();
    };
    let mut parity_blocks = Vec::with_capacity(spectra.len());
    parity_blocks.extend(prefix.iter().map(|spectrum| {
        let mut parity = message.clone();
        parity
            .iter_mut()
            .zip(spectrum)
            .for_each(|(value, multiplier)| *value *= multiplier);
        wht(&mut parity);
        parity
    }));
    message
        .iter_mut()
        .zip(last)
        .for_each(|(value, multiplier)| *value *= multiplier);
    wht(&mut message);
    parity_blocks.push(message);
    parity_blocks
}

#[cfg(test)]
fn encode_parity_blocks_cloned_forward(
    mut message: Vec<Field192>,
    spectra: &[Vec<Field192>],
) -> Vec<Vec<Field192>> {
    wht(&mut message);
    spectra
        .iter()
        .map(|spectrum| {
            let mut parity = message.clone();
            parity
                .iter_mut()
                .zip(spectrum)
                .for_each(|(value, multiplier)| *value *= multiplier);
            wht(&mut parity);
            parity
        })
        .collect()
}

#[cfg(test)]
fn encode_parity_blocks_iterator_wht(
    mut message: Vec<Field192>,
    spectra: &[Vec<Field192>],
) -> Vec<Vec<Field192>> {
    wht_iterator(&mut message);
    let Some((last, prefix)) = spectra.split_last() else {
        return Vec::new();
    };
    let mut parity_blocks = Vec::with_capacity(spectra.len());
    parity_blocks.extend(prefix.iter().map(|spectrum| {
        let mut parity = message.clone();
        parity
            .iter_mut()
            .zip(spectrum)
            .for_each(|(value, multiplier)| *value *= multiplier);
        wht_iterator(&mut parity);
        parity
    }));
    message
        .iter_mut()
        .zip(last)
        .for_each(|(value, multiplier)| *value *= multiplier);
    wht_iterator(&mut message);
    parity_blocks.push(message);
    parity_blocks
}

fn populate_parity_from_systematic(
    level: Level,
    proof_codeword: &mut [Field192],
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
) {
    populate_parity_from_systematic_with_scatter_tile(
        level,
        proof_codeword,
        spectra,
        batch_lanes,
        Some((64, 4)),
    );
}

fn scatter_parity_block_blocked(
    target: &mut [Field192],
    width: usize,
    lane_start: usize,
    encoded: &[Vec<Vec<Field192>>],
    parity_block: usize,
    row_block: usize,
    lane_tile: usize,
) {
    assert!(width > 0 && row_block > 0 && lane_tile > 0);
    assert_eq!(target.len() % width, 0);
    assert!(lane_start + encoded.len() <= width);
    let rows = target.len() / width;
    assert!(encoded.iter().all(|lane| lane
        .get(parity_block)
        .is_some_and(|values| values.len() == rows)));
    target
        .par_chunks_mut(width * row_block)
        .enumerate()
        .for_each(|(block, target_rows)| {
            let row_start = block * row_block;
            for offset_start in (0..encoded.len()).step_by(lane_tile) {
                let offset_end = (offset_start + lane_tile).min(encoded.len());
                for (local_row, target_row) in target_rows.chunks_exact_mut(width).enumerate() {
                    let row = row_start + local_row;
                    for offset in offset_start..offset_end {
                        target_row[lane_start + offset] = encoded[offset][parity_block][row];
                    }
                }
            }
        });
}

fn populate_parity_from_systematic_with_scatter_tile(
    level: Level,
    proof_codeword: &mut [Field192],
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
    scatter_tile: Option<(usize, usize)>,
) {
    assert_eq!(proof_codeword.len(), level.qa_fields());
    assert_eq!(spectra.len(), level.inverse_rate - 1);
    let systematic_fields = level.group * level.width;
    for lane_start in (0..level.width).step_by(batch_lanes) {
        let lane_end = (lane_start + batch_lanes).min(level.width);
        let encoded = {
            let systematic = &proof_codeword[..systematic_fields];
            (lane_start..lane_end)
                .into_par_iter()
                .map(|lane| {
                    let message = systematic
                        .chunks_exact(level.width)
                        .map(|row| row[lane])
                        .collect::<Vec<_>>();
                    encode_parity_blocks(message, spectra)
                })
                .collect::<Vec<_>>()
        };
        for parity_block in 0..level.inverse_rate - 1 {
            let start = (parity_block + 1) * systematic_fields;
            let target = &mut proof_codeword[start..start + systematic_fields];
            if level.group == CARRYOPEN_ROWS {
                if let Some((row_block, lane_tile)) = scatter_tile {
                    scatter_parity_block_blocked(
                        target,
                        level.width,
                        lane_start,
                        &encoded,
                        parity_block,
                        row_block,
                        lane_tile,
                    );
                    continue;
                }
            }
            target
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

#[cfg(test)]
fn scatter_final_parity_block_and_commit(
    target: &mut [Field192],
    width: usize,
    lane_start: usize,
    encoded: &[Vec<Vec<Field192>>],
    parity_block: usize,
    horizontal_spectra: &[Vec<Field192>],
) -> Vec<Digest> {
    const ROW_BLOCK: usize = 64;
    const LANE_TILE: usize = 4;
    assert_eq!(width, CARRYOPEN_WIDTH);
    assert_eq!(target.len() % width, 0);
    assert_eq!(lane_start + encoded.len(), width);
    let rows = target.len() / width;
    assert!(encoded.iter().all(|lane| lane
        .get(parity_block)
        .is_some_and(|values| values.len() == rows)));
    let mut row_roots = vec![[0_u8; 32]; rows];
    target
        .par_chunks_mut(width * ROW_BLOCK)
        .zip(row_roots.par_chunks_mut(ROW_BLOCK))
        .enumerate()
        .for_each_init(
            || {
                (
                    vec![Field192::ZERO; CARRYOPEN_WIDTH + CARRYOPEN_TENSOR_WIDTH],
                    Vec::with_capacity(CARRYOPEN_TENSOR_WIDTH / 4),
                )
            },
            |(field_scratch, digest_scratch), (block, (target_rows, roots))| {
                let row_start = block * ROW_BLOCK;
                for offset_start in (0..encoded.len()).step_by(LANE_TILE) {
                    let offset_end = (offset_start + LANE_TILE).min(encoded.len());
                    for (local_row, target_row) in target_rows.chunks_exact_mut(width).enumerate() {
                        let row = row_start + local_row;
                        for offset in offset_start..offset_end {
                            target_row[lane_start + offset] = encoded[offset][parity_block][row];
                        }
                    }
                }
                let (transformed, horizontal) = field_scratch.split_at_mut(CARRYOPEN_WIDTH);
                for (row, root) in target_rows.chunks_exact(width).zip(roots) {
                    *root = horizontal_encoded_row_root_with_scratch(
                        row,
                        horizontal_spectra,
                        transformed,
                        horizontal,
                        digest_scratch,
                    );
                }
            },
        );
    row_roots
}

fn scatter_parity_block_blocked_uninit(
    target: &mut [MaybeUninit<Field192>],
    width: usize,
    lane_start: usize,
    encoded: &[Vec<Vec<Field192>>],
    parity_block: usize,
) {
    const ROW_BLOCK: usize = 64;
    const LANE_TILE: usize = 4;
    assert_eq!(target.len() % width, 0);
    assert!(lane_start + encoded.len() <= width);
    let rows = target.len() / width;
    assert!(encoded.iter().all(|lane| lane
        .get(parity_block)
        .is_some_and(|values| values.len() == rows)));
    target
        .par_chunks_mut(width * ROW_BLOCK)
        .enumerate()
        .for_each(|(block, target_rows)| {
            let row_start = block * ROW_BLOCK;
            for offset_start in (0..encoded.len()).step_by(LANE_TILE) {
                let offset_end = (offset_start + LANE_TILE).min(encoded.len());
                for (local_row, target_row) in target_rows.chunks_exact_mut(width).enumerate() {
                    let row = row_start + local_row;
                    for offset in offset_start..offset_end {
                        target_row[lane_start + offset].write(encoded[offset][parity_block][row]);
                    }
                }
            }
        });
}

fn scatter_parity_block_uninit(
    target: &mut [MaybeUninit<Field192>],
    width: usize,
    lane_start: usize,
    encoded: &[Vec<Vec<Field192>>],
    parity_block: usize,
) {
    assert_eq!(target.len() % width, 0);
    assert!(lane_start + encoded.len() <= width);
    let rows = target.len() / width;
    assert!(encoded.iter().all(|lane| lane
        .get(parity_block)
        .is_some_and(|values| values.len() == rows)));
    target
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, target)| {
            for (offset, lane) in encoded.iter().enumerate() {
                target[lane_start + offset].write(lane[parity_block][row]);
            }
        });
}

fn scatter_final_parity_block_and_commit_uninit(
    target: &mut [MaybeUninit<Field192>],
    width: usize,
    lane_start: usize,
    encoded: &[Vec<Vec<Field192>>],
    parity_block: usize,
    horizontal_spectra: &[Vec<Field192>],
) -> Vec<Digest> {
    const ROW_BLOCK: usize = 64;
    const LANE_TILE: usize = 4;
    assert_eq!(width, CARRYOPEN_WIDTH);
    assert_eq!(target.len() % width, 0);
    assert_eq!(lane_start + encoded.len(), width);
    let rows = target.len() / width;
    assert!(encoded.iter().all(|lane| lane
        .get(parity_block)
        .is_some_and(|values| values.len() == rows)));
    let mut row_roots = vec![[0_u8; 32]; rows];
    target
        .par_chunks_mut(width * ROW_BLOCK)
        .zip(row_roots.par_chunks_mut(ROW_BLOCK))
        .enumerate()
        .for_each_init(
            || {
                (
                    vec![Field192::ZERO; CARRYOPEN_WIDTH + CARRYOPEN_TENSOR_WIDTH],
                    Vec::with_capacity(CARRYOPEN_TENSOR_WIDTH / 4),
                )
            },
            |(field_scratch, digest_scratch), (block, (target_rows, roots))| {
                let row_start = block * ROW_BLOCK;
                for offset_start in (0..encoded.len()).step_by(LANE_TILE) {
                    let offset_end = (offset_start + LANE_TILE).min(encoded.len());
                    for (local_row, target_row) in target_rows.chunks_exact_mut(width).enumerate() {
                        let row = row_start + local_row;
                        for offset in offset_start..offset_end {
                            target_row[lane_start + offset]
                                .write(encoded[offset][parity_block][row]);
                        }
                    }
                }
                let (transformed, horizontal) = field_scratch.split_at_mut(CARRYOPEN_WIDTH);
                for (row, root) in target_rows.chunks_exact(width).zip(roots) {
                    // All lanes of this row have been written across the completed
                    // lane batches, including the final batch above.  Field192 is
                    // Copy and the allocation remains fixed for the whole pass.
                    let initialized = unsafe {
                        std::slice::from_raw_parts(row.as_ptr().cast::<Field192>(), width)
                    };
                    *root = horizontal_encoded_row_root_with_scratch(
                        initialized,
                        horizontal_spectra,
                        transformed,
                        horizontal,
                        digest_scratch,
                    );
                }
            },
        );
    row_roots
}

#[cfg(test)]
fn populate_parity_and_commit_from_systematic(
    level: Level,
    proof_codeword: &mut [Field192],
    spectra: &[Vec<Field192>],
    horizontal_spectra: &[Vec<Field192>],
    batch_lanes: usize,
    zeros: &[Digest],
) -> Vec<MatrixCommitment> {
    assert_eq!(proof_codeword.len(), level.qa_fields());
    assert_eq!(spectra.len(), level.inverse_rate - 1);
    assert_eq!(horizontal_spectra.len(), level.inverse_rate - 1);
    assert_eq!(level.group, CARRYOPEN_ROWS);
    assert_eq!(level.width, CARRYOPEN_WIDTH);
    assert_eq!(batch_lanes, 64);
    let systematic_fields = level.group * level.width;
    let mut commitments = Vec::with_capacity(level.inverse_rate - 1);
    for lane_start in (0..level.width).step_by(batch_lanes) {
        let lane_end = (lane_start + batch_lanes).min(level.width);
        let encoded = {
            let systematic = &proof_codeword[..systematic_fields];
            (lane_start..lane_end)
                .into_par_iter()
                .map(|lane| {
                    let message = systematic
                        .chunks_exact(level.width)
                        .map(|row| row[lane])
                        .collect::<Vec<_>>();
                    encode_parity_blocks(message, spectra)
                })
                .collect::<Vec<_>>()
        };
        for parity_block in 0..level.inverse_rate - 1 {
            let start = (parity_block + 1) * systematic_fields;
            let target = &mut proof_codeword[start..start + systematic_fields];
            if lane_end == level.width {
                let row_roots = scatter_final_parity_block_and_commit(
                    target,
                    level.width,
                    lane_start,
                    &encoded,
                    parity_block,
                    horizontal_spectra,
                );
                commitments.push(MatrixCommitment {
                    root: combine_equal_subtrees(&row_roots),
                    row_roots,
                    row_domain: CARRYOPEN_ROWS,
                    zero_row_root: zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
                });
            } else {
                scatter_parity_block_blocked(
                    target,
                    level.width,
                    lane_start,
                    &encoded,
                    parity_block,
                    64,
                    4,
                );
            }
        }
    }
    assert_eq!(commitments.len(), level.inverse_rate - 1);
    commitments
}

/// Fill the three parity quarters directly in the reserved spare capacity.
/// The preceding `resize(..., ZERO)` wrote 1.125 GiB that was immediately
/// overwritten. Every parity cell is initialized exactly once per lane, and
/// the vector length is extended only after all cells and row roots exist.
fn populate_parity_and_commit_in_spare_capacity(
    level: Level,
    proof_codeword: &mut Vec<Field192>,
    spectra: &[Vec<Field192>],
    horizontal_spectra: &[Vec<Field192>],
    batch_lanes: usize,
    zeros: &[Digest],
) -> Vec<MatrixCommitment> {
    assert_eq!(proof_codeword.len(), level.group * level.width);
    assert!(proof_codeword.capacity() >= level.qa_fields());
    assert_eq!(spectra.len(), level.inverse_rate - 1);
    assert_eq!(horizontal_spectra.len(), level.inverse_rate - 1);
    assert_eq!(level.group, CARRYOPEN_ROWS);
    assert_eq!(level.width, CARRYOPEN_WIDTH);
    assert_eq!(batch_lanes, 64);
    assert!(!std::mem::needs_drop::<Field192>());
    let systematic_fields = level.group * level.width;
    let parity_fields = level.qa_fields() - systematic_fields;
    let systematic_ptr = proof_codeword.as_ptr();
    let mut commitments = Vec::with_capacity(level.inverse_rate - 1);
    {
        let spare = proof_codeword.spare_capacity_mut();
        assert!(spare.len() >= parity_fields);
        let parity = &mut spare[..parity_fields];
        for lane_start in (0..level.width).step_by(batch_lanes) {
            let lane_end = (lane_start + batch_lanes).min(level.width);
            let encoded = {
                // The immutable systematic prefix and mutable spare capacity are
                // disjoint, and `with_capacity` prevents reallocation here.
                let systematic =
                    unsafe { std::slice::from_raw_parts(systematic_ptr, systematic_fields) };
                (lane_start..lane_end)
                    .into_par_iter()
                    .map(|lane| {
                        let message = systematic
                            .chunks_exact(level.width)
                            .map(|row| row[lane])
                            .collect::<Vec<_>>();
                        encode_parity_blocks(message, spectra)
                    })
                    .collect::<Vec<_>>()
            };
            for parity_block in 0..level.inverse_rate - 1 {
                let start = parity_block * systematic_fields;
                let target = &mut parity[start..start + systematic_fields];
                if lane_end == level.width {
                    let row_roots = scatter_final_parity_block_and_commit_uninit(
                        target,
                        level.width,
                        lane_start,
                        &encoded,
                        parity_block,
                        horizontal_spectra,
                    );
                    commitments.push(MatrixCommitment {
                        root: combine_equal_subtrees(&row_roots),
                        row_roots,
                        row_domain: CARRYOPEN_ROWS,
                        zero_row_root: zeros[CARRYOPEN_TENSOR_WIDTH.trailing_zeros() as usize],
                    });
                } else {
                    scatter_parity_block_blocked_uninit(
                        target,
                        level.width,
                        lane_start,
                        &encoded,
                        parity_block,
                    );
                }
            }
        }
    }
    // SAFETY: each of the `parity_fields` cells was written once in every
    // lane position above; Field192 has no destructor, and no panic path can
    // observe the spare cells as initialized vector elements.
    unsafe { proof_codeword.set_len(level.qa_fields()) };
    assert_eq!(commitments.len(), level.inverse_rate - 1);
    commitments
}

fn populate_parity_in_spare_capacity(
    level: Level,
    proof_codeword: &mut Vec<Field192>,
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
) {
    let systematic_fields = level.group * level.width;
    assert_eq!(proof_codeword.len(), systematic_fields);
    assert!(proof_codeword.capacity() >= level.qa_fields());
    assert_eq!(spectra.len(), level.inverse_rate - 1);
    assert!(batch_lanes > 0);
    assert!(!std::mem::needs_drop::<Field192>());
    let parity_fields = level.qa_fields() - systematic_fields;
    let systematic_ptr = proof_codeword.as_ptr();
    {
        let spare = proof_codeword.spare_capacity_mut();
        assert!(spare.len() >= parity_fields);
        let parity = &mut spare[..parity_fields];
        for lane_start in (0..level.width).step_by(batch_lanes) {
            let lane_end = (lane_start + batch_lanes).min(level.width);
            let encoded = {
                // The initialized prefix and spare output are disjoint and the
                // pre-reserved allocation cannot move during this pass.
                let systematic =
                    unsafe { std::slice::from_raw_parts(systematic_ptr, systematic_fields) };
                (lane_start..lane_end)
                    .into_par_iter()
                    .map(|lane| {
                        let message = systematic
                            .chunks_exact(level.width)
                            .map(|row| row[lane])
                            .collect::<Vec<_>>();
                        encode_parity_blocks(message, spectra)
                    })
                    .collect::<Vec<_>>()
            };
            for parity_block in 0..level.inverse_rate - 1 {
                let start = parity_block * systematic_fields;
                scatter_parity_block_uninit(
                    &mut parity[start..start + systematic_fields],
                    level.width,
                    lane_start,
                    &encoded,
                    parity_block,
                );
            }
        }
    }
    // SAFETY: every spare cell is written once for each lane before the
    // logical length is extended; a panic leaves the cells outside the Vec.
    unsafe { proof_codeword.set_len(level.qa_fields()) };
}

fn encode_full_systematic_codeword(
    level: Level,
    source: &[Field192],
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
) -> Vec<Field192> {
    assert_eq!(source.len(), level.group * level.width);
    let mut codeword = Vec::with_capacity(level.qa_fields());
    codeword.extend_from_slice(source);
    populate_parity_in_spare_capacity(level, &mut codeword, spectra, batch_lanes);
    codeword
}

fn overwrite_padded_proof_codeword(
    level: Level,
    source: &[Field192],
    codeword: &mut Vec<Field192>,
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
) {
    assert_eq!(source.len(), level.raw);
    assert_eq!(level.raw, level.blocks * level.block_semantic);
    let systematic_fields = level.group * level.width;
    codeword.clear();
    if codeword.capacity() < level.qa_fields() {
        codeword.reserve_exact(level.qa_fields());
    }
    {
        let systematic = &mut codeword.spare_capacity_mut()[..systematic_fields];
        systematic
            .par_chunks_mut(level.width)
            .enumerate()
            .for_each(|(row, target)| {
                let block = row / level.row_span;
                let local_row = row % level.row_span;
                for (lane, value) in target.iter_mut().enumerate() {
                    let local = local_row * level.width + lane;
                    value.write(if block < level.blocks && local < level.block_semantic {
                        source[block * level.block_semantic + local]
                    } else {
                        Field192::ZERO
                    });
                }
            });
    }
    // SAFETY: every systematic coordinate, including padding, is written by
    // the disjoint row partition above; a panic leaves the Vec empty.
    unsafe { codeword.set_len(systematic_fields) };
    populate_parity_in_spare_capacity(level, codeword, spectra, batch_lanes);
}

#[cfg(test)]
fn populate_parity_from_systematic_cloned_forward(
    level: Level,
    proof_codeword: &mut [Field192],
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
) {
    assert_eq!(proof_codeword.len(), level.qa_fields());
    assert_eq!(spectra.len(), level.inverse_rate - 1);
    let systematic_fields = level.group * level.width;
    for lane_start in (0..level.width).step_by(batch_lanes) {
        let lane_end = (lane_start + batch_lanes).min(level.width);
        let encoded = {
            let systematic = &proof_codeword[..systematic_fields];
            (lane_start..lane_end)
                .into_par_iter()
                .map(|lane| {
                    let message = systematic
                        .chunks_exact(level.width)
                        .map(|row| row[lane])
                        .collect::<Vec<_>>();
                    encode_parity_blocks_cloned_forward(message, spectra)
                })
                .collect::<Vec<_>>()
        };
        for parity_block in 0..level.inverse_rate - 1 {
            let start = (parity_block + 1) * systematic_fields;
            proof_codeword[start..start + systematic_fields]
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

#[cfg(test)]
fn populate_parity_from_systematic_iterator_wht(
    level: Level,
    proof_codeword: &mut [Field192],
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
) {
    assert_eq!(proof_codeword.len(), level.qa_fields());
    assert_eq!(spectra.len(), level.inverse_rate - 1);
    let systematic_fields = level.group * level.width;
    for lane_start in (0..level.width).step_by(batch_lanes) {
        let lane_end = (lane_start + batch_lanes).min(level.width);
        let encoded = {
            let systematic = &proof_codeword[..systematic_fields];
            (lane_start..lane_end)
                .into_par_iter()
                .map(|lane| {
                    let message = systematic
                        .chunks_exact(level.width)
                        .map(|row| row[lane])
                        .collect::<Vec<_>>();
                    encode_parity_blocks_iterator_wht(message, spectra)
                })
                .collect::<Vec<_>>()
        };
        for parity_block in 0..level.inverse_rate - 1 {
            let start = (parity_block + 1) * systematic_fields;
            let target = &mut proof_codeword[start..start + systematic_fields];
            scatter_parity_block_blocked(
                target,
                level.width,
                lane_start,
                &encoded,
                parity_block,
                64,
                4,
            );
        }
    }
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

    populate_parity_from_systematic(level, proof_codeword, spectra, batch_lanes);
}

#[cfg(test)]
fn populate_proof_codeword_independent_wht(
    level: Level,
    source: &[Field192],
    proof_codeword: &mut [Field192],
    spectra: &[Vec<Field192>],
    batch_lanes: usize,
) {
    assert_eq!(proof_codeword.len(), level.qa_fields());
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
    let (coefficients, alpha) = index_oracle_factors(level_index, level, spectra, transcript_roots);
    index_oracle
        .par_chunks_mut(level.width)
        .enumerate()
        .for_each(|(row, target)| {
            target
                .iter_mut()
                .zip(&alpha)
                .for_each(|(value, lane_weight)| *value = coefficients[row] * *lane_weight);
        });
}

fn overwrite_index_oracle(
    level_index: usize,
    level: Level,
    spectra: &[Vec<Field192>],
    transcript_roots: &[Digest],
    index_oracle: &mut Vec<Field192>,
) {
    let (coefficients, alpha) = index_oracle_factors(level_index, level, spectra, transcript_roots);
    index_oracle.clear();
    if index_oracle.capacity() < level.qa_fields() {
        index_oracle.reserve_exact(level.qa_fields());
    }
    {
        let output = &mut index_oracle.spare_capacity_mut()[..level.qa_fields()];
        output
            .par_chunks_mut(level.width)
            .enumerate()
            .for_each(|(row, target)| {
                target
                    .iter_mut()
                    .zip(&alpha)
                    .for_each(|(value, lane_weight)| {
                        value.write(coefficients[row] * *lane_weight);
                    });
            });
    }
    // SAFETY: the parallel row partition covers every output cell exactly
    // once, and a panic leaves the spare cells outside the vector length.
    unsafe { index_oracle.set_len(level.qa_fields()) };
}

fn materialize_index_oracle(
    level_index: usize,
    level: Level,
    spectra: &[Vec<Field192>],
    transcript_roots: &[Digest],
) -> Vec<Field192> {
    let mut index_oracle = Vec::with_capacity(level.qa_fields());
    overwrite_index_oracle(
        level_index,
        level,
        spectra,
        transcript_roots,
        &mut index_oracle,
    );
    index_oracle
}

/// Return the public rank-one factorization W[row,lane] = c[row] * alpha[lane].
/// The generator is fixed preprocessing and `transcript_roots` is already fixed
/// before the virtual-W descriptor enters Fiat--Shamir.
#[cfg(test)]
fn index_oracle_factors_outer_parallel_for_benchmark(
    level_index: usize,
    level: Level,
    spectra: &[Vec<Field192>],
    transcript_roots: &[Digest],
) -> (Vec<Field192>, Vec<Field192>) {
    index_oracle_factors_with_wht_mode(level_index, level, spectra, transcript_roots, false, false)
}

#[cfg(test)]
fn index_oracle_factors_sequential_parity_for_benchmark(
    level_index: usize,
    level: Level,
    spectra: &[Vec<Field192>],
    transcript_roots: &[Digest],
) -> (Vec<Field192>, Vec<Field192>) {
    let codeword_rows = level.inverse_rate * level.group;
    let position_domain = codeword_rows.next_power_of_two();
    let beta = equality_weights(
        &(0..position_domain.trailing_zeros() as usize)
            .map(|index| semantic_challenge(b"qa-row", level_index, index, transcript_roots))
            .collect::<Vec<_>>(),
    );
    let alpha_point = (0..level.group.trailing_zeros() as usize)
        .map(|index| semantic_challenge(b"qa-lane", level_index, index, transcript_roots))
        .collect::<Vec<_>>();
    let alpha = equality_weights_prefix(&alpha_point, level.width);
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
    let coefficients = (0..codeword_rows)
        .map(|row| {
            if row < level.group {
                beta[row] - message_weights[row]
            } else {
                beta[row]
            }
        })
        .collect();
    (coefficients, alpha)
}

fn index_oracle_factors_with_wht_mode(
    level_index: usize,
    level: Level,
    spectra: &[Vec<Field192>],
    transcript_roots: &[Digest],
    inner_parallel_wht: bool,
    direct_equality_spectrum: bool,
) -> (Vec<Field192>, Vec<Field192>) {
    let codeword_rows = level.inverse_rate * level.group;
    let position_domain = codeword_rows.next_power_of_two();
    let beta_point = (0..position_domain.trailing_zeros() as usize)
        .map(|index| semantic_challenge(b"qa-row", level_index, index, transcript_roots))
        .collect::<Vec<_>>();
    let beta = equality_weights(&beta_point);
    let alpha_point = (0..level.group.trailing_zeros() as usize)
        .map(|index| semantic_challenge(b"qa-lane", level_index, index, transcript_roots))
        .collect::<Vec<_>>();
    let alpha = equality_weights_prefix(&alpha_point, level.width);
    let row_variables = level.group.trailing_zeros() as usize;
    let block_variables = beta_point.len() - row_variables;
    let block_weights = equality_weights(&beta_point[..block_variables]);
    let transformed_parity = spectra
        .par_iter()
        .enumerate()
        .map(|(block, spectrum)| {
            let start = (block + 1) * level.group;
            let mut transformed = if direct_equality_spectrum {
                scaled_equality_walsh_spectrum(
                    &beta_point[block_variables..],
                    block_weights[block + 1],
                )
            } else {
                beta[start..start + level.group].to_vec()
            };
            if direct_equality_spectrum {
                if inner_parallel_wht {
                    transformed
                        .par_iter_mut()
                        .zip(spectrum.par_iter())
                        .for_each(|(value, multiplier)| *value *= multiplier);
                    wht_parallel_for_factors(&mut transformed);
                } else {
                    transformed
                        .iter_mut()
                        .zip(spectrum)
                        .for_each(|(value, multiplier)| *value *= multiplier);
                    wht(&mut transformed);
                }
            } else if inner_parallel_wht {
                apply_encoder_parallel_for_factors(&mut transformed, spectrum);
            } else {
                apply_encoder(&mut transformed, spectrum);
            }
            transformed
        })
        .collect::<Vec<_>>();
    let mut message_weights = beta[..level.group].to_vec();
    message_weights
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, target)| {
            for transformed in &transformed_parity {
                *target += transformed[row];
            }
        });
    let coefficients = (0..codeword_rows)
        .map(|row| {
            if row < level.group {
                beta[row] - message_weights[row]
            } else {
                beta[row]
            }
        })
        .collect();
    (coefficients, alpha)
}

fn index_oracle_factors(
    level_index: usize,
    level: Level,
    spectra: &[Vec<Field192>],
    transcript_roots: &[Digest],
) -> (Vec<Field192>, Vec<Field192>) {
    // CarryOpen has only three parity spectra, leaving enough worker capacity
    // for profitable nested WHT parallelism.  Smaller certificate and strong
    // levels retain the lower-overhead outer-parallel encoder.  Equality-table
    // spectra are generated directly at groups of at least 256; below that
    // size the saved forward WHT is within scheduling noise.
    index_oracle_factors_with_wht_mode(
        level_index,
        level,
        spectra,
        transcript_roots,
        level.group >= 1 << 14,
        level.group >= 1 << 8,
    )
}

fn virtual_index_descriptor(
    level_index: usize,
    level: Level,
    transcript_roots: &[Digest],
) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/public-virtual-W/v1");
    for value in [
        level_index,
        level.inverse_rate,
        level.group,
        level.width,
        transcript_roots.len(),
    ] {
        hasher.update(&(value as u64).to_le_bytes());
    }
    for root in transcript_roots {
        hasher.update(root);
    }
    *hasher.finalize().as_bytes()
}

type VirtualIndexFactors = Arc<(Vec<Field192>, Vec<Field192>)>;
type FixedGeneratorSpectra = Arc<Vec<Vec<Field192>>>;
type FixedGeneratorCache = Mutex<BTreeMap<(usize, usize, usize), FixedGeneratorSpectra>>;
type FixedGeneratorRoots = Arc<Vec<Digest>>;
type FixedGeneratorRootCache = Mutex<BTreeMap<(usize, usize, usize), FixedGeneratorRoots>>;

fn fixed_generator_spectra_cache() -> &'static FixedGeneratorCache {
    static CACHE: OnceLock<FixedGeneratorCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn fixed_generator_roots_cache() -> &'static FixedGeneratorRootCache {
    static CACHE: OnceLock<FixedGeneratorRootCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Return fixed-code generator spectra from shared index preprocessing.  They
/// do not depend on a statement or transcript and therefore belong in the
/// public parameters rather than either online proving or verification.
fn fixed_generator_spectra_for_shape(
    level_index: usize,
    inverse_rate: usize,
    group: usize,
) -> FixedGeneratorSpectra {
    let key = (level_index, inverse_rate, group);
    let cached = {
        let cache = fixed_generator_spectra_cache()
            .lock()
            .expect("fixed-generator cache mutex must not be poisoned");
        cache.get(&key).cloned()
    };
    if let Some(spectra) = cached {
        return spectra;
    }
    let spectra = Arc::new(
        (0..inverse_rate - 1)
            .into_par_iter()
            .map(|block| generator_spectrum(level_index, block, group))
            .collect::<Vec<_>>(),
    );
    fixed_generator_spectra_cache()
        .lock()
        .expect("fixed-generator cache mutex must not be poisoned")
        .insert(key, spectra.clone());
    spectra
}

fn fixed_generator_spectra(level_index: usize, level: Level) -> FixedGeneratorSpectra {
    fixed_generator_spectra_for_shape(level_index, level.inverse_rate, level.group)
}

fn fixed_generator_roots(level_index: usize, level: Level) -> FixedGeneratorRoots {
    let key = (level_index, level.inverse_rate, level.group);
    let cached = {
        let cache = fixed_generator_roots_cache()
            .lock()
            .expect("fixed-generator root cache mutex must not be poisoned");
        cache.get(&key).cloned()
    };
    if let Some(roots) = cached {
        return roots;
    }
    let spectra = fixed_generator_spectra(level_index, level);
    let roots = Arc::new(
        spectra
            .par_iter()
            .map_init(Vec::new, |scratch, spectrum| {
                exact_prefix_root_with_scratch(spectrum, scratch)
            })
            .collect::<Vec<_>>(),
    );
    fixed_generator_roots_cache()
        .lock()
        .expect("fixed-generator root cache mutex must not be poisoned")
        .insert(key, roots.clone());
    roots
}

fn preprocess_canonical_fixed_generators() {
    for (level_index, level) in LEVELS.iter().copied().enumerate() {
        fixed_generator_roots(level_index, level);
    }
    fixed_generator_roots(10, carryopen_level());
    fixed_generator_spectra_for_shape(11, CARRYOPEN_INVERSE_RATE, CARRYOPEN_WIDTH);
    for round in 0..STRONG_ROUNDS {
        fixed_generator_roots(
            12 + round,
            strong_round_level(round).expect("canonical strong round must have a level"),
        );
    }
}

fn canonical_fixed_generator_fields() -> usize {
    LEVELS
        .iter()
        .map(|level| (level.inverse_rate - 1) * level.group)
        .sum::<usize>()
        + (CARRYOPEN_INVERSE_RATE - 1) * CARRYOPEN_ROWS
        + (CARRYOPEN_INVERSE_RATE - 1) * CARRYOPEN_WIDTH
        + (0..STRONG_ROUNDS)
            .map(|round| {
                let level = strong_round_level(round).unwrap();
                (level.inverse_rate - 1) * level.group
            })
            .sum::<usize>()
}

fn canonical_fixed_generator_root_count() -> usize {
    LEVELS
        .iter()
        .map(|level| level.inverse_rate - 1)
        .sum::<usize>()
        + (CARRYOPEN_INVERSE_RATE - 1)
        + (0..STRONG_ROUNDS)
            .map(|round| strong_round_level(round).unwrap().inverse_rate - 1)
            .sum::<usize>()
}

fn canonical_fixed_generator_bytes() -> usize {
    canonical_fixed_generator_fields() * 24 + canonical_fixed_generator_root_count() * 32
}

fn virtual_index_factor_cache() -> &'static Mutex<BTreeMap<Digest, VirtualIndexFactors>> {
    static CACHE: OnceLock<Mutex<BTreeMap<Digest, VirtualIndexFactors>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn virtual_index_row_root_cache() -> &'static Mutex<BTreeMap<(Digest, usize, usize), Digest>> {
    static CACHE: OnceLock<Mutex<BTreeMap<(Digest, usize, usize), Digest>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn cached_virtual_index_factors(
    level_index: usize,
    level: Level,
    transcript_roots: &[Digest],
) -> VirtualIndexFactors {
    let descriptor = virtual_index_descriptor(level_index, level, transcript_roots);
    let cached = {
        let cache = virtual_index_factor_cache()
            .lock()
            .expect("virtual-W cache mutex must not be poisoned");
        cache.get(&descriptor).cloned()
    };
    if let Some(factors) = cached {
        return factors;
    }
    let spectra = fixed_generator_spectra(level_index, level);
    let factors = Arc::new(index_oracle_factors(
        level_index,
        level,
        spectra.as_ref(),
        transcript_roots,
    ));
    virtual_index_factor_cache()
        .lock()
        .expect("virtual-W cache mutex must not be poisoned")
        .insert(descriptor, factors.clone());
    factors
}

fn validated_virtual_index_factors(
    level_index: usize,
    level: Level,
    component_roots: &[Digest],
) -> Option<VirtualIndexFactors> {
    if component_roots.len() != 2 * level.inverse_rate + 1 {
        return None;
    }
    let fixed_roots = fixed_generator_roots(level_index, level);
    if component_roots[..level.inverse_rate - 1] != fixed_roots[..] {
        return None;
    }
    let transcript_roots = &component_roots[..2 * level.inverse_rate];
    (component_roots[2 * level.inverse_rate]
        == virtual_index_descriptor(level_index, level, transcript_roots))
    .then(|| cached_virtual_index_factors(level_index, level, transcript_roots))
}

fn virtual_index_row_matches(
    factors: &VirtualIndexFactors,
    row: usize,
    values: &[Field192],
) -> bool {
    let (coefficients, alpha) = factors.as_ref();
    let Some(coefficient) = coefficients.get(row) else {
        return false;
    };
    values.len() == alpha.len()
        && values
            .iter()
            .zip(alpha)
            .all(|(value, lane)| *value == *coefficient * *lane)
}

fn clear_virtual_index_factor_cache() {
    virtual_index_factor_cache()
        .lock()
        .expect("virtual-W cache mutex must not be poisoned")
        .clear();
    virtual_index_row_root_cache()
        .lock()
        .expect("virtual-W row-root cache mutex must not be poisoned")
        .clear();
}

#[cfg(test)]
fn clear_fixed_generator_spectra_cache() {
    fixed_generator_spectra_cache()
        .lock()
        .expect("fixed-generator cache mutex must not be poisoned")
        .clear();
    fixed_generator_roots_cache()
        .lock()
        .expect("fixed-generator root cache mutex must not be poisoned")
        .clear();
}

fn virtual_index_selected_roots(
    level_index: usize,
    level: Level,
    component_roots: &[Digest],
    selected: &[usize],
    row_capacity: usize,
    zeros: &[Digest],
) -> Option<Vec<Digest>> {
    if component_roots.len() != 2 * level.inverse_rate + 1
        || row_capacity < level.width
        || !row_capacity.is_power_of_two()
    {
        return None;
    }
    let factors = validated_virtual_index_factors(level_index, level, component_roots)?;
    let (coefficients, alpha) = factors.as_ref();
    if selected.iter().any(|row| *row >= coefficients.len()) {
        return None;
    }
    let descriptor = component_roots[2 * level.inverse_rate];
    let mut roots = vec![None; selected.len()];
    {
        let cache = virtual_index_row_root_cache()
            .lock()
            .expect("virtual-W row-root cache mutex must not be poisoned");
        for (slot, row) in roots.iter_mut().zip(selected) {
            *slot = cache.get(&(descriptor, row_capacity, *row)).copied();
        }
    }
    roots
        .par_iter_mut()
        .zip(selected.par_iter())
        .for_each(|(slot, row)| {
            if slot.is_none() {
                *slot = Some(scaled_prefix_root(
                    alpha,
                    coefficients[*row],
                    row_capacity,
                    zeros,
                ));
            }
        });
    let roots = roots.into_iter().collect::<Option<Vec<_>>>()?;
    {
        let mut cache = virtual_index_row_root_cache()
            .lock()
            .expect("virtual-W row-root cache mutex must not be poisoned");
        for (row, root) in selected.iter().zip(&roots) {
            cache.insert((descriptor, row_capacity, *row), *root);
        }
    }
    Some(roots)
}

fn virtual_index_ood_row(
    level_index: usize,
    level: Level,
    component_roots: &[Digest],
    point: &[Field192],
) -> Option<Vec<Field192>> {
    let row_domain = (level.inverse_rate * level.group).next_power_of_two();
    let row_variables = row_domain.trailing_zeros() as usize;
    let lane_domain = level.width.next_power_of_two();
    if point.len() != row_variables + lane_domain.trailing_zeros() as usize {
        return None;
    }
    let factors = validated_virtual_index_factors(level_index, level, component_roots)?;
    let (coefficients, alpha) = factors.as_ref();
    let row_weights = equality_weights(&point[..row_variables]);
    let coefficient = coefficients
        .iter()
        .zip(row_weights)
        .fold(Field192::ZERO, |sum, (value, weight)| sum + *value * weight);
    Some(alpha.iter().map(|weight| coefficient * *weight).collect())
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

    #[cfg(test)]
    fn verify_scalar(&self, expected_root: Digest, row_domain: usize) -> bool {
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
            .collect::<Vec<_>>();
        let mut frontier_position = 0;
        let mut width = row_domain;
        while width > 1 {
            let mut parent_indices = Vec::with_capacity(active.len());
            let mut child_pairs = Vec::with_capacity(active.len());
            let mut position = 0;
            while position < active.len() {
                let (index, root) = active[position];
                let sibling = index ^ 1;
                let sibling_root = if index & 1 == 0
                    && active
                        .get(position + 1)
                        .is_some_and(|(next, _)| *next == sibling)
                {
                    position += 1;
                    active[position].1
                } else {
                    let Some(value) = self.frontier.get(frontier_position) else {
                        return false;
                    };
                    frontier_position += 1;
                    *value
                };
                parent_indices.push(index >> 1);
                child_pairs.push(if index & 1 == 0 {
                    (root, sibling_root)
                } else {
                    (sibling_root, root)
                });
                position += 1;
            }
            let parent_roots = parent_pairs_batched(&child_pairs);
            active = parent_indices.into_iter().zip(parent_roots).collect();
            width >>= 1;
        }
        frontier_position == self.frontier.len()
            && active.len() == 1
            && active[0] == (0, expected_root)
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
}

impl SelectedRowFront {
    const MAGIC: &'static [u8; 8] = b"LILSRF02";

    fn verify(
        &self,
        level_index: usize,
        level: Level,
        component_roots: &[Digest],
        row_capacity: usize,
        zeros: &[Digest],
    ) -> bool {
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
        virtual_index_selected_roots(
            level_index,
            level,
            component_roots,
            &self.selected,
            row_capacity,
            zeros,
        )
        .is_some()
    }

    fn serialize(&self) -> Vec<u8> {
        let proof_payloads = self
            .proof_blocks
            .iter()
            .map(|(block, proof)| (*block, proof.serialize()))
            .collect::<Vec<_>>();
        let size = 24
            + 4 * self.selected.len()
            + proof_payloads
                .iter()
                .map(|(_, proof)| 8 + proof.len())
                .sum::<usize>()
            + 4;
        let mut output = Vec::with_capacity(size);
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.selected.len() as u32).to_le_bytes());
        output.extend_from_slice(&(proof_payloads.len() as u32).to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        for index in &self.selected {
            output.extend_from_slice(&(*index as u32).to_le_bytes());
        }
        for (block, payload) in proof_payloads {
            output.extend_from_slice(&(block as u32).to_le_bytes());
            output.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            output.extend_from_slice(&payload);
        }
        output.extend_from_slice(&0_u32.to_le_bytes());
        output
    }

    fn byte_breakdown(&self) -> (usize, usize, usize) {
        let total = self.serialize().len();
        let proof_rows = self
            .proof_blocks
            .iter()
            .map(|(_, proof)| proof.serialize().len())
            .sum::<usize>();
        let framing = total
            .checked_sub(proof_rows)
            .expect("selected-front components must fit in its serialization");
        (proof_rows, 0, framing)
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
        if declared_index_size != 0 || index_size != 0 || payload.len() != position {
            return None;
        }
        Some(Self {
            selected,
            proof_blocks,
        })
    }
}

fn selected_row_front(
    level: Level,
    proof_commitments: &[MatrixCommitment],
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
    let front = SelectedRowFront {
        selected: selected.to_vec(),
        proof_blocks,
    };
    let payload = front.serialize();
    let parsed = SelectedRowFront::deserialize(&payload, level)
        .expect("canonical selected-row front must parse");
    assert_eq!(parsed, front);
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

fn selected_row_values(values: &[Field192], row_width: usize, indices: &[usize]) -> Vec<Field192> {
    assert!(row_width > 0 && values.len() % row_width == 0);
    assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(indices
        .last()
        .is_none_or(|index| *index < values.len() / row_width));
    let mut selected = Vec::with_capacity(indices.len() * row_width);
    for row_index in indices.iter().copied() {
        let start = row_index * row_width;
        selected.extend_from_slice(&values[start..start + row_width]);
    }
    selected
}

fn derive_next_source_from_selected_rows(
    level: Level,
    selected_proof_rows: &[Field192],
    selected_index_rows: &[Field192],
    proof_ood: &[Field192],
    index_ood: &[Field192],
) -> Vec<Field192> {
    let query_count = level.next_blocks - 1;
    assert_eq!(selected_proof_rows.len(), query_count * level.width);
    assert_eq!(selected_index_rows.len(), selected_proof_rows.len());
    assert_eq!(proof_ood.len(), level.width);
    assert_eq!(index_ood.len(), level.width);

    let mut next = vec![Field192::ZERO; 2 * level.next_blocks * level.width];
    next[..level.width].copy_from_slice(proof_ood);
    next[level.width..2 * level.width].copy_from_slice(index_ood);
    for block in 0..query_count {
        let target = (block + 1) * 2 * level.width;
        let source = block * level.width;
        next[target..target + level.width]
            .copy_from_slice(&selected_proof_rows[source..source + level.width]);
        next[target + level.width..target + 2 * level.width]
            .copy_from_slice(&selected_index_rows[source..source + level.width]);
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
        let spectra = fixed_generator_spectra(level_index, level);
        let generator_roots = fixed_generator_roots(level_index, level);
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
            &current_component_roots,
        );
        breakdown.index_oracle += start.elapsed();
        assert_eq!(
            dot(&left[qa_start..qa_end], &right[qa_start..qa_end]),
            Field192::ZERO
        );

        let start = Instant::now();
        let index_descriptor =
            virtual_index_descriptor(level_index, level, &current_component_roots);
        transcript_roots.push(index_descriptor);
        current_component_roots.push(index_descriptor);
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
            selected_row_front(level, &proof_commitments, &selected);
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
            &transition_proofs[level_index].component_roots,
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

/// Construct only the certificate objects consumed by the recursive proof.
///
/// Unlike [`semantic_vectors`], this path does not pack every local relation
/// into two `PRODUCTION_FIELDS` diagnostic vectors. Each QA or copy relation
/// owns exactly one pair of local vectors and moves them into its sumcheck.
/// Selected rows are extracted before the QA vectors are consumed, so the
/// authenticated successor and every transcript remain unchanged.
fn semantic_certificate_objects(
    batch_lanes: usize,
    direct_whir_tail: bool,
) -> (
    Vec<Digest>,
    Vec<Field192>,
    Option<Digest>,
    Digest,
    Vec<ProductionTransitionProof>,
    SemanticBreakdown,
) {
    assert!(batch_lanes > 0);
    let zeros = zero_roots(PRODUCTION_VARIABLES);
    let mut transcript_roots = Vec::<Digest>::new();
    let mut phi_lengths = Vec::<usize>::with_capacity(LEVELS.len());
    let mut transition_proofs = Vec::with_capacity(LEVELS.len());
    let mut breakdown = SemanticBreakdown::default();
    let mut source = (0..LEVELS[0].raw)
        .into_par_iter()
        .map(|index| semantic_value(0, index))
        .collect::<Vec<_>>();
    let mut relation_left = Vec::<Field192>::new();
    let mut relation_right = Vec::<Field192>::new();

    for (level_index, level) in LEVELS.into_iter().enumerate() {
        assert_eq!(level.raw, level.blocks * level.block_semantic);
        assert_eq!(source.len(), level.raw);

        let start = Instant::now();
        let spectra = fixed_generator_spectra(level_index, level);
        let generator_roots = fixed_generator_roots(level_index, level);
        breakdown.generator_preprocessing += start.elapsed();

        let start = Instant::now();
        overwrite_padded_proof_codeword(
            level,
            &source,
            &mut relation_right,
            spectra.as_ref(),
            batch_lanes,
        );
        breakdown.proof_encoding += start.elapsed();

        let start = Instant::now();
        let (level_roots, proof_commitments) = level_commitment(
            level,
            &source,
            &relation_right,
            generator_roots.as_ref(),
            &zeros,
        );
        let mut current_component_roots = level_roots.clone();
        transcript_roots.extend(level_roots);
        breakdown.proof_commitment += start.elapsed();

        let start = Instant::now();
        overwrite_index_oracle(
            level_index,
            level,
            spectra.as_ref(),
            &current_component_roots,
            &mut relation_left,
        );
        breakdown.index_oracle += start.elapsed();
        assert_eq!(dot(&relation_left, &relation_right), Field192::ZERO);

        let start = Instant::now();
        let index_descriptor =
            virtual_index_descriptor(level_index, level, &current_component_roots);
        transcript_roots.push(index_descriptor);
        current_component_roots.push(index_descriptor);
        breakdown.index_commitment += start.elapsed();

        let selected = selected_rows(
            level_index,
            level.inverse_rate * level.group,
            level.next_blocks - 1,
            &transcript_roots,
        );
        let start = Instant::now();
        let (selected_front, opening_bytes, frontier_hashes) =
            selected_row_front(level, &proof_commitments, &selected);
        breakdown.source_opening += start.elapsed();
        breakdown.source_opening_bytes += opening_bytes;
        breakdown.source_frontier_hashes += frontier_hashes;
        let selected_proof_rows = selected_row_values(&relation_right, level.width, &selected);
        let selected_index_rows = selected_row_values(&relation_left, level.width, &selected);

        let start = Instant::now();
        let qa_tensor = prove_tensor_product_relation_reusing(
            &mut relation_left,
            &mut relation_right,
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

        let start = Instant::now();
        let start_sumcheck = Instant::now();
        // Both full views are deterministic permutations of the same source,
        // so their honest residual is identically zero.  Generate the exact
        // zero-relation transcript without materializing either view, its
        // residual, or the equality table.
        let dual_view_copy = zero_weighted_relation_proof(
            b"copy",
            b"dual-view-copy",
            level_index,
            level.view_capacity(),
            &transcript_roots,
        );
        breakdown.transition_sumchecks += start_sumcheck.elapsed();

        transition_proofs.push(ProductionTransitionProof {
            level: level_index,
            component_roots: current_component_roots,
            ood_block_root: [0_u8; 32],
            next_source_root: [0_u8; 32],
            selected_front,
            algebra: TransitionAlgebraProof {
                level: level_index,
                qa_membership,
                dual_view_copy,
                phi_link: None,
            },
        });

        let expected_next = derive_next_source_from_selected_rows(
            level,
            &selected_proof_rows,
            &selected_index_rows,
            &qa_tensor.right_ood_row,
            &qa_tensor.left_ood_row,
        );
        let ood_block_root = standard_ood_block_root(level, &expected_next, &zeros);
        let next_source_root = derived_next_source_root(
            level_index,
            &transition_proofs[level_index].component_roots,
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
        phi_lengths.push(expected_next.len());
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

    for (level_index, fields) in phi_lengths.into_iter().enumerate() {
        let start_sumcheck = Instant::now();
        transition_proofs[level_index].algebra.phi_link = Some(zero_phi_link_proof(
            b"Phi-link",
            b"Phi-link",
            level_index,
            fields,
            &transcript_roots,
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

    (
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

/// Build the production certificate state without the diagnostic global
/// product-sumcheck. The final recursive proof consumes only the local
/// transition proofs and terminal source; the global fold is not serialized
/// or referenced by any challenge in that proof.
fn run_semantic_certificate_only(batch_lanes: usize, direct_whir_tail: bool) -> KernelMeasurement {
    let start = Instant::now();
    let (
        roots,
        terminal_tail,
        terminal_context,
        terminal_binding_root,
        transition_proofs,
        breakdown,
    ) = semantic_certificate_objects(batch_lanes, direct_whir_tail);
    let setup = start.elapsed();
    let component_root_count = roots.len();
    let roots = aggregate_roots(&roots);
    KernelMeasurement {
        setup,
        polynomial: Duration::ZERO,
        folding: Duration::ZERO,
        total: Duration::ZERO,
        terminal_left: Field192::ZERO,
        terminal_right: Field192::ZERO,
        terminal_claim: Field192::ZERO,
        pairs: Vec::new(),
        transcript_root_count: roots.len(),
        roots,
        fields: PRODUCTION_FIELDS,
        component_root_count,
        semantic_breakdown: Some(breakdown),
        transition_proofs: Some(transition_proofs),
        terminal_tail: Some(terminal_tail),
        terminal_context,
        terminal_binding_root: Some(terminal_binding_root),
    }
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
    assert!(semantic_fields >= 19);
    let mut entries = Vec::with_capacity(19);
    let mut counter = 0_u64;
    while entries.len() < 19 {
        let digest = blake3::hash(&[label, &counter.to_le_bytes()].concat());
        let mut index_bytes = [0_u8; 8];
        index_bytes.copy_from_slice(&digest.as_bytes()[..8]);
        let index = u64::from_le_bytes(index_bytes) as usize % semantic_fields;
        if entries.iter().all(|(existing, _)| *existing != index) {
            entries.push((
                index,
                Field192::from_le_bytes_mod_order(&digest.as_bytes()[8..]),
            ));
        }
        counter += 1;
    }
    entries
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
            && self
                .selected_front
                .verify(12, level, &self.component_roots, level.group, &zeros)
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
                12,
                level,
                &self.component_roots,
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
        self.serialize_unchecked()
    }

    /// Serialize bytes after a verified aggregate has accepted this component.
    fn serialize_unchecked(&self) -> Vec<u8> {
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

    fn terminal_capacity(&self) -> Option<usize> {
        let level = self.level()?;
        Some(strong_round_level(self.round + 1).map_or_else(
            || (2 * level.next_blocks * level.group).next_power_of_two(),
            Level::view_capacity,
        ))
    }

    fn verify(&self) -> bool {
        let Some(level) = self.level() else {
            return false;
        };
        let relation_level = 12 + self.round;
        self.component_roots.len() == 2 * STRONG_INVERSE_RATE + 1
            && self.selected_front.verify(
                relation_level,
                level,
                &self.component_roots,
                level.group,
                &zero_roots(30),
            )
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
                relation_level,
                level,
                &self.component_roots,
                &self.selected_front,
                self.ood_block_root,
                self.terminal_capacity().unwrap(),
                &zero_roots(30),
            ) == Some(self.terminal_source_root)
    }

    #[cfg(test)]
    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        self.serialize_unchecked()
    }

    /// Serialize bytes after a verified recursive aggregate has accepted this transition.
    fn serialize_unchecked(&self) -> Vec<u8> {
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

    /// Structural parser used only inside a verified strong aggregate.
    fn deserialize_unchecked(payload: &[u8]) -> Option<Self> {
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
        Some(proof)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StrongBaseProof {
    private_f_rows: Vec<Field192>,
}

impl StrongBaseProof {
    const MAGIC: &'static [u8; 8] = b"LILSBP03";

    fn restore_terminal_witness(&self, core: &StrongTransitionCore) -> Option<Vec<Field192>> {
        let level = core.level()?;
        if self.private_f_rows.len() != level.next_blocks * level.width {
            return None;
        }
        let point = core.membership.challenges()?;
        let ood_w = virtual_index_ood_row(12 + core.round, level, &core.component_roots, &point)?;
        let factors =
            validated_virtual_index_factors(12 + core.round, level, &core.component_roots)?;
        let (coefficients, alpha) = factors.as_ref();
        let mut terminal = Vec::with_capacity(2 * self.private_f_rows.len());
        for (block, f_row) in self.private_f_rows.chunks_exact(level.width).enumerate() {
            terminal.extend_from_slice(f_row);
            if block == 0 {
                terminal.extend_from_slice(&ood_w);
            } else {
                let row = *core.selected_front.selected.get(block - 1)?;
                let coefficient = *coefficients.get(row)?;
                terminal.extend(alpha.iter().map(|value| coefficient * *value));
            }
        }
        Some(terminal)
    }

    fn verify(&self, core: &StrongTransitionCore) -> bool {
        let Some(level) = core.level() else {
            return false;
        };
        let Some(terminal_witness) = self.restore_terminal_witness(core) else {
            return false;
        };
        self.private_f_rows.len() == level.next_blocks * level.width
            && audit_strong_terminal_restoration(
                12 + core.round,
                level,
                &core.component_roots,
                &core.selected_front,
                &core.membership,
                core.ood_block_root,
                core.terminal_source_root,
                &terminal_witness,
                &zero_roots(30),
            )
    }

    #[cfg(test)]
    fn serialize(&self, core: &StrongTransitionCore) -> Vec<u8> {
        assert!(self.verify(core));
        self.serialize_unchecked()
    }

    /// Serialize bytes after a verified aggregate has accepted this base proof.
    fn serialize_unchecked(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(12 + self.private_f_rows.len() * 24);
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.private_f_rows.len() as u32).to_le_bytes());
        for value in &self.private_f_rows {
            output.extend_from_slice(&canonical_field_bytes(*value));
        }
        output
    }

    /// Structural parser used only inside a verified strong aggregate.
    fn deserialize_unchecked(payload: &[u8], core: &StrongTransitionCore) -> Option<Self> {
        const HEADER: usize = 12;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let fields = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let level = core.level()?;
        if fields != level.next_blocks * level.width || payload.len() != HEADER + fields * 24 {
            return None;
        }
        let mut position = HEADER;
        let mut private_f_rows = Vec::with_capacity(fields);
        for _ in 0..fields {
            let bytes = payload.get(position..position + 24)?;
            let value = Field192::from_le_bytes_mod_order(bytes);
            if canonical_field_bytes(value).as_slice() != bytes {
                return None;
            }
            private_f_rows.push(value);
            position += 24;
        }
        let proof = Self { private_f_rows };
        Some(proof)
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
            .map(ProductionTransitionProof::serialize_unchecked)
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
            transitions.push(ProductionTransitionProof::deserialize_unchecked(
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
    if !transitions
        .par_iter()
        .enumerate()
        .all(|(level_index, transition)| transition.level == level_index && transition.verify())
    {
        return false;
    }
    let mut transcript_roots = Vec::new();
    for (level_index, transition) in transitions.iter().enumerate() {
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
                &self.carryopen.component_roots,
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
        let carryopen = self.carryopen.serialize_unchecked();
        let certificate = self
            .certificate
            .iter()
            .map(ProductionTransitionProof::serialize_unchecked)
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
            certificate.push(ProductionTransitionProof::deserialize_unchecked(
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
        let carry = self.carryopen.serialize_unchecked();
        let certificate = self
            .certificate
            .iter()
            .map(ProductionTransitionProof::serialize_unchecked)
            .collect::<Vec<_>>();
        let strong = self.strong.serialize_unchecked();
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
            certificate.push(ProductionTransitionProof::deserialize_unchecked(
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CanonicalProofByteBreakdown {
    total: usize,
    carry_total: usize,
    certificate_total: usize,
    strong_total: usize,
    base_total: usize,
    proof_row_merkle: usize,
    index_row_merkle: usize,
    selected_front_framing: usize,
    algebra: usize,
    terminal_witness: usize,
    terminal_pcs: usize,
    other: usize,
    proof_row_merkle_by_stage: [usize; 11],
    index_row_merkle_by_stage: [usize; 11],
    selected_front_framing_by_stage: [usize; 11],
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
        if !self
            .strong
            .par_iter()
            .enumerate()
            .all(|(round, proof)| proof.round == round && proof.verify())
        {
            return false;
        }
        if self
            .strong
            .windows(2)
            .any(|pair| pair[1].source_root() != Some(pair[0].terminal_source_root))
        {
            return false;
        }
        self.base.verify(self.strong.last().unwrap())
            && self.carryopen.verify(self.carryopen.terminal_source_root)
            && verify_certificate_core(&self.certificate, certificate_root)
    }

    fn serialize(&self) -> Vec<u8> {
        // Construction verifies the aggregate before emission, and the remote
        // acceptance boundary is `deserialize` plus its semantic pass. Keep a
        // debug-build invariant without charging production serialization for
        // a local verifier execution.
        debug_assert!(self.verify());
        self.serialize_unchecked()
    }

    /// Serialize the already-verified recursive aggregate without repeating
    /// semantic checks in every nested component.
    fn serialize_unchecked(&self) -> Vec<u8> {
        let carry = self.carryopen.serialize_unchecked();
        let certificate = self
            .certificate
            .iter()
            .map(ProductionTransitionProof::serialize_unchecked)
            .collect::<Vec<_>>();
        let strong = self
            .strong
            .iter()
            .map(StrongTransitionCore::serialize_unchecked)
            .collect::<Vec<_>>();
        let base = self.base.serialize_unchecked();
        Self::assemble_serialized(carry, certificate, strong, base)
    }

    #[cfg(test)]
    fn serialize_redundantly_verified(&self) -> Vec<u8> {
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
        Self::assemble_serialized(carry, certificate, strong, base)
    }

    fn assemble_serialized(
        carry: Vec<u8>,
        certificate: Vec<Vec<u8>>,
        strong: Vec<Vec<u8>>,
        base: Vec<u8>,
    ) -> Vec<u8> {
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

    fn byte_breakdown(&self, serialized_len: usize) -> CanonicalProofByteBreakdown {
        let carry_total = self.carryopen.serialize_unchecked().len();
        let certificate_total = self
            .certificate
            .iter()
            .map(|proof| 4 + proof.serialize_unchecked().len())
            .sum::<usize>();
        let strong_total = self
            .strong
            .iter()
            .map(|proof| 4 + proof.serialize_unchecked().len())
            .sum::<usize>();
        let base_total = self.base.serialize_unchecked().len();
        self.byte_breakdown_with_component_totals(
            carry_total,
            certificate_total,
            strong_total,
            base_total,
            serialized_len,
        )
    }

    #[cfg(test)]
    fn byte_breakdown_redundantly_verified(&self) -> CanonicalProofByteBreakdown {
        let carry_total = self.carryopen.serialize().len();
        let certificate_total = self
            .certificate
            .iter()
            .map(|proof| 4 + proof.serialize().len())
            .sum::<usize>();
        let strong_total = self
            .strong
            .iter()
            .map(|proof| 4 + proof.serialize().len())
            .sum::<usize>();
        let base_total = self.base.serialize(self.strong.last().unwrap()).len();
        let serialized_len = self.serialize_redundantly_verified().len();
        self.byte_breakdown_with_component_totals(
            carry_total,
            certificate_total,
            strong_total,
            base_total,
            serialized_len,
        )
    }

    fn byte_breakdown_with_component_totals(
        &self,
        carry_total: usize,
        certificate_total: usize,
        strong_total: usize,
        base_total: usize,
        serialized_len: usize,
    ) -> CanonicalProofByteBreakdown {
        let total = 28 + carry_total + certificate_total + strong_total + base_total;

        let mut proof_row_merkle = 0;
        let mut index_row_merkle = 0;
        let mut selected_front_framing = 0;
        let mut proof_row_merkle_by_stage = [0_usize; 11];
        let mut index_row_merkle_by_stage = [0_usize; 11];
        let mut selected_front_framing_by_stage = [0_usize; 11];
        for (stage, front) in std::iter::once(&self.carryopen.selected_front)
            .chain(self.certificate.iter().map(|proof| &proof.selected_front))
            .chain(self.strong.iter().map(|proof| &proof.selected_front))
            .enumerate()
        {
            let (proof_rows, index_rows, framing) = front.byte_breakdown();
            proof_row_merkle += proof_rows;
            index_row_merkle += index_rows;
            selected_front_framing += framing;
            proof_row_merkle_by_stage[stage] = proof_rows;
            index_row_merkle_by_stage[stage] = index_rows;
            selected_front_framing_by_stage[stage] = framing;
        }

        let carry_algebra = self.carryopen.membership.serialize().len()
            + self.carryopen.evaluation.serialize().len()
            + self.carryopen.phi_link.serialize().len();
        let certificate_algebra = self
            .certificate
            .iter()
            .map(|proof| proof.algebra.serialize().len())
            .sum::<usize>();
        let strong_algebra = self
            .strong
            .iter()
            .map(|proof| proof.membership.serialize().len())
            .sum::<usize>();
        let algebra = carry_algebra + certificate_algebra + strong_algebra;
        let terminal_witness = self.base.private_f_rows.len() * 24;
        let terminal_pcs = 0;
        let named = proof_row_merkle
            + index_row_merkle
            + selected_front_framing
            + algebra
            + terminal_witness
            + terminal_pcs;
        let other = total
            .checked_sub(named)
            .expect("named canonical components must fit in total proof bytes");
        let breakdown = CanonicalProofByteBreakdown {
            total,
            carry_total,
            certificate_total,
            strong_total,
            base_total,
            proof_row_merkle,
            index_row_merkle,
            selected_front_framing,
            algebra,
            terminal_witness,
            terminal_pcs,
            other,
            proof_row_merkle_by_stage,
            index_row_merkle_by_stage,
            selected_front_framing_by_stage,
        };
        assert_eq!(breakdown.total, serialized_len);
        breakdown
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        let proof = Self::deserialize_unchecked(payload)?;
        proof.verify().then_some(proof)
    }

    /// Parse the complete canonical structure without accepting it. The public
    /// `deserialize` wrapper below returns only after one full semantic pass.
    fn deserialize_unchecked(payload: &[u8]) -> Option<Self> {
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
            certificate.push(ProductionTransitionProof::deserialize_unchecked(
                payload.get(position..position + size)?,
            )?);
            position += size;
        }
        let mut strong = Vec::with_capacity(values[2]);
        for _ in 0..values[2] {
            let size =
                u32::from_le_bytes(payload.get(position..position + 4)?.try_into().ok()?) as usize;
            position += 4;
            strong.push(StrongTransitionCore::deserialize_unchecked(
                payload.get(position..position + size)?,
            )?);
            position += size;
        }
        if payload.len() != position + values[3] {
            return None;
        }
        let base = StrongBaseProof::deserialize_unchecked(&payload[position..], strong.last()?)?;
        let proof = Self {
            carryopen,
            certificate,
            strong,
            base,
        };
        Some(proof)
    }

    #[cfg(test)]
    fn deserialize_redundantly_verified(payload: &[u8]) -> Option<Self> {
        let proof = Self::deserialize_unchecked(payload)?;
        if !proof.carryopen.precarry.verify()
            || !proof
                .certificate
                .iter()
                .all(ProductionTransitionProof::verify)
            || !proof.strong.iter().all(StrongTransitionCore::verify)
            || !proof.base.verify(proof.strong.last()?)
        {
            return None;
        }
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
    assert!(!args.final_only || (args.semantic && args.carryopen));
    let fixed_generator_preprocessing_start = Instant::now();
    if args.semantic {
        preprocess_canonical_fixed_generators();
    }
    let fixed_generator_preprocessing = fixed_generator_preprocessing_start.elapsed();
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
    // Final-only always enters the recursive aggregate. `direct_whir_tail` is
    // retained only for the non-final diagnostic joint-WHIR construction.
    let joint_mode = args.carryopen && args.semantic && (args.final_only || args.direct_whir_tail);
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
    let mut canonical_byte_breakdown = None;
    for _ in 0..args.iterations {
        let iteration_start = Instant::now();
        if args.carryopen {
            carryopen_measurements.push(run_production_carryopen(
                args.whir_verifier_repetitions,
                !args.final_only,
            ));
        }
        let measurement = if args.final_only {
            run_semantic_certificate_only(args.batch_lanes, certificate_direct_tail)
        } else if args.semantic && args.relation_whir {
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
        if !args.final_only {
            let sumcheck_payload = measurement.sumcheck_proof().serialize();
            let parsed_sumcheck = PackedSumcheckProof::deserialize(&sumcheck_payload)
                .expect("serialized combined sumcheck must verify");
            assert_eq!(parsed_sumcheck, measurement.sumcheck_proof());
            serialized_sumcheck_bytes = sumcheck_payload.len();
        }
        if let Some(proofs) = &measurement.transition_proofs {
            transition_proof_count = proofs.len();
            transition_proof_bytes = proofs.iter().map(|proof| proof.serialize().len()).sum();
            assert!(proofs.iter().all(|proof| {
                let payload = proof.serialize();
                ProductionTransitionProof::deserialize(&payload).as_ref() == Some(proof)
            }));
        }
        if !args.final_only {
            assert_eq!(measurement.pairs.len(), variables);
        }
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
                canonical_byte_breakdown = Some(proof.byte_breakdown(payload.len()));
                recursive_strong_prover_ms.push(prover_start.elapsed().as_secs_f64() * 1_000.0);
                final_total_prover_ms.push(iteration_start.elapsed().as_secs_f64() * 1_000.0);
                clear_virtual_index_factor_cache();
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
                canonical_byte_breakdown =
                    Some(recursive_proof.byte_breakdown(recursive_payload.len()));
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
        if !args.final_only {
            checksum +=
                measurement.terminal_left + measurement.terminal_right + measurement.terminal_claim;
        }
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
    if args.final_only {
        println!(
            "- reusable relation-local Field192 pair: {:.3} GiB capacity; diagnostic {:.3}-GiB global pair omitted",
            2.0 * LEVELS[0].qa_fields() as f64 * 24.0 / (1_u64 << 30) as f64,
            2.0 * args.fields as f64 * 24.0 / (1_u64 << 30) as f64,
        );
    } else {
        println!(
            "- two live Field192 vectors: {:.3} GiB",
            2.0 * args.fields as f64 * 24.0 / (1_u64 << 30) as f64
        );
    }
    println!(
        "- {} median/p95: {:.3}/{:.3} ms",
        if args.final_only {
            "certificate-state construction"
        } else {
            "witness-vector setup"
        },
        percentile(&setup_ms, 0.5),
        percentile(&setup_ms, 0.95)
    );
    if args.final_only {
        println!(
            "- diagnostic global 26-round sumcheck: skipped (not serialized in the canonical proof)"
        );
    } else {
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
    }
    if args.carryopen {
        let message_and_systematic_commitment_ms = carryopen_measurements
            .iter()
            .map(|item| item.message_and_systematic_commitment.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let precarry_ms = carryopen_measurements
            .iter()
            .map(|item| item.precarry.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let encode_ms = carryopen_measurements
            .iter()
            .map(|item| item.encode_and_commit.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let generator_setup_ms = carryopen_measurements
            .iter()
            .map(|item| item.generator_setup.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let vertical_plus_parity_commitment_ms = carryopen_measurements
            .iter()
            .map(|item| item.vertical_and_parity_commitment.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let front_and_selected_rows_ms = carryopen_measurements
            .iter()
            .map(|item| item.front_and_selected_rows.as_secs_f64() * 1_000.0)
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
        println!(
            "- production CarryOpen code: vertical and horizontal systematic QA rate 1/4, 2^16 x 2^8 message, 1024-field tensor rows, q={CARRYOPEN_QUERIES}"
        );
        println!(
            "- pre-Carry touches three quarters of the reserved 1.500-GiB codeword allocation: message, scratch/weight, and sumcheck copy; the fourth remains uninitialized"
        );
        println!(
            "- CarryOpen M+systematic-root/pre-Carry/encode+commit/algebra/terminal medians: {:.3}/{:.3}/{:.3}/{:.3}/{:.3} ms",
            percentile(&message_and_systematic_commitment_ms, 0.5),
            percentile(&precarry_ms, 0.5),
            percentile(&encode_ms, 0.5),
            percentile(&algebra_ms, 0.5),
            percentile(&terminal_ms, 0.5)
        );
        println!(
            "- CarryOpen encode+commit detail (setup/vertical+parity-root/front) medians: {:.3}/{:.3}/{:.3} ms",
            percentile(&generator_setup_ms, 0.5),
            percentile(&vertical_plus_parity_commitment_ms, 0.5),
            percentile(&front_and_selected_rows_ms, 0.5)
        );
        println!(
            "- CarryOpen terminal fields/padded: {}/{}; serialized component proof: {} B",
            CARRYOPEN_TERMINAL_FIELDS,
            CARRYOPEN_TERMINAL_FIELDS.next_power_of_two(),
            first.serialized_component_bytes()
        );
        println!("- CarryOpen verifier checks the actual M root, pre-Carry claim, QA membership/evaluation, authenticated F rows, public virtual-W rows, Phi link, and recursive terminal restoration; claim mutation rejected");
    }
    if args.semantic {
        println!(
            "- one-time canonical fixed-G preprocessing: {:.3} ms, {:.3} MiB (excluded from online proving and verification)",
            fixed_generator_preprocessing.as_secs_f64() * 1_000.0,
            canonical_fixed_generator_bytes() as f64 / (1_u64 << 20) as f64,
        );
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
            "- public virtual-W descriptors median/p95: {:.3}/{:.3} ms",
            percentile(&index_commitment_ms, 0.5),
            percentile(&index_commitment_ms, 0.95)
        );
        println!(
            "- authenticated F selected-row fronts median/p95: {:.3}/{:.3} ms",
            percentile(&source_opening_ms, 0.5),
            percentile(&source_opening_ms, 0.95)
        );
        println!(
            "- F-row front communication/frontier: {} B / {} hashes",
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
        println!("- all F/Phi challenges and row fronts are root-bound; W is a transcript-bound public rank-one oracle with no commitment or opening");
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
                let bytes = canonical_byte_breakdown
                    .expect("final-only canonical proof must have a byte breakdown");
                let strong_schedule = std::iter::once(STRONG_SOURCE_FIELDS)
                    .chain((0..STRONG_ROUNDS).map(|round| {
                        let level = strong_round_level(round).unwrap();
                        2 * level.next_blocks * level.width
                    }))
                    .map(|fields| fields.to_string())
                    .collect::<Vec<_>>()
                    .join(" -> ");
                println!("- final-only recursive strong schedule: {strong_schedule} fields");
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
                println!(
                    "- canonical proof component totals: CarryOpen={} B; certificate={} B; strong transitions={} B; terminal base={} B",
                    bytes.carry_total,
                    bytes.certificate_total,
                    bytes.strong_total,
                    bytes.base_total
                );
                println!(
                    "- canonical proof byte classes: F-row Merkle={} B; W-row Merkle={} B; selected-front framing={} B; algebra={} B; final witness={} B; terminal PCS={} B; other={} B",
                    bytes.proof_row_merkle,
                    bytes.index_row_merkle,
                    bytes.selected_front_framing,
                    bytes.algebra,
                    bytes.terminal_witness,
                    bytes.terminal_pcs,
                    bytes.other
                );
                for stage in 0..(1 + LEVELS.len() + STRONG_ROUNDS) {
                    let label = match stage {
                        0 => "CarryOpen".to_owned(),
                        1..=4 => format!("certificate-{}", stage - 1),
                        _ => format!("strong-{}", stage - 1 - LEVELS.len()),
                    };
                    println!(
                        "- selected-front {label}: F-row={} B; W-row={} B; framing={} B",
                        bytes.proof_row_merkle_by_stage[stage],
                        bytes.index_row_merkle_by_stage[stage],
                        bytes.selected_front_framing_by_stage[stage]
                    );
                }
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
                    "- recursive strong schedule: {} fields",
                    std::iter::once(STRONG_SOURCE_FIELDS)
                        .chain((0..STRONG_ROUNDS).map(|round| {
                            let level = strong_round_level(round).unwrap();
                            2 * level.next_blocks * level.width
                        }))
                        .map(|fields| fields.to_string())
                        .collect::<Vec<_>>()
                        .join(" -> ")
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
    fn slice_backed_sumcheck_matches_owned_vectors() {
        let left = (0..64)
            .map(|index| Field192::from((7 * index + 3) as u64))
            .collect::<Vec<_>>();
        let right = (0..64)
            .map(|index| Field192::from((11 * index + 5) as u64))
            .collect::<Vec<_>>();
        let claimed_sum = dot(&left, &right);
        let roots = vec![[19_u8; 32], [23_u8; 32]];
        let expected = prove_local_product_relation_with_claim(
            left.clone(),
            right.clone(),
            roots.clone(),
            claimed_sum,
        );
        let mut shared = left;
        shared.extend(right);
        let (slice_left, slice_right) = shared.split_at_mut(64);
        let actual = prove_local_product_relation_with_claim_slices(
            slice_left,
            slice_right,
            roots,
            claimed_sum,
        );
        assert_eq!(actual, expected);
        assert_eq!(actual.serialize(), expected.serialize());
    }

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
    fn tensor_sumcheck_matches_flat_transcript_and_exposes_ood_rows() {
        let rows = 8;
        let width = 4;
        let left = (0..rows * width).map(left_value).collect::<Vec<_>>();
        let right = (0..rows * width).map(right_value).collect::<Vec<_>>();
        let roots = local_relation_roots(b"tensor-flat-equivalence", 0, &[[7_u8; 32]]);
        let claim = dot(&left, &right);
        let flat = prove_local_product_relation_with_claim(
            left.clone(),
            right.clone(),
            roots.clone(),
            claim,
        );
        let tensor = prove_tensor_product_relation(left, right, rows, width, roots);
        assert_eq!(tensor.proof, flat);
        let point = flat.challenges().unwrap();
        let lane_point = &point[rows.trailing_zeros() as usize..];
        assert_eq!(
            evaluate_power_of_two_message(&tensor.left_ood_row, lane_point),
            flat.terminal_left
        );
        assert_eq!(
            evaluate_power_of_two_message(&tensor.right_ood_row, lane_point),
            flat.terminal_right
        );
    }

    #[test]
    fn rank_one_tensor_sumcheck_matches_materialized_transcript() {
        let rows = 8;
        let width = 4;
        let coefficients = (0..rows)
            .map(|index| Field192::from((3 * index + 1) as u64))
            .collect::<Vec<_>>();
        let alpha = (0..width)
            .map(|index| Field192::from((5 * index + 2) as u64))
            .collect::<Vec<_>>();
        let rank_one_matrix = coefficients
            .iter()
            .flat_map(|coefficient| alpha.iter().map(move |value| *coefficient * *value))
            .collect::<Vec<_>>();
        let arbitrary = (0..rows * width).map(right_value).collect::<Vec<_>>();
        let roots = local_relation_roots(b"rank-one-equivalence", 0, &[[9_u8; 32]]);
        let materialized_left = prove_tensor_product_relation(
            rank_one_matrix.clone(),
            arbitrary.clone(),
            rows,
            width,
            roots.clone(),
        );
        let rank_one_left = prove_rank_one_left_tensor_product_relation(
            coefficients.clone(),
            alpha.clone(),
            arbitrary.clone(),
            rows,
            width,
            roots.clone(),
        );
        assert_eq!(rank_one_left.proof, materialized_left.proof);
        assert_eq!(rank_one_left.left_ood_row, materialized_left.left_ood_row);
        assert_eq!(rank_one_left.right_ood_row, materialized_left.right_ood_row);

        let materialized_right = prove_tensor_product_relation(
            arbitrary.clone(),
            rank_one_matrix,
            rows,
            width,
            roots.clone(),
        );
        let rank_one_right = prove_rank_one_right_tensor_product_relation(
            arbitrary,
            coefficients,
            alpha,
            rows,
            width,
            roots,
        );
        assert_eq!(rank_one_right.proof, materialized_right.proof);
        assert_eq!(rank_one_right.left_ood_row, materialized_right.left_ood_row);
        assert_eq!(
            rank_one_right.right_ood_row,
            materialized_right.right_ood_row
        );
    }

    #[test]
    fn fused_dual_rank_one_sumchecks_match_independent_transcripts() {
        let rows = 8;
        let width = 4;
        let membership_coefficients = (0..rows)
            .map(|index| Field192::from((3 * index + 1) as u64))
            .collect::<Vec<_>>();
        let membership_alpha = (0..width)
            .map(|index| Field192::from((5 * index + 2) as u64))
            .collect::<Vec<_>>();
        let evaluation_coefficients = (0..rows)
            .map(|index| Field192::from((7 * index + 4) as u64))
            .collect::<Vec<_>>();
        let evaluation_alpha = (0..width)
            .map(|index| Field192::from((11 * index + 6) as u64))
            .collect::<Vec<_>>();
        let shared = (0..rows * width).map(right_value).collect::<Vec<_>>();
        let membership_roots = local_relation_roots(b"dual-membership", 0, &[[7_u8; 32]]);
        let evaluation_roots = local_relation_roots(b"dual-evaluation", 0, &[[8_u8; 32]]);
        let legacy_membership = prove_rank_one_left_tensor_product_relation(
            membership_coefficients.clone(),
            membership_alpha.clone(),
            shared.clone(),
            rows,
            width,
            membership_roots.clone(),
        );
        let legacy_evaluation = prove_rank_one_right_tensor_product_relation(
            shared.clone(),
            evaluation_coefficients.clone(),
            evaluation_alpha.clone(),
            rows,
            width,
            evaluation_roots.clone(),
        );
        let (fused_membership, fused_evaluation) = prove_dual_rank_one_tensor_product_relations(
            membership_coefficients,
            membership_alpha,
            shared,
            evaluation_coefficients,
            evaluation_alpha,
            rows,
            width,
            membership_roots,
            evaluation_roots,
            legacy_membership.proof.claimed_sum,
            legacy_evaluation.proof.claimed_sum,
        );
        assert_eq!(fused_membership.proof, legacy_membership.proof);
        assert_eq!(
            fused_membership.proof.serialize(),
            legacy_membership.proof.serialize()
        );
        assert_eq!(
            fused_membership.left_ood_row,
            legacy_membership.left_ood_row
        );
        assert_eq!(
            fused_membership.right_ood_row,
            legacy_membership.right_ood_row
        );
        assert_eq!(fused_evaluation.proof, legacy_evaluation.proof);
        assert_eq!(
            fused_evaluation.proof.serialize(),
            legacy_evaluation.proof.serialize()
        );
        assert_eq!(
            fused_evaluation.left_ood_row,
            legacy_evaluation.left_ood_row
        );
        assert_eq!(
            fused_evaluation.right_ood_row,
            legacy_evaluation.right_ood_row
        );
    }

    #[test]
    fn dual_in_place_row_fold_matches_two_independent_folds() {
        for rows in [2_usize, 4, 8, 16] {
            for width in [2_usize, 4, 8] {
                let original = (0..rows * width)
                    .map(|index| Field192::from((17 * index + rows + width) as u64))
                    .collect::<Vec<_>>();
                let first_challenge = Field192::from((3 * rows + width + 1) as u64);
                let second_challenge = Field192::from((rows + 5 * width + 2) as u64);
                let mut first = original.clone();
                let mut second = original.clone();
                let expected_rows = fold_tensor_rows(&mut first, rows, width, first_challenge);
                assert_eq!(
                    fold_tensor_rows(&mut second, rows, width, second_challenge),
                    expected_rows
                );

                let mut packed = original;
                assert_eq!(
                    fold_tensor_rows_dual_in_place(
                        &mut packed,
                        rows,
                        width,
                        first_challenge,
                        second_challenge,
                    ),
                    expected_rows
                );
                let half_fields = expected_rows * width;
                assert_eq!(&packed[..half_fields], &first[..half_fields]);
                assert_eq!(
                    &packed[half_fields..2 * half_fields],
                    &second[..half_fields]
                );
            }
        }
    }

    #[test]
    fn selected_row_successor_matches_full_codeword_derivation() {
        let level = Level {
            raw: 12,
            blocks: 1,
            block_semantic: 12,
            components: &[(12, 16)],
            group: 4,
            width: 3,
            row_span: 4,
            inverse_rate: 2,
            next_blocks: 3,
        };
        let proof_codeword = (0..level.qa_fields())
            .map(|index| Field192::from((7 * index + 1) as u64))
            .collect::<Vec<_>>();
        let index_oracle = (0..level.qa_fields())
            .map(|index| Field192::from((11 * index + 2) as u64))
            .collect::<Vec<_>>();
        let proof_ood = (0..level.width)
            .map(|index| Field192::from((13 * index + 3) as u64))
            .collect::<Vec<_>>();
        let index_ood = (0..level.width)
            .map(|index| Field192::from((17 * index + 4) as u64))
            .collect::<Vec<_>>();
        let selected = [1_usize, 6];
        let selected_proof = selected_row_values(&proof_codeword, level.width, &selected);
        let selected_index = selected_row_values(&index_oracle, level.width, &selected);
        assert_eq!(
            derive_next_source_from_selected_rows(
                level,
                &selected_proof,
                &selected_index,
                &proof_ood,
                &index_ood,
            ),
            derive_next_source(
                level,
                &proof_codeword,
                &index_oracle,
                &proof_ood,
                &index_ood,
                &selected,
            )
        );
    }

    #[test]
    fn cached_message_subtrees_preserve_horizontal_roots() {
        let zeros = zero_roots(12);
        let message = (0..4 * CARRYOPEN_WIDTH)
            .map(|index| Field192::from((7 * index + 3) as u64))
            .collect::<Vec<_>>();
        let (root, row_roots) = exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        assert_eq!(root, prefix_root(&message, message.len(), &zeros));
        assert_eq!(row_roots.len(), 4);

        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let row = &message[..CARRYOPEN_WIDTH];
        let mut legacy = row.to_vec();
        for spectrum in &spectra {
            let mut parity = row.to_vec();
            apply_encoder(&mut parity, spectrum);
            legacy.extend(parity);
        }
        let encoded = encode_horizontal_row(row, &spectra);
        assert_eq!(encoded, legacy);
        let mut transformed = vec![Field192::ZERO; CARRYOPEN_WIDTH];
        let mut scratch_encoded = vec![Field192::ZERO; CARRYOPEN_TENSOR_WIDTH];
        let mut digest_scratch = Vec::new();
        assert_eq!(
            horizontal_encoded_row_root_with_scratch(
                row,
                &spectra,
                &mut transformed,
                &mut scratch_encoded,
                &mut digest_scratch,
            ),
            prefix_root(&encoded, CARRYOPEN_TENSOR_WIDTH, &zeros)
        );
        assert_eq!(
            horizontal_row_root_with_systematic_subtree_scratch(
                row,
                row_roots[0],
                &spectra,
                &mut transformed,
                &mut scratch_encoded,
                &mut digest_scratch,
            ),
            prefix_root(&encoded, CARRYOPEN_TENSOR_WIDTH, &zeros)
        );
        assert_eq!(
            horizontal_row_root_with_systematic_subtree(row, row_roots[0], &spectra, &zeros,),
            prefix_root(&encoded, CARRYOPEN_TENSOR_WIDTH, &zeros)
        );
    }

    #[test]
    fn direct_codeword_and_index_initialization_match_zero_filled_paths() {
        let level = Level {
            raw: 32,
            blocks: 1,
            block_semantic: 32,
            components: &[(32, 32)],
            group: 8,
            width: 4,
            row_span: 8,
            inverse_rate: 4,
            next_blocks: 3,
        };
        let source = (0..level.raw)
            .map(|index| Field192::from((7 * index + 3) as u64))
            .collect::<Vec<_>>();
        let spectra = (0..level.inverse_rate - 1)
            .map(|block| generator_spectrum(21, block, level.group))
            .collect::<Vec<_>>();
        let mut initialized_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &source, &mut initialized_codeword, &spectra, 3);
        assert_eq!(
            encode_full_systematic_codeword(level, &source, &spectra, 3),
            initialized_codeword
        );

        let padded_level = Level {
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
        let padded_source = (0..padded_level.raw)
            .map(|index| Field192::from((11 * index + 5) as u64))
            .collect::<Vec<_>>();
        let padded_spectra = vec![generator_spectrum(22, 0, padded_level.group)];
        let mut initialized_padded = vec![Field192::ZERO; padded_level.qa_fields()];
        populate_proof_codeword(
            padded_level,
            &padded_source,
            &mut initialized_padded,
            &padded_spectra,
            2,
        );
        let mut direct_padded = vec![Field192::ONE; 7];
        overwrite_padded_proof_codeword(
            padded_level,
            &padded_source,
            &mut direct_padded,
            &padded_spectra,
            2,
        );
        assert_eq!(direct_padded, initialized_padded);

        let roots = (0..2 * level.inverse_rate)
            .map(|index| [index as u8; 32])
            .collect::<Vec<_>>();
        let mut initialized_index = vec![Field192::ZERO; level.qa_fields()];
        populate_index_oracle(21, level, &mut initialized_index, &spectra, &roots);
        assert_eq!(
            materialize_index_oracle(21, level, &spectra, &roots),
            initialized_index
        );
    }

    #[test]
    fn shared_forward_wht_matches_independent_parity_encoders() {
        let message = (0..64).map(left_value).collect::<Vec<_>>();
        let spectra = (0..3)
            .map(|block| generator_spectrum(17, block, message.len()))
            .collect::<Vec<_>>();
        let shared = encode_parity_blocks(message.clone(), &spectra);
        let independent = spectra
            .iter()
            .map(|spectrum| {
                let mut parity = message.clone();
                apply_encoder(&mut parity, spectrum);
                parity
            })
            .collect::<Vec<_>>();
        assert_eq!(shared, independent);
    }

    #[test]
    fn interleaved_wht_matches_iterator_wht() {
        let mut interleaved = (0..1024).map(left_value).collect::<Vec<_>>();
        let mut iterator = interleaved.clone();
        wht(&mut interleaved);
        wht_iterator(&mut iterator);
        assert_eq!(interleaved, iterator);
    }

    #[test]
    #[ignore = "full-column four-way interleaved WHT A/B benchmark"]
    fn interleaved_wht_benchmarks_iterator_wht() {
        let input = (0..CARRYOPEN_ROWS).map(left_value).collect::<Vec<_>>();
        let mut expected = input.clone();
        wht_iterator(&mut expected);
        let mut interleaved_ms = Vec::new();
        let mut iterator_ms = Vec::new();
        for trial in 0..12 {
            let run_interleaved = || {
                let mut values = input.clone();
                let start = Instant::now();
                wht(std::hint::black_box(&mut values));
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                assert_eq!(values, expected);
                elapsed
            };
            let run_iterator = || {
                let mut values = input.clone();
                let start = Instant::now();
                wht_iterator(std::hint::black_box(&mut values));
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                assert_eq!(values, expected);
                elapsed
            };
            if trial % 2 == 0 {
                interleaved_ms.push(run_interleaved());
                iterator_ms.push(run_iterator());
            } else {
                iterator_ms.push(run_iterator());
                interleaved_ms.push(run_interleaved());
            }
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        eprintln!(
            "interleaved-wht interleaved-ms={interleaved_ms:?} iterator-ms={iterator_ms:?} interleaved-median={:.3} iterator-median={:.3}",
            median(&interleaved_ms),
            median(&iterator_ms),
        );
    }

    #[test]
    #[ignore = "production-scale interleaved versus iterator WHT crossed A/B"]
    fn interleaved_wht_benchmarks_production_vertical() {
        let level = carryopen_level();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let mut codeword = Vec::with_capacity(level.qa_fields());
        codeword.extend((0..CARRYOPEN_FIELDS).map(precarry_message_value));
        codeword.resize(level.qa_fields(), Field192::ZERO);
        populate_parity_from_systematic(level, &mut codeword, &spectra, 64);
        let expected_codeword = codeword.clone();
        populate_parity_from_systematic_iterator_wht(level, &mut codeword, &spectra, 64);
        assert_eq!(codeword, expected_codeword);
        drop(expected_codeword);

        let sample_step = CARRYOPEN_WIDTH * CARRYOPEN_ROWS / 64;
        let mut interleaved_ms = Vec::new();
        let mut iterator_ms = Vec::new();
        let mut expected_samples = None;
        let mut run = |interleaved: bool| {
            let start = Instant::now();
            if interleaved {
                populate_parity_from_systematic(level, &mut codeword, &spectra, 64);
            } else {
                populate_parity_from_systematic_iterator_wht(level, &mut codeword, &spectra, 64);
            }
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            let samples = codeword
                .iter()
                .step_by(sample_step)
                .copied()
                .collect::<Vec<_>>();
            if let Some(expected) = &expected_samples {
                assert_eq!(&samples, expected);
            } else {
                expected_samples = Some(samples);
            }
            elapsed
        };
        for trial in 0..9 {
            if trial % 2 == 0 {
                interleaved_ms.push(run(true));
                iterator_ms.push(run(false));
                iterator_ms.push(run(false));
                interleaved_ms.push(run(true));
            } else {
                iterator_ms.push(run(false));
                interleaved_ms.push(run(true));
                interleaved_ms.push(run(true));
                iterator_ms.push(run(false));
            }
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        eprintln!(
            "production-interleaved-wht interleaved-ms={interleaved_ms:?} iterator-ms={iterator_ms:?} interleaved-median={:.3} iterator-median={:.3}",
            median(&interleaved_ms),
            median(&iterator_ms),
        );
    }

    #[test]
    fn blocked_parity_scatter_matches_row_scatter() {
        let rows = 37;
        let width = 13;
        let lane_start = 3;
        let lanes = 7;
        let parity_blocks = 3;
        let encoded = (0..lanes)
            .map(|lane| {
                (0..parity_blocks)
                    .map(|parity| {
                        (0..rows)
                            .map(|row| Field192::from((1 + lane + 17 * parity + 101 * row) as u64))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for parity_block in 0..parity_blocks {
            let mut expected = vec![Field192::ZERO; rows * width];
            for row in 0..rows {
                for (offset, lane) in encoded.iter().enumerate() {
                    expected[row * width + lane_start + offset] = lane[parity_block][row];
                }
            }
            let mut blocked = vec![Field192::ZERO; rows * width];
            scatter_parity_block_blocked(
                &mut blocked,
                width,
                lane_start,
                &encoded,
                parity_block,
                8,
                3,
            );
            assert_eq!(blocked, expected);
        }
    }

    #[test]
    fn fixed_generator_preprocessing_matches_direct_spectra() {
        assert_eq!(canonical_fixed_generator_fields(), 252_928);
        assert_eq!(canonical_fixed_generator_root_count(), 150);
        assert_eq!(canonical_fixed_generator_bytes(), 6_075_072);
        let zeros = zero_roots(30);
        for (level_index, level) in LEVELS
            .iter()
            .copied()
            .enumerate()
            .chain(std::iter::once((10, carryopen_level())))
            .chain((0..STRONG_ROUNDS).map(|round| {
                (
                    12 + round,
                    strong_round_level(round).expect("strong round must have a level"),
                )
            }))
        {
            let cached = fixed_generator_spectra(level_index, level);
            let cached_roots = fixed_generator_roots(level_index, level);
            assert_eq!(cached.len(), level.inverse_rate - 1);
            assert_eq!(cached_roots.len(), cached.len());
            for (block, spectrum) in cached.iter().enumerate() {
                assert_eq!(
                    spectrum,
                    &generator_spectrum(level_index, block, level.group),
                );
                assert_eq!(
                    cached_roots[block],
                    prefix_root(spectrum, level.group, &zeros)
                );
            }
        }
        let horizontal =
            fixed_generator_spectra_for_shape(11, CARRYOPEN_INVERSE_RATE, CARRYOPEN_WIDTH);
        assert_eq!(horizontal.len(), CARRYOPEN_INVERSE_RATE - 1);
        for (block, spectrum) in horizontal.iter().enumerate() {
            assert_eq!(spectrum, &generator_spectrum(11, block, CARRYOPEN_WIDTH));
        }
    }

    #[test]
    fn verifier_enforces_preprocessed_generator_roots() {
        let level_index = 12 + STRONG_ROUNDS - 1;
        let level = strong_round_level(STRONG_ROUNDS - 1).unwrap();
        let fixed_roots = fixed_generator_roots(level_index, level);
        let mut transcript_roots = fixed_roots.as_ref().clone();
        while transcript_roots.len() < 2 * level.inverse_rate {
            transcript_roots.push([transcript_roots.len() as u8 + 1; 32]);
        }
        let descriptor = virtual_index_descriptor(level_index, level, &transcript_roots);
        let mut component_roots = transcript_roots;
        component_roots.push(descriptor);
        assert!(validated_virtual_index_factors(level_index, level, &component_roots).is_some());

        component_roots[0][0] ^= 1;
        component_roots[2 * level.inverse_rate] = virtual_index_descriptor(
            level_index,
            level,
            &component_roots[..2 * level.inverse_rate],
        );
        assert!(validated_virtual_index_factors(level_index, level, &component_roots).is_none());
    }

    #[test]
    #[ignore = "canonical prover fixed-G cache versus online reconstruction benchmark"]
    fn fixed_generator_cache_benchmarks_online_reconstruction() {
        let levels = LEVELS
            .iter()
            .copied()
            .enumerate()
            .chain(std::iter::once((10, carryopen_level())))
            .chain((0..STRONG_ROUNDS).map(|round| {
                (
                    12 + round,
                    strong_round_level(round).expect("strong round must have a level"),
                )
            }))
            .collect::<Vec<_>>();
        let zeros = zero_roots(30);
        clear_fixed_generator_spectra_cache();
        preprocess_canonical_fixed_generators();

        let reconstruct = || {
            let data = levels
                .iter()
                .map(|(level_index, level)| {
                    let spectra = (0..level.inverse_rate - 1)
                        .map(|block| generator_spectrum(*level_index, block, level.group))
                        .collect::<Vec<_>>();
                    let roots = spectra
                        .iter()
                        .map(|spectrum| prefix_root(spectrum, level.group, &zeros))
                        .collect::<Vec<_>>();
                    (spectra, roots)
                })
                .collect::<Vec<_>>();
            let horizontal = (0..CARRYOPEN_INVERSE_RATE - 1)
                .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
                .collect::<Vec<_>>();
            (data, horizontal)
        };
        let cached = || {
            let data = levels
                .iter()
                .map(|(level_index, level)| {
                    (
                        fixed_generator_spectra(*level_index, *level),
                        fixed_generator_roots(*level_index, *level),
                    )
                })
                .collect::<Vec<_>>();
            let horizontal =
                fixed_generator_spectra_for_shape(11, CARRYOPEN_INVERSE_RATE, CARRYOPEN_WIDTH);
            (data, horizontal)
        };

        let expected = reconstruct();
        let retained = cached();
        for ((spectra, roots), (cached_spectra, cached_roots)) in
            expected.0.iter().zip(retained.0.iter())
        {
            assert_eq!(spectra, cached_spectra.as_ref());
            assert_eq!(roots, cached_roots.as_ref());
        }
        assert_eq!(&expected.1, retained.1.as_ref());

        let run = |use_cache: bool| {
            let start = Instant::now();
            let result = std::hint::black_box(if use_cache {
                let (data, horizontal) = cached();
                (data.len(), horizontal.len())
            } else {
                let (data, horizontal) = reconstruct();
                (data.len(), horizontal.len())
            });
            assert_eq!(result, (levels.len(), CARRYOPEN_INVERSE_RATE - 1));
            start.elapsed().as_secs_f64() * 1_000.0
        };
        let mut online = Vec::with_capacity(8);
        let mut preprocessed = Vec::with_capacity(8);
        for trial in 0..4 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            for use_cache in order {
                if use_cache {
                    preprocessed.push(run(true));
                } else {
                    online.push(run(false));
                }
            }
        }
        eprintln!("fixed-generator-prover online-ms={online:?} preprocessed-ms={preprocessed:?}");
    }

    #[test]
    fn equality_weight_prefix_matches_full_table() {
        let point = (0..16)
            .map(|index| fixed_challenge(b"equality-prefix-test", 0, index))
            .collect::<Vec<_>>();
        let full = equality_weights(&point);
        for length in [0, 1, 7, 8, 255, 256, 583, 823, 1369, 2438, 65536] {
            assert_eq!(equality_weights_prefix(&point, length), full[..length]);
        }
    }

    #[test]
    fn direct_equality_walsh_spectrum_matches_transform() {
        for variables in 0..=12 {
            let point = (0..variables)
                .map(|index| fixed_challenge(b"direct-equality-WHT", variables, index))
                .collect::<Vec<_>>();
            let scale = fixed_challenge(b"direct-equality-WHT-scale", variables, 0);
            let mut expected = equality_weights(&point);
            expected.iter_mut().for_each(|value| *value *= scale);
            wht(&mut expected);
            assert_eq!(scaled_equality_walsh_spectrum(&point, scale), expected);
        }
    }

    #[test]
    fn direct_first_evaluation_round_matches_copy_then_fold() {
        for variables in 1..=12 {
            let fields = 1usize << variables;
            let message = (0..fields)
                .map(|index| Field192::from((17 * index + 9) as u64))
                .collect::<Vec<_>>();
            let point = (0..variables)
                .map(|index| fixed_challenge(b"direct-first-evaluation-round", variables, index))
                .collect::<Vec<_>>();
            let mut copied = vec![Field192::ZERO; fields];
            let expected =
                evaluate_power_of_two_message_with_scratch(&message, &point, &mut copied);
            let mut direct = vec![Field192::ONE; fields];
            let actual =
                evaluate_power_of_two_message_direct_first_round(&message, &point, &mut direct);
            assert_eq!(actual, expected);
            assert_eq!(direct[0], copied[0]);
            let mut spare = vec![MaybeUninit::<Field192>::uninit(); fields];
            let spare_actual = evaluate_power_of_two_message_direct_first_round_in_spare(
                &message, &point, &mut spare,
            );
            assert_eq!(spare_actual, expected);
        }
    }

    #[test]
    fn block_local_equality_accumulation_matches_global_tables() {
        for variables in [10_usize, 11, 12] {
            let fields = 1usize << variables;
            let points = (0..5)
                .map(|opening| {
                    (0..variables)
                        .map(|coordinate| {
                            fixed_challenge(
                                b"block-local-equality-accumulation",
                                opening,
                                coordinate,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let coefficients = (0..points.len())
                .map(|index| Field192::from((13 * index + 7) as u64))
                .collect::<Vec<_>>();
            let mut scratch = vec![Field192::ZERO; fields];
            let mut expected = vec![Field192::ZERO; fields];
            for (point, coefficient) in points.iter().zip(&coefficients) {
                accumulate_scaled_equality_weights(
                    point,
                    *coefficient,
                    &mut scratch,
                    &mut expected,
                );
            }
            let mut block_local = vec![Field192::ZERO; fields];
            accumulate_equality_weights_block_local(&points, &coefficients, &mut block_local);
            assert_eq!(block_local, expected);
        }
    }

    #[test]
    fn symbolic_zero_relation_matches_materialized_sumcheck() {
        let transcript_roots = (0..9).map(|index| [index as u8; 32]).collect::<Vec<_>>();
        for fields in [2_usize, 3, 7, 8, 19, 64, 583, 823, 1_369, 2_438] {
            let variables = fields.next_power_of_two().trailing_zeros() as usize;
            let point = (0..variables)
                .map(|index| semantic_challenge(b"copy", 2, index, &transcript_roots))
                .collect::<Vec<_>>();
            let roots = local_relation_roots(b"dual-view-copy", 2, &transcript_roots);
            let materialized = prove_local_product_relation(
                vec![Field192::ZERO; fields],
                equality_weights_prefix(&point, fields),
                roots,
            );
            let symbolic = zero_weighted_relation_proof(
                b"copy",
                b"dual-view-copy",
                2,
                fields,
                &transcript_roots,
            );
            assert_eq!(symbolic, materialized);

            let challenges = symbolic.challenges().unwrap();
            let expected = dot(
                &equality_weights_prefix(&point, fields),
                &equality_weights_prefix(&challenges, fields),
            );
            assert_eq!(
                equality_prefix_inner_product(&point, &challenges, fields),
                expected,
            );
        }
    }

    #[test]
    #[ignore = "production CarryOpen Phi materialized-versus-symbolic A/B"]
    fn carryopen_phi_uses_symbolic_zero_relation() {
        let final_roots = (0..10).map(|index| [index as u8; 32]).collect::<Vec<_>>();
        let fields = CARRYOPEN_TERMINAL_FIELDS;
        let variables = fields.next_power_of_two().trailing_zeros() as usize;
        let point = (0..variables)
            .map(|index| semantic_challenge(b"CarryOpen-Phi-link", 10, index, &final_roots))
            .collect::<Vec<_>>();
        let expected = prove_local_product_relation(
            vec![Field192::ZERO; fields],
            equality_weights(&point)[..fields].to_vec(),
            local_relation_roots(b"CarryOpen-Phi-link", 10, &final_roots),
        );
        let run = |symbolic: bool| {
            let start = Instant::now();
            let proof = if symbolic {
                zero_weighted_relation_proof(
                    b"CarryOpen-Phi-link",
                    b"CarryOpen-Phi-link",
                    10,
                    fields,
                    &final_roots,
                )
            } else {
                prove_local_product_relation(
                    vec![Field192::ZERO; fields],
                    equality_weights(&point)[..fields].to_vec(),
                    local_relation_roots(b"CarryOpen-Phi-link", 10, &final_roots),
                )
            };
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            assert_eq!(proof, expected);
            elapsed
        };
        let mut materialized_ms = Vec::with_capacity(16);
        let mut symbolic_ms = Vec::with_capacity(16);
        for trial in 0..8 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            let mut trial_materialized = Vec::with_capacity(2);
            let mut trial_symbolic = Vec::with_capacity(2);
            for symbolic in order {
                let elapsed = run(symbolic);
                if symbolic {
                    symbolic_ms.push(elapsed);
                    trial_symbolic.push(elapsed);
                } else {
                    materialized_ms.push(elapsed);
                    trial_materialized.push(elapsed);
                }
            }
            eprintln!(
                "carryopen-phi trial={} materialized={:.3}/{:.3} ms symbolic={:.3}/{:.3} ms",
                trial + 1,
                trial_materialized[0],
                trial_materialized[1],
                trial_symbolic[0],
                trial_symbolic[1],
            );
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "carryopen-phi materialized-ms={materialized_ms:?} symbolic-ms={symbolic_ms:?} materialized-median={:.3} symbolic-median={:.3} materialized-mean={:.3} symbolic-mean={:.3}",
            median(&materialized_ms),
            median(&symbolic_ms),
            mean(&materialized_ms),
            mean(&symbolic_ms),
        );
    }

    #[test]
    #[ignore = "production CarryOpen parallel-versus-sequential factor benchmark"]
    fn parallel_parity_factors_benchmark_sequential_carryopen() {
        let level = carryopen_level();
        let spectra = fixed_generator_spectra(10, level);
        let roots = (0..2 * level.inverse_rate)
            .map(|index| {
                *blake3::hash(
                    &[
                        b"LiLAC/factor-benchmark/v1".as_slice(),
                        &index.to_le_bytes(),
                    ]
                    .concat(),
                )
                .as_bytes()
            })
            .collect::<Vec<_>>();
        let parallel = || {
            index_oracle_factors_outer_parallel_for_benchmark(10, level, spectra.as_ref(), &roots)
        };
        let sequential = || {
            index_oracle_factors_sequential_parity_for_benchmark(
                10,
                level,
                spectra.as_ref(),
                &roots,
            )
        };
        let expected = sequential();
        assert_eq!(parallel(), expected);

        let mut parallel_ms = Vec::with_capacity(8);
        let mut sequential_ms = Vec::with_capacity(8);
        for trial in 0..4 {
            let order = if trial % 2 == 0 {
                [true, false, false, true]
            } else {
                [false, true, true, false]
            };
            for is_parallel in order {
                let start = Instant::now();
                let factors = std::hint::black_box(if is_parallel {
                    parallel()
                } else {
                    sequential()
                });
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                assert_eq!(factors, expected);
                if is_parallel {
                    parallel_ms.push(elapsed);
                } else {
                    sequential_ms.push(elapsed);
                }
            }
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        eprintln!(
            "carryopen-index-factors parallel-ms={parallel_ms:?} sequential-ms={sequential_ms:?} parallel-median={:.3} sequential-median={:.3}",
            median(&parallel_ms),
            median(&sequential_ms),
        );
    }

    fn parallel_wht_factor_fixture() -> (Level, Arc<Vec<Vec<Field192>>>, Vec<Digest>) {
        let level = carryopen_level();
        let spectra = fixed_generator_spectra(10, level);
        let roots = (0..2 * level.inverse_rate)
            .map(|index| {
                *blake3::hash(
                    &[
                        b"LiLAC/parallel-WHT-factor-benchmark/v1".as_slice(),
                        &index.to_le_bytes(),
                    ]
                    .concat(),
                )
                .as_bytes()
            })
            .collect::<Vec<_>>();
        (level, spectra, roots)
    }

    #[test]
    fn parallel_wht_factors_match_outer_parallel() {
        let (level, spectra, roots) = parallel_wht_factor_fixture();
        let expected =
            index_oracle_factors_outer_parallel_for_benchmark(10, level, spectra.as_ref(), &roots);
        assert_eq!(
            index_oracle_factors(10, level, spectra.as_ref(), &roots),
            expected
        );
        assert_eq!(
            index_oracle_factors_with_wht_mode(10, level, spectra.as_ref(), &roots, true, true),
            expected
        );
    }

    #[test]
    #[ignore = "production CarryOpen nested-parallel WHT factor benchmark"]
    fn parallel_wht_factors_benchmark_outer_parallel() {
        let (level, spectra, roots) = parallel_wht_factor_fixture();
        let outer = || {
            index_oracle_factors_outer_parallel_for_benchmark(10, level, spectra.as_ref(), &roots)
        };
        let nested =
            || index_oracle_factors_with_wht_mode(10, level, spectra.as_ref(), &roots, true, false);
        let expected = outer();
        assert_eq!(nested(), expected);
        let mut outer_ms = Vec::with_capacity(8);
        let mut nested_ms = Vec::with_capacity(8);
        for trial in 0..4 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            for use_nested in order {
                let start = Instant::now();
                let factors = std::hint::black_box(if use_nested { nested() } else { outer() });
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                assert_eq!(factors, expected);
                if use_nested {
                    nested_ms.push(elapsed);
                } else {
                    outer_ms.push(elapsed);
                }
            }
        }
        eprintln!("carryopen-parallel-WHT-factors outer-ms={outer_ms:?} nested-ms={nested_ms:?}");
    }

    #[test]
    #[ignore = "production CarryOpen direct equality-spectrum factor benchmark"]
    fn direct_equality_spectrum_benchmarks_nested_forward_wht() {
        let (level, spectra, roots) = parallel_wht_factor_fixture();
        let nested =
            || index_oracle_factors_with_wht_mode(10, level, spectra.as_ref(), &roots, true, false);
        let direct =
            || index_oracle_factors_with_wht_mode(10, level, spectra.as_ref(), &roots, true, true);
        let expected = nested();
        assert_eq!(direct(), expected);
        let mut nested_ms = Vec::with_capacity(8);
        let mut direct_ms = Vec::with_capacity(8);
        for trial in 0..4 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            for use_direct in order {
                let start = Instant::now();
                let factors = std::hint::black_box(if use_direct { direct() } else { nested() });
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                assert_eq!(factors, expected);
                if use_direct {
                    direct_ms.push(elapsed);
                } else {
                    nested_ms.push(elapsed);
                }
            }
        }
        eprintln!(
            "carryopen-direct-equality-spectrum nested-ms={nested_ms:?} direct-ms={direct_ms:?}"
        );
    }

    #[test]
    #[ignore = "certificate and strong direct equality-spectrum threshold benchmark"]
    fn direct_equality_spectrum_benchmarks_smaller_levels() {
        let levels = LEVELS
            .iter()
            .copied()
            .enumerate()
            .chain((0..STRONG_ROUNDS).map(|round| {
                (
                    12 + round,
                    strong_round_level(round).expect("strong round must have a level"),
                )
            }));
        for (level_index, level) in levels {
            let spectra = fixed_generator_spectra(level_index, level);
            let roots = (0..2 * level.inverse_rate)
                .map(|index| {
                    *blake3::hash(
                        &[
                            b"LiLAC/direct-equality-small-level/v1".as_slice(),
                            &level_index.to_le_bytes(),
                            &index.to_le_bytes(),
                        ]
                        .concat(),
                    )
                    .as_bytes()
                })
                .collect::<Vec<_>>();
            let run = |direct: bool| {
                index_oracle_factors_with_wht_mode(
                    level_index,
                    level,
                    spectra.as_ref(),
                    &roots,
                    false,
                    direct,
                )
            };
            let expected = run(false);
            assert_eq!(run(true), expected);
            let mut wht_ms = Vec::with_capacity(8);
            let mut direct_ms = Vec::with_capacity(8);
            for trial in 0..4 {
                let order = if trial % 2 == 0 {
                    [false, true, true, false]
                } else {
                    [true, false, false, true]
                };
                for direct in order {
                    let start = Instant::now();
                    assert_eq!(std::hint::black_box(run(direct)), expected);
                    let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                    if direct {
                        direct_ms.push(elapsed);
                    } else {
                        wht_ms.push(elapsed);
                    }
                }
            }
            eprintln!(
                "direct-equality-small-level level={level_index} group={} spectra={} wht-ms={wht_ms:?} direct-ms={direct_ms:?}",
                level.group,
                spectra.len()
            );
        }
    }

    #[test]
    #[ignore = "small certificate-level nested-parallel WHT threshold benchmark"]
    fn parallel_wht_factors_benchmark_certificate_threshold() {
        for (level_index, level) in LEVELS.iter().copied().enumerate() {
            let spectra = fixed_generator_spectra(level_index, level);
            let roots = (0..2 * level.inverse_rate)
                .map(|index| {
                    *blake3::hash(
                        &[
                            b"LiLAC/certificate-WHT-threshold/v1".as_slice(),
                            &level_index.to_le_bytes(),
                            &index.to_le_bytes(),
                        ]
                        .concat(),
                    )
                    .as_bytes()
                })
                .collect::<Vec<_>>();
            let run = |nested: bool| {
                index_oracle_factors_with_wht_mode(
                    level_index,
                    level,
                    spectra.as_ref(),
                    &roots,
                    nested,
                    false,
                )
            };
            let expected = run(false);
            assert_eq!(run(true), expected);
            let mut outer_ms = Vec::with_capacity(8);
            let mut nested_ms = Vec::with_capacity(8);
            for trial in 0..4 {
                let order = if trial % 2 == 0 {
                    [false, true, true, false]
                } else {
                    [true, false, false, true]
                };
                for nested in order {
                    let start = Instant::now();
                    assert_eq!(std::hint::black_box(run(nested)), expected);
                    let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                    if nested {
                        nested_ms.push(elapsed);
                    } else {
                        outer_ms.push(elapsed);
                    }
                }
            }
            eprintln!(
                "certificate-parallel-WHT level={level_index} group={} spectra={} outer-ms={outer_ms:?} nested-ms={nested_ms:?}",
                level.group,
                spectra.len()
            );
        }
    }

    #[test]
    #[ignore = "production-scale vertical-encoding A/B benchmark"]
    fn shared_forward_wht_benchmarks_independent_vertical_encoder() {
        let level = carryopen_level();
        let source = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let mut codeword = vec![Field192::ZERO; level.qa_fields()];

        let start = Instant::now();
        populate_proof_codeword(level, &source, &mut codeword, &spectra, 64);
        let shared_first = start.elapsed();
        let sample_step = CARRYOPEN_WIDTH * CARRYOPEN_ROWS / 64;
        let shared_samples = codeword
            .iter()
            .step_by(sample_step)
            .copied()
            .collect::<Vec<_>>();
        let start = Instant::now();
        populate_proof_codeword_independent_wht(level, &source, &mut codeword, &spectra, 64);
        let independent_first = start.elapsed();
        assert_eq!(
            codeword
                .iter()
                .step_by(sample_step)
                .copied()
                .collect::<Vec<_>>(),
            shared_samples
        );
        let start = Instant::now();
        populate_proof_codeword_independent_wht(level, &source, &mut codeword, &spectra, 64);
        let independent_second = start.elapsed();
        let start = Instant::now();
        populate_proof_codeword(level, &source, &mut codeword, &spectra, 64);
        let shared_second = start.elapsed();
        let checksum = codeword
            .iter()
            .step_by(sample_step)
            .copied()
            .sum::<Field192>();
        assert_ne!(checksum, Field192::ZERO);
        eprintln!(
            "shared={:.3}/{:.3} ms independent={:.3}/{:.3} ms",
            shared_first.as_secs_f64() * 1_000.0,
            shared_second.as_secs_f64() * 1_000.0,
            independent_first.as_secs_f64() * 1_000.0,
            independent_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "full-column last-parity forward-buffer reuse A/B benchmark"]
    fn last_parity_reuses_forward_buffer() {
        let group = CARRYOPEN_ROWS;
        let message = (0..group).map(left_value).collect::<Vec<_>>();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, group))
            .collect::<Vec<_>>();
        let expected = encode_parity_blocks_cloned_forward(message.clone(), &spectra);
        assert_eq!(encode_parity_blocks(message.clone(), &spectra), expected);

        let mut reused_ms = Vec::new();
        let mut cloned_ms = Vec::new();
        for trial in 0..10 {
            let run_reused = || {
                let start = Instant::now();
                let output = std::hint::black_box(encode_parity_blocks(
                    std::hint::black_box(message.clone()),
                    std::hint::black_box(&spectra),
                ));
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                assert_eq!(output, expected);
                elapsed
            };
            let run_cloned = || {
                let start = Instant::now();
                let output = std::hint::black_box(encode_parity_blocks_cloned_forward(
                    std::hint::black_box(message.clone()),
                    std::hint::black_box(&spectra),
                ));
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                assert_eq!(output, expected);
                elapsed
            };
            if trial % 2 == 0 {
                reused_ms.push(run_reused());
                cloned_ms.push(run_cloned());
            } else {
                cloned_ms.push(run_cloned());
                reused_ms.push(run_reused());
            }
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        eprintln!(
            "last-parity-reuse reused-ms={reused_ms:?} cloned-ms={cloned_ms:?} reused-median={:.3} cloned-median={:.3}",
            median(&reused_ms),
            median(&cloned_ms),
        );
    }

    #[test]
    #[ignore = "production-scale last-parity buffer-reuse A/B benchmark"]
    fn last_parity_reuse_benchmarks_production_vertical_encoder() {
        let level = carryopen_level();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let mut codeword = Vec::with_capacity(level.qa_fields());
        codeword.extend((0..CARRYOPEN_FIELDS).map(precarry_message_value));
        codeword.resize(level.qa_fields(), Field192::ZERO);
        let sample_step = CARRYOPEN_WIDTH * CARRYOPEN_ROWS / 64;
        let mut reused_ms = Vec::new();
        let mut cloned_ms = Vec::new();
        let mut expected_samples = None;
        for trial in 0..4 {
            let run_reused = |codeword: &mut [Field192]| {
                let start = Instant::now();
                populate_parity_from_systematic(level, codeword, &spectra, 64);
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                let samples = codeword
                    .iter()
                    .step_by(sample_step)
                    .copied()
                    .collect::<Vec<_>>();
                (elapsed, samples)
            };
            let run_cloned = |codeword: &mut [Field192]| {
                let start = Instant::now();
                populate_parity_from_systematic_cloned_forward(level, codeword, &spectra, 64);
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                let samples = codeword
                    .iter()
                    .step_by(sample_step)
                    .copied()
                    .collect::<Vec<_>>();
                (elapsed, samples)
            };
            let ((reused, reused_samples), (cloned, cloned_samples)) = if trial % 2 == 0 {
                (run_reused(&mut codeword), run_cloned(&mut codeword))
            } else {
                let cloned = run_cloned(&mut codeword);
                let reused = run_reused(&mut codeword);
                (reused, cloned)
            };
            assert_eq!(reused_samples, cloned_samples);
            if let Some(expected) = &expected_samples {
                assert_eq!(&reused_samples, expected);
            } else {
                expected_samples = Some(reused_samples);
            }
            reused_ms.push(reused);
            cloned_ms.push(cloned);
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        eprintln!(
            "production-last-parity-reuse reused-ms={reused_ms:?} cloned-ms={cloned_ms:?} reused-median={:.3} cloned-median={:.3}",
            median(&reused_ms),
            median(&cloned_ms),
        );
    }

    #[test]
    #[ignore = "production-scale vertical lane-batch sweep"]
    fn vertical_lane_batch_sweep() {
        let level = carryopen_level();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let mut codeword = Vec::with_capacity(level.qa_fields());
        codeword.extend((0..CARRYOPEN_FIELDS).map(precarry_message_value));
        codeword.resize(level.qa_fields(), Field192::ZERO);
        let sample_step = CARRYOPEN_WIDTH * CARRYOPEN_ROWS / 64;
        let batch_sizes = [8_usize, 10, 12, 16, 20, 24, 32, 48, 64, 96];
        let mut measurements = BTreeMap::<usize, Vec<f64>>::new();
        let mut expected_samples = None;
        for order in [
            batch_sizes.to_vec(),
            batch_sizes.into_iter().rev().collect(),
        ] {
            for batch_lanes in order {
                let start = Instant::now();
                populate_parity_from_systematic(level, &mut codeword, &spectra, batch_lanes);
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                let samples = codeword
                    .iter()
                    .step_by(sample_step)
                    .copied()
                    .collect::<Vec<_>>();
                if let Some(expected) = &expected_samples {
                    assert_eq!(&samples, expected);
                } else {
                    expected_samples = Some(samples);
                }
                measurements.entry(batch_lanes).or_default().push(elapsed);
            }
        }
        for (batch_lanes, samples) in measurements {
            let mut sorted = samples.clone();
            sorted.sort_by(f64::total_cmp);
            let median = (sorted[0] + sorted[1]) / 2.0;
            let live_mib = batch_lanes
                * (CARRYOPEN_INVERSE_RATE - 1)
                * CARRYOPEN_ROWS
                * std::mem::size_of::<Field192>();
            eprintln!(
                "vertical-lane-batch lanes={batch_lanes} live-scratch-mib={:.1} samples-ms={samples:?} midpoint-ms={median:.3}",
                live_mib as f64 / (1_u64 << 20) as f64,
            );
        }
    }

    #[test]
    #[ignore = "production-scale blocked vertical scatter sweep"]
    fn blocked_vertical_scatter_sweep() {
        let level = carryopen_level();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let mut codeword = Vec::with_capacity(level.qa_fields());
        codeword.extend((0..CARRYOPEN_FIELDS).map(precarry_message_value));
        codeword.resize(level.qa_fields(), Field192::ZERO);
        let sample_step = CARRYOPEN_WIDTH * CARRYOPEN_ROWS / 64;
        let tiles = [
            (0_usize, 0_usize),
            (64, 4),
            (64, 8),
            (256, 4),
            (256, 8),
            (256, 16),
            (1024, 8),
        ];
        let mut measurements = BTreeMap::<(usize, usize), Vec<f64>>::new();
        let mut expected_samples = None;
        for order in [tiles.to_vec(), tiles.into_iter().rev().collect()] {
            for tile in order {
                let scatter_tile = (tile != (0, 0)).then_some(tile);
                let start = Instant::now();
                populate_parity_from_systematic_with_scatter_tile(
                    level,
                    &mut codeword,
                    &spectra,
                    64,
                    scatter_tile,
                );
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                let samples = codeword
                    .iter()
                    .step_by(sample_step)
                    .copied()
                    .collect::<Vec<_>>();
                if let Some(expected) = &expected_samples {
                    assert_eq!(&samples, expected);
                } else {
                    expected_samples = Some(samples);
                }
                measurements.entry(tile).or_default().push(elapsed);
            }
        }
        for (tile, samples) in measurements {
            let mut sorted = samples.clone();
            sorted.sort_by(f64::total_cmp);
            let midpoint = (sorted[0] + sorted[1]) / 2.0;
            eprintln!(
                "vertical-scatter-tile rows={} lanes={} samples-ms={samples:?} midpoint-ms={midpoint:.3}",
                tile.0, tile.1,
            );
        }
    }

    #[test]
    #[ignore = "production-scale 64-row four-lane blocked scatter crossed A/B"]
    fn blocked_vertical_scatter_benchmarks_row_scatter() {
        let level = carryopen_level();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let mut codeword = Vec::with_capacity(level.qa_fields());
        codeword.extend((0..CARRYOPEN_FIELDS).map(precarry_message_value));
        codeword.resize(level.qa_fields(), Field192::ZERO);
        let sample_step = CARRYOPEN_WIDTH * CARRYOPEN_ROWS / 64;
        let mut blocked_ms = Vec::new();
        let mut rows_ms = Vec::new();
        let mut expected_samples = None;
        populate_parity_from_systematic_with_scatter_tile(
            level,
            &mut codeword,
            &spectra,
            64,
            Some((64, 4)),
        );
        let expected_codeword = codeword.clone();
        populate_parity_from_systematic_with_scatter_tile(level, &mut codeword, &spectra, 64, None);
        assert_eq!(codeword, expected_codeword);
        drop(expected_codeword);
        let mut run = |scatter_tile: Option<(usize, usize)>| {
            let start = Instant::now();
            populate_parity_from_systematic_with_scatter_tile(
                level,
                &mut codeword,
                &spectra,
                64,
                scatter_tile,
            );
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            let samples = codeword
                .iter()
                .step_by(sample_step)
                .copied()
                .collect::<Vec<_>>();
            if let Some(expected) = &expected_samples {
                assert_eq!(&samples, expected);
            } else {
                expected_samples = Some(samples);
            }
            elapsed
        };
        for trial in 0..9 {
            if trial % 2 == 0 {
                blocked_ms.push(run(Some((64, 4))));
                rows_ms.push(run(None));
                rows_ms.push(run(None));
                blocked_ms.push(run(Some((64, 4))));
            } else {
                rows_ms.push(run(None));
                blocked_ms.push(run(Some((64, 4))));
                blocked_ms.push(run(Some((64, 4))));
                rows_ms.push(run(None));
            }
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let trim_two = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[2..sorted.len() - 2].iter().sum::<f64>() / (sorted.len() - 4) as f64
        };
        eprintln!(
            "blocked-vertical-scatter blocked-ms={blocked_ms:?} rows-ms={rows_ms:?} blocked-median={:.3} rows-median={:.3} blocked-trim2={:.3} rows-trim2={:.3}",
            median(&blocked_ms),
            median(&rows_ms),
            trim_two(&blocked_ms),
            trim_two(&rows_ms),
        );
    }

    #[test]
    #[ignore = "production-scale preallocated systematic-prefix versus separate-message benchmark"]
    fn preallocated_systematic_prefix_benchmarks_separate_message() {
        let level = carryopen_level();
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let run = |preallocated: bool| {
            if preallocated {
                let mut codeword = Vec::with_capacity(level.qa_fields());
                codeword.extend_from_slice(&message);
                let start = Instant::now();
                codeword.resize(level.qa_fields(), Field192::ZERO);
                populate_parity_from_systematic(level, &mut codeword, &spectra, 64);
                (codeword, start.elapsed())
            } else {
                let start = Instant::now();
                let mut codeword = vec![Field192::ZERO; level.qa_fields()];
                populate_proof_codeword(level, &message, &mut codeword, &spectra, 64);
                (codeword, start.elapsed())
            }
        };

        let (preallocated_first, preallocated_first_time) = run(true);
        let (separate_first, separate_first_time) = run(false);
        assert_eq!(preallocated_first, separate_first);
        drop(preallocated_first);
        drop(separate_first);
        let (separate_second, separate_second_time) = run(false);
        let (preallocated_second, preallocated_second_time) = run(true);
        assert_eq!(preallocated_second, separate_second);
        eprintln!(
            "preallocated-systematic-prefix={:.3}/{:.3} ms separate-message={:.3}/{:.3} ms",
            preallocated_first_time.as_secs_f64() * 1_000.0,
            preallocated_second_time.as_secs_f64() * 1_000.0,
            separate_first_time.as_secs_f64() * 1_000.0,
            separate_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale fused message and systematic tensor commitment A/B"]
    fn message_rows_fuse_systematic_tensor_commitment() {
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let run = |fused: bool| {
            let start = Instant::now();
            let result = if fused {
                message_and_systematic_tensor_commitment(&message, &horizontal_spectra, &zeros)
            } else {
                let (message_root, message_row_roots) =
                    exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
                let commitment = systematic_tensor_row_commitment(
                    &message,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                );
                (message_root, message_row_roots, commitment)
            };
            (result, start.elapsed())
        };
        let retained_first = run(false);
        let fused_first = run(true);
        let fused_second = run(true);
        let retained_second = run(false);
        for candidate in [&fused_first.0, &fused_second.0, &retained_second.0] {
            assert_eq!(candidate.0, retained_first.0 .0);
            assert_eq!(candidate.1, retained_first.0 .1);
            assert_eq!(candidate.2.root, retained_first.0 .2.root);
            assert_eq!(candidate.2.row_roots, retained_first.0 .2.row_roots);
            assert_eq!(candidate.2.row_domain, retained_first.0 .2.row_domain);
            assert_eq!(candidate.2.zero_row_root, retained_first.0 .2.zero_row_root);
        }
        eprintln!(
            "message-systematic-commit retained={:.3}/{:.3} ms fused={:.3}/{:.3} ms",
            retained_first.1.as_secs_f64() * 1_000.0,
            retained_second.1.as_secs_f64() * 1_000.0,
            fused_first.1.as_secs_f64() * 1_000.0,
            fused_second.1.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale final vertical scatter plus horizontal commitment A/B"]
    fn final_vertical_scatter_fuses_parity_commitments() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = Vec::with_capacity(level.qa_fields());
        proof_codeword.extend_from_slice(&message);
        proof_codeword.resize(level.qa_fields(), Field192::ZERO);

        let run = |fused: bool, proof_codeword: &mut [Field192]| {
            let start = Instant::now();
            let commitments = if fused {
                let parity = populate_parity_and_commit_from_systematic(
                    level,
                    proof_codeword,
                    &spectra,
                    &horizontal_spectra,
                    64,
                    &zeros,
                );
                let systematic = systematic_tensor_row_commitment(
                    proof_codeword,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                );
                std::iter::once(systematic)
                    .chain(parity)
                    .collect::<Vec<_>>()
            } else {
                populate_parity_from_systematic(level, proof_codeword, &spectra, 64);
                tensor_row_commitments(
                    proof_codeword,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                )
            };
            (commitments, start.elapsed())
        };
        let retained_first = run(false, &mut proof_codeword);
        let fused_first = run(true, &mut proof_codeword);
        let fused_second = run(true, &mut proof_codeword);
        let retained_second = run(false, &mut proof_codeword);
        let assert_same = |candidate: &[MatrixCommitment], expected: &[MatrixCommitment]| {
            assert_eq!(candidate.len(), expected.len());
            for (candidate, expected) in candidate.iter().zip(expected) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
                assert_eq!(candidate.row_domain, expected.row_domain);
                assert_eq!(candidate.zero_row_root, expected.zero_row_root);
            }
        };
        assert_same(&retained_first.0, &fused_first.0);
        assert_same(&retained_first.0, &fused_second.0);
        assert_same(&retained_first.0, &retained_second.0);
        eprintln!(
            "final-scatter-commit retained={:.3}/{:.3} ms fused={:.3}/{:.3} ms",
            retained_first.1.as_secs_f64() * 1_000.0,
            retained_second.1.as_secs_f64() * 1_000.0,
            fused_first.1.as_secs_f64() * 1_000.0,
            fused_second.1.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale zero-fill versus spare-capacity parity A/B"]
    fn parity_spare_capacity_avoids_zero_fill() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut initialized = Vec::with_capacity(level.qa_fields());
        initialized.extend_from_slice(&message);
        let mut spare = Vec::with_capacity(level.qa_fields());
        spare.extend_from_slice(&message);
        let run_initialized = |codeword: &mut Vec<Field192>| {
            codeword.truncate(CARRYOPEN_FIELDS);
            let start = Instant::now();
            codeword.resize(level.qa_fields(), Field192::ZERO);
            let commitments = populate_parity_and_commit_from_systematic(
                level,
                codeword,
                &spectra,
                &horizontal_spectra,
                64,
                &zeros,
            );
            (commitments, start.elapsed().as_secs_f64() * 1_000.0)
        };
        let run_spare = |codeword: &mut Vec<Field192>| {
            codeword.truncate(CARRYOPEN_FIELDS);
            let start = Instant::now();
            let commitments = populate_parity_and_commit_in_spare_capacity(
                level,
                codeword,
                &spectra,
                &horizontal_spectra,
                64,
                &zeros,
            );
            (commitments, start.elapsed().as_secs_f64() * 1_000.0)
        };
        let assert_same = |candidate: &[MatrixCommitment], expected: &[MatrixCommitment]| {
            assert_eq!(candidate.len(), expected.len());
            for (candidate, expected) in candidate.iter().zip(expected) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
                assert_eq!(candidate.row_domain, expected.row_domain);
                assert_eq!(candidate.zero_row_root, expected.zero_row_root);
            }
        };
        let mut initialized_ms = Vec::with_capacity(16);
        let mut spare_ms = Vec::with_capacity(16);
        for trial in 0..8 {
            let (initialized_first, spare_first, spare_second, initialized_second) =
                if trial % 2 == 0 {
                    (
                        run_initialized(&mut initialized),
                        run_spare(&mut spare),
                        run_spare(&mut spare),
                        run_initialized(&mut initialized),
                    )
                } else {
                    let spare_first = run_spare(&mut spare);
                    let initialized_first = run_initialized(&mut initialized);
                    let initialized_second = run_initialized(&mut initialized);
                    let spare_second = run_spare(&mut spare);
                    (
                        initialized_first,
                        spare_first,
                        spare_second,
                        initialized_second,
                    )
                };
            assert_same(&spare_first.0, &initialized_first.0);
            assert_same(&spare_second.0, &initialized_first.0);
            assert_same(&initialized_second.0, &initialized_first.0);
            assert_eq!(spare, initialized);
            initialized_ms.extend([initialized_first.1, initialized_second.1]);
            spare_ms.extend([spare_first.1, spare_second.1]);
            eprintln!(
                "parity-spare trial={} initialized={:.3}/{:.3} ms spare={:.3}/{:.3} ms",
                trial + 1,
                initialized_first.1,
                initialized_second.1,
                spare_first.1,
                spare_second.1,
            );
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "parity-spare initialized-ms={initialized_ms:?} spare-ms={spare_ms:?} initialized-median={:.3} spare-median={:.3} initialized-mean={:.3} spare-mean={:.3}",
            median(&initialized_ms),
            median(&spare_ms),
            mean(&initialized_ms),
            mean(&spare_ms),
        );
    }

    #[test]
    #[ignore = "production six-round strong codeword/index initialization A/B"]
    fn strong_schedule_avoids_zero_filled_codeword_and_index() {
        let inputs = (0..STRONG_ROUNDS)
            .map(|round| {
                let level = strong_round_level(round).unwrap();
                let relation_level = 12 + round;
                let source = (0..level.raw)
                    .into_par_iter()
                    .map(|index| semantic_value(relation_level, index))
                    .collect::<Vec<_>>();
                let spectra = (0..level.inverse_rate - 1)
                    .map(|block| generator_spectrum(relation_level, block, level.group))
                    .collect::<Vec<_>>();
                let roots = (0..2 * level.inverse_rate)
                    .map(|index| {
                        *blake3::hash(
                            &[
                                b"LiLAC/strong-initialization-benchmark/v1".as_slice(),
                                &round.to_le_bytes(),
                                &index.to_le_bytes(),
                            ]
                            .concat(),
                        )
                        .as_bytes()
                    })
                    .collect::<Vec<_>>();
                (level, relation_level, source, spectra, roots)
            })
            .collect::<Vec<_>>();
        let run = |direct: bool| {
            let start = Instant::now();
            let output = inputs
                .iter()
                .map(|(level, relation_level, source, spectra, roots)| {
                    let codeword = if direct {
                        encode_full_systematic_codeword(*level, source, spectra, 64)
                    } else {
                        let mut codeword = vec![Field192::ZERO; level.qa_fields()];
                        populate_proof_codeword(*level, source, &mut codeword, spectra, 64);
                        codeword
                    };
                    let index_oracle = if direct {
                        materialize_index_oracle(*relation_level, *level, spectra, roots)
                    } else {
                        let mut index_oracle = vec![Field192::ZERO; level.qa_fields()];
                        populate_index_oracle(
                            *relation_level,
                            *level,
                            &mut index_oracle,
                            spectra,
                            roots,
                        );
                        index_oracle
                    };
                    (codeword, index_oracle)
                })
                .collect::<Vec<_>>();
            (output, start.elapsed().as_secs_f64() * 1_000.0)
        };
        let mut initialized_ms = Vec::with_capacity(16);
        let mut direct_ms = Vec::with_capacity(16);
        for trial in 0..8 {
            let orders = if trial % 2 == 0 {
                [[false, true], [true, false]]
            } else {
                [[true, false], [false, true]]
            };
            let mut trial_initialized = Vec::with_capacity(2);
            let mut trial_direct = Vec::with_capacity(2);
            for pair in orders {
                let first = run(pair[0]);
                let second = run(pair[1]);
                assert_eq!(first.0, second.0);
                for (direct, elapsed) in [(pair[0], first.1), (pair[1], second.1)] {
                    if direct {
                        direct_ms.push(elapsed);
                        trial_direct.push(elapsed);
                    } else {
                        initialized_ms.push(elapsed);
                        trial_initialized.push(elapsed);
                    }
                }
            }
            eprintln!(
                "strong-initialization trial={} initialized={:.3}/{:.3} ms direct={:.3}/{:.3} ms",
                trial + 1,
                trial_initialized[0],
                trial_initialized[1],
                trial_direct[0],
                trial_direct[1],
            );
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "strong-initialization initialized-ms={initialized_ms:?} direct-ms={direct_ms:?} initialized-median={:.3} direct-median={:.3} initialized-mean={:.3} direct-mean={:.3}",
            median(&initialized_ms),
            median(&direct_ms),
            mean(&initialized_ms),
            mean(&direct_ms),
        );
    }

    #[test]
    #[ignore = "production four-level certificate codeword/index initialization A/B"]
    fn certificate_schedule_avoids_zero_filled_codeword_and_index() {
        let inputs = LEVELS
            .into_iter()
            .enumerate()
            .map(|(level_index, level)| {
                let source = (0..level.raw)
                    .into_par_iter()
                    .map(|index| semantic_value(level_index, index))
                    .collect::<Vec<_>>();
                let spectra = (0..level.inverse_rate - 1)
                    .map(|block| generator_spectrum(level_index, block, level.group))
                    .collect::<Vec<_>>();
                let roots = (0..2 * level.inverse_rate)
                    .map(|index| {
                        *blake3::hash(
                            &[
                                b"LiLAC/certificate-initialization-benchmark/v1".as_slice(),
                                &level_index.to_le_bytes(),
                                &index.to_le_bytes(),
                            ]
                            .concat(),
                        )
                        .as_bytes()
                    })
                    .collect::<Vec<_>>();
                (level, level_index, source, spectra, roots)
            })
            .collect::<Vec<_>>();
        let expected = inputs
            .iter()
            .map(|(level, level_index, source, spectra, roots)| {
                let mut codeword = vec![Field192::ZERO; level.qa_fields()];
                populate_proof_codeword(*level, source, &mut codeword, spectra, 64);
                let mut index_oracle = vec![Field192::ZERO; level.qa_fields()];
                populate_index_oracle(*level_index, *level, &mut index_oracle, spectra, roots);
                (codeword, index_oracle)
            })
            .collect::<Vec<_>>();
        let run = |direct: bool| {
            let mut codeword = Vec::new();
            let mut index_oracle = Vec::new();
            let mut elapsed = Duration::ZERO;
            for (ordinal, (level, level_index, source, spectra, roots)) in inputs.iter().enumerate()
            {
                let start = Instant::now();
                if direct {
                    overwrite_padded_proof_codeword(*level, source, &mut codeword, spectra, 64);
                    overwrite_index_oracle(*level_index, *level, spectra, roots, &mut index_oracle);
                } else {
                    codeword.resize(level.qa_fields(), Field192::ZERO);
                    codeword.fill(Field192::ZERO);
                    populate_proof_codeword(*level, source, &mut codeword, spectra, 64);
                    index_oracle.resize(level.qa_fields(), Field192::ZERO);
                    index_oracle.fill(Field192::ZERO);
                    populate_index_oracle(*level_index, *level, &mut index_oracle, spectra, roots);
                }
                elapsed += start.elapsed();
                assert_eq!(codeword, expected[ordinal].0);
                assert_eq!(index_oracle, expected[ordinal].1);
            }
            elapsed.as_secs_f64() * 1_000.0
        };
        let mut initialized_ms = Vec::with_capacity(16);
        let mut direct_ms = Vec::with_capacity(16);
        for trial in 0..8 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            let mut trial_initialized = Vec::with_capacity(2);
            let mut trial_direct = Vec::with_capacity(2);
            for direct in order {
                let elapsed = run(direct);
                if direct {
                    direct_ms.push(elapsed);
                    trial_direct.push(elapsed);
                } else {
                    initialized_ms.push(elapsed);
                    trial_initialized.push(elapsed);
                }
            }
            eprintln!(
                "certificate-initialization trial={} initialized={:.3}/{:.3} ms direct={:.3}/{:.3} ms",
                trial + 1,
                trial_initialized[0],
                trial_initialized[1],
                trial_direct[0],
                trial_direct[1],
            );
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "certificate-initialization initialized-ms={initialized_ms:?} direct-ms={direct_ms:?} initialized-median={:.3} direct-median={:.3} initialized-mean={:.3} direct-mean={:.3}",
            median(&initialized_ms),
            median(&direct_ms),
            mean(&initialized_ms),
            mean(&direct_ms),
        );
    }

    #[test]
    #[ignore = "production four-level materialized-versus-symbolic zero relations"]
    fn certificate_zero_relations_avoid_materialized_residuals() {
        let inputs = LEVELS
            .into_iter()
            .enumerate()
            .map(|(level_index, level)| {
                let source = (0..level.raw)
                    .into_par_iter()
                    .map(|index| semantic_value(level_index, index))
                    .collect::<Vec<_>>();
                let roots = (0..2 * level.inverse_rate + 1)
                    .map(|index| {
                        *blake3::hash(
                            &[
                                b"LiLAC/copy-relation-benchmark/v1".as_slice(),
                                &level_index.to_le_bytes(),
                                &index.to_le_bytes(),
                            ]
                            .concat(),
                        )
                        .as_bytes()
                    })
                    .collect::<Vec<_>>();
                (level_index, level, source, roots)
            })
            .collect::<Vec<_>>();
        let expected = inputs
            .iter()
            .map(|(level_index, level, _, roots)| {
                (
                    zero_weighted_relation_proof(
                        b"copy",
                        b"dual-view-copy",
                        *level_index,
                        level.view_capacity(),
                        roots,
                    ),
                    zero_weighted_relation_proof(
                        b"Phi-link",
                        b"Phi-link",
                        *level_index,
                        2 * level.next_blocks * level.width,
                        roots,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let run = |symbolic: bool| {
            let mut left = Vec::new();
            let mut right = Vec::new();
            let start = Instant::now();
            let proofs = inputs
                .iter()
                .map(|(level_index, level, source, roots)| {
                    let copy = if symbolic {
                        zero_weighted_relation_proof(
                            b"copy",
                            b"dual-view-copy",
                            *level_index,
                            level.view_capacity(),
                            roots,
                        )
                    } else {
                        left.resize(level.view_capacity(), Field192::ZERO);
                        right.resize(level.view_capacity(), Field192::ZERO);
                        left.fill(Field192::ZERO);
                        right.fill(Field192::ZERO);
                        populate_copy_relation(
                            *level_index,
                            *level,
                            source,
                            &mut left,
                            &mut right,
                            roots,
                        );
                        prove_local_product_relation_with_claim_reusing(
                            &mut left,
                            &mut right,
                            local_relation_roots(b"dual-view-copy", *level_index, roots),
                            Field192::ZERO,
                        )
                    };
                    let phi_fields = 2 * level.next_blocks * level.width;
                    let phi = if symbolic {
                        zero_weighted_relation_proof(
                            b"Phi-link",
                            b"Phi-link",
                            *level_index,
                            phi_fields,
                            roots,
                        )
                    } else {
                        let point = (0..phi_fields.next_power_of_two().trailing_zeros() as usize)
                            .map(|index| {
                                semantic_challenge(b"Phi-link", *level_index, index, roots)
                            })
                            .collect::<Vec<_>>();
                        prove_local_product_relation(
                            vec![Field192::ZERO; phi_fields],
                            equality_weights_prefix(&point, phi_fields),
                            local_relation_roots(b"Phi-link", *level_index, roots),
                        )
                    };
                    (copy, phi)
                })
                .collect::<Vec<_>>();
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            assert_eq!(proofs, expected);
            elapsed
        };
        let mut materialized_ms = Vec::with_capacity(16);
        let mut symbolic_ms = Vec::with_capacity(16);
        for trial in 0..8 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            let mut trial_materialized = Vec::with_capacity(2);
            let mut trial_symbolic = Vec::with_capacity(2);
            for symbolic in order {
                let elapsed = run(symbolic);
                if symbolic {
                    symbolic_ms.push(elapsed);
                    trial_symbolic.push(elapsed);
                } else {
                    materialized_ms.push(elapsed);
                    trial_materialized.push(elapsed);
                }
            }
            eprintln!(
                "zero-relations trial={} materialized={:.3}/{:.3} ms symbolic={:.3}/{:.3} ms",
                trial + 1,
                trial_materialized[0],
                trial_materialized[1],
                trial_symbolic[0],
                trial_symbolic[1],
            );
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "zero-relations materialized-ms={materialized_ms:?} symbolic-ms={symbolic_ms:?} materialized-median={:.3} symbolic-median={:.3} materialized-mean={:.3} symbolic-mean={:.3}",
            median(&materialized_ms),
            median(&symbolic_ms),
            mean(&materialized_ms),
            mean(&symbolic_ms),
        );
    }

    #[test]
    #[ignore = "production-scale tensor-commitment A/B benchmark"]
    fn reusable_horizontal_scratch_matches_and_benchmarks_materialized_rows() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &spectra, 64);

        let start = Instant::now();
        let scratch_first = tensor_row_commitments(
            &proof_codeword,
            &message_row_roots,
            &horizontal_spectra,
            &zeros,
        );
        let scratch_first_time = start.elapsed();
        let start = Instant::now();
        let materialized_first = tensor_row_commitments_materialized(
            &proof_codeword,
            &message_row_roots,
            &horizontal_spectra,
            &zeros,
        );
        let materialized_first_time = start.elapsed();
        let start = Instant::now();
        let materialized_second = tensor_row_commitments_materialized(
            &proof_codeword,
            &message_row_roots,
            &horizontal_spectra,
            &zeros,
        );
        let materialized_second_time = start.elapsed();
        let start = Instant::now();
        let scratch_second = tensor_row_commitments(
            &proof_codeword,
            &message_row_roots,
            &horizontal_spectra,
            &zeros,
        );
        let scratch_second_time = start.elapsed();

        for candidate in [&materialized_first, &materialized_second, &scratch_second] {
            assert_eq!(candidate.len(), scratch_first.len());
            for (candidate, expected) in candidate.iter().zip(&scratch_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "scratch={:.3}/{:.3} ms materialized={:.3}/{:.3} ms",
            scratch_first_time.as_secs_f64() * 1_000.0,
            scratch_second_time.as_secs_f64() * 1_000.0,
            materialized_first_time.as_secs_f64() * 1_000.0,
            materialized_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale tensor encoding/hash split diagnostic"]
    fn tensor_commitment_reports_encoding_hash_split() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);

        let encode_only = || {
            proof_codeword
                .par_chunks_exact(CARRYOPEN_WIDTH)
                .enumerate()
                .map_init(
                    || {
                        (
                            vec![Field192::ZERO; CARRYOPEN_WIDTH],
                            vec![Field192::ZERO; CARRYOPEN_TENSOR_WIDTH],
                        )
                    },
                    |(transformed, encoded), (index, row)| {
                        transformed.copy_from_slice(row);
                        wht(transformed);
                        let mut checksum = if index < CARRYOPEN_ROWS {
                            Field192::ZERO
                        } else {
                            encoded[..CARRYOPEN_WIDTH].copy_from_slice(row);
                            std::hint::black_box(&encoded[..CARRYOPEN_WIDTH]);
                            encoded[0]
                        };
                        for (block, spectrum) in horizontal_spectra.iter().enumerate() {
                            let start = if index < CARRYOPEN_ROWS {
                                0
                            } else {
                                (block + 1) * CARRYOPEN_WIDTH
                            };
                            let parity = &mut encoded[start..start + CARRYOPEN_WIDTH];
                            parity.copy_from_slice(transformed);
                            parity
                                .iter_mut()
                                .zip(spectrum)
                                .for_each(|(value, multiplier)| *value *= multiplier);
                            wht(parity);
                            std::hint::black_box(&*parity);
                            checksum += parity[0];
                        }
                        checksum
                    },
                )
                .sum::<Field192>()
        };

        let start = Instant::now();
        let encoding_first = encode_only();
        let encoding_first_time = start.elapsed();
        type RowRoot = fn(&[Field192], usize, &[Digest]) -> Digest;
        let run_commitment = |row_root: Option<RowRoot>| {
            let start = Instant::now();
            let commitment = if let Some(row_root) = row_root {
                tensor_row_commitments_with_root(
                    &proof_codeword,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                    row_root,
                    combine_equal_subtrees,
                )
            } else {
                tensor_row_commitments(
                    &proof_codeword,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                )
            };
            (commitment, start.elapsed())
        };

        let (optimized_first, optimized_first_time) = run_commitment(None);
        let (leaf_first, leaf_first_time) = run_commitment(Some(
            prefix_root_materialized_leaves_for_benchmark as RowRoot,
        ));
        let (parent_first, parent_first_time) = run_commitment(Some(
            prefix_root_materialized_parents_for_benchmark as RowRoot,
        ));
        let (baseline_first, baseline_first_time) = run_commitment(Some(
            prefix_root_materialized_blocks_for_benchmark as RowRoot,
        ));
        let (baseline_second, baseline_second_time) = run_commitment(Some(
            prefix_root_materialized_blocks_for_benchmark as RowRoot,
        ));
        let (parent_second, parent_second_time) = run_commitment(Some(
            prefix_root_materialized_parents_for_benchmark as RowRoot,
        ));
        let (leaf_second, leaf_second_time) = run_commitment(Some(
            prefix_root_materialized_leaves_for_benchmark as RowRoot,
        ));
        let (optimized_second, optimized_second_time) = run_commitment(None);
        let start = Instant::now();
        let encoding_second = encode_only();
        let encoding_second_time = start.elapsed();

        assert_eq!(encoding_first, encoding_second);
        assert_ne!(encoding_first, Field192::ZERO);
        for candidate in [
            &baseline_first,
            &baseline_second,
            &parent_first,
            &parent_second,
            &leaf_first,
            &leaf_second,
            &optimized_second,
        ] {
            assert_eq!(candidate.len(), optimized_first.len());
            for (candidate, expected) in candidate.iter().zip(&optimized_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "rows={} encoding_only={:.3}/{:.3} ms direct_all={:.3}/{:.3} ms materialized_leaves={:.3}/{:.3} ms materialized_parents={:.3}/{:.3} ms materialized_all={:.3}/{:.3} ms",
            proof_codeword.len() / CARRYOPEN_WIDTH,
            encoding_first_time.as_secs_f64() * 1_000.0,
            encoding_second_time.as_secs_f64() * 1_000.0,
            optimized_first_time.as_secs_f64() * 1_000.0,
            optimized_second_time.as_secs_f64() * 1_000.0,
            leaf_first_time.as_secs_f64() * 1_000.0,
            leaf_second_time.as_secs_f64() * 1_000.0,
            parent_first_time.as_secs_f64() * 1_000.0,
            parent_second_time.as_secs_f64() * 1_000.0,
            baseline_first_time.as_secs_f64() * 1_000.0,
            baseline_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "production tensor roots with four-lane interleaved Field192 canonicalization"]
    fn four_lane_canonicalization_benchmarks_scalar_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);

        let run = |candidate: bool| {
            let root = if candidate {
                prefix_root as fn(&[Field192], usize, &[Digest]) -> Digest
            } else {
                prefix_root_scalar_canonical_for_benchmark
                    as fn(&[Field192], usize, &[Digest]) -> Digest
            };
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed().as_secs_f64() * 1_000.0)
        };
        let assert_same = |candidate: &[MatrixCommitment], expected: &[MatrixCommitment]| {
            assert_eq!(candidate.len(), expected.len());
            for (candidate, expected) in candidate.iter().zip(expected) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        };

        let reference = run(false).0;
        let mut scalar = Vec::with_capacity(8);
        let mut production = Vec::with_capacity(8);
        for trial in 0..4 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            for candidate in order {
                let result = run(candidate);
                assert_same(&result.0, &reference);
                if candidate {
                    production.push(result.1);
                } else {
                    scalar.push(result.1);
                }
            }
        }
        eprintln!(
            "four-lane-canonical tensor-roots scalar-ms={scalar:?} production-ms={production:?}"
        );
    }

    #[test]
    #[ignore = "production-scale small-subtree reducer crossed A/B benchmark"]
    fn small_subtree_fast_path_benchmarks_parallel_reducer() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);

        type CombineRoots = fn(&[Digest]) -> Digest;
        let run = |combine_roots: CombineRoots| {
            let start = Instant::now();
            let commitment = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                prefix_root,
                combine_roots,
            );
            (commitment, start.elapsed())
        };
        let (fast_first, fast_first_time) = run(combine_equal_subtrees);
        let (parallel_first, parallel_first_time) =
            run(combine_equal_subtrees_parallel_for_benchmark);
        let (parallel_second, parallel_second_time) =
            run(combine_equal_subtrees_parallel_for_benchmark);
        let (fast_second, fast_second_time) = run(combine_equal_subtrees);

        for candidate in [&parallel_first, &parallel_second, &fast_second] {
            assert_eq!(candidate.len(), fast_first.len());
            for (candidate, expected) in candidate.iter().zip(&fast_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "small-fast={:.3}/{:.3} ms parallel-legacy={:.3}/{:.3} ms",
            fast_first_time.as_secs_f64() * 1_000.0,
            fast_second_time.as_secs_f64() * 1_000.0,
            parallel_first_time.as_secs_f64() * 1_000.0,
            parallel_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale worker-local versus TLS digest-scratch crossed A/B benchmark"]
    fn worker_digest_scratch_benchmarks_tls_scratch() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);

        let run = |worker_scratch: bool| {
            let start = Instant::now();
            let commitment = if worker_scratch {
                tensor_row_commitments(
                    &proof_codeword,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                )
            } else {
                tensor_row_commitments_with_root(
                    &proof_codeword,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                    prefix_root,
                    combine_equal_subtrees,
                )
            };
            (commitment, start.elapsed())
        };
        let (worker_first, worker_first_time) = run(true);
        let (tls_first, tls_first_time) = run(false);
        let (tls_second, tls_second_time) = run(false);
        let (worker_second, worker_second_time) = run(true);

        for candidate in [&tls_first, &tls_second, &worker_second] {
            assert_eq!(candidate.len(), worker_first.len());
            for (candidate, expected) in candidate.iter().zip(&worker_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "worker-digest-scratch={:.3}/{:.3} ms tls-digest-scratch={:.3}/{:.3} ms",
            worker_first_time.as_secs_f64() * 1_000.0,
            worker_second_time.as_secs_f64() * 1_000.0,
            tls_first_time.as_secs_f64() * 1_000.0,
            tls_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale fused versus unfused first Merkle level benchmark"]
    fn fused_first_parent_level_benchmarks_unfused_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);

        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitment = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitment, start.elapsed())
        };
        let (fused_first, fused_first_time) = run(prefix_root);
        let (unfused_first, unfused_first_time) = run(prefix_root_unfused_for_benchmark);
        let (unfused_second, unfused_second_time) = run(prefix_root_unfused_for_benchmark);
        let (fused_second, fused_second_time) = run(prefix_root);

        for candidate in [&unfused_first, &unfused_second, &fused_second] {
            assert_eq!(candidate.len(), fused_first.len());
            for (candidate, expected) in candidate.iter().zip(&fused_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "fused-level1={:.3}/{:.3} ms unfused={:.3}/{:.3} ms",
            fused_first_time.as_secs_f64() * 1_000.0,
            fused_second_time.as_secs_f64() * 1_000.0,
            unfused_first_time.as_secs_f64() * 1_000.0,
            unfused_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale NEON transposed-output versus scalar-scatter benchmark"]
    fn transposed_output_benchmarks_scatter_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);

        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitment = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitment, start.elapsed())
        };
        let (transposed_first, transposed_first_time) = run(prefix_root);
        let (scatter_first, scatter_first_time) = run(prefix_root_scatter_for_benchmark);
        let (scatter_second, scatter_second_time) = run(prefix_root_scatter_for_benchmark);
        let (transposed_second, transposed_second_time) = run(prefix_root);

        for candidate in [&scatter_first, &scatter_second, &transposed_second] {
            assert_eq!(candidate.len(), transposed_first.len());
            for (candidate, expected) in candidate.iter().zip(&transposed_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "transposed-output={:.3}/{:.3} ms scalar-scatter={:.3}/{:.3} ms",
            transposed_first_time.as_secs_f64() * 1_000.0,
            transposed_second_time.as_secs_f64() * 1_000.0,
            scatter_first_time.as_secs_f64() * 1_000.0,
            scatter_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale borrowed versus copied parent-input benchmark"]
    fn borrowed_parent_inputs_benchmark_copied_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(0, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(1, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };

        let (borrowed_first, borrowed_first_time) = run(prefix_root);
        let (copied_first, copied_first_time) = run(prefix_root_copied_parents_for_benchmark);
        let (copied_second, copied_second_time) = run(prefix_root_copied_parents_for_benchmark);
        let (borrowed_second, borrowed_second_time) = run(prefix_root);
        for candidate in [&copied_first, &copied_second, &borrowed_second] {
            assert_eq!(candidate.len(), borrowed_first.len());
            for (candidate, expected) in candidate.iter().zip(&borrowed_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "borrowed-parent-inputs={:.3}/{:.3} ms copied-parent-inputs={:.3}/{:.3} ms",
            borrowed_first_time.as_secs_f64() * 1_000.0,
            borrowed_second_time.as_secs_f64() * 1_000.0,
            copied_first_time.as_secs_f64() * 1_000.0,
            copied_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale register-resident versus materialized parent-CV benchmark"]
    fn register_resident_parent_cv_benchmarks_materialized_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };

        let materialized_first_order =
            std::env::var_os("LILAC_PARENT_CV_MATERIALIZED_FIRST").is_some();
        let (
            (packed_first, packed_first_time),
            (materialized_first, materialized_first_time),
            (materialized_second, materialized_second_time),
            (packed_second, packed_second_time),
        ) = if materialized_first_order {
            let materialized_first = run(prefix_root_materialized_cv_for_benchmark);
            let packed_first = run(prefix_root);
            let packed_second = run(prefix_root);
            let materialized_second = run(prefix_root_materialized_cv_for_benchmark);
            (
                packed_first,
                materialized_first,
                materialized_second,
                packed_second,
            )
        } else {
            let packed_first = run(prefix_root);
            let materialized_first = run(prefix_root_materialized_cv_for_benchmark);
            let materialized_second = run(prefix_root_materialized_cv_for_benchmark);
            let packed_second = run(prefix_root);
            (
                packed_first,
                materialized_first,
                materialized_second,
                packed_second,
            )
        };
        for candidate in [&materialized_first, &materialized_second, &packed_second] {
            assert_eq!(candidate.len(), packed_first.len());
            for (candidate, expected) in candidate.iter().zip(&packed_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "materialized-first-order={materialized_first_order} register-resident-parent-cv={:.3}/{:.3} ms materialized-parent-cv={:.3}/{:.3} ms",
            packed_first_time.as_secs_f64() * 1_000.0,
            packed_second_time.as_secs_f64() * 1_000.0,
            materialized_first_time.as_secs_f64() * 1_000.0,
            materialized_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale vector-load versus scalar-gather parent-message benchmark"]
    fn vector_parent_messages_benchmark_scalar_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };

        let scalar_first_order = std::env::var_os("LILAC_PARENT_MESSAGES_SCALAR_FIRST").is_some();
        let (
            (vector_first, vector_first_time),
            (scalar_first, scalar_first_time),
            (scalar_second, scalar_second_time),
            (vector_second, vector_second_time),
        ) = if scalar_first_order {
            let scalar_first = run(prefix_root_scalar_parent_messages_for_benchmark);
            let vector_first = run(prefix_root);
            let vector_second = run(prefix_root);
            let scalar_second = run(prefix_root_scalar_parent_messages_for_benchmark);
            (vector_first, scalar_first, scalar_second, vector_second)
        } else {
            let vector_first = run(prefix_root);
            let scalar_first = run(prefix_root_scalar_parent_messages_for_benchmark);
            let scalar_second = run(prefix_root_scalar_parent_messages_for_benchmark);
            let vector_second = run(prefix_root);
            (vector_first, scalar_first, scalar_second, vector_second)
        };
        for candidate in [&scalar_first, &scalar_second, &vector_second] {
            assert_eq!(candidate.len(), vector_first.len());
            for (candidate, expected) in candidate.iter().zip(&vector_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "scalar-first-order={scalar_first_order} vector-parent-messages={:.3}/{:.3} ms scalar-parent-messages={:.3}/{:.3} ms",
            vector_first_time.as_secs_f64() * 1_000.0,
            vector_second_time.as_secs_f64() * 1_000.0,
            scalar_first_time.as_secs_f64() * 1_000.0,
            scalar_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale vector-load versus scalar-gather leaf-message benchmark"]
    fn vector_leaf_messages_benchmark_scalar_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };

        let scalar_first_order = std::env::var_os("LILAC_LEAF_MESSAGES_SCALAR_FIRST").is_some();
        let (
            (vector_first, vector_first_time),
            (scalar_first, scalar_first_time),
            (scalar_second, scalar_second_time),
            (vector_second, vector_second_time),
        ) = if scalar_first_order {
            let scalar_first = run(prefix_root_scalar_leaf_messages_for_benchmark);
            let vector_first = run(prefix_root);
            let vector_second = run(prefix_root);
            let scalar_second = run(prefix_root_scalar_leaf_messages_for_benchmark);
            (vector_first, scalar_first, scalar_second, vector_second)
        } else {
            let vector_first = run(prefix_root);
            let scalar_first = run(prefix_root_scalar_leaf_messages_for_benchmark);
            let scalar_second = run(prefix_root_scalar_leaf_messages_for_benchmark);
            let vector_second = run(prefix_root);
            (vector_first, scalar_first, scalar_second, vector_second)
        };
        for candidate in [&scalar_first, &scalar_second, &vector_second] {
            assert_eq!(candidate.len(), vector_first.len());
            for (candidate, expected) in candidate.iter().zip(&vector_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "scalar-first-order={scalar_first_order} vector-leaf-messages={:.3}/{:.3} ms scalar-leaf-messages={:.3}/{:.3} ms",
            vector_first_time.as_secs_f64() * 1_000.0,
            vector_second_time.as_secs_f64() * 1_000.0,
            scalar_first_time.as_secs_f64() * 1_000.0,
            scalar_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale fused versus materialized leaf-to-parent benchmark"]
    fn fused_leaf_parent_benchmark_unfused_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };

        let unfused_first_order = std::env::var_os("LILAC_LEAF_PARENT_UNFUSED_FIRST").is_some();
        let (
            (fused_first, fused_first_time),
            (unfused_first, unfused_first_time),
            (unfused_second, unfused_second_time),
            (fused_second, fused_second_time),
        ) = if unfused_first_order {
            let unfused_first = run(prefix_root_unfused_leaf_parent_io_for_benchmark);
            let fused_first = run(prefix_root);
            let fused_second = run(prefix_root);
            let unfused_second = run(prefix_root_unfused_leaf_parent_io_for_benchmark);
            (fused_first, unfused_first, unfused_second, fused_second)
        } else {
            let fused_first = run(prefix_root);
            let unfused_first = run(prefix_root_unfused_leaf_parent_io_for_benchmark);
            let unfused_second = run(prefix_root_unfused_leaf_parent_io_for_benchmark);
            let fused_second = run(prefix_root);
            (fused_first, unfused_first, unfused_second, fused_second)
        };
        for candidate in [&unfused_first, &unfused_second, &fused_second] {
            assert_eq!(candidate.len(), fused_first.len());
            for (candidate, expected) in candidate.iter().zip(&fused_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "unfused-first-order={unfused_first_order} fused-leaf-parent={:.3}/{:.3} ms unfused-leaf-parent={:.3}/{:.3} ms",
            fused_first_time.as_secs_f64() * 1_000.0,
            fused_second_time.as_secs_f64() * 1_000.0,
            unfused_first_time.as_secs_f64() * 1_000.0,
            unfused_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale fused versus materialized upper-parent benchmark"]
    fn fused_parent_levels_benchmark_unfused_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };

        let unfused_first_order = std::env::var_os("LILAC_PARENT_LEVELS_UNFUSED_FIRST").is_some();
        let (
            (fused_first, fused_first_time),
            (unfused_first, unfused_first_time),
            (unfused_second, unfused_second_time),
            (fused_second, fused_second_time),
        ) = if unfused_first_order {
            let unfused_first = run(prefix_root_unfused_parent_levels_for_benchmark);
            let fused_first = run(prefix_root);
            let fused_second = run(prefix_root);
            let unfused_second = run(prefix_root_unfused_parent_levels_for_benchmark);
            (fused_first, unfused_first, unfused_second, fused_second)
        } else {
            let fused_first = run(prefix_root);
            let unfused_first = run(prefix_root_unfused_parent_levels_for_benchmark);
            let unfused_second = run(prefix_root_unfused_parent_levels_for_benchmark);
            let fused_second = run(prefix_root);
            (fused_first, unfused_first, unfused_second, fused_second)
        };
        for candidate in [&unfused_first, &unfused_second, &fused_second] {
            assert_eq!(candidate.len(), fused_first.len());
            for (candidate, expected) in candidate.iter().zip(&fused_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "unfused-first-order={unfused_first_order} fused-parent-levels={:.3}/{:.3} ms unfused-parent-levels={:.3}/{:.3} ms",
            fused_first_time.as_secs_f64() * 1_000.0,
            fused_second_time.as_secs_f64() * 1_000.0,
            unfused_first_time.as_secs_f64() * 1_000.0,
            unfused_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale one-block-node v2 versus two-block v1 tensor roots"]
    fn one_block_v2_benchmarks_two_block_v1_tensor_roots() {
        let level = carryopen_level();
        let zeros_v1 = zero_roots(30);
        let zeros_v2 = zero_roots_one_block_nodes_v2_for_benchmark(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let message_row_roots_v1 = message
            .par_chunks_exact(CARRYOPEN_WIDTH)
            .map(|row| {
                prefix_root_two_block_nodes_v1_for_benchmark(row, CARRYOPEN_WIDTH, &zeros_v1)
            })
            .collect::<Vec<_>>();
        let message_row_roots_v2 = message
            .par_chunks_exact(CARRYOPEN_WIDTH)
            .map(|row| {
                prefix_root_one_block_nodes_v2_for_benchmark(row, CARRYOPEN_WIDTH, &zeros_v2)
            })
            .collect::<Vec<_>>();
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |message_roots: &[Digest],
                   zeros: &[Digest],
                   row_root: fn(&[Field192], usize, &[Digest]) -> Digest,
                   combine_roots: fn(&[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                message_roots,
                &horizontal_spectra,
                zeros,
                row_root,
                combine_roots,
            );
            (commitments, start.elapsed())
        };
        let v1_first = run(
            &message_row_roots_v1,
            &zeros_v1,
            prefix_root_two_block_nodes_v1_for_benchmark,
            combine_equal_subtrees_two_block_nodes_v1_for_benchmark,
        );
        let v2_first = run(
            &message_row_roots_v2,
            &zeros_v2,
            prefix_root_one_block_nodes_v2_for_benchmark,
            combine_equal_subtrees_one_block_nodes_v2_for_benchmark,
        );
        let v2_second = run(
            &message_row_roots_v2,
            &zeros_v2,
            prefix_root_one_block_nodes_v2_for_benchmark,
            combine_equal_subtrees_one_block_nodes_v2_for_benchmark,
        );
        let v1_second = run(
            &message_row_roots_v1,
            &zeros_v1,
            prefix_root_two_block_nodes_v1_for_benchmark,
            combine_equal_subtrees_two_block_nodes_v1_for_benchmark,
        );
        for (candidate, expected) in v2_second.0.iter().zip(&v2_first.0) {
            assert_eq!(candidate.root, expected.root);
            assert_eq!(candidate.row_roots, expected.row_roots);
        }
        for (candidate, expected) in v1_second.0.iter().zip(&v1_first.0) {
            assert_eq!(candidate.root, expected.root);
            assert_eq!(candidate.row_roots, expected.row_roots);
        }
        assert!(v1_first
            .0
            .iter()
            .zip(&v2_first.0)
            .all(|(v1, v2)| v1.root != v2.root && v1.row_roots != v2.row_roots));
        eprintln!(
            "one-block-v2 tensor-roots v1={:.3}/{:.3} ms v2={:.3}/{:.3} ms",
            v1_first.1.as_secs_f64() * 1_000.0,
            v1_second.1.as_secs_f64() * 1_000.0,
            v2_first.1.as_secs_f64() * 1_000.0,
            v2_second.1.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale fused field-level-2 versus former field-level-1 v2 tensor roots"]
    fn fused_field_level2_v2_benchmarks_field_level1_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let message_row_roots = message
            .par_chunks_exact(CARRYOPEN_WIDTH)
            .map(|row| prefix_root(row, CARRYOPEN_WIDTH, &zeros))
            .collect::<Vec<_>>();
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };
        let level1_first = run(prefix_root_one_block_nodes_v2_for_benchmark);
        let level2_first = run(prefix_root_fused_field_level2_v2_for_benchmark);
        let level2_second = run(prefix_root_fused_field_level2_v2_for_benchmark);
        let level1_second = run(prefix_root_one_block_nodes_v2_for_benchmark);
        let assert_same = |candidate: &[MatrixCommitment], expected: &[MatrixCommitment]| {
            assert_eq!(candidate.len(), expected.len());
            for (candidate, expected) in candidate.iter().zip(expected) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
                assert_eq!(candidate.row_domain, expected.row_domain);
                assert_eq!(candidate.zero_row_root, expected.zero_row_root);
            }
        };
        assert_same(&level1_first.0, &level2_first.0);
        assert_same(&level1_first.0, &level2_second.0);
        assert_same(&level1_first.0, &level1_second.0);
        eprintln!(
            "fused-field-level2-v2 tensor-roots level1={:.3}/{:.3} ms level2={:.3}/{:.3} ms",
            level1_first.1.as_secs_f64() * 1_000.0,
            level1_second.1.as_secs_f64() * 1_000.0,
            level2_first.1.as_secs_f64() * 1_000.0,
            level2_second.1.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale word-major versus lane-major upper v2 tensor roots"]
    fn word_major_upper_v2_benchmarks_lane_major_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let message_row_roots = message
            .par_chunks_exact(CARRYOPEN_WIDTH)
            .map(|row| prefix_root(row, CARRYOPEN_WIDTH, &zeros))
            .collect::<Vec<_>>();
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };
        let lane_first = run(prefix_root_fused_field_level2_v2_for_benchmark);
        let word_first = run(prefix_root_word_major_upper_v2_for_benchmark);
        let word_second = run(prefix_root_word_major_upper_v2_for_benchmark);
        let lane_second = run(prefix_root_fused_field_level2_v2_for_benchmark);
        let assert_same = |candidate: &[MatrixCommitment], expected: &[MatrixCommitment]| {
            assert_eq!(candidate.len(), expected.len());
            for (candidate, expected) in candidate.iter().zip(expected) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
                assert_eq!(candidate.row_domain, expected.row_domain);
                assert_eq!(candidate.zero_row_root, expected.zero_row_root);
            }
        };
        assert_same(&lane_first.0, &word_first.0);
        assert_same(&lane_first.0, &word_second.0);
        assert_same(&lane_first.0, &lane_second.0);
        eprintln!(
            "word-major-upper-v2 tensor-roots lane={:.3}/{:.3} ms word={:.3}/{:.3} ms",
            lane_first.1.as_secs_f64() * 1_000.0,
            lane_second.1.as_secs_f64() * 1_000.0,
            word_first.1.as_secs_f64() * 1_000.0,
            word_second.1.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale direct word-major field-level-2 handoff benchmark"]
    fn direct_word_major_field2_benchmarks_transposed_tensor_roots() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let message_row_roots = message
            .par_chunks_exact(CARRYOPEN_WIDTH)
            .map(|row| prefix_root(row, CARRYOPEN_WIDTH, &zeros))
            .collect::<Vec<_>>();
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |row_root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            let commitments = tensor_row_commitments_with_root(
                &proof_codeword,
                &message_row_roots,
                &horizontal_spectra,
                &zeros,
                row_root,
                combine_equal_subtrees,
            );
            (commitments, start.elapsed())
        };
        let direct_first_order = std::env::var_os("LILAC_DIRECT_WORD_MAJOR_FIELD2_FIRST").is_some();
        let (
            (transposed_first, transposed_first_time),
            (direct_first, direct_first_time),
            (direct_second, direct_second_time),
            (transposed_second, transposed_second_time),
        ) = if direct_first_order {
            let direct_first = run(prefix_root);
            let transposed_first = run(prefix_root_word_major_upper_v2_for_benchmark);
            let transposed_second = run(prefix_root_word_major_upper_v2_for_benchmark);
            let direct_second = run(prefix_root);
            (
                transposed_first,
                direct_first,
                direct_second,
                transposed_second,
            )
        } else {
            let transposed_first = run(prefix_root_word_major_upper_v2_for_benchmark);
            let direct_first = run(prefix_root);
            let direct_second = run(prefix_root);
            let transposed_second = run(prefix_root_word_major_upper_v2_for_benchmark);
            (
                transposed_first,
                direct_first,
                direct_second,
                transposed_second,
            )
        };
        for candidate in [&direct_first, &direct_second, &transposed_second] {
            assert_eq!(candidate.len(), transposed_first.len());
            for (candidate, expected) in candidate.iter().zip(&transposed_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
                assert_eq!(candidate.row_domain, expected.row_domain);
                assert_eq!(candidate.zero_row_root, expected.zero_row_root);
            }
        }
        eprintln!(
            "direct-word-major-field2-first={direct_first_order} transposed={:.3}/{:.3} ms direct={:.3}/{:.3} ms",
            transposed_first_time.as_secs_f64() * 1_000.0,
            transposed_second_time.as_secs_f64() * 1_000.0,
            direct_first_time.as_secs_f64() * 1_000.0,
            direct_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale matrix-local versus flat row-root staging benchmark"]
    fn matrix_local_row_roots_benchmark_flat_staging() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(0, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(1, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);
        let run = |matrix_local: bool| {
            let start = Instant::now();
            let commitments = if matrix_local {
                tensor_row_commitments(
                    &proof_codeword,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                )
            } else {
                tensor_row_commitments_flat_staging(
                    &proof_codeword,
                    &message_row_roots,
                    &horizontal_spectra,
                    &zeros,
                )
            };
            (commitments, start.elapsed())
        };

        let (matrix_first, matrix_first_time) = run(true);
        let (flat_first, flat_first_time) = run(false);
        let (flat_second, flat_second_time) = run(false);
        let (matrix_second, matrix_second_time) = run(true);
        for candidate in [&flat_first, &flat_second, &matrix_second] {
            assert_eq!(candidate.len(), matrix_first.len());
            for (candidate, expected) in candidate.iter().zip(&matrix_first) {
                assert_eq!(candidate.root, expected.root);
                assert_eq!(candidate.row_roots, expected.row_roots);
            }
        }
        eprintln!(
            "matrix-local-row-roots={:.3}/{:.3} ms flat-staging={:.3}/{:.3} ms",
            matrix_first_time.as_secs_f64() * 1_000.0,
            matrix_second_time.as_secs_f64() * 1_000.0,
            flat_first_time.as_secs_f64() * 1_000.0,
            flat_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-scale coalesced versus separate folder scratch"]
    fn coalesced_tensor_scratch_benchmarks_separate_allocations() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let (_, message_row_roots) =
            exact_root_with_row_subtrees(&message, CARRYOPEN_WIDTH, &zeros);
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(10, component, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|component| generator_spectrum(11, component, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let mut proof_codeword = vec![Field192::ZERO; level.qa_fields()];
        populate_proof_codeword(level, &message, &mut proof_codeword, &vertical_spectra, 64);

        let mut coalesced_ms = Vec::with_capacity(4);
        let mut separate_ms = Vec::with_capacity(4);
        let mut expected: Option<Vec<(Digest, Vec<Digest>)>> = None;
        for order in [[true, false, false, true], [false, true, true, false]] {
            for coalesced in order {
                let start = Instant::now();
                let candidate = if coalesced {
                    tensor_row_commitments(
                        &proof_codeword,
                        &message_row_roots,
                        &horizontal_spectra,
                        &zeros,
                    )
                } else {
                    tensor_row_commitments_separate_scratch_for_benchmark(
                        &proof_codeword,
                        &message_row_roots,
                        &horizontal_spectra,
                        &zeros,
                    )
                };
                let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
                if let Some(expected) = &expected {
                    assert_eq!(candidate.len(), expected.len());
                    for (candidate, (expected_root, expected_rows)) in
                        candidate.iter().zip(expected)
                    {
                        assert_eq!(candidate.root, *expected_root);
                        assert_eq!(candidate.row_roots, *expected_rows);
                    }
                } else {
                    expected = Some(
                        candidate
                            .iter()
                            .map(|commitment| (commitment.root, commitment.row_roots.clone()))
                            .collect(),
                    );
                }
                if coalesced {
                    coalesced_ms.push(elapsed_ms);
                } else {
                    separate_ms.push(elapsed_ms);
                }
            }
        }
        eprintln!(
            "coalesced-tensor-scratch-ms={coalesced_ms:?} separate-scratch-ms={separate_ms:?}"
        );
    }

    #[test]
    fn reusable_precarry_scratch_matches_materialized_helpers() {
        let point = [
            Field192::from(2_u64),
            Field192::from(3_u64),
            Field192::from(5_u64),
        ];
        let message = (0..8).map(left_value).collect::<Vec<_>>();
        let mut scratch = vec![Field192::ZERO; message.len()];
        assert_eq!(
            evaluate_power_of_two_message_with_scratch(&message, &point, &mut scratch),
            evaluate_power_of_two_message(&message, &point)
        );

        let scale = Field192::from(17_u64);
        fill_scaled_equality_weights(&point, scale, &mut scratch);
        let materialized = equality_weights(&point)
            .into_iter()
            .map(|weight| scale * weight)
            .collect::<Vec<_>>();
        assert_eq!(scratch, materialized);
        let mut accumulated = vec![Field192::ZERO; materialized.len()];
        accumulate_scaled_equality_weights(&point, scale, &mut scratch, &mut accumulated);
        assert_eq!(accumulated, materialized);
        let mut one_round = vec![Field192::ZERO; materialized.len()];
        accumulate_scaled_equality_weights_one_round(&point, scale, &mut scratch, &mut one_round);
        assert_eq!(one_round, materialized);
    }

    #[test]
    #[ignore = "production pre-Carry fused final equality-weight round benchmark"]
    fn fused_equality_weight_accumulation_benchmarks_separate_pass() {
        let points = precarry_opening_points(&[37_u8; 32], 16);
        let coefficients = precarry_coefficients(&[91_u8; 32], points.len());
        let run = |fused: bool| {
            let start = Instant::now();
            let mut scratch = vec![Field192::ZERO; CARRYOPEN_FIELDS];
            let mut combined = vec![Field192::ZERO; CARRYOPEN_FIELDS];
            for (point, coefficient) in points.iter().zip(&coefficients) {
                if fused {
                    accumulate_scaled_equality_weights(
                        point,
                        *coefficient,
                        &mut scratch,
                        &mut combined,
                    );
                } else {
                    fill_scaled_equality_weights(point, *coefficient, &mut scratch);
                    combined
                        .par_iter_mut()
                        .zip(scratch.par_iter())
                        .for_each(|(target, weight)| *target += *weight);
                }
            }
            (combined, start.elapsed())
        };

        let (fused_first, fused_first_time) = run(true);
        let (separate_first, separate_first_time) = run(false);
        assert_eq!(separate_first, fused_first);
        drop(separate_first);
        let (separate_second, separate_second_time) = run(false);
        assert_eq!(separate_second, fused_first);
        drop(separate_second);
        let (fused_second, fused_second_time) = run(true);
        assert_eq!(fused_second, fused_first);
        eprintln!(
            "fused-accumulation={:.3}/{:.3} ms separate-pass={:.3}/{:.3} ms",
            fused_first_time.as_secs_f64() * 1_000.0,
            fused_second_time.as_secs_f64() * 1_000.0,
            separate_first_time.as_secs_f64() * 1_000.0,
            separate_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production pre-Carry two-round versus one-round accumulation benchmark"]
    fn two_round_equality_accumulation_benchmarks_one_round() {
        let points = precarry_opening_points(&[43_u8; 32], 16);
        let coefficients = precarry_coefficients(&[97_u8; 32], points.len());
        let run = |two_rounds: bool| {
            let start = Instant::now();
            let mut scratch = vec![Field192::ZERO; CARRYOPEN_FIELDS];
            let mut combined = vec![Field192::ZERO; CARRYOPEN_FIELDS];
            for (point, coefficient) in points.iter().zip(&coefficients) {
                if two_rounds {
                    accumulate_scaled_equality_weights(
                        point,
                        *coefficient,
                        &mut scratch,
                        &mut combined,
                    );
                } else {
                    accumulate_scaled_equality_weights_one_round(
                        point,
                        *coefficient,
                        &mut scratch,
                        &mut combined,
                    );
                }
            }
            (combined, start.elapsed())
        };

        let (two_first, two_first_time) = run(true);
        let (one_first, one_first_time) = run(false);
        assert_eq!(one_first, two_first);
        drop(one_first);
        let (one_second, one_second_time) = run(false);
        assert_eq!(one_second, two_first);
        drop(one_second);
        let (two_second, two_second_time) = run(true);
        assert_eq!(two_second, two_first);
        eprintln!(
            "two-round-accumulation={:.3}/{:.3} ms one-round-accumulation={:.3}/{:.3} ms",
            two_first_time.as_secs_f64() * 1_000.0,
            two_second_time.as_secs_f64() * 1_000.0,
            one_first_time.as_secs_f64() * 1_000.0,
            one_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production pre-Carry end-to-end fused-accumulation benchmark"]
    fn fused_equality_weight_accumulation_benchmarks_full_precarry() {
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let commitment_root = prefix_root(
            &message,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let run = |fused: bool| {
            let start = Instant::now();
            let proof = if fused {
                build_production_precarry(&message, commitment_root)
            } else {
                build_production_precarry_separate_pass(&message, commitment_root)
            };
            (proof, start.elapsed())
        };

        let (fused_first, fused_first_time) = run(true);
        let (separate_first, separate_first_time) = run(false);
        let (separate_second, separate_second_time) = run(false);
        let (fused_second, fused_second_time) = run(true);
        assert_eq!(separate_first, fused_first);
        assert_eq!(separate_second, fused_first);
        assert_eq!(fused_second, fused_first);
        eprintln!(
            "precarry-fused={:.3}/{:.3} ms precarry-separate={:.3}/{:.3} ms",
            fused_first_time.as_secs_f64() * 1_000.0,
            fused_second_time.as_secs_f64() * 1_000.0,
            separate_first_time.as_secs_f64() * 1_000.0,
            separate_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production direct-first-round pre-Carry evaluation A/B"]
    fn direct_first_round_benchmarks_precarry_evaluations() {
        let mut original = Vec::with_capacity(4 * CARRYOPEN_FIELDS);
        original.resize(CARRYOPEN_FIELDS, Field192::ZERO);
        original
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = precarry_message_value(index));
        let commitment_root = prefix_root(
            &original,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let expected = {
            let mut codeword = original.clone();
            codeword.reserve_exact(3 * CARRYOPEN_FIELDS);
            build_production_precarry_in_codeword_with_modes(
                &mut codeword,
                commitment_root,
                false,
                false,
            )
        };
        let run = |direct_first_round: bool| {
            let mut codeword = original.clone();
            codeword.reserve_exact(3 * CARRYOPEN_FIELDS);
            let start = Instant::now();
            let proof = build_production_precarry_in_codeword_with_modes(
                &mut codeword,
                commitment_root,
                direct_first_round,
                false,
            );
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            assert_eq!(proof, expected);
            assert_eq!(codeword, original);
            elapsed
        };
        let mut copied_ms = Vec::with_capacity(16);
        let mut direct_ms = Vec::with_capacity(16);
        for trial in 0..8 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            let mut trial_copied = Vec::with_capacity(2);
            let mut trial_direct = Vec::with_capacity(2);
            for direct_first_round in order {
                let elapsed = run(direct_first_round);
                if direct_first_round {
                    direct_ms.push(elapsed);
                    trial_direct.push(elapsed);
                } else {
                    copied_ms.push(elapsed);
                    trial_copied.push(elapsed);
                }
            }
            eprintln!(
                "precarry-evaluation trial={} copied={:.3}/{:.3} ms direct={:.3}/{:.3} ms",
                trial + 1,
                trial_copied[0],
                trial_copied[1],
                trial_direct[0],
                trial_direct[1],
            );
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "precarry-evaluation copied-ms={copied_ms:?} direct-ms={direct_ms:?} copied-median={:.3} direct-median={:.3} copied-mean={:.3} direct-mean={:.3}",
            median(&copied_ms),
            median(&direct_ms),
            mean(&copied_ms),
            mean(&direct_ms),
        );
    }

    #[test]
    #[ignore = "production compact versus four-quarter pre-Carry workspace A/B"]
    fn compact_precarry_workspace_benchmarks_four_quarters() {
        let mut original = Vec::with_capacity(4 * CARRYOPEN_FIELDS);
        original.resize(CARRYOPEN_FIELDS, Field192::ZERO);
        original
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = precarry_message_value(index));
        let commitment_root = prefix_root(
            &original,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let expected = {
            let mut codeword = original.clone();
            codeword.reserve_exact(3 * CARRYOPEN_FIELDS);
            build_production_precarry_in_codeword_with_modes(
                &mut codeword,
                commitment_root,
                true,
                true,
            )
        };
        let run = |compact: bool| {
            let mut codeword = original.clone();
            codeword.reserve_exact(3 * CARRYOPEN_FIELDS);
            let start = Instant::now();
            let proof = if compact {
                build_production_precarry_in_codeword_compact(&mut codeword, commitment_root)
            } else {
                build_production_precarry_in_codeword_with_modes(
                    &mut codeword,
                    commitment_root,
                    true,
                    true,
                )
            };
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            assert_eq!(proof, expected);
            assert_eq!(codeword, original);
            elapsed
        };
        let mut four_quarter_ms = Vec::with_capacity(16);
        let mut compact_ms = Vec::with_capacity(16);
        let mut compact_midpoints = 0_usize;
        for trial in 0..8 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            let mut trial_four_quarter = Vec::with_capacity(2);
            let mut trial_compact = Vec::with_capacity(2);
            for compact in order {
                let elapsed = run(compact);
                if compact {
                    compact_ms.push(elapsed);
                    trial_compact.push(elapsed);
                } else {
                    four_quarter_ms.push(elapsed);
                    trial_four_quarter.push(elapsed);
                }
            }
            let four_quarter_midpoint = (trial_four_quarter[0] + trial_four_quarter[1]) / 2.0;
            let compact_midpoint = (trial_compact[0] + trial_compact[1]) / 2.0;
            compact_midpoints += usize::from(compact_midpoint < four_quarter_midpoint);
            eprintln!(
                "precarry-workspace trial={} four-quarter={:.3}/{:.3} ms compact={:.3}/{:.3} ms",
                trial + 1,
                trial_four_quarter[0],
                trial_four_quarter[1],
                trial_compact[0],
                trial_compact[1],
            );
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "precarry-workspace four-quarter-ms={four_quarter_ms:?} compact-ms={compact_ms:?} four-quarter-median={:.3} compact-median={:.3} four-quarter-mean={:.3} compact-mean={:.3} compact-midpoints={compact_midpoints}/8",
            median(&four_quarter_ms),
            median(&compact_ms),
            mean(&four_quarter_ms),
            mean(&compact_ms),
        );
    }

    #[test]
    #[ignore = "production four-quarter pre-Carry workspace memory profile"]
    fn four_quarter_precarry_workspace_memory_profile() {
        let mut codeword = Vec::with_capacity(4 * CARRYOPEN_FIELDS);
        codeword.resize(CARRYOPEN_FIELDS, Field192::ZERO);
        codeword
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = precarry_message_value(index));
        let commitment_root = prefix_root(
            &codeword,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let proof = build_production_precarry_in_codeword_with_modes(
            &mut codeword,
            commitment_root,
            true,
            true,
        );
        assert!(proof.verify());
        assert_eq!(codeword.len(), CARRYOPEN_FIELDS);
        std::hint::black_box((&codeword, proof));
    }

    #[test]
    #[ignore = "production compact pre-Carry workspace memory profile"]
    fn compact_precarry_workspace_memory_profile() {
        let mut codeword = Vec::with_capacity(4 * CARRYOPEN_FIELDS);
        codeword.resize(CARRYOPEN_FIELDS, Field192::ZERO);
        codeword
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = precarry_message_value(index));
        let commitment_root = prefix_root(
            &codeword,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let proof = build_production_precarry_in_codeword_compact(&mut codeword, commitment_root);
        assert!(proof.verify());
        assert_eq!(codeword.len(), CARRYOPEN_FIELDS);
        std::hint::black_box((&codeword, proof));
    }

    #[test]
    #[ignore = "production block-local equality-weight accumulation A/B"]
    fn block_local_weights_benchmark_global_precarry_tables() {
        let mut original = Vec::with_capacity(4 * CARRYOPEN_FIELDS);
        original.resize(CARRYOPEN_FIELDS, Field192::ZERO);
        original
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = precarry_message_value(index));
        let commitment_root = prefix_root(
            &original,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let global_proof = {
            let mut codeword = original.clone();
            codeword.reserve_exact(3 * CARRYOPEN_FIELDS);
            build_production_precarry_in_codeword_with_modes(
                &mut codeword,
                commitment_root,
                true,
                false,
            )
        };
        let block_local_proof = {
            let mut codeword = original.clone();
            codeword.reserve_exact(3 * CARRYOPEN_FIELDS);
            build_production_precarry_in_codeword_with_modes(
                &mut codeword,
                commitment_root,
                true,
                true,
            )
        };
        assert_eq!(block_local_proof, global_proof);

        let points = precarry_opening_points(&commitment_root, 16);
        let coefficients = precarry_coefficients(&[0x71_u8; 32], points.len());
        let mut scratch = vec![Field192::ZERO; CARRYOPEN_FIELDS];
        let mut target = vec![Field192::ZERO; CARRYOPEN_FIELDS];
        let mut expected = vec![Field192::ZERO; CARRYOPEN_FIELDS];
        for (point, coefficient) in points.iter().zip(&coefficients) {
            accumulate_scaled_equality_weights(point, *coefficient, &mut scratch, &mut expected);
        }
        let mut run = |block_local: bool| {
            target.fill(Field192::ZERO);
            let start = Instant::now();
            if block_local {
                accumulate_equality_weights_block_local(&points, &coefficients, &mut target);
            } else {
                for (point, coefficient) in points.iter().zip(&coefficients) {
                    accumulate_scaled_equality_weights(
                        point,
                        *coefficient,
                        &mut scratch,
                        &mut target,
                    );
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            assert_eq!(target, expected);
            elapsed
        };
        let mut global_ms = Vec::with_capacity(16);
        let mut block_local_ms = Vec::with_capacity(16);
        for trial in 0..8 {
            let order = if trial % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            let mut trial_global = Vec::with_capacity(2);
            let mut trial_block_local = Vec::with_capacity(2);
            for block_local in order {
                let elapsed = run(block_local);
                if block_local {
                    block_local_ms.push(elapsed);
                    trial_block_local.push(elapsed);
                } else {
                    global_ms.push(elapsed);
                    trial_global.push(elapsed);
                }
            }
            eprintln!(
                "precarry-weight trial={} global={:.3}/{:.3} ms block-local={:.3}/{:.3} ms",
                trial + 1,
                trial_global[0],
                trial_global[1],
                trial_block_local[0],
                trial_block_local[1],
            );
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "precarry-weight global-ms={global_ms:?} block-local-ms={block_local_ms:?} global-median={:.3} block-local-median={:.3} global-mean={:.3} block-local-mean={:.3}",
            median(&global_ms),
            median(&block_local_ms),
            mean(&global_ms),
            mean(&block_local_ms),
        );
    }

    #[test]
    #[ignore = "production in-allocation versus separate-allocation pre-Carry benchmark"]
    fn in_allocation_precarry_matches_and_benchmarks_separate_allocations() {
        let mut original = Vec::with_capacity(4 * CARRYOPEN_FIELDS);
        original.resize(CARRYOPEN_FIELDS, Field192::ZERO);
        original
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = precarry_message_value(index));
        let commitment_root = prefix_root(
            &original,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let run_in_allocation = || {
            let mut codeword = original.clone();
            codeword.reserve_exact(3 * CARRYOPEN_FIELDS);
            let start = Instant::now();
            let proof = build_production_precarry_in_codeword(&mut codeword, commitment_root);
            let elapsed = start.elapsed();
            assert_eq!(codeword, original);
            (proof, elapsed)
        };
        let run_separate = || {
            let start = Instant::now();
            let proof = build_production_precarry(&original, commitment_root);
            (proof, start.elapsed())
        };

        let (in_allocation_first, in_allocation_first_time) = run_in_allocation();
        let (separate_first, separate_first_time) = run_separate();
        let (separate_second, separate_second_time) = run_separate();
        let (in_allocation_second, in_allocation_second_time) = run_in_allocation();
        assert_eq!(in_allocation_first, separate_first);
        assert_eq!(in_allocation_first, separate_second);
        assert_eq!(in_allocation_first, in_allocation_second);
        eprintln!(
            "precarry-in-allocation={:.3}/{:.3} ms separate-allocation={:.3}/{:.3} ms",
            in_allocation_first_time.as_secs_f64() * 1_000.0,
            in_allocation_second_time.as_secs_f64() * 1_000.0,
            separate_first_time.as_secs_f64() * 1_000.0,
            separate_second_time.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production separate-allocation pre-Carry memory profile"]
    fn separate_allocation_precarry_memory_profile() {
        let mut codeword = Vec::with_capacity(4 * CARRYOPEN_FIELDS);
        codeword.resize(CARRYOPEN_FIELDS, Field192::ZERO);
        codeword
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = precarry_message_value(index));
        let commitment_root = prefix_root(
            &codeword,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let proof = build_production_precarry(&codeword, commitment_root);
        codeword.resize(4 * CARRYOPEN_FIELDS, Field192::ZERO);
        assert!(proof.verify());
        assert_eq!(codeword.len(), 4 * CARRYOPEN_FIELDS);
    }

    #[test]
    #[ignore = "production in-allocation pre-Carry memory profile"]
    fn in_allocation_precarry_memory_profile() {
        let mut codeword = Vec::with_capacity(4 * CARRYOPEN_FIELDS);
        codeword.resize(CARRYOPEN_FIELDS, Field192::ZERO);
        codeword
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = precarry_message_value(index));
        let commitment_root = prefix_root(
            &codeword,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let proof = build_production_precarry_in_codeword(&mut codeword, commitment_root);
        codeword.resize(4 * CARRYOPEN_FIELDS, Field192::ZERO);
        assert!(proof.verify());
        assert_eq!(codeword.len(), 4 * CARRYOPEN_FIELDS);
    }

    #[test]
    #[ignore = "production pre-Carry two-round versus one-round full-proof benchmark"]
    fn two_round_equality_accumulation_benchmarks_full_precarry() {
        let message = (0..CARRYOPEN_FIELDS)
            .into_par_iter()
            .map(precarry_message_value)
            .collect::<Vec<_>>();
        let commitment_root = prefix_root(
            &message,
            CARRYOPEN_FIELDS,
            &zero_roots(CARRYOPEN_FIELDS.trailing_zeros() as usize),
        );
        let run = |two_rounds: bool| {
            let start = Instant::now();
            let proof = if two_rounds {
                build_production_precarry(&message, commitment_root)
            } else {
                build_production_precarry_one_round(&message, commitment_root)
            };
            (proof, start.elapsed())
        };

        let (two_first, two_first_time) = run(true);
        let (one_first, one_first_time) = run(false);
        let (one_second, one_second_time) = run(false);
        let (two_second, two_second_time) = run(true);
        assert_eq!(one_first, two_first);
        assert_eq!(one_second, two_first);
        assert_eq!(two_second, two_first);
        eprintln!(
            "precarry-two-round={:.3}/{:.3} ms precarry-one-round={:.3}/{:.3} ms",
            two_first_time.as_secs_f64() * 1_000.0,
            two_second_time.as_secs_f64() * 1_000.0,
            one_first_time.as_secs_f64() * 1_000.0,
            one_second_time.as_secs_f64() * 1_000.0,
        );
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
    fn public_rank_one_rows_match_their_scaled_merkle_roots() {
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
        let mut transcript_roots = fixed_generator_roots(0, level).as_ref().clone();
        transcript_roots.resize(2 * level.inverse_rate, [7_u8; 32]);
        let mut component_roots = transcript_roots.clone();
        component_roots.push(virtual_index_descriptor(0, level, &transcript_roots));
        let factors = validated_virtual_index_factors(0, level, &component_roots).unwrap();
        let (coefficients, alpha) = factors.as_ref();

        for row in [0, 3, level.inverse_rate * level.group - 1] {
            let values = alpha
                .iter()
                .map(|lane| coefficients[row] * *lane)
                .collect::<Vec<_>>();
            assert!(virtual_index_row_matches(&factors, row, &values));
            assert_eq!(
                prefix_root(&values, level.group, &zeros),
                scaled_prefix_root(alpha, coefficients[row], level.group, &zeros)
            );
            let mut changed = values;
            changed[1] += Field192::ONE;
            assert!(!virtual_index_row_matches(&factors, row, &changed));
        }
    }

    #[test]
    #[ignore = "production-width public-W field-check versus duplicate-hash benchmark"]
    fn public_rank_one_row_checks_benchmark_duplicate_hashing() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let transcript_roots = vec![[11_u8; 32]; 2 * level.inverse_rate];
        let mut component_roots = transcript_roots.clone();
        component_roots.push(virtual_index_descriptor(10, level, &transcript_roots));
        let factors = validated_virtual_index_factors(10, level, &component_roots).unwrap();
        let (coefficients, alpha) = factors.as_ref();
        let selected = (0..CARRYOPEN_QUERIES)
            .map(|index| index * coefficients.len() / CARRYOPEN_QUERIES)
            .collect::<Vec<_>>();
        let rows = selected
            .iter()
            .map(|row| {
                alpha
                    .iter()
                    .map(|lane| coefficients[*row] * *lane)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected_roots = selected
            .iter()
            .map(|row| scaled_prefix_root(alpha, coefficients[*row], CARRYOPEN_WIDTH, &zeros))
            .collect::<Vec<_>>();
        let repetitions = 20;

        let start = Instant::now();
        for _ in 0..repetitions {
            assert!(selected
                .iter()
                .zip(&rows)
                .all(|(row, values)| virtual_index_row_matches(&factors, *row, values)));
        }
        let direct_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..repetitions {
            assert!(rows
                .iter()
                .zip(&expected_roots)
                .all(|(values, root)| prefix_root(values, CARRYOPEN_WIDTH, &zeros) == *root));
        }
        let hashed_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..repetitions {
            assert!(rows
                .iter()
                .zip(&expected_roots)
                .all(|(values, root)| prefix_root(values, CARRYOPEN_WIDTH, &zeros) == *root));
        }
        let hashed_second = start.elapsed();
        let start = Instant::now();
        for _ in 0..repetitions {
            assert!(selected
                .iter()
                .zip(&rows)
                .all(|(row, values)| virtual_index_row_matches(&factors, *row, values)));
        }
        let direct_second = start.elapsed();

        eprintln!(
            "direct={:.3}/{:.3} ms duplicate-hash={:.3}/{:.3} ms repetitions={repetitions}",
            direct_first.as_secs_f64() * 1_000.0,
            direct_second.as_secs_f64() * 1_000.0,
            hashed_first.as_secs_f64() * 1_000.0,
            hashed_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production-geometry parallel-vs-sequential restoration-row benchmark"]
    fn parallel_restoration_rows_benchmark_sequential_audit() {
        let level = carryopen_level();
        let zeros = zero_roots(30);
        let horizontal_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(11, block, CARRYOPEN_WIDTH))
            .collect::<Vec<_>>();
        let transcript_roots = vec![[13_u8; 32]; 2 * level.inverse_rate];
        let mut component_roots = transcript_roots.clone();
        component_roots.push(virtual_index_descriptor(10, level, &transcript_roots));
        let factors = validated_virtual_index_factors(10, level, &component_roots).unwrap();
        let (coefficients, alpha) = factors.as_ref();
        let selected = (0..CARRYOPEN_QUERIES)
            .map(|index| index * coefficients.len() / CARRYOPEN_QUERIES)
            .collect::<Vec<_>>();
        let f_rows = selected
            .iter()
            .map(|row| {
                let base = (0..CARRYOPEN_WIDTH)
                    .map(|lane| Field192::from((*row + lane + 1) as u64))
                    .collect::<Vec<_>>();
                encode_horizontal_row(&base, &horizontal_spectra)
            })
            .collect::<Vec<_>>();
        let f_roots = f_rows
            .iter()
            .map(|row| prefix_root(row, CARRYOPEN_TENSOR_WIDTH, &zeros))
            .collect::<Vec<_>>();
        let w_rows = selected
            .iter()
            .map(|row| {
                alpha
                    .iter()
                    .map(|lane| coefficients[*row] * *lane)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let check = |ordinal: usize| {
            let f_row = &f_rows[ordinal];
            encode_horizontal_row(&f_row[..CARRYOPEN_WIDTH], &horizontal_spectra).as_slice()
                == f_row.as_slice()
                && prefix_root(f_row, CARRYOPEN_TENSOR_WIDTH, &zeros) == f_roots[ordinal]
                && virtual_index_row_matches(&factors, selected[ordinal], &w_rows[ordinal])
        };
        let repetitions = 10;

        let start = Instant::now();
        for _ in 0..repetitions {
            assert!((0..selected.len()).all(&check));
        }
        let sequential_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..repetitions {
            assert!((0..selected.len()).into_par_iter().all(&check));
        }
        let parallel_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..repetitions {
            assert!((0..selected.len()).into_par_iter().all(&check));
        }
        let parallel_second = start.elapsed();
        let start = Instant::now();
        for _ in 0..repetitions {
            assert!((0..selected.len()).all(&check));
        }
        let sequential_second = start.elapsed();

        eprintln!(
            "parallel={:.3}/{:.3} ms sequential={:.3}/{:.3} ms repetitions={repetitions}",
            parallel_first.as_secs_f64() * 1_000.0,
            parallel_second.as_secs_f64() * 1_000.0,
            sequential_first.as_secs_f64() * 1_000.0,
            sequential_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production canonical single-pass serialization/verifier A/B benchmark"]
    fn canonical_verifier_benchmarks_single_semantic_pass() {
        let carry = run_production_carryopen(1, false);
        let certificate = run_semantic_certificate_only(64, false);
        let proof = build_recursive_strong_end_to_end(&carry, &certificate, 1);
        let payload = proof.serialize();
        assert_eq!(payload.len(), 303_204);

        let breakdown = proof.byte_breakdown(payload.len());
        assert_eq!(breakdown.total, payload.len());
        assert_eq!(breakdown.base_total, 12_300);
        assert_eq!(breakdown.terminal_witness, 12_288);
        assert_eq!(breakdown.terminal_pcs, 0);

        let mut changed_base = proof.clone();
        changed_base.base.private_f_rows[0] += Field192::ONE;
        assert!(!changed_base.verify());

        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let serialized_first = proof.serialize();
        let serialized_first_time = start.elapsed();
        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let redundant_serialized_first = proof.serialize_redundantly_verified();
        let redundant_serialized_first_time = start.elapsed();
        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let redundant_serialized_second = proof.serialize_redundantly_verified();
        let redundant_serialized_second_time = start.elapsed();
        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let serialized_second = proof.serialize();
        let serialized_second_time = start.elapsed();
        assert_eq!(serialized_first, payload);
        assert_eq!(redundant_serialized_first, payload);
        assert_eq!(redundant_serialized_second, payload);
        assert_eq!(serialized_second, payload);

        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let emitted_first = proof.serialize();
        let breakdown_first = proof.byte_breakdown(emitted_first.len());
        let emitted_first_time = start.elapsed();
        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let redundant_emitted_first = proof.serialize_redundantly_verified();
        let redundant_breakdown_first = proof.byte_breakdown_redundantly_verified();
        let redundant_emitted_first_time = start.elapsed();
        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let redundant_emitted_second = proof.serialize_redundantly_verified();
        let redundant_breakdown_second = proof.byte_breakdown_redundantly_verified();
        let redundant_emitted_second_time = start.elapsed();
        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let emitted_second = proof.serialize();
        let breakdown_second = proof.byte_breakdown(emitted_second.len());
        let emitted_second_time = start.elapsed();
        assert_eq!(emitted_first, payload);
        assert_eq!(redundant_emitted_first, payload);
        assert_eq!(redundant_emitted_second, payload);
        assert_eq!(emitted_second, payload);
        for breakdown in [
            breakdown_first,
            redundant_breakdown_first,
            redundant_breakdown_second,
            breakdown_second,
        ] {
            assert_eq!(breakdown.total, payload.len());
        }
        let carry_offset = 8 + 5 * 4;
        let precarry_size = u32::from_le_bytes(
            payload[carry_offset + 8..carry_offset + 12]
                .try_into()
                .unwrap(),
        ) as usize;
        let first_carry_component_root = carry_offset + 8 + 7 * 4 + precarry_size;
        let mut changed = payload.clone();
        changed[first_carry_component_root] ^= 1;
        assert!(RecursiveStrongEndToEndProof::deserialize_unchecked(&changed).is_some());
        assert!(RecursiveStrongEndToEndProof::deserialize(&changed).is_none());
        assert!(RecursiveStrongEndToEndProof::deserialize_redundantly_verified(&changed).is_none());

        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let single_first = RecursiveStrongEndToEndProof::deserialize(&payload).unwrap();
        let single_first_time = start.elapsed();
        assert_eq!(single_first, proof);

        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let redundant_first =
            RecursiveStrongEndToEndProof::deserialize_redundantly_verified(&payload).unwrap();
        let redundant_first_time = start.elapsed();
        assert_eq!(redundant_first, proof);

        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let redundant_second =
            RecursiveStrongEndToEndProof::deserialize_redundantly_verified(&payload).unwrap();
        let redundant_second_time = start.elapsed();
        assert_eq!(redundant_second, proof);

        clear_virtual_index_factor_cache();
        let start = Instant::now();
        let single_second = RecursiveStrongEndToEndProof::deserialize(&payload).unwrap();
        let single_second_time = start.elapsed();
        assert_eq!(single_second, proof);

        eprintln!(
            "serialize-single={:.3}/{:.3} ms serialize-redundant={:.3}/{:.3} ms emit-and-account-single={:.3}/{:.3} ms emit-and-account-redundant={:.3}/{:.3} ms verifier-single={:.3}/{:.3} ms verifier-redundant={:.3}/{:.3} ms proof={} B",
            serialized_first_time.as_secs_f64() * 1_000.0,
            serialized_second_time.as_secs_f64() * 1_000.0,
            redundant_serialized_first_time.as_secs_f64() * 1_000.0,
            redundant_serialized_second_time.as_secs_f64() * 1_000.0,
            emitted_first_time.as_secs_f64() * 1_000.0,
            emitted_second_time.as_secs_f64() * 1_000.0,
            redundant_emitted_first_time.as_secs_f64() * 1_000.0,
            redundant_emitted_second_time.as_secs_f64() * 1_000.0,
            single_first_time.as_secs_f64() * 1_000.0,
            single_second_time.as_secs_f64() * 1_000.0,
            redundant_first_time.as_secs_f64() * 1_000.0,
            redundant_second_time.as_secs_f64() * 1_000.0,
            payload.len(),
        );
    }

    #[test]
    #[ignore = "production scalar versus batched row-subtree path verifier benchmark"]
    fn canonical_row_subtree_paths_benchmark_batched_parents() {
        use std::hint::black_box;

        let carry = run_production_carryopen(1, false);
        let certificate = run_semantic_certificate_only(64, false);
        let proof = build_recursive_strong_end_to_end(&carry, &certificate, 1);
        let payload = proof.serialize();
        assert_eq!(payload.len(), 303_204);

        let mut fronts = Vec::<(&SelectedRowFront, Level, &[Digest])>::with_capacity(11);
        fronts.push((
            &proof.carryopen.selected_front,
            carryopen_level(),
            &proof.carryopen.component_roots,
        ));
        fronts.extend(proof.certificate.iter().map(|transition| {
            (
                &transition.selected_front,
                LEVELS[transition.level],
                transition.component_roots.as_slice(),
            )
        }));
        fronts.extend(proof.strong.iter().map(|transition| {
            (
                &transition.selected_front,
                transition.level().unwrap(),
                transition.component_roots.as_slice(),
            )
        }));
        assert_eq!(fronts.len(), 11);

        let check = |batched: bool| {
            fronts.iter().all(|(front, level, roots)| {
                let proof_roots = &roots[level.inverse_rate..2 * level.inverse_rate];
                front.proof_blocks.iter().all(|(block, path)| {
                    if batched {
                        path.verify(proof_roots[*block], level.group)
                    } else {
                        path.verify_scalar(proof_roots[*block], level.group)
                    }
                })
            })
        };
        assert!(check(false));
        assert!(check(true));

        let repetitions = 200;
        let run = |batched| {
            let start = Instant::now();
            for _ in 0..repetitions {
                assert!(black_box(check(black_box(batched))));
            }
            start.elapsed().as_secs_f64() * 1_000.0
        };
        let mut scalar = Vec::with_capacity(8);
        let mut batched = Vec::with_capacity(8);
        for trial in 0..8 {
            if trial % 2 == 0 {
                scalar.push(run(false));
                batched.push(run(true));
            } else {
                batched.push(run(true));
                scalar.push(run(false));
            }
        }
        eprintln!(
            "canonical-row-paths repetitions={repetitions} scalar-ms={scalar:?} batched-ms={batched:?}"
        );
    }

    #[test]
    #[ignore = "production canonical verifier stage profile"]
    fn canonical_verifier_reports_stage_profile() {
        let carry = run_production_carryopen(1, false);
        let certificate = run_semantic_certificate_only(64, false);
        let proof = build_recursive_strong_end_to_end(&carry, &certificate, 1);
        let payload = proof.serialize();
        assert_eq!(payload.len(), 303_204);

        clear_fixed_generator_spectra_cache();
        let preprocessing_start = Instant::now();
        preprocess_canonical_fixed_generators();
        let preprocessing_ms = preprocessing_start.elapsed().as_secs_f64() * 1_000.0;

        let repetitions = 6;
        let mut parse_ms = Vec::with_capacity(repetitions);
        let mut linkage_ms = Vec::with_capacity(repetitions);
        let mut strong_ms = vec![Vec::with_capacity(repetitions); STRONG_ROUNDS];
        let mut base_ms = Vec::with_capacity(repetitions);
        let mut carry_ms = Vec::with_capacity(repetitions);
        let mut certificate_ms = Vec::with_capacity(repetitions);
        let mut total_ms = Vec::with_capacity(repetitions);

        for _ in 0..repetitions {
            clear_virtual_index_factor_cache();
            let total_start = Instant::now();

            let start = Instant::now();
            let parsed = RecursiveStrongEndToEndProof::deserialize_unchecked(&payload).unwrap();
            parse_ms.push(start.elapsed().as_secs_f64() * 1_000.0);

            let start = Instant::now();
            let certificate_root = parsed.certificate.last().unwrap().next_source_root;
            let linked = parsed.strong.len() == STRONG_ROUNDS
                && parsed.certificate.len() == LEVELS.len()
                && parsed.strong[0].source_root()
                    == Some(parent(
                        certificate_root,
                        parsed.carryopen.terminal_source_root,
                    ));
            linkage_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            assert!(linked);

            for round in 0..STRONG_ROUNDS {
                let start = Instant::now();
                let valid = parsed.strong[round].round == round
                    && parsed.strong[round].verify()
                    && (round == 0
                        || parsed.strong[round].source_root()
                            == Some(parsed.strong[round - 1].terminal_source_root));
                strong_ms[round].push(start.elapsed().as_secs_f64() * 1_000.0);
                assert!(valid);
            }

            let start = Instant::now();
            assert!(parsed.base.verify(parsed.strong.last().unwrap()));
            base_ms.push(start.elapsed().as_secs_f64() * 1_000.0);

            let start = Instant::now();
            assert!(parsed
                .carryopen
                .verify(parsed.carryopen.terminal_source_root));
            carry_ms.push(start.elapsed().as_secs_f64() * 1_000.0);

            let start = Instant::now();
            assert!(verify_certificate_core(
                &parsed.certificate,
                certificate_root,
            ));
            certificate_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            total_ms.push(total_start.elapsed().as_secs_f64() * 1_000.0);
        }

        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        let strong_medians = strong_ms
            .iter()
            .map(|samples| median(samples))
            .collect::<Vec<_>>();
        eprintln!(
            "canonical-verifier-stage-profile repetitions={repetitions} proof={} B parse-ms={parse_ms:?} linkage-ms={linkage_ms:?} strong-ms={strong_ms:?} base-ms={base_ms:?} carry-ms={carry_ms:?} certificate-ms={certificate_ms:?} total-ms={total_ms:?}",
            payload.len(),
        );
        eprintln!(
            "canonical-verifier-stage-medians preprocessing={preprocessing_ms:.3} parse={:.3} linkage={:.3} strong={strong_medians:?} strong-total={:.3} base={:.3} carry={:.3} certificate={:.3} total={:.3} ms",
            median(&parse_ms),
            median(&linkage_ms),
            strong_medians.iter().sum::<f64>(),
            median(&base_ms),
            median(&carry_ms),
            median(&certificate_ms),
            median(&total_ms),
        );

        let level = carryopen_level();
        let zeros = zero_roots(30);
        let start = Instant::now();
        assert!(proof.carryopen.precarry.verify());
        let precarry_ms = start.elapsed().as_secs_f64() * 1_000.0;

        let start = Instant::now();
        let spectra = (0..level.inverse_rate - 1)
            .map(|block| generator_spectrum(10, block, level.group))
            .collect::<Vec<_>>();
        let generator_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let start = Instant::now();
        let factors = index_oracle_factors(
            10,
            level,
            &spectra,
            &proof.carryopen.component_roots[..2 * level.inverse_rate],
        );
        let factor_math_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(factors.0.len(), level.inverse_rate * level.group);

        clear_virtual_index_factor_cache();
        let start = Instant::now();
        assert!(
            validated_virtual_index_factors(10, level, &proof.carryopen.component_roots,).is_some()
        );
        let factor_total_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let start = Instant::now();
        assert!(virtual_index_selected_roots(
            10,
            level,
            &proof.carryopen.component_roots,
            &proof.carryopen.selected_front.selected,
            CARRYOPEN_WIDTH,
            &zeros,
        )
        .is_some());
        let selected_roots_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let start = Instant::now();
        assert!(proof.carryopen.selected_front.verify(
            10,
            level,
            &proof.carryopen.component_roots,
            CARRYOPEN_WIDTH,
            &zeros,
        ));
        let cached_front_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let start = Instant::now();
        assert_eq!(
            derived_carry_terminal_root(
                &proof.carryopen.component_roots,
                &proof.carryopen.selected_front,
                proof.carryopen.membership_ood_root,
                proof.carryopen.evaluation_ood_root,
                &zeros,
            ),
            Some(proof.carryopen.terminal_source_root),
        );
        let cached_derived_ms = start.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "carry-substages precarry={precarry_ms:.3} generator={generator_ms:.3} factor-math={factor_math_ms:.3} factor-total-cold={factor_total_ms:.3} selected-roots-after-factor={selected_roots_ms:.3} cached-front={cached_front_ms:.3} cached-derived={cached_derived_ms:.3} ms"
        );

        clear_virtual_index_factor_cache();
        let mut transition_ms = Vec::with_capacity(LEVELS.len());
        for transition in &proof.certificate {
            let start = Instant::now();
            assert!(transition.verify());
            transition_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        eprintln!("certificate-transition-ms={transition_ms:?}");
    }

    #[test]
    #[ignore = "production certificate padded scaled-root crossed benchmark"]
    fn dyadic_scaled_roots_benchmark_scalar_certificate_rows() {
        let measurement = run_semantic_certificate_only(64, false);
        let transitions = measurement
            .transition_proofs
            .as_ref()
            .expect("semantic certificate must retain transitions");
        let transition = &transitions[0];
        let level = LEVELS[0];
        let zeros = zero_roots(PRODUCTION_VARIABLES);
        preprocess_canonical_fixed_generators();
        clear_virtual_index_factor_cache();
        let factors = validated_virtual_index_factors(0, level, &transition.component_roots)
            .expect("level-0 public-W factors must validate");
        let (coefficients, alpha) = factors.as_ref();
        let selected = &transition.selected_front.selected;
        assert_eq!(selected.len(), level.next_blocks - 1);

        let dyadic = || {
            selected
                .par_iter()
                .map(|row| scaled_prefix_root(alpha, coefficients[*row], level.group, &zeros))
                .collect::<Vec<_>>()
        };
        let scalar = || {
            selected
                .par_iter()
                .map(|row| {
                    scaled_prefix_root_sequential_for_benchmark(
                        alpha,
                        coefficients[*row],
                        level.group,
                        &zeros,
                    )
                })
                .collect::<Vec<_>>()
        };
        let expected = scalar();
        assert_eq!(dyadic(), expected);

        let mut dyadic_ms = Vec::with_capacity(8);
        let mut scalar_ms = Vec::with_capacity(8);
        for trial in 0..4 {
            let order = if trial % 2 == 0 {
                [true, false, false, true]
            } else {
                [false, true, true, false]
            };
            for is_dyadic in order {
                let start = Instant::now();
                let roots = std::hint::black_box(if is_dyadic { dyadic() } else { scalar() });
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                assert_eq!(roots, expected);
                if is_dyadic {
                    dyadic_ms.push(elapsed);
                } else {
                    scalar_ms.push(elapsed);
                }
            }
        }
        let median = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
        };
        eprintln!(
            "certificate-level0-scaled-roots rows={} width={} capacity={} dyadic-ms={dyadic_ms:?} scalar-ms={scalar_ms:?} dyadic-median={:.3} scalar-median={:.3}",
            selected.len(),
            alpha.len(),
            level.group,
            median(&dyadic_ms),
            median(&scalar_ms),
        );
    }

    #[test]
    #[ignore = "production terminal-cutoff byte audit"]
    fn terminal_cutoff_byte_audit() {
        let carry = run_production_carryopen(1, false);
        let certificate = run_semantic_certificate_only(64, false);
        let proof = build_recursive_strong_end_to_end(&carry, &certificate, 1);
        let payload = proof.serialize();
        assert_eq!(payload.len(), 303_204);

        let strong_sizes = proof
            .strong
            .iter()
            .map(|core| 4 + core.serialize_unchecked().len())
            .collect::<Vec<_>>();
        let current_base = proof.base.serialize_unchecked().len();
        let brakedown_distance = 0.95_f64;
        let brakedown_target_bits = 131.162_f64;
        let brakedown_columns =
            (brakedown_target_bits / -(1.0 - brakedown_distance / 3.0).log2()).ceil() as usize;
        assert_eq!(brakedown_columns, 239);
        eprintln!("strong-core-framed-bytes={strong_sizes:?}");
        eprintln!(
            "brakedown-equal-security-target={brakedown_target_bits:.3} bits distance={brakedown_distance:.2} columns={brakedown_columns}"
        );
        for round in 0..STRONG_ROUNDS {
            let level = proof.strong[round].level().unwrap();
            let terminal_fields = level.next_blocks * level.width;
            let direct_base = 12 + terminal_fields * 24;
            let removed_later = strong_sizes[round + 1..].iter().sum::<usize>();
            let projected_total = payload.len() - removed_later - current_base + direct_base;
            let retained_prefix = payload.len() - removed_later - current_base;
            let backend_budget = payload.len() - retained_prefix;
            let (brakedown_fields, brakedown_rows, brakedown_row_width) = (1..=terminal_fields)
                .map(|rows| {
                    let row_width = terminal_fields.div_ceil(rows);
                    (2 * row_width + rows * brakedown_columns, rows, row_width)
                })
                .min()
                .unwrap();
            let brakedown_bare_bytes = brakedown_fields * 24;
            assert!(brakedown_bare_bytes >= backend_budget);
            eprintln!(
                "cutoff-round={round} terminal-fields={terminal_fields} direct-base-bytes={direct_base} projected-total-bytes={projected_total} backend-budget={backend_budget} brakedown-best=({brakedown_rows},{brakedown_row_width}) brakedown-bare-fields={brakedown_fields} brakedown-bare-bytes={brakedown_bare_bytes}"
            );
        }

        let context = *blake3::hash(b"LiLAC/terminal-backend-size-audit/v1").as_bytes();
        let whir = run_direct_whir_tail(proof.base.private_f_rows.clone(), context, 1);
        eprintln!(
            "terminal-backend-whir-optimistic semantic-fields={} padded-fields={} raw-proof-bytes={} framed-artifact-bytes={} commit-ms={:.3} prove-ms={:.3} verify-ms={:.3}",
            whir.semantic_fields,
            whir.padded_fields,
            whir.proof_bytes,
            whir.artifact.serialize().len(),
            whir.commit.as_secs_f64() * 1_000.0,
            whir.prove.as_secs_f64() * 1_000.0,
            whir.verify.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "production canonical payload digest checkpoint"]
    fn canonical_payload_digest_checkpoint() {
        let carry = run_production_carryopen(1, false);
        let certificate = run_semantic_certificate_only(64, false);
        let proof = build_recursive_strong_end_to_end(&carry, &certificate, 1);
        let payload = proof.serialize();
        let digest = blake3::hash(&payload);
        eprintln!(
            "canonical-payload-bytes={} blake3={}",
            payload.len(),
            digest.to_hex()
        );
        assert_eq!(payload.len(), 303_204);
        assert_eq!(
            digest.as_bytes(),
            &hex::decode("20809657e67ec20c3db9096bd7abe05baae2b4840951b5b4a3b2a1b9ed83ae09")
                .unwrap()[..]
        );
    }

    #[test]
    #[ignore = "production packed-global certificate memory profile"]
    fn packed_global_certificate_memory_profile() {
        let (_, _, roots, source, _, terminal_root, proofs, _) = semantic_vectors(64, false);
        assert_eq!(roots.len(), 31);
        assert_eq!(source.len(), 247_192);
        assert_eq!(roots.last(), Some(&terminal_root));
        assert_eq!(
            proofs
                .iter()
                .map(|proof| proof.serialize().len())
                .sum::<usize>(),
            180_156
        );
    }

    #[test]
    #[ignore = "production relation-local certificate memory profile"]
    fn relation_local_certificate_memory_profile() {
        let (roots, source, _, terminal_root, proofs, _) = semantic_certificate_objects(64, false);
        assert_eq!(roots.len(), 31);
        assert_eq!(source.len(), 247_192);
        assert_eq!(roots.last(), Some(&terminal_root));
        assert_eq!(
            proofs
                .iter()
                .map(|proof| proof.serialize().len())
                .sum::<usize>(),
            180_156
        );
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
        assert!(proof.verify_scalar(commitment.root, commitment.row_domain));
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
        assert!(!parsed.verify_scalar(commitment.root, commitment.row_domain));
        let mut changed = proof;
        changed.frontier[0][0] ^= 1;
        assert!(!changed.verify(commitment.root, commitment.row_domain));
        assert!(!changed.verify_scalar(commitment.root, commitment.row_domain));
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
        let mut membership = PackedSumcheckProof {
            fields: CARRYOPEN_INVERSE_RATE * CARRYOPEN_FIELDS,
            roots: vec![[7_u8; 32]],
            claimed_sum: Field192::ZERO,
            pairs: vec![(Field192::ZERO, Field192::ZERO); 26],
            terminal_left: Field192::ZERO,
            terminal_right: Field192::ZERO,
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
        let mut component_roots = fixed_generator_roots(10, level).as_ref().clone();
        component_roots.extend(
            (component_roots.len()..2 * CARRYOPEN_INVERSE_RATE).map(|value| [value as u8 + 1; 32]),
        );
        component_roots.push(virtual_index_descriptor(10, level, &component_roots));
        let vertical_spectra = (0..CARRYOPEN_INVERSE_RATE - 1)
            .map(|block| generator_spectrum(10, block, CARRYOPEN_ROWS))
            .collect::<Vec<_>>();
        let (w_coefficients, w_lanes) = index_oracle_factors(
            10,
            level,
            &vertical_spectra,
            &component_roots[..2 * CARRYOPEN_INVERSE_RATE],
        );
        let selected = (0..CARRYOPEN_QUERIES).collect::<Vec<_>>();
        let mut terminal = Vec::with_capacity(CARRYOPEN_TERMINAL_FIELDS);
        terminal.extend(encode_horizontal_row(
            &vec![Field192::ZERO; CARRYOPEN_WIDTH],
            &horizontal_spectra,
        ));
        let membership_point = membership.challenges().unwrap();
        let public_ood_w =
            virtual_index_ood_row(10, level, &component_roots, &membership_point).unwrap();
        let row_variables = (CARRYOPEN_INVERSE_RATE * CARRYOPEN_ROWS).trailing_zeros() as usize;
        membership.terminal_left =
            evaluate_power_of_two_message(&public_ood_w, &membership_point[row_variables..]);
        assert!(membership.challenges().is_some());
        terminal.extend(public_ood_w);
        terminal.extend(encode_horizontal_row(
            &vec![Field192::ZERO; CARRYOPEN_WIDTH],
            &horizontal_spectra,
        ));
        let mut f_roots = Vec::with_capacity(selected.len());
        for ordinal in 0..selected.len() {
            let f = (0..CARRYOPEN_WIDTH)
                .map(|lane| Field192::from((ordinal * CARRYOPEN_WIDTH + lane + 17) as u64))
                .collect::<Vec<_>>();
            let encoded = encode_horizontal_row(&f, &horizontal_spectra);
            f_roots.push(prefix_root(&encoded, CARRYOPEN_TENSOR_WIDTH, &zeros));
            terminal.extend(encoded);
            let w = w_lanes
                .iter()
                .map(|lane| w_coefficients[ordinal] * *lane)
                .collect::<Vec<_>>();
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
        };
        assert!(audit_tensor_carry_restoration(
            level,
            &component_roots,
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
            &component_roots,
            &front,
            &parity_mutation,
            &horizontal_spectra,
            &membership,
            &evaluation,
            &zeros,
        ));

        let mut w_mutation = terminal.clone();
        w_mutation[CARRYOPEN_OOD_FIELDS + CARRYOPEN_TENSOR_WIDTH] += Field192::ONE;
        assert!(!audit_tensor_carry_restoration(
            level,
            &component_roots,
            &front,
            &w_mutation,
            &horizontal_spectra,
            &membership,
            &evaluation,
            &zeros,
        ));

        let mut descriptor_mutation = component_roots.clone();
        descriptor_mutation.last_mut().unwrap()[0] ^= 1;
        assert!(!audit_tensor_carry_restoration(
            level,
            &descriptor_mutation,
            &front,
            &terminal,
            &horizontal_spectra,
            &membership,
            &evaluation,
            &zeros,
        ));
        let (_, virtual_w_bytes, _) = front.byte_breakdown();
        assert_eq!(virtual_w_bytes, 0);
    }

    #[test]
    fn direct_tail_sparse_entries_are_distinct_at_small_terminals() {
        let entries = direct_tail_sparse_entries(b"distinct-sparse-test", 32);
        let mut indices = entries.iter().map(|(index, _)| *index).collect::<Vec<_>>();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(entries.len(), 19);
        assert_eq!(indices.len(), entries.len());
    }
}
