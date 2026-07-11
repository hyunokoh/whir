use std::{
    convert::TryInto,
    mem::size_of,
    time::{Duration, Instant},
};

use ark_ff::{AdditiveGroup, BigInteger, Field, PrimeField};
use clap::Parser;
use rayon::prelude::*;
use whir::algebra::{fields::Field192, sumcheck::compute_sumcheck_polynomial};
use whir::lilac_merkle::{prefix_root, zero_roots};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 24)]
    variables: usize,
    #[arg(long, default_value_t = 16)]
    openings: usize,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
}

#[derive(Debug)]
struct Measurement {
    message_setup: Duration,
    claim_evaluation: Duration,
    weight_construction: Duration,
    polynomial: Duration,
    folding: Duration,
    prover_total: Duration,
    verifier: Duration,
    proof_bytes: usize,
    checksum: Field192,
    artifact: PreCarryArtifact,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreCarryArtifact {
    variables: usize,
    openings: usize,
    commitment: [u8; 32],
    points: Vec<Vec<Field192>>,
    claims: Vec<Field192>,
    pairs: Vec<(Field192, Field192)>,
    terminal_message: Field192,
    terminal_weight: Field192,
}

fn canonical_field_bytes(value: Field192) -> [u8; 24] {
    field_bytes(value)
        .try_into()
        .expect("Field192 canonical encoding must be 24 bytes")
}

impl PreCarryArtifact {
    const MAGIC: &'static [u8; 8] = b"LILPRE01";

    fn terminal_point(&self) -> Option<Vec<Field192>> {
        if self.points.len() != self.openings
            || self.claims.len() != self.openings
            || self.pairs.len() != self.variables
            || self
                .points
                .iter()
                .any(|point| point.len() != self.variables)
        {
            return None;
        }
        let statement = statement_root(&self.commitment, &self.points, &self.claims);
        let coefficients = batching_coefficients(&statement, self.openings);
        let batched_claim = self
            .claims
            .iter()
            .zip(&coefficients)
            .map(|(claim, coefficient)| *claim * coefficient)
            .sum::<Field192>();
        let mut claim = batched_claim;
        let mut challenges = Vec::with_capacity(self.variables);
        for prefix in 1..=self.pairs.len() {
            let (constant, quadratic) = self.pairs[prefix - 1];
            let challenge = transcript_challenge(&statement, batched_claim, &self.pairs[..prefix]);
            let linear = claim - constant.double() - quadratic;
            claim = (quadratic * challenge + linear) * challenge + constant;
            challenges.push(challenge);
        }
        let expected_weight = public_weight_at(&self.points, &coefficients, &challenges);
        (expected_weight == self.terminal_weight
            && claim == self.terminal_message * self.terminal_weight)
            .then_some(challenges)
    }

    fn verify(&self) -> bool {
        self.variables > 0
            && self.openings > 1
            && self.commitment != [0_u8; 32]
            && self.terminal_point().is_some()
    }

    fn serialize(&self) -> Vec<u8> {
        assert!(self.verify());
        let field_count = self.openings * self.variables + self.openings + 2 * self.variables + 2;
        let mut output = Vec::with_capacity(56 + field_count * 24);
        output.extend_from_slice(Self::MAGIC);
        output.extend_from_slice(&(self.variables as u32).to_le_bytes());
        output.extend_from_slice(&(self.openings as u32).to_le_bytes());
        output.extend_from_slice(&self.commitment);
        output.extend_from_slice(&(self.pairs.len() as u32).to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        for point in &self.points {
            for value in point {
                output.extend_from_slice(&canonical_field_bytes(*value));
            }
        }
        for value in &self.claims {
            output.extend_from_slice(&canonical_field_bytes(*value));
        }
        for (constant, quadratic) in &self.pairs {
            output.extend_from_slice(&canonical_field_bytes(*constant));
            output.extend_from_slice(&canonical_field_bytes(*quadratic));
        }
        output.extend_from_slice(&canonical_field_bytes(self.terminal_message));
        output.extend_from_slice(&canonical_field_bytes(self.terminal_weight));
        output
    }

    fn deserialize(payload: &[u8]) -> Option<Self> {
        const HEADER: usize = 8 + 4 + 4 + 32 + 4 + 4;
        if payload.len() < HEADER || &payload[..8] != Self::MAGIC {
            return None;
        }
        let variables = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
        let openings = u32::from_le_bytes(payload[12..16].try_into().ok()?) as usize;
        let commitment = payload[16..48].try_into().ok()?;
        let pair_count = u32::from_le_bytes(payload[48..52].try_into().ok()?) as usize;
        let reserved = u32::from_le_bytes(payload[52..56].try_into().ok()?);
        let field_count = openings.checked_mul(variables)? + openings + 2 * pair_count + 2;
        if variables == 0
            || openings <= 1
            || pair_count != variables
            || reserved != 0
            || payload.len() != HEADER + field_count * 24
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
        let pairs = (0..pair_count)
            .map(|_| Some((take_field()?, take_field()?)))
            .collect::<Option<Vec<_>>>()?;
        let terminal_message = take_field()?;
        let terminal_weight = take_field()?;
        let artifact = Self {
            variables,
            openings,
            commitment,
            points,
            claims,
            pairs,
            terminal_message,
            terminal_weight,
        };
        artifact.verify().then_some(artifact)
    }
}

fn percentile(values: &[f64], probability: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * probability).round() as usize]
}

fn field_bytes(value: Field192) -> Vec<u8> {
    value.into_bigint().to_bytes_le()
}

fn hash_to_field(parts: &[&[u8]]) -> Field192 {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes())
}

fn message_value(index: usize) -> Field192 {
    Field192::from(
        (index as u64 + 1)
            .wrapping_mul(0x9e37_79b1)
            .wrapping_add(0x4c49_4c41),
    )
}

fn commitment_root(message: &[Field192]) -> [u8; 32] {
    assert!(!message.is_empty() && message.len().is_power_of_two());
    prefix_root(
        message,
        message.len(),
        &zero_roots(message.len().trailing_zeros() as usize),
    )
}

fn opening_points(root: &[u8; 32], openings: usize, variables: usize) -> Vec<Vec<Field192>> {
    (0..openings)
        .map(|opening| {
            (0..variables)
                .map(|coordinate| {
                    hash_to_field(&[
                        b"LiLAC/pre-Carry/fixed-opening-point/v1",
                        root,
                        &(opening as u64).to_le_bytes(),
                        &(coordinate as u64).to_le_bytes(),
                    ])
                })
                .collect()
        })
        .collect()
}

fn fold_power_of_two(values: &mut [Field192], active: usize, challenge: Field192) -> usize {
    assert!(active > 1 && active.is_power_of_two() && active <= values.len());
    let half = active / 2;
    let (low, high) = values[..active].split_at_mut(half);
    low.par_iter_mut()
        .zip(high.par_iter())
        .for_each(|(low, high)| *low += (*high - *low) * challenge);
    half
}

fn evaluate_message(
    message: &[Field192],
    point: &[Field192],
    scratch: &mut [Field192],
) -> Field192 {
    assert_eq!(message.len(), scratch.len());
    assert_eq!(message.len(), 1_usize << point.len());
    scratch.copy_from_slice(message);
    let mut active = message.len();
    for challenge in point {
        active = fold_power_of_two(scratch, active, *challenge);
    }
    assert_eq!(active, 1);
    scratch[0]
}

fn statement_root(
    commitment: &[u8; 32],
    points: &[Vec<Field192>],
    claims: &[Field192],
) -> [u8; 32] {
    assert_eq!(points.len(), claims.len());
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/pre-Carry/batch-statement/v1");
    hasher.update(commitment);
    hasher.update(&(points.len() as u64).to_le_bytes());
    for (point, claim) in points.iter().zip(claims) {
        for coordinate in point {
            hasher.update(&field_bytes(*coordinate));
        }
        hasher.update(&field_bytes(*claim));
    }
    *hasher.finalize().as_bytes()
}

fn batching_coefficients(root: &[u8; 32], openings: usize) -> Vec<Field192> {
    (0..openings)
        .map(|opening| {
            let value = hash_to_field(&[
                b"LiLAC/pre-Carry/batch-coefficient/v1",
                root,
                &(opening as u64).to_le_bytes(),
            ]);
            if value == Field192::ZERO {
                Field192::ONE
            } else {
                value
            }
        })
        .collect()
}

// Fill alpha * chi_point over the Boolean cube. Processing coordinates in
// reverse lets each round split the live prefix in place. Each parent needs
// one multiplication: high = parent*r and low = parent-high.
fn scaled_equality_weights(point: &[Field192], alpha: Field192, target: &mut [Field192]) {
    assert_eq!(target.len(), 1_usize << point.len());
    target[0] = alpha;
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

fn dot(left: &[Field192], right: &[Field192]) -> Field192 {
    left.par_iter()
        .zip(right)
        .map(|(left, right)| *left * right)
        .reduce(|| Field192::ZERO, |left, right| left + right)
}

fn transcript_challenge(
    statement: &[u8; 32],
    claimed_sum: Field192,
    pairs: &[(Field192, Field192)],
) -> Field192 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/pre-Carry/product-sumcheck/v1");
    hasher.update(statement);
    hasher.update(&field_bytes(claimed_sum));
    for (constant, quadratic) in pairs {
        hasher.update(&field_bytes(*constant));
        hasher.update(&field_bytes(*quadratic));
    }
    Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes())
}

fn public_weight_at(
    points: &[Vec<Field192>],
    coefficients: &[Field192],
    evaluation_point: &[Field192],
) -> Field192 {
    points
        .iter()
        .zip(coefficients)
        .map(|(point, coefficient)| {
            point
                .iter()
                .zip(evaluation_point)
                .fold(*coefficient, |weight, (fixed, terminal)| {
                    weight
                        * ((Field192::ONE - *fixed) * (Field192::ONE - *terminal)
                            + *fixed * *terminal)
                })
        })
        .sum()
}

fn run(variables: usize, openings: usize) -> Measurement {
    assert!(variables > 0 && variables < usize::BITS as usize);
    assert!(openings > 1);
    let fields = 1_usize << variables;

    let prover_start = Instant::now();
    let start = Instant::now();
    let message = (0..fields)
        .into_par_iter()
        .map(message_value)
        .collect::<Vec<_>>();
    let commitment = commitment_root(&message);
    let points = opening_points(&commitment, openings, variables);
    let message_setup = start.elapsed();

    let start = Instant::now();
    let mut scratch = vec![Field192::ZERO; fields];
    let claims = points
        .iter()
        .map(|point| evaluate_message(&message, point, &mut scratch))
        .collect::<Vec<_>>();
    let claim_evaluation = start.elapsed();

    let statement = statement_root(&commitment, &points, &claims);
    let coefficients = batching_coefficients(&statement, openings);
    let batched_claim = claims
        .iter()
        .zip(&coefficients)
        .map(|(claim, coefficient)| *claim * coefficient)
        .sum::<Field192>();

    let start = Instant::now();
    let mut combined_weight = vec![Field192::ZERO; fields];
    for (point, coefficient) in points.iter().zip(&coefficients) {
        scaled_equality_weights(point, *coefficient, &mut scratch);
        combined_weight
            .par_iter_mut()
            .zip(scratch.par_iter())
            .for_each(|(combined, weight)| *combined += *weight);
    }
    let weight_construction = start.elapsed();
    drop(scratch);
    assert_eq!(dot(&message, &combined_weight), batched_claim);

    let mut left = message;
    let mut right = combined_weight;
    let mut claim = batched_claim;
    let mut pairs = Vec::with_capacity(variables);
    let mut challenges = Vec::with_capacity(variables);
    let mut polynomial = Duration::ZERO;
    let mut folding = Duration::ZERO;
    let mut active = fields;
    for _ in 0..variables {
        let start = Instant::now();
        let (constant, quadratic) = compute_sumcheck_polynomial(&left[..active], &right[..active]);
        polynomial += start.elapsed();
        pairs.push((constant, quadratic));
        let challenge = transcript_challenge(&statement, batched_claim, &pairs);
        challenges.push(challenge);
        let linear = claim - constant.double() - quadratic;

        let start = Instant::now();
        let next_left = fold_power_of_two(&mut left, active, challenge);
        let next_right = fold_power_of_two(&mut right, active, challenge);
        folding += start.elapsed();
        assert_eq!(next_left, next_right);
        active = next_left;
        claim = (quadratic * challenge + linear) * challenge + constant;
    }
    assert_eq!(active, 1);
    assert_eq!(claim, left[0] * right[0]);
    let prover_total = prover_start.elapsed();

    let verifier_start = Instant::now();
    let mut verifier_claim = batched_claim;
    let mut verifier_challenges = Vec::with_capacity(variables);
    for prefix in 1..=pairs.len() {
        let (constant, quadratic) = pairs[prefix - 1];
        let challenge = transcript_challenge(&statement, batched_claim, &pairs[..prefix]);
        let linear = verifier_claim - constant.double() - quadratic;
        verifier_claim = (quadratic * challenge + linear) * challenge + constant;
        verifier_challenges.push(challenge);
    }
    let expected_weight = public_weight_at(&points, &coefficients, &verifier_challenges);
    assert_eq!(verifier_challenges, challenges);
    assert_eq!(right[0], expected_weight);
    assert_eq!(verifier_claim, left[0] * expected_weight);
    let verifier = verifier_start.elapsed();

    let artifact = PreCarryArtifact {
        variables,
        openings,
        commitment,
        points,
        claims,
        pairs,
        terminal_message: left[0],
        terminal_weight: right[0],
    };
    assert!(artifact.verify());
    let payload = artifact.serialize();
    assert_eq!(
        PreCarryArtifact::deserialize(&payload),
        Some(artifact.clone())
    );
    let mut changed = payload;
    *changed.last_mut().unwrap() ^= 1;
    assert!(PreCarryArtifact::deserialize(&changed).is_none());

    Measurement {
        message_setup,
        claim_evaluation,
        weight_construction,
        polynomial,
        folding,
        prover_total,
        verifier,
        proof_bytes: 16 + (2 * variables + 2) * 24,
        checksum: left[0] + right[0] + verifier_claim,
        artifact,
    }
}

fn main() {
    let args = Args::parse();
    assert!(args.iterations > 0);
    let fields = 1_usize << args.variables;
    let mut setup_ms = Vec::with_capacity(args.iterations);
    let mut claims_ms = Vec::with_capacity(args.iterations);
    let mut weights_ms = Vec::with_capacity(args.iterations);
    let mut polynomial_ms = Vec::with_capacity(args.iterations);
    let mut folding_ms = Vec::with_capacity(args.iterations);
    let mut total_ms = Vec::with_capacity(args.iterations);
    let mut verifier_ms = Vec::with_capacity(args.iterations);
    let mut proof_bytes = 0;
    let mut artifact_bytes = 0;
    let mut checksum = Field192::ZERO;
    for _ in 0..args.iterations {
        let measurement = run(args.variables, args.openings);
        setup_ms.push(measurement.message_setup.as_secs_f64() * 1_000.0);
        claims_ms.push(measurement.claim_evaluation.as_secs_f64() * 1_000.0);
        weights_ms.push(measurement.weight_construction.as_secs_f64() * 1_000.0);
        polynomial_ms.push(measurement.polynomial.as_secs_f64() * 1_000.0);
        folding_ms.push(measurement.folding.as_secs_f64() * 1_000.0);
        total_ms.push(measurement.prover_total.as_secs_f64() * 1_000.0);
        verifier_ms.push(measurement.verifier.as_secs_f64() * 1_000.0);
        proof_bytes = measurement.proof_bytes;
        artifact_bytes = measurement.artifact.serialize().len();
        checksum += measurement.checksum;
    }

    println!("LiLAC production pre-Carry batch product-sumcheck");
    println!(
        "- fixed evaluations / variables / fields: {}/{}/{}",
        args.openings, args.variables, fields
    );
    println!("- Field192 element size: {} B", size_of::<Field192>());
    println!(
        "- peak three-vector storage: {:.3} GiB",
        3.0 * fields as f64 * size_of::<Field192>() as f64 / (1_u64 << 30) as f64
    );
    println!(
        "- message setup median/p95: {:.3}/{:.3} ms",
        percentile(&setup_ms, 0.5),
        percentile(&setup_ms, 0.95)
    );
    println!(
        "- sixteen claimed evaluations median/p95: {:.3}/{:.3} ms",
        percentile(&claims_ms, 0.5),
        percentile(&claims_ms, 0.95)
    );
    println!(
        "- random-linear weight construction median/p95: {:.3}/{:.3} ms",
        percentile(&weights_ms, 0.5),
        percentile(&weights_ms, 0.95)
    );
    println!(
        "- round-polynomial generation median/p95: {:.3}/{:.3} ms",
        percentile(&polynomial_ms, 0.5),
        percentile(&polynomial_ms, 0.95)
    );
    println!(
        "- in-place folding median/p95: {:.3}/{:.3} ms",
        percentile(&folding_ms, 0.5),
        percentile(&folding_ms, 0.95)
    );
    println!(
        "- end-to-end prover median/p95: {:.3}/{:.3} ms",
        percentile(&total_ms, 0.5),
        percentile(&total_ms, 0.95)
    );
    println!(
        "- verifier median/p95: {:.3}/{:.3} ms",
        percentile(&verifier_ms, 0.5),
        percentile(&verifier_ms, 0.95)
    );
    println!("- compressed arithmetic proof: {proof_bytes} B");
    println!("- canonical statement+proof artifact: {artifact_bytes} B");
    println!("- checksum nonzero: {}", checksum != Field192::ZERO);
    println!("- statement is bound to the actual Field192 Merkle root of M; the remaining terminal claim still requires the CarryOpen transition");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_weights_match_direct_boolean_values() {
        let point = vec![
            Field192::from(2_u64),
            Field192::from(3_u64),
            Field192::from(5_u64),
        ];
        let alpha = Field192::from(7_u64);
        let mut weights = vec![Field192::ZERO; 8];
        scaled_equality_weights(&point, alpha, &mut weights);
        for (index, weight) in weights.iter().enumerate() {
            let expected = point
                .iter()
                .enumerate()
                .fold(alpha, |value, (coordinate, r)| {
                    let bit = (index >> (point.len() - 1 - coordinate)) & 1;
                    value * if bit == 0 { Field192::ONE - *r } else { *r }
                });
            assert_eq!(*weight, expected);
        }
    }

    #[test]
    fn small_witness_backed_batch_verifies() {
        let measurement = run(8, 16);
        assert_eq!(measurement.proof_bytes, 448);
        assert_ne!(measurement.checksum, Field192::ZERO);
        let payload = measurement.artifact.serialize();
        assert_eq!(
            PreCarryArtifact::deserialize(&payload),
            Some(measurement.artifact)
        );
    }

    #[test]
    fn commitment_is_the_actual_field_merkle_root() {
        let mut message = (0..256).map(message_value).collect::<Vec<_>>();
        let root = commitment_root(&message);
        message[17] += Field192::ONE;
        assert_ne!(root, commitment_root(&message));
    }
}
