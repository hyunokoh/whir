use std::time::{Duration, Instant};

use ark_ff::{Field, PrimeField, UniformRand, Zero};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use clap::Parser;
use rayon::prelude::*;
use whir::algebra::fields::Field192;
use whir::lilac_merkle::{
    combine_equal_subtrees, field_leaf, prefix_root, zero_roots, Digest, MerkleAccumulator,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 64)]
    batch_lanes: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    #[arg(long, default_value_t = 5)]
    residual_iterations: usize,
}

const BLOCKS: usize = 207;
const BLOCK_SEMANTIC: usize = 28_711;
const BLOCK_CAPACITY: usize = 1 << 16;
const GROUP_SIZE: usize = 1 << 12;
const ROW_WIDTH: usize = 2_438;
const ROW_CAPACITY: usize = 1 << 12;
const ROWS_PER_BLOCK: usize = 16;
const VIEW_CAPACITY: usize = GROUP_SIZE * ROW_CAPACITY;
const COMPONENTS: [(usize, usize); 4] = [
    (16_406, 1 << 15),
    (8_203, 1 << 14),
    (4_102, 1 << 13),
    (0, 1 << 13),
];

fn percentile(values: &[f64], probability: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * probability).round() as usize]
}

fn transcript_challenge(label: &[u8], roots: &[Digest], index: usize) -> Field192 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/level0-post-commitment-challenge/v1");
    hasher.update(label);
    hasher.update(&(index as u64).to_le_bytes());
    for root in roots {
        hasher.update(root);
    }
    Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes())
}

fn equality_weights(point: &[Field192]) -> Vec<Field192> {
    let mut weights = vec![Field192::from(1_u64)];
    for coordinate in point {
        let one_minus = Field192::from(1_u64) - *coordinate;
        let mut next = Vec::with_capacity(weights.len() * 2);
        for weight in weights {
            next.push(weight * one_minus);
            next.push(weight * *coordinate);
        }
        weights = next;
    }
    weights
}

fn source_value(index: usize) -> Field192 {
    Field192::from(((index as u64 + 1) * 0x1_0001 + (index as u64 + 7) * 0x101) % 0xffff_ffff)
}

fn wht(values: &mut [Field192]) {
    assert!(values.len().is_power_of_two());
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

fn systematic_lane(source: &[Field192], lane: usize) -> Vec<Field192> {
    assert!(lane < ROW_WIDTH && source.len() == BLOCKS * BLOCK_SEMANTIC);
    let mut message = vec![Field192::from(0_u64); GROUP_SIZE];
    for block in 0..BLOCKS {
        let source_start = block * BLOCK_SEMANTIC;
        let target_start = block * ROWS_PER_BLOCK;
        for row in 0..ROWS_PER_BLOCK {
            let local = row * ROW_WIDTH + lane;
            if local < BLOCK_SEMANTIC {
                message[target_start + row] = source[source_start + local];
            }
        }
    }
    message
}

#[cfg(test)]
fn parity_lane(source: &[Field192], lane: usize, generator_spectrum: &[Field192]) -> Vec<Field192> {
    let mut message = systematic_lane(source, lane);
    wht(&mut message);
    for (value, multiplier) in message.iter_mut().zip(generator_spectrum) {
        *value *= multiplier;
    }
    wht(&mut message);
    message
}

fn splice_block_root(values: &[Field192], zeros: &[Digest]) -> Digest {
    let mut position = 0;
    let roots = COMPONENTS
        .iter()
        .map(|(semantic, capacity)| {
            let root = prefix_root(&values[position..position + semantic], *capacity, zeros);
            position += semantic;
            root
        })
        .collect::<Vec<_>>();
    let mut accumulator = MerkleAccumulator::new(BLOCK_CAPACITY.trailing_zeros() as usize);
    for (root, (_, capacity)) in roots.into_iter().zip(COMPONENTS) {
        accumulator.append_subtree(root, capacity.trailing_zeros() as usize);
    }
    accumulator.finish(BLOCK_CAPACITY, zeros)
}

fn systematic_block_root(values: &[Field192], zeros: &[Digest]) -> Digest {
    let mut rows = values
        .chunks(ROW_WIDTH)
        .map(|row| prefix_root(row, ROW_CAPACITY, zeros))
        .collect::<Vec<_>>();
    rows.resize(
        ROWS_PER_BLOCK,
        zeros[ROW_CAPACITY.trailing_zeros() as usize],
    );
    combine_equal_subtrees(&rows)
}

fn top_from_block_roots(roots: &[Digest], zeros: &[Digest]) -> Digest {
    let mut accumulator = MerkleAccumulator::new(VIEW_CAPACITY.trailing_zeros() as usize);
    let height = BLOCK_CAPACITY.trailing_zeros() as usize;
    for root in roots {
        accumulator.append_subtree(*root, height);
    }
    accumulator.finish(VIEW_CAPACITY, zeros)
}

fn splice_root_from_source(source: &[Field192], zeros: &[Digest]) -> Digest {
    let roots = source
        .par_chunks_exact(BLOCK_SEMANTIC)
        .map(|block| splice_block_root(block, zeros))
        .collect::<Vec<_>>();
    top_from_block_roots(&roots, zeros)
}

fn systematic_root(source: &[Field192], zeros: &[Digest]) -> Digest {
    let roots = source
        .par_chunks_exact(BLOCK_SEMANTIC)
        .map(|block| systematic_block_root(block, zeros))
        .collect::<Vec<_>>();
    top_from_block_roots(&roots, zeros)
}

fn parity_commitment(
    source: &[Field192],
    generator_spectrum: &[Field192],
    zeros: &[Digest],
    batch_lanes: usize,
) -> (Duration, Duration, Duration, Digest) {
    let mut accumulators = (0..GROUP_SIZE)
        .map(|_| MerkleAccumulator::new(ROW_CAPACITY.trailing_zeros() as usize))
        .collect::<Vec<_>>();
    let mut layout = Duration::ZERO;
    let mut encoding = Duration::ZERO;
    let mut hashing = Duration::ZERO;

    for lane_start in (0..ROW_WIDTH).step_by(batch_lanes) {
        let lane_end = (lane_start + batch_lanes).min(ROW_WIDTH);
        let start = Instant::now();
        let messages = (lane_start..lane_end)
            .into_par_iter()
            .map(|lane| systematic_lane(source, lane))
            .collect::<Vec<_>>();
        layout += start.elapsed();

        let start = Instant::now();
        let parities = messages
            .into_par_iter()
            .map(|mut message| {
                wht(&mut message);
                for (value, multiplier) in message.iter_mut().zip(generator_spectrum) {
                    *value *= multiplier;
                }
                wht(&mut message);
                message
            })
            .collect::<Vec<_>>();
        encoding += start.elapsed();

        let start = Instant::now();
        accumulators
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, accumulator)| {
                for parity in &parities {
                    accumulator.append_leaf(field_leaf(parity[row]));
                }
            });
        hashing += start.elapsed();
    }

    let start = Instant::now();
    let row_roots = accumulators
        .into_par_iter()
        .map(|accumulator| accumulator.finish(ROW_CAPACITY, zeros))
        .collect::<Vec<_>>();
    let root = combine_equal_subtrees(&row_roots);
    hashing += start.elapsed();
    (layout, encoding, hashing, root)
}

fn transpose_encoding_weights(
    beta_systematic: &[Field192],
    beta_parity: &[Field192],
    generator_spectrum: &[Field192],
) -> Vec<Field192> {
    let mut transformed = beta_parity.to_vec();
    wht(&mut transformed);
    for (value, multiplier) in transformed.iter_mut().zip(generator_spectrum) {
        *value *= multiplier;
    }
    wht(&mut transformed);
    transformed
        .into_iter()
        .zip(beta_systematic)
        .map(|(parity, systematic)| parity + *systematic)
        .collect()
}

fn membership_residual(
    source: &[Field192],
    generator_spectrum: &[Field192],
    beta: &[Field192],
    alpha: &[Field192],
) -> Field192 {
    let (beta_systematic, beta_parity) = beta.split_at(GROUP_SIZE);
    let message_weights =
        transpose_encoding_weights(beta_systematic, beta_parity, generator_spectrum);
    (0..ROW_WIDTH)
        .into_par_iter()
        .map(|lane| {
            let message = systematic_lane(source, lane);
            let mut parity = message.clone();
            wht(&mut parity);
            for (value, multiplier) in parity.iter_mut().zip(generator_spectrum) {
                *value *= multiplier;
            }
            wht(&mut parity);
            let left = message
                .iter()
                .zip(beta_systematic)
                .fold(Field192::zero(), |sum, (value, weight)| {
                    sum + *value * weight
                })
                + parity
                    .iter()
                    .zip(beta_parity)
                    .fold(Field192::zero(), |sum, (value, weight)| {
                        sum + *value * weight
                    });
            let right = message
                .iter()
                .zip(&message_weights)
                .fold(Field192::zero(), |sum, (value, weight)| {
                    sum + *value * weight
                });
            alpha[lane] * (left - right)
        })
        .reduce(Field192::zero, |left, right| left + right)
}

fn main() {
    let args = Args::parse();
    assert!(args.batch_lanes > 0 && args.iterations > 0 && args.residual_iterations > 0);
    let zeros = zero_roots(VIEW_CAPACITY.trailing_zeros() as usize);
    let source = (0..BLOCKS * BLOCK_SEMANTIC)
        .into_par_iter()
        .map(source_value)
        .collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(0x4c49_4c41_4351_4130);
    let inverse_size = Field192::from(GROUP_SIZE as u64).inverse().unwrap();
    let generator_spectrum = (0..GROUP_SIZE)
        .map(|_| Field192::rand(&mut rng) * inverse_size)
        .collect::<Vec<_>>();
    let generator_root = prefix_root(&generator_spectrum, GROUP_SIZE, &zeros);

    // Extractor/setup audit: recompute the old subtree view once. The current
    // prover normally reuses these authenticated roots and derives c_splice.
    let start = Instant::now();
    let splice_root = splice_root_from_source(&source, &zeros);
    let splice_setup = start.elapsed();

    let mut systematic_ms = Vec::with_capacity(args.iterations);
    let mut layout_ms = Vec::with_capacity(args.iterations);
    let mut encoding_ms = Vec::with_capacity(args.iterations);
    let mut parity_hash_ms = Vec::with_capacity(args.iterations);
    let mut total_ms = Vec::with_capacity(args.iterations);
    let mut last_roots = None;
    let mut checksum = 0_u8;
    for _ in 0..args.iterations {
        let start = Instant::now();
        let systematic_root = systematic_root(&source, &zeros);
        let systematic = start.elapsed();
        let (layout, encoding, parity_hash, parity_root) =
            parity_commitment(&source, &generator_spectrum, &zeros, args.batch_lanes);
        assert_ne!(splice_root, systematic_root);
        assert_ne!(systematic_root, parity_root);
        checksum ^= splice_root[0] ^ systematic_root[0] ^ parity_root[0];
        last_roots = Some([splice_root, systematic_root, parity_root, generator_root]);
        systematic_ms.push(systematic.as_secs_f64() * 1_000.0);
        layout_ms.push(layout.as_secs_f64() * 1_000.0);
        encoding_ms.push(encoding.as_secs_f64() * 1_000.0);
        parity_hash_ms.push(parity_hash.as_secs_f64() * 1_000.0);
        total_ms.push((systematic + layout + encoding + parity_hash).as_secs_f64() * 1_000.0);
    }

    let roots = last_roots.unwrap();
    let row_point = (0..13)
        .map(|index| transcript_challenge(b"row", &roots, index))
        .collect::<Vec<_>>();
    let lane_point = (0..12)
        .map(|index| transcript_challenge(b"lane", &roots, index))
        .collect::<Vec<_>>();
    let start = Instant::now();
    let beta = equality_weights(&row_point);
    let alpha = equality_weights(&lane_point);
    let challenge_weights = start.elapsed();
    let mut residual_ms = Vec::with_capacity(args.residual_iterations);
    for _ in 0..args.residual_iterations {
        let start = Instant::now();
        let residual =
            membership_residual(&source, &generator_spectrum, &beta, &alpha[..ROW_WIDTH]);
        residual_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert!(residual.is_zero());
    }

    println!("LiLAC level-0 QA dual-view systematic/parity commitment pipeline");
    println!(
        "- raw/systematic capacity: {}/{} fields",
        source.len(),
        VIEW_CAPACITY
    );
    println!("- QA group/rate/lanes: {GROUP_SIZE}/1/2/{ROW_WIDTH}");
    println!("- lane batch: {}", args.batch_lanes);
    println!(
        "- old splice subtree recomputation (setup/extractor): {:.3} ms",
        splice_setup.as_secs_f64() * 1_000.0
    );
    println!(
        "- reused splice + systematic row root median/p95: {:.3}/{:.3} ms",
        percentile(&systematic_ms, 0.5),
        percentile(&systematic_ms, 0.95)
    );
    println!(
        "- systematic lane layout median/p95: {:.3}/{:.3} ms",
        percentile(&layout_ms, 0.5),
        percentile(&layout_ms, 0.95)
    );
    println!(
        "- QA parity encoding median/p95: {:.3}/{:.3} ms",
        percentile(&encoding_ms, 0.5),
        percentile(&encoding_ms, 0.95)
    );
    println!(
        "- parity field-Merkle median/p95: {:.3}/{:.3} ms",
        percentile(&parity_hash_ms, 0.5),
        percentile(&parity_hash_ms, 0.95)
    );
    println!(
        "- current-prover three-root pipeline median/p95: {:.3}/{:.3} ms",
        percentile(&total_ms, 0.5),
        percentile(&total_ms, 0.95)
    );
    println!("- splice/systematic/parity roots distinct: true");
    println!(
        "- post-root equality-weight generation: {:.3} ms",
        challenge_weights.as_secs_f64() * 1_000.0
    );
    println!(
        "- post-root QA membership residual median/p95: {:.3}/{:.3} ms",
        percentile(&residual_ms, 0.5),
        percentile(&residual_ms, 0.95)
    );
    println!("- checksum byte: {checksum}");
    println!("- excludes copy residual, sumcheck polynomial generation, and recursive NIRK proof");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_layout_matches_block_repacking() {
        let source = (0..BLOCKS * BLOCK_SEMANTIC)
            .map(source_value)
            .collect::<Vec<_>>();
        for lane in [0, ROW_WIDTH - 1] {
            let message = systematic_lane(&source, lane);
            for block in [0, BLOCKS - 1] {
                let target = block * ROWS_PER_BLOCK;
                assert_eq!(message[target], source[block * BLOCK_SEMANTIC + lane]);
                assert_eq!(message[target + ROWS_PER_BLOCK - 1], Field192::from(0_u64));
            }
        }
    }

    #[test]
    fn wht_squared_is_scaled_identity() {
        let original = (1_u64..=8).map(Field192::from).collect::<Vec<_>>();
        let mut transformed = original.clone();
        wht(&mut transformed);
        wht(&mut transformed);
        let scale = Field192::from(original.len() as u64);
        assert_eq!(
            transformed,
            original
                .iter()
                .map(|value| *value * scale)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn parity_lane_changes_after_source_mutation() {
        let mut source = (0..BLOCKS * BLOCK_SEMANTIC)
            .map(source_value)
            .collect::<Vec<_>>();
        let spectrum = vec![Field192::from(1_u64); GROUP_SIZE];
        let original = parity_lane(&source, 0, &spectrum);
        source[0] += Field192::from(1_u64);
        assert_ne!(original, parity_lane(&source, 0, &spectrum));
    }

    #[test]
    fn transpose_weights_match_parity_inner_product() {
        let message = (1_u64..=8).map(Field192::from).collect::<Vec<_>>();
        let beta = (11_u64..=18).map(Field192::from).collect::<Vec<_>>();
        let inverse = Field192::from(8_u64).inverse().unwrap();
        let spectrum = (21_u64..=28)
            .map(|value| Field192::from(value) * inverse)
            .collect::<Vec<_>>();
        let mut parity = message.clone();
        wht(&mut parity);
        for (value, multiplier) in parity.iter_mut().zip(&spectrum) {
            *value *= multiplier;
        }
        wht(&mut parity);
        let transformed = transpose_encoding_weights(&vec![Field192::zero(); 8], &beta, &spectrum);
        let left = parity
            .iter()
            .zip(&beta)
            .fold(Field192::zero(), |sum, (value, weight)| {
                sum + *value * weight
            });
        let right = message
            .iter()
            .zip(&transformed)
            .fold(Field192::zero(), |sum, (value, weight)| {
                sum + *value * weight
            });
        assert_eq!(left, right);
    }
}
