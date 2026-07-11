use std::time::{Duration, Instant};

use ark_ff::PrimeField;
use clap::Parser;
use rayon::prelude::*;
use whir::algebra::fields::Field192;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 205)]
    queries: usize,
    #[arg(long, default_value_t = 7)]
    iterations: usize,
}

const SOURCE_WIDTHS: [usize; 3] = [4_102, 8_203, 16_406];

fn percentile(values: &[f64], probability: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * probability).round() as usize]
}

fn canonical_update(hasher: &mut blake3::Hasher, value: Field192) {
    let bigint = value.into_bigint();
    for limb in bigint.as_ref() {
        hasher.update(&limb.to_le_bytes());
    }
}

fn merkle_root(mut roots: Vec<[u8; 32]>) -> [u8; 32] {
    roots.resize(
        roots.len().next_power_of_two(),
        *blake3::hash(b"LiLAC/CarryLin/bundle-padding/v1").as_bytes(),
    );
    while roots.len() > 1 {
        roots = roots
            .par_chunks_exact(2)
            .map(|pair| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"LiLAC/CarryLin/bundle-parent/v1");
                hasher.update(&pair[0]);
                hasher.update(&pair[1]);
                *hasher.finalize().as_bytes()
            })
            .collect();
    }
    roots[0]
}

fn run_iteration(args: &Args) -> (Duration, Duration, Duration, [u8; 32]) {
    let row_width: usize = SOURCE_WIDTHS.iter().sum();
    // The half-distance first state contains q selected bundle rows and two
    // fresh fold/weight rows of the same flattened width.
    let rows = args.queries + 2;

    let start = Instant::now();
    let state = (0..rows)
        .into_par_iter()
        .map(|row| {
            (0..row_width)
                .map(|column| {
                    Field192::from(
                        ((row as u64 + 1) * 0x1_0001 + (column as u64 + 3) * 0x101) % 0xffff_ffff,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let materialize = start.elapsed();

    let start = Instant::now();
    let residuals = state
        .par_iter()
        .enumerate()
        .map(|(row, values)| {
            let mut left = Field192::from(0_u64);
            let mut right = Field192::from(0_u64);
            for (column, value) in values.iter().enumerate() {
                let coefficient = Field192::from(
                    ((row as u64 + 5) * 17 + (column as u64 + 11) * 29) % 0xffff_ffff,
                );
                left += *value * coefficient;
                right += (*value + Field192::from(1_u64)) * coefficient;
            }
            (left, right)
        })
        .collect::<Vec<_>>();
    let residual = start.elapsed();

    let start = Instant::now();
    let roots = state
        .par_iter()
        .enumerate()
        .map(|(row, values)| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"LiLAC/CarryLin/carried-bundle-row/v1");
            hasher.update(&(row as u64).to_le_bytes());
            for value in values {
                canonical_update(&mut hasher, *value);
            }
            // Bind the two residuals to keep the arithmetic pass live.
            canonical_update(&mut hasher, residuals[row].0);
            canonical_update(&mut hasher, residuals[row].1);
            *hasher.finalize().as_bytes()
        })
        .collect::<Vec<_>>();
    let root = merkle_root(roots);
    let hash_merkle = start.elapsed();
    (materialize, residual, hash_merkle, root)
}

fn main() {
    let args = Args::parse();
    assert!(args.queries > 0 && args.iterations > 0);
    let row_width: usize = SOURCE_WIDTHS.iter().sum();
    let fields = (args.queries + 2) * row_width;
    let mut materialize_ms = Vec::with_capacity(args.iterations);
    let mut residual_ms = Vec::with_capacity(args.iterations);
    let mut hash_ms = Vec::with_capacity(args.iterations);
    let mut checksum = 0_u8;
    for _ in 0..args.iterations {
        let (materialize, residual, hash_merkle, root) = run_iteration(&args);
        materialize_ms.push(materialize.as_secs_f64() * 1_000.0);
        residual_ms.push(residual.as_secs_f64() * 1_000.0);
        hash_ms.push(hash_merkle.as_secs_f64() * 1_000.0);
        checksum ^= root[0];
    }
    let total = materialize_ms
        .iter()
        .zip(&residual_ms)
        .zip(&hash_ms)
        .map(|((materialize, residual), hash)| materialize + residual + hash)
        .collect::<Vec<_>>();
    println!("LiLAC Field192 CarryLin selected-bundle pass");
    println!("- source widths: {SOURCE_WIDTHS:?} (flattened {row_width})");
    println!("- selected/fresh rows: {}/2", args.queries);
    println!("- carried state fields: {fields}");
    println!(
        "- materialize median/p95: {:.3}/{:.3} ms",
        percentile(&materialize_ms, 0.5),
        percentile(&materialize_ms, 0.95)
    );
    println!(
        "- two-residual pass median/p95: {:.3}/{:.3} ms",
        percentile(&residual_ms, 0.5),
        percentile(&residual_ms, 0.95)
    );
    println!(
        "- row-hash+Merkle median/p95: {:.3}/{:.3} ms",
        percentile(&hash_ms, 0.5),
        percentile(&hash_ms, 0.95)
    );
    println!(
        "- combined median/p95: {:.3}/{:.3} ms",
        percentile(&total, 0.5),
        percentile(&total, 0.95)
    );
    println!("- checksum byte: {checksum}");
    println!("- excludes source-oracle construction and the successor QA encoding");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_half_distance_first_state() {
        let row_width: usize = SOURCE_WIDTHS.iter().sum();
        assert_eq!(row_width, 28_711);
        assert_eq!((205 + 2) * row_width, 5_943_177);
    }

    #[test]
    fn merkle_is_order_sensitive() {
        let a = *blake3::hash(b"a").as_bytes();
        let b = *blake3::hash(b"b").as_bytes();
        assert_ne!(merkle_root(vec![a, b]), merkle_root(vec![b, a]));
    }
}
