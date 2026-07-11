use std::time::{Duration, Instant};

use ark_ff::{Field, PrimeField, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use clap::Parser;
use rayon::prelude::*;
use whir::algebra::fields::Field192;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    message_length: usize,
    #[arg(long)]
    inverse_rate: usize,
    #[arg(long)]
    lanes: usize,
    #[arg(long, default_value_t = 64)]
    batch_lanes: usize,
    #[arg(long, default_value_t = 7)]
    iterations: usize,
}

fn percentile(values: &[f64], probability: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * probability).round() as usize]
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

fn encode_lane(message: &[Field192], generator_spectra: &[Vec<Field192>]) -> Vec<Field192> {
    let mut spectrum = message.to_vec();
    wht(&mut spectrum);
    let mut encoded = Vec::with_capacity(message.len() * (generator_spectra.len() + 1));
    encoded.extend_from_slice(message);
    for generator in generator_spectra {
        let mut block = spectrum
            .iter()
            .zip(generator)
            .map(|(value, multiplier)| *value * multiplier)
            .collect::<Vec<_>>();
        wht(&mut block);
        encoded.extend(block);
    }
    encoded
}

fn update_row_hashers(hashers: &mut [blake3::Hasher], encoded_lanes: &[Vec<Field192>]) {
    hashers
        .par_iter_mut()
        .enumerate()
        .for_each(|(position, hasher)| {
            for lane in encoded_lanes {
                let bigint = lane[position].into_bigint();
                for limb in bigint.as_ref() {
                    hasher.update(&limb.to_le_bytes());
                }
            }
        });
}

fn merkle_root(mut roots: Vec<[u8; 32]>) -> [u8; 32] {
    let padded = roots.len().next_power_of_two();
    let padding = *blake3::hash(b"TensorCarry/QA-row-padding/v1").as_bytes();
    roots.resize(padded, padding);
    while roots.len() > 1 {
        roots = roots
            .par_chunks_exact(2)
            .map(|pair| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"TensorCarry/QA-row-parent/v1");
                hasher.update(&pair[0]);
                hasher.update(&pair[1]);
                *hasher.finalize().as_bytes()
            })
            .collect();
    }
    roots[0]
}

fn run_iteration(
    args: &Args,
    padded: usize,
    generator_spectra: &[Vec<Field192>],
) -> (Duration, Duration, Duration, [u8; 32]) {
    let codeword_length = args.inverse_rate * padded;
    let mut row_hashers = (0..codeword_length)
        .map(|position| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"TensorCarry/QA-row/v1");
            hasher.update(&(position as u64).to_le_bytes());
            hasher
        })
        .collect::<Vec<_>>();
    let mut encoding = Duration::ZERO;
    let mut hashing = Duration::ZERO;

    for lane_start in (0..args.lanes).step_by(args.batch_lanes) {
        let lane_count = args.batch_lanes.min(args.lanes - lane_start);
        let messages = (0..lane_count)
            .map(|batch_lane| {
                let lane = lane_start + batch_lane;
                (0..padded)
                    .map(|index| {
                        if index < args.message_length {
                            Field192::from((index + 17 * lane + 1) as u64)
                        } else {
                            Field192::from(0_u64)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let start = Instant::now();
        let encoded = messages
            .par_iter()
            .map(|message| encode_lane(message, generator_spectra))
            .collect::<Vec<_>>();
        encoding += start.elapsed();

        let start = Instant::now();
        update_row_hashers(&mut row_hashers, &encoded);
        hashing += start.elapsed();
    }

    let start = Instant::now();
    let leaf_roots = row_hashers
        .into_par_iter()
        .map(|hasher| *hasher.finalize().as_bytes())
        .collect::<Vec<_>>();
    let root = merkle_root(leaf_roots);
    let merkle = start.elapsed();
    (encoding, hashing, merkle, root)
}

fn main() {
    let args = Args::parse();
    assert!(
        args.message_length > 0
            && args.inverse_rate >= 2
            && args.lanes > 0
            && args.batch_lanes > 0
            && args.iterations > 0
    );
    let padded = args.message_length.next_power_of_two();
    let mut rng = StdRng::seed_from_u64(0x5443_514d);
    let inverse_size = Field192::from(padded as u64).inverse().unwrap();
    let generator_spectra = (0..args.inverse_rate - 1)
        .map(|_| {
            (0..padded)
                .map(|_| Field192::rand(&mut rng) * inverse_size)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut encode_ms = Vec::with_capacity(args.iterations);
    let mut hash_ms = Vec::with_capacity(args.iterations);
    let mut merkle_ms = Vec::with_capacity(args.iterations);
    let mut checksum = 0_u8;
    for _ in 0..args.iterations {
        let (encoding, hashing, merkle, root) = run_iteration(&args, padded, &generator_spectra);
        encode_ms.push(encoding.as_secs_f64() * 1_000.0);
        hash_ms.push(hashing.as_secs_f64() * 1_000.0);
        merkle_ms.push(merkle.as_secs_f64() * 1_000.0);
        checksum ^= root[0];
    }
    let output_fields = args.lanes * args.inverse_rate * padded;
    println!("Field192 streaming QA encode-rowhash-Merkle benchmark");
    println!("- message/padded length: {}/{padded}", args.message_length);
    println!("- inverse rate: {}", args.inverse_rate);
    println!("- lanes/batch lanes: {}/{}", args.lanes, args.batch_lanes);
    println!("- output fields: {output_fields}");
    println!(
        "- encode median/p95: {:.3}/{:.3} ms",
        percentile(&encode_ms, 0.5),
        percentile(&encode_ms, 0.95)
    );
    println!(
        "- row-hash median/p95: {:.3}/{:.3} ms",
        percentile(&hash_ms, 0.5),
        percentile(&hash_ms, 0.95)
    );
    println!(
        "- Merkle median/p95: {:.3}/{:.3} ms",
        percentile(&merkle_ms, 0.5),
        percentile(&merkle_ms, 0.95)
    );
    let total = encode_ms
        .iter()
        .zip(&hash_ms)
        .zip(&merkle_ms)
        .map(|((encode, hash), merkle)| encode + hash + merkle)
        .collect::<Vec<_>>();
    println!(
        "- pipeline median/p95: {:.3}/{:.3} ms",
        percentile(&total, 0.5),
        percentile(&total, 0.95)
    );
    println!("- checksum byte: {checksum}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_root_is_deterministic_and_order_sensitive() {
        let roots = (0_u8..3)
            .map(|value| *blake3::hash(&[value]).as_bytes())
            .collect::<Vec<_>>();
        let original = merkle_root(roots.clone());
        assert_eq!(original, merkle_root(roots.clone()));
        let mut reversed = roots;
        reversed.reverse();
        assert_ne!(original, merkle_root(reversed));
    }

    #[test]
    fn streaming_hash_matches_manual_lane_order() {
        let encoded = vec![
            vec![Field192::from(1_u64), Field192::from(2_u64)],
            vec![Field192::from(3_u64), Field192::from(4_u64)],
        ];
        let mut hashers = (0..2)
            .map(|position| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"TensorCarry/QA-row/v1");
                hasher.update(&(position as u64).to_le_bytes());
                hasher
            })
            .collect::<Vec<_>>();
        update_row_hashers(&mut hashers, &encoded);
        assert_ne!(hashers[0].finalize(), hashers[1].finalize());
    }
}
