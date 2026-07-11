use std::time::Instant;

use ark_ff::{Field, UniformRand};
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
    #[arg(long, default_value_t = 8)]
    lanes: usize,
    #[arg(long, default_value_t = 15)]
    iterations: usize,
}

fn percentile(values: &[f64], probability: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * probability).round() as usize]
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

fn main() {
    let args = Args::parse();
    assert!(
        args.message_length > 0 && args.inverse_rate >= 2 && args.lanes > 0 && args.iterations > 0
    );
    let padded = args.message_length.next_power_of_two();
    let mut rng = StdRng::seed_from_u64(0x5443_5141);
    let inverse_size = Field192::from(padded as u64)
        .inverse()
        .expect("nonzero WHT size");
    // Include the inverse WHT scale in the fixed generator spectrum, so each
    // parity block costs one pointwise multiplication and one WHT.
    let generator_spectra = (0..args.inverse_rate - 1)
        .map(|_| {
            (0..padded)
                .map(|_| Field192::rand(&mut rng) * inverse_size)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let messages = (0..args.lanes)
        .map(|lane| {
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
    let mut timings = Vec::with_capacity(args.iterations);
    let mut checksum = Field192::from(0_u64);
    for _ in 0..args.iterations {
        let start = Instant::now();
        let encoded = messages
            .par_iter()
            .map(|message| encode_lane(message, &generator_spectra))
            .collect::<Vec<_>>();
        timings.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert!(encoded
            .iter()
            .all(|lane| lane.len() == args.inverse_rate * padded));
        checksum += encoded[encoded.len() / 2][encoded[0].len() / 2];
    }
    let output_fields = args.lanes * args.inverse_rate * padded;
    let median = percentile(&timings, 0.5);
    let p95 = percentile(&timings, 0.95);
    println!("Field192 quasi-Abelian WHT encoder benchmark");
    println!("- message/padded length: {}/{padded}", args.message_length);
    println!("- inverse rate: {}", args.inverse_rate);
    println!("- interleaved lanes: {}", args.lanes);
    println!("- output fields: {output_fields}");
    println!("- encode median/p95: {median:.3}/{p95:.3} ms");
    println!(
        "- median throughput: {:.3} million output fields/s",
        output_fields as f64 / median / 1_000.0
    );
    println!("- checksum nonzero: {}", checksum != Field192::from(0_u64));
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn spectral_encoder_matches_xor_convolution() {
        let message = [1_u64, 2, 4, 8]
            .into_iter()
            .map(Field192::from)
            .collect::<Vec<_>>();
        let generator = [3_u64, 5, 7, 11]
            .into_iter()
            .map(Field192::from)
            .collect::<Vec<_>>();
        let mut spectrum = generator.clone();
        wht(&mut spectrum);
        let inverse_size = Field192::from(message.len() as u64).inverse().unwrap();
        for value in &mut spectrum {
            *value *= inverse_size;
        }
        let encoded = encode_lane(&message, &[spectrum]);
        assert_eq!(&encoded[..message.len()], message);
        let direct = (0..message.len())
            .map(|output| {
                (0..message.len()).fold(Field192::from(0_u64), |sum, input| {
                    sum + message[input] * generator[output ^ input]
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(&encoded[message.len()..], direct);
    }
}
