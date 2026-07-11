use std::time::Instant;

use clap::Parser;
use whir::algebra::{fields::Field192, ntt};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    message_length: usize,
    #[arg(long)]
    log_inv_rate: usize,
    #[arg(long, default_value_t = 8)]
    lanes: usize,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
}

fn percentile(values: &[f64], probability: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * probability).round() as usize]
}

fn main() {
    let args = Args::parse();
    assert!(args.message_length > 0 && args.lanes > 0 && args.iterations > 0);
    let requested = args
        .message_length
        .checked_shl(args.log_inv_rate as u32)
        .expect("requested codeword length overflow");
    let codeword_length = ntt::next_order::<Field192>(requested)
        .expect("requested codeword length exceeds Field192 NTT order");
    let messages = (0..args.lanes)
        .map(|lane| {
            (0..args.message_length)
                .map(|index| Field192::from((index + 17 * lane + 1) as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let references = messages.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut timings = Vec::with_capacity(args.iterations);
    let mut checksum = Field192::from(0_u64);
    for _ in 0..args.iterations {
        let start = Instant::now();
        let encoded = ntt::interleaved_rs_encode(&references, &[], codeword_length);
        timings.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert_eq!(encoded.len(), args.lanes * codeword_length);
        checksum += encoded[encoded.len() / 2];
    }
    let output_fields = args.lanes * codeword_length;
    let median = percentile(&timings, 0.5);
    let p95 = percentile(&timings, 0.95);
    println!("Field192 late RS encoder benchmark");
    println!("- message length: {}", args.message_length);
    println!("- requested inverse rate: 2^{}", args.log_inv_rate);
    println!("- requested/supported codeword length: {requested}/{codeword_length}");
    println!("- interleaved lanes: {}", args.lanes);
    println!("- output fields: {output_fields}");
    println!("- encode median/p95: {median:.3}/{p95:.3} ms");
    println!(
        "- median throughput: {:.3} million output fields/s",
        output_fields as f64 / median / 1_000.0
    );
    println!("- checksum nonzero: {}", checksum != Field192::from(0_u64));
}
