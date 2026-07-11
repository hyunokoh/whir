use std::{borrow::Cow, time::Instant};

use ark_ff::FftField;
use ark_std::rand::distributions::{Distribution, Standard};
use clap::Parser;
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field128, Field192, Field256, Field64, Field64_2, Field64_3},
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension, SparseCovector},
        MultilinearPoint,
    },
    bits::Bits,
    cmdline_utils::{AvailableFields, AvailableHash},
    hash::HASH_COUNTER,
    parameters::ProtocolParameters,
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "128")]
    security_level: usize,

    /// Maximum proof of work difficulty in bits.
    #[arg(short = 'p', long, default_value = "20")]
    pow_bits: usize,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(long = "linear-constraints", default_value = "0")]
    num_linear_constraints: usize,

    /// Number of sparse arbitrary linear constraints.
    #[arg(long = "sparse-linear-constraints", default_value = "0")]
    num_sparse_linear_constraints: usize,

    /// Number of nonzero coefficients in each sparse linear constraint.
    #[arg(long = "sparse-width", default_value = "17")]
    sparse_width: usize,

    /// Number of source-field vectors committed in each Merkle leaf.
    #[arg(long = "batch-size", default_value = "1")]
    batch_size: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(long = "reps", default_value = "1000")]
    verifier_repetitions: usize,

    #[arg(short = 'i', long = "initfold", default_value = "4")]
    first_round_folding_factor: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    /// Restrict PCS to the Unique Decoding regime. LDT is always UD.
    #[arg(long = "unique-decoding", default_value_t = false)]
    unique_decoding: bool,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks3")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    hash: AvailableHash,

    /// Domain-separation seed for reproducible transcript distributions.
    #[arg(long = "session-seed", default_value = "0")]
    session_seed: u64,

    /// Use LiLAC's exact 19-product packed-terminal endpoint positions and two
    /// transcript-derived sparse forms. Intended for the 2^21 Field192 audit.
    #[arg(long = "lilac-terminal-layout", default_value_t = false)]
    lilac_terminal_layout: bool,

    /// Number of packed CarryOpen terminal segments in LiLAC layout mode.
    #[arg(long = "lilac-opening-count", default_value = "16")]
    lilac_opening_count: usize,

    #[arg(long = "zk")]
    zk: bool,
}

const LILAC_CERTIFICATE_CAPACITY: usize = 1 << 18;
const LILAC_OPENING_CAPACITY: usize = 1 << 16;

fn lilac_terminal_pairs(opening_count: usize) -> Vec<(usize, usize)> {
    assert!(opening_count > 0);
    let mut pairs = vec![(2, 3), (4, 5), (6, 7)];
    pairs.extend((0..opening_count).map(|opening| {
        let left = LILAC_CERTIFICATE_CAPACITY + opening * LILAC_OPENING_CAPACITY;
        (left, left + 1)
    }));
    pairs
}

fn lilac_terminal_domain(opening_count: usize) -> usize {
    (LILAC_CERTIFICATE_CAPACITY + opening_count * LILAC_OPENING_CAPACITY).next_power_of_two()
}

fn lilac_scalar(label: &[u8], seed: u64, ordinal: usize) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/packed-terminal-layout/v1");
    hasher.update(label);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&ordinal.to_le_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    u64::from_le_bytes(bytes).max(1)
}

fn lilac_layout_digest(seed: u64, opening_count: usize) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/packed-terminal-layout-digest/v1");
    hasher.update(&seed.to_le_bytes());
    hasher.update(&opening_count.to_le_bytes());
    for (ordinal, (left, right)) in lilac_terminal_pairs(opening_count).into_iter().enumerate() {
        hasher.update(&left.to_le_bytes());
        hasher.update(&right.to_le_bytes());
        hasher.update(&lilac_scalar(b"left-form", seed, ordinal).to_le_bytes());
        hasher.update(&lilac_scalar(b"right-form", seed, ordinal).to_le_bytes());
        hasher.update(&lilac_scalar(b"left-value", seed, ordinal).to_le_bytes());
        hasher.update(&lilac_scalar(b"right-value", seed, ordinal).to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn main() {
    use AvailableFields as AF;
    let args = Args::parse();
    let field = args.field;

    // Dispatch on embedding
    if args.zk {
        #[cfg(not(feature = "rs_in_order"))]
        panic!("ZK requires --features rs_in_order");
        #[cfg(feature = "rs_in_order")]
        match field {
            AF::Goldilocks1 => run_whir_zk::<Field64>(&args),
            AF::Goldilocks2 => run_whir_zk::<Field64_2>(&args),
            AF::Goldilocks3 => run_whir_zk::<Field64_3>(&args),
            AF::Field128 => run_whir_zk::<Field128>(&args),
            AF::Field192 => run_whir_zk::<Field192>(&args),
            AF::Field256 => run_whir_zk::<Field256>(&args),
        }
    } else {
        match field {
            AF::Goldilocks1 => run_whir::<Identity<Field64>>(&args),
            AF::Goldilocks2 => run_whir::<Basefield<Field64_2>>(&args),
            AF::Goldilocks3 => run_whir::<Basefield<Field64_3>>(&args),
            AF::Field128 => run_whir::<Identity<Field128>>(&args),
            AF::Field192 => run_whir::<Identity<Field192>>(&args),
            AF::Field256 => run_whir::<Identity<Field256>>(&args),
        }
    }
}

#[allow(clippy::too_many_lines)]
fn run_whir<M>(args: &Args)
where
    Standard: Distribution<M::Source> + Distribution<M::Target>,
    M: Embedding + Default,
    M::Source: FftField,
    M::Target: FftField + Codec,
{
    use whir::protocols::whir::Config;

    // Runs as a PCS
    let security_level = args.security_level;
    let pow_bits = args.pow_bits;
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let unique_decoding = args.unique_decoding;
    let num_evaluations = if args.lilac_terminal_layout {
        0
    } else {
        args.num_evaluations
    };
    let num_linear_constraints = if args.lilac_terminal_layout {
        0
    } else {
        args.num_linear_constraints
    };
    let num_sparse_linear_constraints = if args.lilac_terminal_layout {
        0
    } else {
        args.num_sparse_linear_constraints
    };
    let hash_id = args.hash.hash_id();
    let batch_size = args.batch_size;

    if num_evaluations + num_linear_constraints + num_sparse_linear_constraints == 0
        && !args.lilac_terminal_layout
    {
        println!("No constraints specified, running as low-degree-test.");
    }

    let num_coeffs = 1 << num_variables;

    let whir_params = ProtocolParameters {
        security_level,
        pow_bits,
        initial_folding_factor: first_round_folding_factor,
        folding_factor,
        unique_decoding,
        starting_log_inv_rate: starting_rate,
        batch_size,
        hash_id,
    };

    let params = Config::<M>::new(1 << num_variables, &whir_params);

    let ds = DomainSeparator::protocol(&params)
        .session(&format!(
            "Example at {}:{} seed={} lilac-layout={} layout-digest={}",
            file!(),
            line!(),
            args.session_seed,
            args.lilac_terminal_layout,
            if args.lilac_terminal_layout {
                lilac_layout_digest(args.session_seed, args.lilac_opening_count)
            } else {
                String::from("none")
            }
        ))
        .instance(&Empty);

    let mut prover_state = ProverState::new_std(&ds);

    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params.check_max_pow_bits(Bits::new(whir_params.pow_bits as f64)) {
        println!("WARN: more PoW bits required than specified.");
    }

    assert!(batch_size > 0);
    let vectors = if args.lilac_terminal_layout {
        assert_eq!(
            batch_size, 1,
            "LiLAC terminal layout uses one packed vector"
        );
        assert_eq!(
            num_coeffs,
            lilac_terminal_domain(args.lilac_opening_count),
            "LiLAC terminal layout domain does not match opening count"
        );
        let mut vector = vec![M::Source::from(0_u64); num_coeffs];
        for (ordinal, (left, right)) in lilac_terminal_pairs(args.lilac_opening_count)
            .into_iter()
            .enumerate()
        {
            vector[left] = M::Source::from(lilac_scalar(b"left-value", args.session_seed, ordinal));
            vector[right] =
                M::Source::from(lilac_scalar(b"right-value", args.session_seed, ordinal));
        }
        vec![vector]
    } else {
        (0..batch_size)
            .map(|lane| {
                (0..num_coeffs)
                    .map(|value| M::Source::from((value + lane) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let vector_refs = vectors.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let whir_commit_time = Instant::now();
    let witness = params.commit(&mut prover_state, &vector_refs);
    let whir_commit_time = whir_commit_time.elapsed();

    // Allocate constraints
    let mut linear_forms: Vec<Box<dyn Evaluate<M>>> = Vec::new();
    let mut prove_linear_forms: Vec<Box<dyn LinearForm<M::Target>>> = Vec::new();
    let mut evaluations = Vec::new();

    // Linear constraint
    // We do these first to benefit from buffer recycling.
    for _ in 0..num_linear_constraints {
        let linear_form = Box::new(Covector {
            vector: (0..num_coeffs)
                .map(|value| M::Target::from(value as u64))
                .collect(),
        });
        evaluations.extend(
            vectors
                .iter()
                .map(|vector| linear_form.evaluate(params.embedding(), vector)),
        );
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    for constraint in 0..num_sparse_linear_constraints {
        assert!(args.sparse_width <= num_coeffs);
        let entries = (0..args.sparse_width)
            .map(|entry| {
                let index = (constraint + entry * 65_537) % num_coeffs;
                (index, M::Target::from((entry + 1) as u64))
            })
            .collect();
        let linear_form = Box::new(SparseCovector::new(num_coeffs, entries));
        evaluations.extend(
            vectors
                .iter()
                .map(|vector| linear_form.evaluate(params.embedding(), vector)),
        );
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    if args.lilac_terminal_layout {
        for (label, side) in [
            (b"left-form".as_slice(), 0_usize),
            (b"right-form".as_slice(), 1),
        ] {
            let entries = lilac_terminal_pairs(args.lilac_opening_count)
                .into_iter()
                .enumerate()
                .map(|(ordinal, pair)| {
                    let position = if side == 0 { pair.0 } else { pair.1 };
                    (
                        position,
                        M::Target::from(lilac_scalar(label, args.session_seed, ordinal)),
                    )
                })
                .collect();
            let linear_form = Box::new(SparseCovector::new(num_coeffs, entries));
            evaluations.extend(
                vectors
                    .iter()
                    .map(|vector| linear_form.evaluate(params.embedding(), vector)),
            );
            linear_forms.push(linear_form.clone());
            prove_linear_forms.push(linear_form);
        }
        let width = 3 + args.lilac_opening_count;
        println!("LiLAC terminal layout: {width} endpoint pairs, two width-{width} sparse forms");
        println!(
            "LiLAC layout digest: {}",
            lilac_layout_digest(args.session_seed, args.lilac_opening_count)
        );
    }

    // Evaluation constraint
    let points: Vec<_> = (0..num_evaluations)
        .map(|x| MultilinearPoint(vec![M::Target::from(x as u64); num_variables]))
        .collect();
    for point in &points {
        let linear_form = Box::new(MultilinearExtension::new(point.0.clone()));
        evaluations.extend(
            vectors
                .iter()
                .map(|vector| linear_form.evaluate(params.embedding(), vector)),
        );
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    let whir_prove_time = Instant::now();
    let _ = params.prove(
        &mut prover_state,
        vectors
            .iter()
            .map(|vector| Cow::Borrowed(vector.as_slice()))
            .collect(),
        vec![Cow::Owned(witness)],
        prove_linear_forms,
        Cow::Borrowed(evaluations.as_slice()),
    );
    let whir_prove_time = whir_prove_time.elapsed();

    let proof = prover_state.proof();
    println!(
        "Prover time: {whir_commit_time:.1?} + {whir_prove_time:.1?} = {:.1?}",
        whir_commit_time + whir_prove_time,
    );
    println!(
        "Proof size: {:.1} KiB",
        (proof.narg_string.len() + proof.hints.len()) as f64 / 1024.0
    );

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let commitment = params.receive_commitment(&mut verifier_state).unwrap();
        let final_claim = params
            .verify(&mut verifier_state, &[&commitment], &evaluations)
            .unwrap();
        final_claim
            .verify(
                linear_forms
                    .iter()
                    .map(|w| w.as_ref() as &dyn LinearForm<M::Target>),
            )
            .unwrap();
    }
    println!(
        "Verifier time: {:.1?}",
        whir_verifier_time.elapsed() / reps as u32
    );
    println!(
        "Average hashes: {:.1}k",
        (HASH_COUNTER.get() as f64 / reps as f64) / 1000.0
    );
}

#[cfg(test)]
mod lilac_terminal_layout_tests {
    use super::*;

    #[test]
    fn exact_endpoint_positions_match_batch_contract() {
        let pairs = lilac_terminal_pairs(16);
        assert_eq!(pairs.len(), 19);
        assert_eq!(&pairs[..3], &[(2, 3), (4, 5), (6, 7)]);
        assert_eq!(pairs[3], (1 << 18, (1 << 18) + 1));
        assert_eq!(
            pairs[18],
            ((1 << 18) + 15 * (1 << 16), (1 << 18) + 15 * (1 << 16) + 1)
        );
        assert!(pairs
            .iter()
            .all(|(_, right)| *right < lilac_terminal_domain(16)));
        assert_eq!(lilac_terminal_domain(16), 1 << 21);
        assert_eq!(lilac_terminal_domain(1), 1 << 19);
        assert_eq!(lilac_terminal_pairs(1).len(), 4);
    }

    #[test]
    fn transcript_seed_changes_layout_digest() {
        assert_ne!(lilac_layout_digest(0, 16), lilac_layout_digest(1, 16));
        assert_ne!(lilac_layout_digest(0, 1), lilac_layout_digest(0, 16));
        assert_eq!(lilac_layout_digest(7, 16), lilac_layout_digest(7, 16));
    }
}

#[cfg(feature = "rs_in_order")]
#[allow(clippy::too_many_lines)]
fn run_whir_zk<F>(args: &Args)
where
    Standard: Distribution<F>,
    F: FftField + Codec,
{
    use whir::protocols::whir_zk::Config;

    let security_level = args.security_level;
    let pow_bits = args.pow_bits;
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let num_evaluations = args.num_evaluations;
    let num_linear_constraints = args.num_linear_constraints;
    let hash_id = args.hash.hash_id();

    if num_evaluations + num_linear_constraints == 0 {
        println!("No constraints specified, running as low-degree-test.");
    }

    let num_coeffs = 1 << num_variables;

    let whir_params = ProtocolParameters {
        unique_decoding: args.unique_decoding,
        security_level,
        pow_bits,
        initial_folding_factor: first_round_folding_factor,
        folding_factor,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        hash_id,
    };

    let params = Config::<F>::new(1 << num_variables, &whir_params, 1);

    let ds = DomainSeparator::protocol(&params)
        .session(&format!(
            "Example at {}:{} seed={}",
            file!(),
            line!(),
            args.session_seed
        ))
        .instance(&Empty);

    let mut prover_state = ProverState::new_std(&ds);

    println!("=========================================");
    println!("Whir (PCS + ZK) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params
        .blinded_commitment
        .check_max_pow_bits(Bits::new(whir_params.pow_bits as f64))
    {
        println!("WARN: more PoW bits required than specified.");
    }

    let embedding = Identity::<F>::new();
    let vector = (0..num_coeffs).map(F::from).collect::<Vec<_>>();

    // Allocate constraints
    let mut linear_forms: Vec<Box<dyn Evaluate<Basefield<F>>>> = Vec::new();
    let mut prove_linear_forms: Vec<Box<dyn LinearForm<F>>> = Vec::new();
    let mut evaluations = Vec::new();

    // Linear constraint
    // We do these first to benefit from buffer recycling.
    for _ in 0..num_linear_constraints {
        let linear_form = Box::new(Covector {
            vector: (0..num_coeffs).map(F::from).collect(),
        });
        evaluations.push(linear_form.evaluate(&embedding, &vector));
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    // Evaluation constraint
    let points: Vec<_> = (0..num_evaluations)
        .map(|x| MultilinearPoint(vec![F::from(x as u64); num_variables]))
        .collect();
    for point in &points {
        let linear_form = Box::new(MultilinearExtension::new(point.0.clone()));
        evaluations.push(linear_form.evaluate(&embedding, &vector));
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    let whir_commit_time = Instant::now();
    let witness = params.commit(&mut prover_state, &[vector.as_slice()]);
    let whir_commit_time = whir_commit_time.elapsed();

    let whir_prove_time = Instant::now();
    let _ = params.prove(
        &mut prover_state,
        vec![Cow::Borrowed(&vector)],
        witness,
        prove_linear_forms,
        Cow::Borrowed(&evaluations),
    );
    let whir_prove_time = whir_prove_time.elapsed();

    let proof = prover_state.proof();
    println!(
        "Prover time: {whir_commit_time:.1?} + {whir_prove_time:.1?} = {:.1?}",
        whir_commit_time + whir_prove_time,
    );
    println!(
        "Proof size: {:.1} KiB",
        (proof.narg_string.len() + proof.hints.len()) as f64 / 1024.0
    );

    let weight_dyn_refs = linear_forms
        .iter()
        .map(|w| w.as_ref() as &dyn LinearForm<F>)
        .collect::<Vec<_>>();

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params.receive_commitments(&mut verifier_state, 1).unwrap();
        params
            .verify(
                &mut verifier_state,
                &weight_dyn_refs,
                &evaluations,
                &commitment,
            )
            .unwrap();
    }
    println!(
        "Verifier time: {:.1?}",
        whir_verifier_time.elapsed() / reps as u32
    );
    println!(
        "Average hashes: {:.1}k",
        (HASH_COUNTER.get() as f64 / reps as f64) / 1000.0
    );
}
