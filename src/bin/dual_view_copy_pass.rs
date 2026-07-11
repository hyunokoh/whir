use std::time::{Duration, Instant};

use ark_ff::{PrimeField, Zero};
use clap::Parser;
use rayon::prelude::*;
use whir::algebra::fields::Field192;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 0)]
    level: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    #[arg(long, default_value_t = false)]
    mle: bool,
    #[arg(long, default_value_t = 5)]
    mle_iterations: usize,
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    raw: usize,
    blocks: usize,
    block_semantic: usize,
    row_width: usize,
    row_capacity: usize,
    row_span: usize,
    group_size: usize,
}

const GEOMETRIES: [Geometry; 4] = [
    Geometry {
        raw: 5_943_177,
        blocks: 207,
        block_semantic: 28_711,
        row_width: 2_438,
        row_capacity: 4_096,
        row_span: 16,
        group_size: 4_096,
    },
    Geometry {
        raw: 1_872_384,
        blocks: 384,
        block_semantic: 4_876,
        row_width: 1_369,
        row_capacity: 2_048,
        row_span: 4,
        group_size: 2_048,
    },
    Geometry {
        raw: 676_286,
        blocks: 247,
        block_semantic: 2_738,
        row_width: 823,
        row_capacity: 1_024,
        row_span: 4,
        group_size: 1_024,
    },
    Geometry {
        raw: 339_076,
        blocks: 206,
        block_semantic: 1_646,
        row_width: 583,
        row_capacity: 1_024,
        row_span: 4,
        group_size: 1_024,
    },
];

impl Geometry {
    fn view_capacity(self) -> usize {
        self.group_size * self.row_capacity
    }

    fn row_position(self, semantic_index: usize) -> usize {
        assert!(semantic_index < self.raw);
        let block = semantic_index / self.block_semantic;
        let local = semantic_index % self.block_semantic;
        let row = local / self.row_width;
        let column = local % self.row_width;
        block * self.row_span * self.row_capacity + row * self.row_capacity + column
    }

    fn source_position(self, row_position: usize) -> Option<usize> {
        assert!(row_position < self.view_capacity());
        let block_capacity = self.row_span * self.row_capacity;
        let block = row_position / block_capacity;
        if block >= self.blocks {
            return None;
        }
        let local = row_position % block_capacity;
        let row = local / self.row_capacity;
        let column = local % self.row_capacity;
        if column >= self.row_width {
            return None;
        }
        let semantic_local = row * self.row_width + column;
        (semantic_local < self.block_semantic)
            .then_some(block * self.block_semantic + semantic_local)
    }

    fn valid(self) -> bool {
        self.raw == self.blocks * self.block_semantic
            && self.row_capacity.is_power_of_two()
            && self.row_span.is_power_of_two()
            && self.blocks * self.row_span <= self.group_size
            && (self.block_semantic + self.row_width - 1) / self.row_width <= self.row_span
    }
}

fn percentile(values: &[f64], probability: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * probability).round() as usize]
}

fn source_value(index: usize) -> Field192 {
    Field192::from(((index as u64 + 1) * 0x1_0001 + (index as u64 + 7) * 0x101) % 0xffff_ffff)
}

fn coefficient(index: usize) -> Field192 {
    Field192::from(((index as u64 + 11) * 0x1_000_003) % 0xffff_ffff)
}

fn materialize_row_view(geometry: Geometry, source: &[Field192]) -> Vec<Field192> {
    let block_capacity = geometry.row_span * geometry.row_capacity;
    let mut row_view = vec![Field192::zero(); geometry.view_capacity()];
    row_view
        .par_chunks_mut(block_capacity)
        .take(geometry.blocks)
        .enumerate()
        .for_each(|(block, target)| {
            let source_start = block * geometry.block_semantic;
            let source_block = &source[source_start..source_start + geometry.block_semantic];
            for (local, value) in source_block.iter().enumerate() {
                let row = local / geometry.row_width;
                let column = local % geometry.row_width;
                target[row * geometry.row_capacity + column] = *value;
            }
        });
    row_view
}

fn copy_residual(geometry: Geometry, source: &[Field192], row_view: &[Field192]) -> Field192 {
    source
        .par_iter()
        .enumerate()
        .map(|(index, value)| {
            (row_view[geometry.row_position(index)] - *value) * coefficient(index)
        })
        .reduce(Field192::zero, |left, right| left + right)
}

fn transcript_point(geometry: Geometry) -> Vec<Field192> {
    let roots = (0_u64..4)
        .map(|index| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"LiLAC/copy-mle/committed-root-placeholder/v1");
            hasher.update(&(geometry.raw as u64).to_le_bytes());
            hasher.update(&index.to_le_bytes());
            *hasher.finalize().as_bytes()
        })
        .collect::<Vec<_>>();
    (0..geometry.view_capacity().trailing_zeros() as usize)
        .map(|index| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"LiLAC/copy-mle/challenge/v1");
            hasher.update(&(index as u64).to_le_bytes());
            for root in &roots {
                hasher.update(root);
            }
            Field192::from_le_bytes_mod_order(hasher.finalize().as_bytes())
        })
        .collect()
}

fn prepare_copy_difference(geometry: Geometry, source: &[Field192], row_view: &mut [Field192]) {
    row_view
        .par_iter_mut()
        .enumerate()
        .for_each(|(position, value)| {
            if let Some(source_position) = geometry.source_position(position) {
                *value -= source[source_position];
            }
        });
}

fn evaluate_mle_in_place(values: &mut [Field192], point: &[Field192]) -> Field192 {
    assert!(values.len().is_power_of_two());
    assert_eq!(values.len().trailing_zeros() as usize, point.len());
    let mut active = values.len();
    for coordinate in point {
        let half = active / 2;
        let (left, right) = values[..active].split_at_mut(half);
        let one_minus = Field192::from(1_u64) - *coordinate;
        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(low, high)| *low = *low * one_minus + *high * *coordinate);
        active = half;
    }
    values[0]
}

fn run_iteration(geometry: Geometry, source: &[Field192]) -> (Duration, Duration, Field192) {
    let start = Instant::now();
    let row_view = materialize_row_view(geometry, source);
    let materialize = start.elapsed();
    let start = Instant::now();
    let residual = copy_residual(geometry, source, &row_view);
    let copy = start.elapsed();
    (materialize, copy, residual)
}

fn main() {
    let args = Args::parse();
    assert!(args.level < GEOMETRIES.len() && args.iterations > 0);
    let geometry = GEOMETRIES[args.level];
    assert!(geometry.valid());
    let source = (0..geometry.raw)
        .into_par_iter()
        .map(source_value)
        .collect::<Vec<_>>();
    let mut materialize_ms = Vec::with_capacity(args.iterations);
    let mut copy_ms = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let (materialize, copy, residual) = run_iteration(geometry, &source);
        assert!(residual.is_zero());
        materialize_ms.push(materialize.as_secs_f64() * 1_000.0);
        copy_ms.push(copy.as_secs_f64() * 1_000.0);
    }
    println!("LiLAC Field192 dual-view copy/unpadding pass");
    println!("- level: {}", args.level);
    println!("- semantic fields: {}", geometry.raw);
    println!(
        "- blocks/semantic fields each: {}/{}",
        geometry.blocks, geometry.block_semantic
    );
    println!(
        "- row semantic/capacity/span: {}/{}/{}",
        geometry.row_width, geometry.row_capacity, geometry.row_span
    );
    println!(
        "- row-view used/capacity fields: {}/{}",
        geometry.blocks * geometry.row_span * geometry.row_capacity,
        geometry.view_capacity()
    );
    println!(
        "- row-view materialize median/p95: {:.3}/{:.3} ms",
        percentile(&materialize_ms, 0.5),
        percentile(&materialize_ms, 0.95)
    );
    println!(
        "- copy residual median/p95: {:.3}/{:.3} ms",
        percentile(&copy_ms, 0.5),
        percentile(&copy_ms, 0.95)
    );
    let totals = materialize_ms
        .iter()
        .zip(&copy_ms)
        .map(|(materialize, copy)| materialize + copy)
        .collect::<Vec<_>>();
    println!(
        "- combined median/p95: {:.3}/{:.3} ms",
        percentile(&totals, 0.5),
        percentile(&totals, 0.95)
    );
    println!("- excludes both Merkle commitments, QA encoding, and the NIRK transcript");

    if args.mle {
        assert!(args.mle_iterations > 0);
        let start = Instant::now();
        let point = transcript_point(geometry);
        let challenge_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let mut staging_ms = Vec::with_capacity(args.mle_iterations);
        let mut prepare_ms = Vec::with_capacity(args.mle_iterations);
        let mut fold_ms = Vec::with_capacity(args.mle_iterations);
        for _ in 0..args.mle_iterations {
            let start = Instant::now();
            let mut difference = materialize_row_view(geometry, &source);
            staging_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            let start = Instant::now();
            prepare_copy_difference(geometry, &source, &mut difference);
            prepare_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            let start = Instant::now();
            let residual = evaluate_mle_in_place(&mut difference, &point);
            fold_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            assert!(residual.is_zero());
        }
        let totals = staging_ms
            .iter()
            .zip(&prepare_ms)
            .zip(&fold_ms)
            .map(|((staging, prepare), fold)| staging + prepare + fold)
            .collect::<Vec<_>>();
        println!("- root-bound FS MLE variables: {}", point.len());
        println!("- FS challenge derivation: {challenge_ms:.3} ms");
        println!(
            "- copy-MLE row staging median/p95: {:.3}/{:.3} ms",
            percentile(&staging_ms, 0.5),
            percentile(&staging_ms, 0.95)
        );
        println!(
            "- copy/unpadding difference median/p95: {:.3}/{:.3} ms",
            percentile(&prepare_ms, 0.5),
            percentile(&prepare_ms, 0.95)
        );
        println!(
            "- {}-round MLE fold median/p95: {:.3}/{:.3} ms",
            point.len(),
            percentile(&fold_ms, 0.5),
            percentile(&fold_ms, 0.95)
        );
        println!(
            "- complete copy-MLE pass median/p95: {:.3}/{:.3} ms",
            percentile(&totals, 0.5),
            percentile(&totals, 0.95)
        );
        println!("- challenge roots are fixed committed-root placeholders; hashing commitments is excluded");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_geometries_match_contract() {
        assert!(GEOMETRIES.into_iter().all(Geometry::valid));
        assert_eq!(GEOMETRIES[0].blocks * GEOMETRIES[0].row_span, 3_312);
        assert_eq!(GEOMETRIES[3].blocks * GEOMETRIES[3].row_span, 824);
    }

    #[test]
    fn honest_copy_accepts_and_one_field_tamper_fails() {
        let geometry = Geometry {
            raw: 18,
            blocks: 3,
            block_semantic: 6,
            row_width: 2,
            row_capacity: 4,
            row_span: 4,
            group_size: 16,
        };
        assert!(geometry.valid());
        let source = (0..geometry.raw).map(source_value).collect::<Vec<_>>();
        let mut row_view = materialize_row_view(geometry, &source);
        assert!(copy_residual(geometry, &source, &row_view).is_zero());
        let position = geometry.row_position(7);
        row_view[position] += Field192::from(1_u64);
        assert!(!copy_residual(geometry, &source, &row_view).is_zero());
    }

    #[test]
    fn copy_mle_checks_semantic_values_and_padding() {
        let geometry = Geometry {
            raw: 18,
            blocks: 3,
            block_semantic: 6,
            row_width: 2,
            row_capacity: 4,
            row_span: 4,
            group_size: 16,
        };
        let source = (0..geometry.raw).map(source_value).collect::<Vec<_>>();
        let point = transcript_point(geometry);
        let mut honest = materialize_row_view(geometry, &source);
        prepare_copy_difference(geometry, &source, &mut honest);
        assert!(evaluate_mle_in_place(&mut honest, &point).is_zero());

        let mut semantic_fault = materialize_row_view(geometry, &source);
        semantic_fault[geometry.row_position(7)] += Field192::from(1_u64);
        prepare_copy_difference(geometry, &source, &mut semantic_fault);
        assert!(!evaluate_mle_in_place(&mut semantic_fault, &point).is_zero());

        let mut padding_fault = materialize_row_view(geometry, &source);
        padding_fault[3] += Field192::from(1_u64);
        assert_eq!(geometry.source_position(3), None);
        prepare_copy_difference(geometry, &source, &mut padding_fault);
        assert!(!evaluate_mle_in_place(&mut padding_fault, &point).is_zero());
    }
}
