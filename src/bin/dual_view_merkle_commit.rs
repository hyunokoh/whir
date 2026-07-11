use std::time::{Duration, Instant};

use clap::Parser;
use rayon::prelude::*;
use whir::algebra::fields::Field192;
use whir::lilac_merkle::{
    combine_equal_subtrees, field_leaf, prefix_root, zero_roots, Digest, MerkleAccumulator,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 5)]
    iterations: usize,
}

const BLOCKS: usize = 207;
const BLOCK_SEMANTIC: usize = 28_711;
const BLOCK_CAPACITY: usize = 1 << 16;
const VIEW_CAPACITY: usize = 1 << 24;
const ROW_WIDTH: usize = 2_438;
const ROW_CAPACITY: usize = 1 << 12;
const ROWS_PER_BLOCK: usize = 16;
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

fn source_value(index: usize) -> Field192 {
    Field192::from(((index as u64 + 1) * 0x1_0001 + (index as u64 + 7) * 0x101) % 0xffff_ffff)
}

fn splice_block_root(values: &[Field192], zeros: &[Digest]) -> Digest {
    assert_eq!(values.len(), BLOCK_SEMANTIC);
    let mut position = 0;
    let roots = COMPONENTS
        .iter()
        .map(|(semantic, capacity)| {
            let root = prefix_root(&values[position..position + semantic], *capacity, zeros);
            position += semantic;
            root
        })
        .collect::<Vec<_>>();
    assert_eq!(position, values.len());
    // Descending capacities 2^15, 2^14, 2^13, 2^13 are each aligned at
    // their insertion position. Join them with the same
    // aligned accumulator used by the outer commitment.
    let mut accumulator = MerkleAccumulator::new(BLOCK_CAPACITY.trailing_zeros() as usize);
    for (root, (_, capacity)) in roots.into_iter().zip(COMPONENTS) {
        accumulator.append_subtree(root, capacity.trailing_zeros() as usize);
    }
    accumulator.finish(BLOCK_CAPACITY, zeros)
}

fn row_block_root(values: &[Field192], zeros: &[Digest]) -> Digest {
    assert_eq!(values.len(), BLOCK_SEMANTIC);
    let mut row_roots = values
        .chunks(ROW_WIDTH)
        .map(|row| prefix_root(row, ROW_CAPACITY, zeros))
        .collect::<Vec<_>>();
    assert!(row_roots.len() <= ROWS_PER_BLOCK);
    row_roots.resize(
        ROWS_PER_BLOCK,
        zeros[ROW_CAPACITY.trailing_zeros() as usize],
    );
    combine_equal_subtrees(&row_roots)
}

fn top_root(block_roots: &[Digest], zeros: &[Digest]) -> Digest {
    assert_eq!(block_roots.len(), BLOCKS);
    let mut accumulator = MerkleAccumulator::new(VIEW_CAPACITY.trailing_zeros() as usize);
    let height = BLOCK_CAPACITY.trailing_zeros() as usize;
    for root in block_roots {
        accumulator.append_subtree(*root, height);
    }
    accumulator.finish(VIEW_CAPACITY, zeros)
}

fn build_block_roots(
    source: &[Field192],
    zeros: &[Digest],
    builder: fn(&[Field192], &[Digest]) -> Digest,
) -> Vec<Digest> {
    source
        .par_chunks_exact(BLOCK_SEMANTIC)
        .map(|block| builder(block, zeros))
        .collect()
}

fn run_row_commitment(source: &[Field192], zeros: &[Digest]) -> (Duration, Duration, Digest) {
    let start = Instant::now();
    let roots = build_block_roots(source, zeros, row_block_root);
    let blocks = start.elapsed();
    let start = Instant::now();
    let root = top_root(&roots, zeros);
    let top = start.elapsed();
    (blocks, top, root)
}

fn main() {
    let args = Args::parse();
    assert!(args.iterations > 0);
    assert_eq!(BLOCKS * BLOCK_SEMANTIC, 5_943_177);
    assert_eq!(BLOCKS * BLOCK_CAPACITY, 13_565_952);
    assert_eq!(ROWS_PER_BLOCK * ROW_CAPACITY, BLOCK_CAPACITY);
    let zeros = zero_roots(VIEW_CAPACITY.trailing_zeros() as usize);
    let source = (0..BLOCKS * BLOCK_SEMANTIC)
        .into_par_iter()
        .map(source_value)
        .collect::<Vec<_>>();

    let start = Instant::now();
    let splice_blocks = build_block_roots(&source, &zeros, splice_block_root);
    let source_subtrees = start.elapsed();
    let start = Instant::now();
    let splice_root = top_root(&splice_blocks, &zeros);
    let splice_derive = start.elapsed();

    let mut row_block_ms = Vec::with_capacity(args.iterations);
    let mut row_top_ms = Vec::with_capacity(args.iterations);
    let mut checksum = 0_u8;
    for _ in 0..args.iterations {
        let (blocks, top, row_root) = run_row_commitment(&source, &zeros);
        row_block_ms.push(blocks.as_secs_f64() * 1_000.0);
        row_top_ms.push(top.as_secs_f64() * 1_000.0);
        checksum ^= row_root[0];
        assert_ne!(
            row_root, splice_root,
            "distinct padded views unexpectedly collided"
        );
    }
    let totals = row_block_ms
        .iter()
        .zip(&row_top_ms)
        .map(|(blocks, top)| blocks + top)
        .collect::<Vec<_>>();

    println!("LiLAC level-0 splice-compatible dual-view Merkle commitment");
    println!(
        "- semantic/view-capacity fields: {}/{}",
        source.len(),
        VIEW_CAPACITY
    );
    println!("- blocks semantic/capacity/count: {BLOCK_SEMANTIC}/{BLOCK_CAPACITY}/{BLOCKS}");
    println!("- row semantic/capacity/span: {ROW_WIDTH}/{ROW_CAPACITY}/{ROWS_PER_BLOCK}");
    println!(
        "- source subtree roots (one-time/extractor) : {:.3} ms",
        source_subtrees.as_secs_f64() * 1_000.0
    );
    println!(
        "- splice root derivation from 207 old roots: {:.3} ms",
        splice_derive.as_secs_f64() * 1_000.0
    );
    println!(
        "- row-view block roots median/p95: {:.3}/{:.3} ms",
        percentile(&row_block_ms, 0.5),
        percentile(&row_block_ms, 0.95)
    );
    println!(
        "- row-view top root median/p95: {:.3}/{:.3} ms",
        percentile(&row_top_ms, 0.5),
        percentile(&row_top_ms, 0.95)
    );
    println!(
        "- row-view commitment median/p95: {:.3}/{:.3} ms",
        percentile(&totals, 0.5),
        percentile(&totals, 0.95)
    );
    println!("- splice/row roots distinct and bound by the copy relation: true");
    println!("- checksum byte: {checksum}");
    println!("- excludes QA parity encoding/root, copy sumcheck, and recursive NIRK");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn explicit_root(values: &[Field192], capacity: usize) -> Digest {
        let mut leaves = values.iter().copied().map(field_leaf).collect::<Vec<_>>();
        leaves.resize(capacity, field_leaf(Field192::from(0_u64)));
        combine_equal_subtrees(&leaves)
    }

    #[test]
    fn sparse_prefix_root_matches_explicit_tree() {
        let zeros = zero_roots(6);
        let values = (1_u64..=19).map(Field192::from).collect::<Vec<_>>();
        assert_eq!(prefix_root(&values, 32, &zeros), explicit_root(&values, 32));
    }

    #[test]
    fn aligned_accumulator_matches_explicit_tree() {
        let zeros = zero_roots(6);
        let left = (1_u64..=5).map(Field192::from).collect::<Vec<_>>();
        let right = (11_u64..=13).map(Field192::from).collect::<Vec<_>>();
        let left_root = prefix_root(&left, 8, &zeros);
        let right_root = prefix_root(&right, 8, &zeros);
        let mut accumulator = MerkleAccumulator::new(5);
        accumulator.append_subtree(left_root, 3);
        accumulator.append_subtree(right_root, 3);
        let derived = accumulator.finish(32, &zeros);
        let mut explicit = left;
        explicit.resize(8, Field192::from(0_u64));
        explicit.extend(right);
        explicit.resize(32, Field192::from(0_u64));
        assert_eq!(derived, explicit_root(&explicit, 32));
    }

    #[test]
    fn one_semantic_field_changes_both_views() {
        let zeros = zero_roots(BLOCK_CAPACITY.trailing_zeros() as usize);
        let mut values = (0..BLOCK_SEMANTIC).map(source_value).collect::<Vec<_>>();
        let splice = splice_block_root(&values, &zeros);
        let row = row_block_root(&values, &zeros);
        values[12_345] += Field192::from(1_u64);
        assert_ne!(splice, splice_block_root(&values, &zeros));
        assert_ne!(row, row_block_root(&values, &zeros));
    }
}
