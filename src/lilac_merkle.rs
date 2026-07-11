use ark_ff::PrimeField;
use rayon::prelude::*;

use crate::algebra::fields::Field192;

pub type Digest = [u8; 32];

const FIELD_LEAF_DOMAIN: &[u8; 19] = b"LiLAC/field-leaf/v1";
const FIELD_NODE_DOMAIN: &[u8; 19] = b"LiLAC/field-node/v1";

pub fn field_leaf(value: Field192) -> Digest {
    let mut input = [0_u8; FIELD_LEAF_DOMAIN.len() + 24];
    input[..FIELD_LEAF_DOMAIN.len()].copy_from_slice(FIELD_LEAF_DOMAIN);
    let bigint = value.into_bigint();
    for (index, limb) in bigint.as_ref().iter().enumerate() {
        let start = FIELD_LEAF_DOMAIN.len() + index * 8;
        input[start..start + 8].copy_from_slice(&limb.to_le_bytes());
    }
    *blake3::hash(&input).as_bytes()
}

pub fn parent(left: Digest, right: Digest) -> Digest {
    let mut input = [0_u8; FIELD_NODE_DOMAIN.len() + 2 * 32];
    input[..FIELD_NODE_DOMAIN.len()].copy_from_slice(FIELD_NODE_DOMAIN);
    input[FIELD_NODE_DOMAIN.len()..FIELD_NODE_DOMAIN.len() + 32].copy_from_slice(&left);
    input[FIELD_NODE_DOMAIN.len() + 32..].copy_from_slice(&right);
    *blake3::hash(&input).as_bytes()
}

pub fn zero_roots(max_height: usize) -> Vec<Digest> {
    let mut roots = Vec::with_capacity(max_height + 1);
    roots.push(field_leaf(Field192::from(0_u64)));
    for height in 0..max_height {
        roots.push(parent(roots[height], roots[height]));
    }
    roots
}

#[derive(Debug)]
pub struct MerkleAccumulator {
    stack: Vec<Option<Digest>>,
    leaves: usize,
}

impl MerkleAccumulator {
    pub fn new(max_height: usize) -> Self {
        Self {
            stack: vec![None; max_height + 1],
            leaves: 0,
        }
    }

    pub fn append_subtree(&mut self, mut root: Digest, mut height: usize) {
        let span = 1usize << height;
        assert_eq!(self.leaves % span, 0, "unaligned appended subtree");
        self.leaves += span;
        loop {
            if self.stack[height].is_none() {
                self.stack[height] = Some(root);
                return;
            }
            let left = self.stack[height].take().unwrap();
            root = parent(left, root);
            height += 1;
        }
    }

    pub fn append_leaf(&mut self, leaf: Digest) {
        self.append_subtree(leaf, 0);
    }

    pub fn fill_zeros_to(&mut self, capacity: usize, zeros: &[Digest]) {
        assert!(capacity.is_power_of_two() && self.leaves <= capacity);
        while self.leaves < capacity {
            let remaining = capacity - self.leaves;
            let alignment = if self.leaves == 0 {
                capacity
            } else {
                1usize << self.leaves.trailing_zeros()
            };
            let span = alignment.min(1usize << (usize::BITS - 1 - remaining.leading_zeros()));
            let height = span.trailing_zeros() as usize;
            self.append_subtree(zeros[height], height);
        }
    }

    pub fn finish(mut self, capacity: usize, zeros: &[Digest]) -> Digest {
        self.fill_zeros_to(capacity, zeros);
        let height = capacity.trailing_zeros() as usize;
        assert_eq!(self.leaves, capacity);
        assert!(self
            .stack
            .iter()
            .enumerate()
            .all(|(index, value)| index == height || value.is_none()));
        self.stack[height].take().unwrap()
    }
}

pub fn prefix_root(values: &[Field192], capacity: usize, zeros: &[Digest]) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    const PARALLEL_MIN_FIELDS: usize = 1 << 18;
    const CHUNK_FIELDS: usize = 1 << 12;
    if values.len() >= PARALLEL_MIN_FIELDS {
        return chunked_prefix_root(values, capacity, zeros, CHUNK_FIELDS);
    }
    sequential_prefix_root(values, capacity, zeros)
}

fn sequential_prefix_root(values: &[Field192], capacity: usize, zeros: &[Digest]) -> Digest {
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for value in values {
        accumulator.append_leaf(field_leaf(*value));
    }
    accumulator.finish(capacity, zeros)
}

fn chunked_prefix_root(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
    chunk_fields: usize,
) -> Digest {
    assert!(
        values.len() <= capacity
            && capacity.is_power_of_two()
            && chunk_fields.is_power_of_two()
            && chunk_fields <= capacity
    );
    let chunk_roots = values
        .par_chunks(chunk_fields)
        .map(|chunk| sequential_prefix_root(chunk, chunk_fields, zeros))
        .collect::<Vec<_>>();
    assert!(chunk_roots.len() * chunk_fields <= capacity);
    let chunk_height = chunk_fields.trailing_zeros() as usize;
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for root in chunk_roots {
        accumulator.append_subtree(root, chunk_height);
    }
    accumulator.finish(capacity, zeros)
}

/// Compute the ordinary field-Merkle root of `scale * values` without
/// materializing the scaled row. This preserves the exact leaf and node
/// format used by [`prefix_root`].
pub fn scaled_prefix_root(
    values: &[Field192],
    scale: Field192,
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for value in values {
        accumulator.append_leaf(field_leaf(scale * *value));
    }
    accumulator.finish(capacity, zeros)
}

pub fn combine_equal_subtrees(roots: &[Digest]) -> Digest {
    assert!(!roots.is_empty() && roots.len().is_power_of_two());
    let mut level = roots.to_vec();
    while level.len() > 1 {
        level = level
            .par_chunks_exact(2)
            .map(|pair| parent(pair[0], pair[1]))
            .collect();
    }
    level[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_field_leaf(value: Field192) -> Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(FIELD_LEAF_DOMAIN);
        let bigint = value.into_bigint();
        for limb in bigint.as_ref() {
            hasher.update(&limb.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    fn legacy_parent(left: Digest, right: Digest) -> Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(FIELD_NODE_DOMAIN);
        hasher.update(&left);
        hasher.update(&right);
        *hasher.finalize().as_bytes()
    }

    #[test]
    fn one_shot_hashes_match_legacy_transcript_bytes() {
        let left = legacy_field_leaf(Field192::from(17_u64));
        let right = legacy_field_leaf(Field192::from(29_u64));
        assert_eq!(field_leaf(Field192::from(17_u64)), left);
        assert_eq!(field_leaf(Field192::from(29_u64)), right);
        assert_eq!(parent(left, right), legacy_parent(left, right));
    }

    #[test]
    fn chunked_prefix_root_matches_sequential_root() {
        let zeros = zero_roots(8);
        let values = (0..128)
            .map(|index| Field192::from((index + 1) as u64))
            .collect::<Vec<_>>();
        for length in [0, 1, 7, 8, 9, 63, 64, 65, 127, 128] {
            assert_eq!(
                chunked_prefix_root(&values[..length], 128, &zeros, 8),
                sequential_prefix_root(&values[..length], 128, &zeros)
            );
        }
    }

    #[test]
    fn scaled_prefix_root_matches_materialized_row() {
        let zeros = zero_roots(7);
        let values = (0..73)
            .map(|index| Field192::from((3 * index + 5) as u64))
            .collect::<Vec<_>>();
        let scale = Field192::from(17_u64);
        let scaled = values
            .iter()
            .map(|value| scale * *value)
            .collect::<Vec<_>>();
        assert_eq!(
            scaled_prefix_root(&values, scale, 128, &zeros),
            sequential_prefix_root(&scaled, 128, &zeros)
        );
    }
}
