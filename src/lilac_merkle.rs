use ark_ff::PrimeField;
use rayon::prelude::*;
use std::{cell::RefCell, sync::OnceLock};

use crate::algebra::fields::Field192;

pub type Digest = [u8; 32];

const FIELD_LEAF_DOMAIN: &[u8; 19] = b"LiLAC/field-leaf/v1";
const FIELD_NODE_DOMAIN: &[u8; 19] = b"LiLAC/field-node/v1";
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const BLAKE3_CHUNK_START: u8 = 1 << 0;
const BLAKE3_CHUNK_END: u8 = 1 << 1;
const BLAKE3_ROOT: u8 = 1 << 3;

fn blake3_platform() -> blake3::platform::Platform {
    // `platform` is a doc-hidden benchmark API. Cargo.toml pins Blake3 1.8.3,
    // and the tests below compare both fixed transcript lengths against the
    // stable `blake3::hash` entry point over thousands of inputs.
    static PLATFORM: OnceLock<blake3::platform::Platform> = OnceLock::new();
    *PLATFORM.get_or_init(blake3::platform::Platform::detect)
}

fn fixed_blake3_hash_43(block: &[u8; 64]) -> Digest {
    let output = blake3_platform().compress_xof(
        &BLAKE3_IV,
        block,
        43,
        0,
        BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT,
    );
    output[..32].try_into().unwrap()
}

fn fixed_blake3_hash_83(first: &[u8; 64], second: &[u8; 64]) -> Digest {
    let platform = blake3_platform();
    let mut cv = BLAKE3_IV;
    platform.compress_in_place(&mut cv, first, 64, 0, BLAKE3_CHUNK_START);
    let output = platform.compress_xof(&cv, second, 19, 0, BLAKE3_CHUNK_END | BLAKE3_ROOT);
    output[..32].try_into().unwrap()
}

fn node_blocks(left: Digest, right: Digest) -> ([u8; 64], [u8; 64]) {
    let mut first = [0_u8; 64];
    first[..FIELD_NODE_DOMAIN.len()].copy_from_slice(FIELD_NODE_DOMAIN);
    first[FIELD_NODE_DOMAIN.len()..FIELD_NODE_DOMAIN.len() + 32].copy_from_slice(&left);
    first[FIELD_NODE_DOMAIN.len() + 32..].copy_from_slice(&right[..13]);
    let mut second = [0_u8; 64];
    second[..19].copy_from_slice(&right[13..]);
    (first, second)
}

fn parent4(children: &[Digest; 8]) -> [Digest; 4] {
    let platform = blake3_platform();
    if platform.simd_degree() < 4 {
        return std::array::from_fn(|index| parent(children[2 * index], children[2 * index + 1]));
    }
    let blocks = std::array::from_fn::<_, 4, _>(|index| {
        node_blocks(children[2 * index], children[2 * index + 1])
    });
    let first = [&blocks[0].0, &blocks[1].0, &blocks[2].0, &blocks[3].0];
    let mut chaining_values = [0_u8; 4 * 32];
    platform.hash_many(
        &first,
        &BLAKE3_IV,
        0,
        blake3::IncrementCounter::No,
        0,
        BLAKE3_CHUNK_START,
        0,
        &mut chaining_values,
    );
    std::array::from_fn(|index| {
        let bytes = &chaining_values[index * 32..(index + 1) * 32];
        let cv = std::array::from_fn(|word| {
            u32::from_le_bytes(bytes[word * 4..(word + 1) * 4].try_into().unwrap())
        });
        let output =
            platform.compress_xof(&cv, &blocks[index].1, 19, 0, BLAKE3_CHUNK_END | BLAKE3_ROOT);
        output[..32].try_into().unwrap()
    })
}

pub fn field_leaf(value: Field192) -> Digest {
    let mut input = [0_u8; 64];
    input[..FIELD_LEAF_DOMAIN.len()].copy_from_slice(FIELD_LEAF_DOMAIN);
    let bigint = value.into_bigint();
    for (index, limb) in bigint.as_ref().iter().enumerate() {
        let start = FIELD_LEAF_DOMAIN.len() + index * 8;
        input[start..start + 8].copy_from_slice(&limb.to_le_bytes());
    }
    fixed_blake3_hash_43(&input)
}

pub fn parent(left: Digest, right: Digest) -> Digest {
    let (first, second) = node_blocks(left, right);
    fixed_blake3_hash_83(&first, &second)
}

thread_local! {
    static EXACT_ROOT_SCRATCH: RefCell<Vec<Digest>> = const { RefCell::new(Vec::new()) };
}

fn reduce_exact_digests(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active > 1 {
        let pairs = active / 2;
        let batched = pairs / 4 * 4;
        for pair in (0..batched).step_by(4) {
            let child = 2 * pair;
            let children = [
                scratch[child],
                scratch[child + 1],
                scratch[child + 2],
                scratch[child + 3],
                scratch[child + 4],
                scratch[child + 5],
                scratch[child + 6],
                scratch[child + 7],
            ];
            let roots = parent4(&children);
            scratch[pair..pair + 4].copy_from_slice(&roots);
        }
        for pair in batched..pairs {
            scratch[pair] = parent(scratch[2 * pair], scratch[2 * pair + 1]);
        }
        active = pairs;
    }
    scratch[0]
}

fn exact_prefix_root_batched(values: &[Field192]) -> Digest {
    assert!(values.len() >= 8 && values.len().is_power_of_two());
    EXACT_ROOT_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.clear();
        scratch.extend(values.iter().map(|value| field_leaf(*value)));
        reduce_exact_digests(&mut scratch)
    })
}

fn exact_scaled_prefix_root_batched(values: &[Field192], scale: Field192) -> Digest {
    assert!(values.len() >= 8 && values.len().is_power_of_two());
    EXACT_ROOT_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.clear();
        scratch.extend(values.iter().map(|value| field_leaf(scale * *value)));
        reduce_exact_digests(&mut scratch)
    })
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
    if values.len() == capacity && values.len() >= 8 {
        return exact_prefix_root_batched(values);
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
        .map(|chunk| {
            if chunk.len() == chunk_fields && chunk.len() >= 8 {
                exact_prefix_root_batched(chunk)
            } else {
                sequential_prefix_root(chunk, chunk_fields, zeros)
            }
        })
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
    if values.len() == capacity && values.len() >= 8 {
        return exact_scaled_prefix_root_batched(values, scale);
    }
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
    fn fixed_length_hashes_match_blake3_over_many_inputs() {
        for index in 0..4096_u64 {
            let seed = blake3::hash(&index.to_le_bytes());
            let value = Field192::from_le_bytes_mod_order(seed.as_bytes());
            assert_eq!(field_leaf(value), legacy_field_leaf(value));

            let mut left_input = [0_u8; 16];
            left_input[..8].copy_from_slice(&index.to_le_bytes());
            left_input[8..].copy_from_slice(&index.wrapping_mul(3).to_le_bytes());
            let left = *blake3::hash(&left_input).as_bytes();
            let mut right_input = [0_u8; 16];
            right_input[..8].copy_from_slice(&index.wrapping_add(1).to_le_bytes());
            right_input[8..].copy_from_slice(&index.wrapping_mul(7).to_le_bytes());
            let right = *blake3::hash(&right_input).as_bytes();
            assert_eq!(parent(left, right), legacy_parent(left, right));
        }
    }

    #[test]
    fn four_way_parents_match_scalar_parents() {
        for batch in 0..1024_u64 {
            let children = std::array::from_fn(|index| {
                let value = batch * 8 + index as u64;
                *blake3::hash(&value.to_le_bytes()).as_bytes()
            });
            let batched = parent4(&children);
            let scalar =
                std::array::from_fn(|index| parent(children[2 * index], children[2 * index + 1]));
            assert_eq!(batched, scalar);
        }
    }

    #[test]
    fn exact_batched_roots_match_sequential_roots() {
        let zeros = zero_roots(12);
        let values = (0..4096)
            .map(|index| Field192::from((17 * index + 11) as u64))
            .collect::<Vec<_>>();
        for length in [8, 16, 64, 256, 1024, 4096] {
            assert_eq!(
                exact_prefix_root_batched(&values[..length]),
                sequential_prefix_root(&values[..length], length, &zeros)
            );
        }
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
        let values = (0..128)
            .map(|index| Field192::from((3 * index + 5) as u64))
            .collect::<Vec<_>>();
        let scale = Field192::from(17_u64);
        let scaled = values
            .iter()
            .map(|value| scale * *value)
            .collect::<Vec<_>>();
        for length in [73, 128] {
            assert_eq!(
                scaled_prefix_root(&values[..length], scale, 128, &zeros),
                sequential_prefix_root(&scaled[..length], 128, &zeros)
            );
        }
    }
}
