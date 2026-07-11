use ark_ff::PrimeField;
use rayon::prelude::*;

use crate::algebra::fields::Field192;

pub type Digest = [u8; 32];

pub fn field_leaf(value: Field192) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/field-leaf/v1");
    let bigint = value.into_bigint();
    for limb in bigint.as_ref() {
        hasher.update(&limb.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

pub fn parent(left: Digest, right: Digest) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LiLAC/field-node/v1");
    hasher.update(&left);
    hasher.update(&right);
    *hasher.finalize().as_bytes()
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
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for value in values {
        accumulator.append_leaf(field_leaf(*value));
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
