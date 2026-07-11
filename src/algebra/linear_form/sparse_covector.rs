use ark_ff::{AdditiveGroup, Field};

use super::{Evaluate, LinearForm};
use crate::algebra::Embedding;

/// A linear form represented by its nonzero covector entries.
///
/// This keeps verifier work proportional to `entries.len() * log(size)` when
/// [`LinearForm::mle_evaluate`] is used in WHIR's final-claim check. The prover
/// still accumulates the sparse entries into its ordinary dense sumcheck
/// covector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseCovector<F: Field> {
    size: usize,
    entries: Vec<(usize, F)>,
}

impl<F: Field> SparseCovector<F> {
    pub fn new(size: usize, mut entries: Vec<(usize, F)>) -> Self {
        assert!(size > 0, "sparse covector must have positive size");
        entries.sort_unstable_by_key(|(index, _)| *index);
        assert!(
            entries.iter().all(|(index, _)| *index < size),
            "sparse covector index out of range"
        );
        assert!(
            entries.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "sparse covector indices must be distinct"
        );
        entries.retain(|(_, value)| !value.is_zero());
        Self { size, entries }
    }

    pub fn entries(&self) -> &[(usize, F)] {
        &self.entries
    }

    fn equality_weight(point: &[F], index: usize) -> F {
        point
            .iter()
            .enumerate()
            .fold(F::ONE, |weight, (coordinate, value)| {
                let shift = point.len() - coordinate - 1;
                if (index >> shift) & 1 == 1 {
                    weight * value
                } else {
                    weight * (F::ONE - value)
                }
            })
    }
}

impl<F: Field> LinearForm<F> for SparseCovector<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        assert!(
            point.len() < usize::BITS as usize && (1usize << point.len()) >= self.size,
            "multilinear point is too short for sparse covector"
        );
        self.entries.iter().fold(F::ZERO, |sum, (index, value)| {
            sum + *value * Self::equality_weight(point, *index)
        })
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        assert_eq!(accumulator.len(), self.size);
        for (index, value) in &self.entries {
            accumulator[*index] += scalar * value;
        }
    }
}

impl<M: Embedding> Evaluate<M> for SparseCovector<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        assert_eq!(vector.len(), self.size);
        self.entries
            .iter()
            .fold(M::Target::ZERO, |sum, (index, value)| {
                sum + embedding.mixed_mul(*value, vector[*index])
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{embedding::Identity, fields::Field64, linear_form::Covector};

    #[test]
    fn sparse_matches_dense_covector() {
        let entries = vec![
            (0, Field64::from(3)),
            (5, Field64::from(7)),
            (15, Field64::from(11)),
        ];
        let sparse = SparseCovector::new(16, entries.clone());
        let mut dense_values = vec![Field64::ZERO; 16];
        for (index, value) in entries {
            dense_values[index] = value;
        }
        let dense = Covector::new(dense_values);
        let point = [
            Field64::from(2),
            Field64::from(3),
            Field64::from(5),
            Field64::from(7),
        ];
        assert_eq!(sparse.mle_evaluate(&point), dense.mle_evaluate(&point));

        let vector = (0..16).map(Field64::from).collect::<Vec<_>>();
        let embedding = Identity::<Field64>::new();
        assert_eq!(
            sparse.evaluate(&embedding, &vector),
            dense.evaluate(&embedding, &vector)
        );

        let mut sparse_accumulator = vec![Field64::ZERO; 16];
        let mut dense_accumulator = vec![Field64::ZERO; 16];
        sparse.accumulate(&mut sparse_accumulator, Field64::from(13));
        dense.accumulate(&mut dense_accumulator, Field64::from(13));
        assert_eq!(sparse_accumulator, dense_accumulator);
    }
}
