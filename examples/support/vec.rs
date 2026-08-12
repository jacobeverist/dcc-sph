// Segmented hypervectors — a port of `demos/vec.h` from OgmaNeoDemos.
//
// A `SegVec<S, L>` is `S` segments, each holding one index in `0..L`. Written out
// one-hot it would be an `S * L` binary vector with exactly one bit set per
// segment, which is what makes it a CSDR: `S` columns of `L` cells. That is the
// point of the representation — a hypervector and a CSDR are the same object here,
// so one can be fed straight to a `Hierarchy` as column indices.
//
// The algebra is Vector Symbolic Architecture:
//
//   bind    a * b   elementwise (a + b) mod L    — invertible, similarity-destroying
//   unbind  a / b   elementwise (a - b) mod L    — exact inverse of bind
//   bundle  a + b   superposition, in a `Bundle` — similarity-preserving
//   permute a.permute(k)                          — cheap "position" operator
//
// Bind and unbind let you attach a value to a role and get it back (`(k * v) / k
// == v`); bundling lets you hold several such pairs in one vector and recover each
// approximately. `thin()` collapses a bundle back to a single vector by taking the
// argmax within each segment.
//
// Upstream names these `Vec` and uses `*` and `/`. Renamed here because shadowing
// `std::vec::Vec` in Rust would be a permanent nuisance; the operators are provided
// as well as the named methods so ported code reads the same.
//
// This is used by `vsa_char` and `explore`.

use std::ops::{Add, AddAssign, Div, Mul};

use crate::support::rng::Rng;

/// `S` segments of `L` possible values each.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SegVec<const S: usize, const L: usize> {
    data: [u8; S],
}

impl<const S: usize, const L: usize> SegVec<S, L> {
    /// All segments at index 0 — the bind identity.
    pub fn zero() -> Self {
        SegVec { data: [0u8; S] }
    }

    pub fn filled(value: u8) -> Self {
        assert!((value as usize) < L, "segment value {value} outside 0..{L}");
        SegVec { data: [value; S] }
    }

    /// A uniformly random vector. Two of these are near-orthogonal: they agree on
    /// about `S / L` segments by chance.
    pub fn randomized(rng: &mut Rng) -> Self {
        let mut data = [0u8; S];
        for d in data.iter_mut() {
            *d = rng.below(L) as u8;
        }
        SegVec { data }
    }

    pub const fn segments() -> usize {
        S
    }

    pub const fn length() -> usize {
        L
    }

    /// The one-hot width, `S * L`.
    pub const fn size() -> usize {
        S * L
    }

    pub fn get(&self, i: usize) -> u8 {
        self.data[i]
    }

    pub fn set(&mut self, i: usize, v: u8) {
        debug_assert!((v as usize) < L);
        self.data[i] = v;
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// The vector as column indices, ready to hand to a `Hierarchy` IO port of size
    /// `(x, y, L)` with `x * y == S`.
    pub fn to_cis(self) -> Vec<i32> {
        self.data.iter().map(|&v| v as i32).collect()
    }

    /// Build from column indices, the inverse of [`to_cis`](Self::to_cis).
    pub fn from_cis(cis: &[i32]) -> Self {
        assert_eq!(cis.len(), S, "expected {S} column indices");
        let mut data = [0u8; S];
        for (i, &c) in cis.iter().enumerate() {
            data[i] = c.rem_euclid(L as i32) as u8;
        }
        SegVec { data }
    }

    /// Bind: elementwise `(a + b) mod L`. Invertible, and the result resembles
    /// neither operand.
    pub fn bind(&self, other: &Self) -> Self {
        let mut data = [0u8; S];
        for i in 0..S {
            let v = self.data[i] as usize + other.data[i] as usize;
            data[i] = (if v >= L { v - L } else { v }) as u8;
        }
        SegVec { data }
    }

    /// Unbind: elementwise `(a - b) mod L`, the exact inverse of [`bind`](Self::bind).
    pub fn unbind(&self, other: &Self) -> Self {
        let mut data = [0u8; S];
        for i in 0..S {
            let v = self.data[i] as usize + L - other.data[i] as usize;
            data[i] = (if v >= L { v - L } else { v }) as u8;
        }
        SegVec { data }
    }

    /// Rotate the segments. Used as a position operator: `v.permute(k)` is
    /// dissimilar to `v` but recoverable by permuting back.
    pub fn permute(&self, shift: isize) -> Self {
        let mut data = [0u8; S];
        let s = S as isize;
        for i in 0..S {
            let idx = ((i as isize + shift) % s + s) % s;
            data[i] = self.data[idx as usize];
        }
        SegVec { data }
    }

    /// Number of segments on which two vectors agree. `S` for identical vectors,
    /// about `S / L` for unrelated ones.
    pub fn dot(&self, other: &Self) -> usize {
        (0..S).filter(|&i| self.data[i] == other.data[i]).count()
    }

    /// [`dot`](Self::dot) as a fraction of `S`.
    pub fn similarity(&self, other: &Self) -> f32 {
        self.dot(other) as f32 / S as f32
    }
}

impl<const S: usize, const L: usize> Default for SegVec<S, L> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const S: usize, const L: usize> Mul for SegVec<S, L> {
    type Output = SegVec<S, L>;
    fn mul(self, rhs: Self) -> Self {
        self.bind(&rhs)
    }
}

impl<const S: usize, const L: usize> Div for SegVec<S, L> {
    type Output = SegVec<S, L>;
    fn div(self, rhs: Self) -> Self {
        self.unbind(&rhs)
    }
}

impl<const S: usize, const L: usize> Add for SegVec<S, L> {
    type Output = Bundle<S, L>;
    /// Superposition of two vectors, as a `Bundle`.
    fn add(self, rhs: Self) -> Bundle<S, L> {
        let mut b = Bundle::zero();
        b.add_vec(&self, 1.0);
        b.add_vec(&rhs, 1.0);
        b
    }
}

/// A weighted superposition of vectors: one accumulator per (segment, value).
#[derive(Clone, Copy, Debug)]
pub struct Bundle<const S: usize, const L: usize> {
    counts: [[f32; L]; S],
}

impl<const S: usize, const L: usize> Bundle<S, L> {
    pub fn zero() -> Self {
        Bundle { counts: [[0.0f32; L]; S] }
    }

    pub fn get(&self, segment: usize, value: usize) -> f32 {
        self.counts[segment][value]
    }

    /// Superpose `v` with the given weight.
    pub fn add_vec(&mut self, v: &SegVec<S, L>, weight: f32) {
        for i in 0..S {
            self.counts[i][v.get(i) as usize] += weight;
        }
    }

    /// Move every accumulator a fraction of the way toward `v` — upstream's
    /// `space += rate * (item - space)` idiom for a slowly-updated memory.
    pub fn blend_vec(&mut self, v: &SegVec<S, L>, rate: f32) {
        for i in 0..S {
            for j in 0..L {
                let target = if j == v.get(i) as usize { 1.0 } else { 0.0 };
                self.counts[i][j] += rate * (target - self.counts[i][j]);
            }
        }
    }

    pub fn scale(&mut self, factor: f32) {
        for seg in self.counts.iter_mut() {
            for c in seg.iter_mut() {
                *c *= factor;
            }
        }
    }

    /// Collapse back to a single vector by taking each segment's argmax.
    ///
    /// Ties are broken deterministically by upstream's "context-dependent thinning":
    /// the tied indices are summed, and that sum modulo the number of ties selects
    /// which one wins. This is not decoration — an *empty* bundle has every value
    /// tied at zero, and without it every segment would collapse to 0 and every
    /// empty bundle would thin to the same vector.
    pub fn thin(&self) -> SegVec<S, L> {
        let mut out = SegVec::<S, L>::zero();

        for i in 0..S {
            let seg = &self.counts[i];

            let mut mv = 0.0f32;
            let mut mi = 0usize;
            for (j, &v) in seg.iter().enumerate() {
                if v > mv {
                    mv = v;
                    mi = j;
                }
            }

            let mut index_sum = 0usize;
            let mut count = 0usize;
            for (j, &v) in seg.iter().enumerate() {
                if v == mv {
                    index_sum += j;
                    count += 1;
                }
            }

            if count > 0 {
                let p = index_sum % count;
                let mut seen = 0usize;
                for (j, &v) in seg.iter().enumerate() {
                    if v == mv {
                        if seen == p {
                            mi = j;
                            break;
                        }
                        seen += 1;
                    }
                }
            }

            out.set(i, mi as u8);
        }

        out
    }
}

impl<const S: usize, const L: usize> Default for Bundle<S, L> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const S: usize, const L: usize> AddAssign<SegVec<S, L>> for Bundle<S, L> {
    fn add_assign(&mut self, rhs: SegVec<S, L>) {
        self.add_vec(&rhs, 1.0);
    }
}

impl<const S: usize, const L: usize> AddAssign<Bundle<S, L>> for Bundle<S, L> {
    fn add_assign(&mut self, rhs: Bundle<S, L>) {
        for i in 0..S {
            for j in 0..L {
                self.counts[i][j] += rhs.counts[i][j];
            }
        }
    }
}

/// Nearest entry in a codebook, by [`SegVec::dot`], with its similarity.
///
/// This is the "cleanup memory" every VSA design needs: unbinding recovers a noisy
/// version of what was stored, and only a comparison against the known symbols turns
/// it back into one of them.
pub fn cleanup<const S: usize, const L: usize>(
    query: &SegVec<S, L>,
    codebook: &[SegVec<S, L>],
) -> Option<(usize, f32)> {
    let mut best: Option<(usize, usize)> = None;
    for (i, c) in codebook.iter().enumerate() {
        let d = query.dot(c);
        if best.is_none_or(|(_, bd)| d > bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, d)| (i, d as f32 / S as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::Rng;

    const S: usize = 256;
    const L: usize = 8;
    type V = SegVec<S, L>;

    #[test]
    fn bind_then_unbind_is_exact() {
        let mut rng = Rng::new(1);
        for _ in 0..64 {
            let a = V::randomized(&mut rng);
            let k = V::randomized(&mut rng);
            assert_eq!((a.bind(&k)).unbind(&k), a);
            // And via the operators, so ported code reads the same.
            assert_eq!((a * k) / k, a);
        }
    }

    #[test]
    fn binding_destroys_similarity_to_both_operands() {
        let mut rng = Rng::new(2);
        let a = V::randomized(&mut rng);
        let k = V::randomized(&mut rng);
        let bound = a.bind(&k);

        let chance = 1.0 / L as f32;
        assert!(bound.similarity(&a) < chance * 3.0, "bound stayed similar to a");
        assert!(bound.similarity(&k) < chance * 3.0, "bound stayed similar to k");
    }

    #[test]
    fn random_vectors_agree_at_about_one_in_l() {
        let mut rng = Rng::new(3);
        let mut total = 0.0f32;
        let n = 200;
        for _ in 0..n {
            let a = V::randomized(&mut rng);
            let b = V::randomized(&mut rng);
            total += a.similarity(&b);
        }
        let mean = total / n as f32;
        let chance = 1.0 / L as f32;
        assert!(
            (mean - chance).abs() < chance * 0.25,
            "mean similarity {mean} is not near chance {chance}"
        );
    }

    #[test]
    fn a_vector_is_identical_to_itself() {
        let mut rng = Rng::new(4);
        let a = V::randomized(&mut rng);
        assert_eq!(a.dot(&a), S);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn permute_is_invertible_and_dissimilar() {
        let mut rng = Rng::new(5);
        let a = V::randomized(&mut rng);
        let p = a.permute(1);
        assert_eq!(p.permute(-1), a);
        assert!(p.similarity(&a) < 3.0 / L as f32, "permutation stayed similar");
    }

    #[test]
    fn a_bundle_of_k_vectors_keeps_each_one_recoverable() {
        // The capacity property: bundling superposes, and each member stays the
        // nearest codebook entry to the bundle until there are too many.
        let mut rng = Rng::new(6);
        let members: Vec<V> = (0..4).map(|_| V::randomized(&mut rng)).collect();
        let distractors: Vec<V> = (0..20).map(|_| V::randomized(&mut rng)).collect();

        let mut b = Bundle::<S, L>::zero();
        for m in &members {
            b.add_vec(m, 1.0);
        }
        let thinned = b.thin();

        for m in members.iter() {
            let member_sim = thinned.similarity(m);
            let best_distractor = distractors
                .iter()
                .map(|d| thinned.similarity(d))
                .fold(0.0f32, f32::max);
            assert!(
                member_sim > best_distractor,
                "a bundled member ({member_sim}) lost to a distractor ({best_distractor})"
            );
        }
    }

    #[test]
    fn bundle_capacity_degrades_as_more_is_added() {
        let mut rng = Rng::new(7);
        let sim_at = |k: usize, rng: &mut Rng| {
            let members: Vec<V> = (0..k).map(|_| V::randomized(rng)).collect();
            let mut b = Bundle::<S, L>::zero();
            for m in &members {
                b.add_vec(m, 1.0);
            }
            let t = b.thin();
            members.iter().map(|m| t.similarity(m)).sum::<f32>() / k as f32
        };

        let few = sim_at(2, &mut rng);
        let many = sim_at(24, &mut rng);
        assert!(few > many, "capacity did not degrade: {few} vs {many}");
        assert!(few > 0.5, "even two bundled vectors were not recoverable: {few}");
    }

    #[test]
    fn key_value_pairs_round_trip_through_a_bundle() {
        // The whole reason for the algebra: store several role-filler pairs in one
        // vector and get each filler back by unbinding its role.
        let mut rng = Rng::new(8);
        let keys: Vec<V> = (0..3).map(|_| V::randomized(&mut rng)).collect();
        let values: Vec<V> = (0..3).map(|_| V::randomized(&mut rng)).collect();

        let mut b = Bundle::<S, L>::zero();
        for i in 0..3 {
            b.add_vec(&keys[i].bind(&values[i]), 1.0);
        }
        let record = b.thin();

        for i in 0..3 {
            let recovered = record.unbind(&keys[i]);
            let (best, _) = cleanup(&recovered, &values).unwrap();
            assert_eq!(best, i, "unbinding key {i} recovered the wrong value");
        }
    }

    #[test]
    fn thinning_an_empty_bundle_is_deterministic_and_not_all_zero() {
        // Upstream's context-dependent tie-breaking. Without it every segment would
        // collapse to 0 and every empty bundle would thin to the same vector.
        let a = Bundle::<S, L>::zero().thin();
        let b = Bundle::<S, L>::zero().thin();
        assert_eq!(a, b, "thinning is not deterministic");

        let expected = (0..L).sum::<usize>() % L;
        assert_eq!(a.get(0) as usize, expected, "tie-break did not follow index_sum % count");
    }

    #[test]
    fn cleanup_finds_the_planted_entry() {
        let mut rng = Rng::new(9);
        let book: Vec<V> = (0..16).map(|_| V::randomized(&mut rng)).collect();
        for (i, entry) in book.iter().enumerate() {
            let (found, sim) = cleanup(entry, &book).unwrap();
            assert_eq!(found, i);
            assert!((sim - 1.0).abs() < 1e-6);
        }
        assert!(cleanup::<S, L>(&book[0], &[]).is_none());
    }

    #[test]
    fn cis_round_trip_for_feeding_a_hierarchy() {
        let mut rng = Rng::new(10);
        let a = V::randomized(&mut rng);
        let cis = a.to_cis();
        assert_eq!(cis.len(), S);
        assert!(cis.iter().all(|&c| (0..L as i32).contains(&c)));
        assert_eq!(V::from_cis(&cis), a);
    }
}
