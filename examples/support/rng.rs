// A thin ergonomic wrapper over the crate's own PCG32.
//
// The demos deliberately do not pull in `rand`: `src/helpers.rs` already carries
// the PCG32 that AOgmaNeo uses, and driving environment randomness from the same
// generator means `--seed` reproduces a run exactly — the same property
// `tests/support/fidelity_scenario.rs` relies on.
//
// Note this is a *separate* stream from the library's thread-local global state.
// Seeding both (see `seed_everything`) is what makes a whole demo reproducible.

use dcc_sph::helpers::{
    rand_get_state, rand_normalf_step, rand_step, randf_range_step, randf_step, set_global_state,
};

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: rand_get_state(seed) }
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        randf_step(&mut self.state)
    }

    /// Uniform in `[low, high)`.
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        randf_range_step(low, high, &mut self.state)
    }

    /// Standard normal, via Box–Muller.
    pub fn normal(&mut self) -> f32 {
        rand_normalf_step(&mut self.state)
    }

    /// Uniform integer in `[0, n)`. Returns 0 for `n == 0` rather than dividing by zero.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (rand_step(&mut self.state) as usize) % n
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.unit() < p
    }
}

/// Seed both the library's global RNG (used for weight initialisation) and return
/// a demo-local stream for environment randomness.
///
/// Every demo calls this once at startup so that `--seed N` fully determines the run.
pub fn seed_everything(seed: u64) -> Rng {
    set_global_state(rand_get_state(seed));
    // Offset the environment stream so it does not replay the same sequence the
    // hierarchy's initialisation just consumed.
    Rng::new(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_the_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..64 {
            assert_eq!(a.unit().to_bits(), b.unit().to_bits());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let da: Vec<u32> = (0..32).map(|_| a.unit().to_bits()).collect();
        let db: Vec<u32> = (0..32).map(|_| b.unit().to_bits()).collect();
        assert_ne!(da, db);
    }

    #[test]
    fn below_stays_in_range_and_handles_zero() {
        let mut r = Rng::new(7);
        for _ in 0..500 {
            assert!(r.below(5) < 5);
        }
        assert_eq!(r.below(0), 0);
    }

    #[test]
    fn unit_stays_in_zero_one() {
        let mut r = Rng::new(9);
        for _ in 0..1000 {
            let v = r.unit();
            assert!((0.0..1.0).contains(&v), "{v}");
        }
    }
}
