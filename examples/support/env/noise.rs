// Corrupted-pattern source for `noise_robustness`.
//
// RUST-ONLY. This has no upstream counterpart in OgmaNeoDemos — it exists so the
// three ports (`dcc-sph`, `dcc-sparsey`, `dcc-htm`) can each answer the same
// question in the same shape: once a representation has been learned, how much
// input corruption does it survive? See `doc/Demos.md`.
//
// The world is deliberately trivial — a fixed book of random patterns, each held
// for a few steps — because the *task* is not the point. What is being measured is
// the degradation curve, and anything with interesting dynamics of its own would
// confound it.

use crate::support::rng::Rng;
use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use std::collections::HashMap;

/// Observation grid width, in columns.
pub const GRID_W: i32 = 4;
/// Observation grid height, in columns.
pub const GRID_H: i32 = 4;
/// Total observation columns.
pub const COLUMNS: usize = (GRID_W * GRID_H) as usize;
/// Cells per observation column.
pub const COLUMN_SIZE: i32 = 16;

/// A fixed set of patterns to be memorised, one per class.
pub struct PatternBook {
    patterns: Vec<Vec<i32>>,
}

impl PatternBook {
    /// Draw `count` patterns of `COLUMNS` columns, each column an independent
    /// uniform cell index.
    pub fn generate(count: usize, rng: &mut Rng) -> Self {
        let patterns = (0..count)
            .map(|_| {
                (0..COLUMNS)
                    .map(|_| rng.below(COLUMN_SIZE as usize) as i32)
                    .collect()
            })
            .collect();
        PatternBook { patterns }
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    pub fn get(&self, i: usize) -> &[i32] {
        &self.patterns[i]
    }

    /// The smallest number of differing columns between any two patterns.
    ///
    /// Worth reporting rather than assuming: with `COLUMNS` columns drawn
    /// independently, two patterns *can* land close together, and if they do the
    /// task is unwinnable at a corruption level that would otherwise be easy. A
    /// surprising result is much easier to read when this number is on the page.
    pub fn min_separation(&self) -> usize {
        let mut worst = COLUMNS;
        for i in 0..self.patterns.len() {
            for j in (i + 1)..self.patterns.len() {
                let d = self.patterns[i]
                    .iter()
                    .zip(self.patterns[j].iter())
                    .filter(|(a, b)| a != b)
                    .count();
                worst = worst.min(d);
            }
        }
        worst
    }
}

/// Replaces a fixed number of columns with a *different* cell index.
///
/// Two decisions here are load-bearing. The count is `round(fraction * COLUMNS)`
/// rather than a per-column coin flip, so every test step at a given `--noise` is
/// corrupted by exactly the same amount and the sweep measures the corruption
/// level rather than the variance around it. And the replacement is drawn from the
/// `COLUMN_SIZE - 1` values the column does *not* currently hold, so "corrupt 25%
/// of columns" corrupts 25% — a uniform redraw would silently pick the original
/// value about 1 time in `COLUMN_SIZE` and make the effective level `fraction *
/// (1 - 1/COLUMN_SIZE)`.
pub struct Corruptor {
    order: Vec<usize>,
}

impl Corruptor {
    pub fn new() -> Self {
        Corruptor {
            order: (0..COLUMNS).collect(),
        }
    }

    /// Write a corrupted copy of `clean` into `out`, returning how many columns
    /// were changed.
    pub fn apply(&mut self, clean: &[i32], fraction: f32, rng: &mut Rng, out: &mut [i32]) -> usize {
        out.copy_from_slice(clean);

        let k = ((fraction.clamp(0.0, 1.0) * COLUMNS as f32) + 0.5) as usize;
        if k == 0 {
            return 0;
        }

        // Partial Fisher-Yates: pick `k` distinct columns without allocating.
        for i in 0..k {
            let j = i + rng.below(COLUMNS - i);
            self.order.swap(i, j);
            let col = self.order[i];
            // Draw from the values this column does not already hold.
            let mut v = rng.below(COLUMN_SIZE as usize - 1) as i32;
            if v >= clean[col] {
                v += 1;
            }
            out[col] = v;
        }
        k
    }
}

impl Default for Corruptor {
    fn default() -> Self {
        Self::new()
    }
}

/// Exact-match memorisation — the control this demo exists to be compared against.
///
/// It is perfect on a clean pattern and blind to everything else, so its curve is
/// the shape a system that has memorised rather than generalised produces. It is
/// not a competitor to beat; it is the null hypothesis for the *shape* of the
/// degradation curve.
#[derive(Default)]
pub struct LookupTable {
    table: HashMap<Vec<i32>, usize>,
}

impl LookupTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn learn(&mut self, pattern: &[i32], label: usize) {
        self.table.insert(pattern.to_vec(), label);
    }

    pub fn classify(&self, pattern: &[i32]) -> Option<usize> {
        self.table.get(pattern).copied()
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

/// Observation port plus a label port, over a stack where layer `l` updates once
/// every `2^l` bottom-layer steps.
///
/// The tick stack is the same choice `wavy_classify` makes and for the same
/// reason: a pattern is held for several steps, so a layer that integrates over a
/// longer window sees the whole presentation rather than one frame of it.
///
/// Defined here rather than in the demo so a sweep, a checkpoint round trip and any
/// future viewer all drive exactly the same configuration.
pub fn build_hierarchy(classes: usize, num_layers: usize) -> Hierarchy {
    let io_descs = vec![
        IoDesc {
            size: Int3::new(GRID_W, GRID_H, COLUMN_SIZE),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 4,
            up_radius: 2,
            down_radius: 2,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, 1, classes as i32),
            io_type: IoType::Prediction,
            ..Default::default()
        },
    ];

    let layer_descs: Vec<LayerDesc> = (0..num_layers)
        .map(|l| LayerDesc {
            hidden_size: Int3::new(5, 5, 64),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: 0,
            down_radius: 2,
            ticks_per_update: 1usize << l,
            top_feedback: false,
        })
        .collect();

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::Rng;

    #[test]
    fn corruption_changes_exactly_the_requested_column_count() {
        let mut rng = Rng::new(1);
        let book = PatternBook::generate(1, &mut rng);
        let clean = book.get(0).to_vec();
        let mut out = vec![0i32; COLUMNS];
        let mut c = Corruptor::new();

        for (fraction, expected) in [(0.0, 0), (0.25, 4), (0.5, 8), (1.0, 16)] {
            let reported = c.apply(&clean, fraction, &mut rng, &mut out);
            let actual = clean.iter().zip(out.iter()).filter(|(a, b)| a != b).count();
            assert_eq!(reported, expected, "reported count at {fraction}");
            // The real point: a uniform redraw would make this flaky, because it
            // would sometimes pick the value already there.
            assert_eq!(actual, expected, "actual changed columns at {fraction}");
        }
    }

    #[test]
    fn corrupted_columns_stay_in_range() {
        let mut rng = Rng::new(7);
        let book = PatternBook::generate(1, &mut rng);
        let clean = book.get(0).to_vec();
        let mut out = vec![0i32; COLUMNS];
        let mut c = Corruptor::new();
        for _ in 0..200 {
            c.apply(&clean, 0.5, &mut rng, &mut out);
            assert!(out.iter().all(|&v| (0..COLUMN_SIZE).contains(&v)));
        }
    }

    #[test]
    fn lookup_is_perfect_on_clean_input_and_blind_off_it() {
        let mut rng = Rng::new(3);
        let book = PatternBook::generate(8, &mut rng);
        let mut table = LookupTable::new();
        for c in 0..book.len() {
            table.learn(book.get(c), c);
        }

        for c in 0..book.len() {
            assert_eq!(table.classify(book.get(c)), Some(c));
        }

        let mut out = vec![0i32; COLUMNS];
        let mut corruptor = Corruptor::new();
        corruptor.apply(book.get(0), 0.25, &mut rng, &mut out);
        assert_eq!(table.classify(&out), None, "a corrupted pattern must miss");
    }

    #[test]
    fn generated_patterns_are_separated() {
        let mut rng = Rng::new(11);
        let book = PatternBook::generate(8, &mut rng);
        // Not a tight bound — just a guard that the book is not degenerate, which
        // would make every accuracy number meaningless.
        assert!(
            book.min_separation() >= 4,
            "patterns too close: min separation {}",
            book.min_separation()
        );
    }
}
