// Signal generation for `wavy_line` and `wavy_classify`.
//
// Ported from `demos/Wavy_Line.cpp` and `demos/Wavy_Classify.cpp` in
// jacobeverist/OgmaNeoDemos @ aogmaneo. The constants are transcribed from those
// sources; they are arbitrary but the demos are calibrated around them, so
// changing them changes how hard the prediction problem is.

use crate::support::rng::Rng;

const PI: f32 = std::f32::consts::PI;

// --- Wavy_Line ---

/// Value range the `wavy_line` signals are binned over.
///
/// Upstream uses `[-1.25, 1.25]` against signals bounded by ±1, leaving headroom
/// for the injected noise. Note upstream does *not* clamp — a large enough noise
/// spike produces an out-of-range column index there. [`crate::support::encode`]
/// clamps, so this port saturates instead of corrupting memory.
pub const LINE_MIN: f32 = -1.25;
pub const LINE_MAX: f32 = 1.25;

/// The multi-sine generator from `Wavy_Line.cpp`.
///
/// Signal 0 is a product of three sines — aperiodic-looking over short windows,
/// which is what makes it a non-trivial prediction target. The remaining signals
/// are simple sums whose frequency scales with the channel index.
pub struct WavyLine {
    /// The phase counter. Public because the random jump below is the point of
    /// the demo: the hierarchy has to notice the context changed and recover.
    pub index: i64,
    num_inputs: usize,
    /// Amplitude of additive uniform noise. Upstream toggles this between 0 and
    /// 0.01 with the `n`/`c` keys.
    pub noise: f32,
    /// Per-step probability of teleporting `index`. Upstream hard-codes 0.001.
    pub jump_chance: f32,
    jumps: u64,
}

impl WavyLine {
    pub fn new(num_inputs: usize) -> Self {
        WavyLine { index: 0, num_inputs, noise: 0.0, jump_chance: 0.001, jumps: 0 }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// How many context jumps have happened so far.
    pub fn jumps(&self) -> u64 {
        self.jumps
    }

    /// Advance one step and return the current value of every channel.
    pub fn advance(&mut self, rng: &mut Rng) -> Vec<f32> {
        self.index += 1;

        if rng.chance(self.jump_chance) {
            self.index = rng.below(1001) as i64;
            self.jumps += 1;
        }

        self.sample(rng)
    }

    /// Evaluate the channels at the current index without advancing. Noise is
    /// still applied, so this is not pure — it matches upstream, where noise is
    /// added once per frame after the signals are computed.
    fn sample(&mut self, rng: &mut Rng) -> Vec<f32> {
        let t = self.index as f32;
        let mut out = Vec::with_capacity(self.num_inputs);

        for i in 0..self.num_inputs {
            let mut v = if i == 0 {
                (0.0125 * PI * t * 0.5 + 0.25).sin()
                    * (0.03 * PI * t + 1.5).sin()
                    * (0.025 * PI * t - 0.1).sin()
            } else {
                let fi = i as f32;
                0.8 * (0.02 * fi * PI * t).cos() + 0.2 * (0.05 * fi * PI * t).sin()
            };

            if self.noise > 0.0 {
                v += self.noise * rng.range(-1.0, 1.0);
            }

            out.push(v);
        }

        out
    }
}

// --- Wavy_Classify ---

/// Value range the `wavy_classify` signal is binned over.
pub const CLASS_MIN: f32 = -3.0;
pub const CLASS_MAX: f32 = 3.0;

/// Number of generated classes. Upstream's `numLabels` under the default
/// (non-`USE_SENSOR_DATA`) build.
pub const NUM_CLASSES: usize = 5;

/// The five-way classification signal from `Wavy_Classify.cpp`.
///
/// All five classes are built from the same three sines, so they are genuinely
/// confusable: classes 3 and 4 differ only by the presence of `in2`, and a
/// classifier has to integrate over time to tell them apart. That is the point —
/// a memoryless classifier cannot do better than chance on a single sample.
pub struct WavyClassify {
    pub index: i64,
}

impl WavyClassify {
    pub fn new() -> Self {
        WavyClassify { index: 0 }
    }

    pub fn advance(&mut self) {
        self.index += 1;
    }

    /// The signal value for `class` at the current index.
    pub fn value(&self, class: usize) -> f32 {
        let t = self.index as f32;
        let in0 = (0.025 * PI * t + 0.25).sin();
        let in1 = (0.09 * PI * t + 1.5).sin();
        let in2 = (0.05 * PI * t - 0.1).sin();

        match class {
            0 => in0,
            1 => in1,
            2 => in2,
            3 => in0 + in1,
            4 => in0 + in1 + in2,
            _ => panic!("class {class} out of range (expected 0..{NUM_CLASSES})"),
        }
    }
}

impl Default for WavyClassify {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::Rng;

    #[test]
    fn line_signals_stay_inside_the_encoding_range() {
        let mut rng = Rng::new(1);
        let mut w = WavyLine::new(2);
        w.jump_chance = 0.0;
        for _ in 0..5000 {
            for v in w.advance(&mut rng) {
                assert!((LINE_MIN..=LINE_MAX).contains(&v), "{v} escaped [{LINE_MIN}, {LINE_MAX}]");
            }
        }
    }

    #[test]
    fn jumps_are_recorded_and_land_in_range() {
        let mut rng = Rng::new(3);
        let mut w = WavyLine::new(1);
        w.jump_chance = 1.0;
        for _ in 0..200 {
            w.advance(&mut rng);
            assert!((0..=1000).contains(&w.index), "index {} out of jump range", w.index);
        }
        assert_eq!(w.jumps(), 200);
    }

    #[test]
    fn classify_values_stay_inside_the_encoding_range() {
        let mut c = WavyClassify::new();
        for _ in 0..5000 {
            c.advance();
            for class in 0..NUM_CLASSES {
                let v = c.value(class);
                assert!((CLASS_MIN..=CLASS_MAX).contains(&v), "class {class} produced {v}");
            }
        }
    }

    #[test]
    fn classes_are_actually_distinguishable_over_a_window() {
        // Sanity check that the five classes are not accidentally identical.
        let mut c = WavyClassify::new();
        let mut traces = vec![Vec::new(); NUM_CLASSES];
        for _ in 0..200 {
            c.advance();
            for (class, trace) in traces.iter_mut().enumerate() {
                trace.push(c.value(class));
            }
        }
        for a in 0..NUM_CLASSES {
            for b in (a + 1)..NUM_CLASSES {
                let diff: f32 = traces[a]
                    .iter()
                    .zip(&traces[b])
                    .map(|(x, y)| (x - y).abs())
                    .sum();
                assert!(diff > 1.0, "classes {a} and {b} are indistinguishable");
            }
        }
    }
}
