// Text output for the headless demos.
//
// The demos run without a window by default, so this module is what makes them
// worth watching: rolling statistics, and small ASCII renderings of the things a
// windowed demo would draw (a signal trace, a predicted frame, a scatter of
// learned receptive fields).

use std::collections::VecDeque;
use std::fmt::Write as _;

/// Windowed mean plus an exponential moving average.
///
/// The window answers "how is it doing right now", the EMA answers "which way is
/// it trending" — RL demos need both, because a windowed mean over a sparse
/// reward is mostly zeros.
pub struct Rolling {
    window: VecDeque<f32>,
    capacity: usize,
    sum: f64,
    ema: f32,
    alpha: f32,
    seeded: bool,
    count: u64,
}

impl Rolling {
    pub fn new(capacity: usize, alpha: f32) -> Self {
        Rolling {
            window: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
            ema: 0.0,
            alpha,
            seeded: false,
            count: 0,
        }
    }

    pub fn push(&mut self, v: f32) {
        self.count += 1;

        if self.window.len() == self.capacity {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old as f64;
            }
        }
        self.window.push_back(v);
        self.sum += v as f64;

        if self.seeded {
            self.ema += self.alpha * (v - self.ema);
        } else {
            self.ema = v;
            self.seeded = true;
        }
    }

    pub fn mean(&self) -> f32 {
        if self.window.is_empty() {
            0.0
        } else {
            (self.sum / self.window.len() as f64) as f32
        }
    }

    pub fn ema(&self) -> f32 {
        self.ema
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn as_slice(&self) -> Vec<f32> {
        self.window.iter().copied().collect()
    }
}

const BLOCKS: [char; 8] = ['\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];
const RAMP: &[u8] = b" .:-=+*#%@";

/// One-line Unicode block sparkline, auto-scaled to the data's own range.
///
/// A flat series renders as a row of the lowest block rather than dividing by a
/// zero range.
pub fn sparkline(values: &[f32]) -> String {
    if values.is_empty() {
        return String::new();
    }

    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return "?".repeat(values.len());
    }

    let lo = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = hi - lo;

    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return '?';
            }
            if span <= f32::EPSILON {
                return BLOCKS[0];
            }
            let t = ((v - lo) / span * (BLOCKS.len() - 1) as f32).round() as usize;
            BLOCKS[t.min(BLOCKS.len() - 1)]
        })
        .collect()
}

/// Render a value in `[0, 1]` as a fixed-width bar.
///
/// Moved here verbatim from `examples/wave_prediction.rs`, which now imports it.
pub fn ascii_bar(x: f32) -> String {
    let filled = ((x * 16.0 + 0.5) as usize).min(16);
    let mut s = String::with_capacity(16);
    for i in 0..16 {
        s.push(if i < filled { '\u{2588}' } else { '\u{2591}' });
    }
    s
}

/// Render a grayscale byte buffer as ASCII, box-filter downsampled to
/// `out_w` × `out_h`.
///
/// `pixels` is indexed `y + x * h` — the `address2` layout the image demos use
/// throughout, matching upstream's `imgb[y + x * height]`.
///
/// Terminal cells are roughly twice as tall as they are wide, so a square image
/// wants `out_w ≈ 2 * out_h` to look square.
pub fn ascii_image(pixels: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> String {
    assert_eq!(pixels.len(), w * h, "pixel buffer is not {w}x{h}");

    let mut out = String::with_capacity((out_w + 1) * out_h);

    for oy in 0..out_h {
        let y0 = oy * h / out_h;
        let y1 = (((oy + 1) * h).div_ceil(out_h)).max(y0 + 1).min(h);

        for ox in 0..out_w {
            let x0 = ox * w / out_w;
            let x1 = (((ox + 1) * w).div_ceil(out_w)).max(x0 + 1).min(w);

            let mut sum = 0u32;
            let mut n = 0u32;
            for x in x0..x1 {
                for y in y0..y1 {
                    sum += pixels[y + x * h] as u32;
                    n += 1;
                }
            }

            let mean = if n == 0 { 0 } else { sum / n };
            let idx = (mean as usize * (RAMP.len() - 1)) / 255;
            out.push(RAMP[idx] as char);
        }
        out.push('\n');
    }

    out
}

/// Lay two multi-line blocks out side by side, for comparing a predicted frame
/// against the real one.
pub fn side_by_side(left: &str, right: &str, gap: usize) -> String {
    let l: Vec<&str> = left.lines().collect();
    let r: Vec<&str> = right.lines().collect();
    let width = l.iter().map(|s| s.chars().count()).max().unwrap_or(0);

    let mut out = String::new();
    for i in 0..l.len().max(r.len()) {
        let lhs = l.get(i).copied().unwrap_or("");
        let pad = width.saturating_sub(lhs.chars().count());
        out.push_str(lhs);
        out.push_str(&" ".repeat(pad + gap));
        out.push_str(r.get(i).copied().unwrap_or(""));
        out.push('\n');
    }
    out
}

/// Axis-aligned bounds for [`ascii_scatter`].
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

impl Bounds {
    pub fn square(r: f32) -> Self {
        Bounds { min_x: -r, max_x: r, min_y: -r, max_y: r }
    }

    pub fn of(points: &[(f32, f32)]) -> Self {
        let mut b = Bounds { min_x: f32::INFINITY, max_x: f32::NEG_INFINITY, min_y: f32::INFINITY, max_y: f32::NEG_INFINITY };
        for &(x, y) in points {
            if x.is_finite() && y.is_finite() {
                b.min_x = b.min_x.min(x);
                b.max_x = b.max_x.max(x);
                b.min_y = b.min_y.min(y);
                b.max_y = b.max_y.max(y);
            }
        }
        if !b.min_x.is_finite() {
            return Bounds::square(1.0);
        }
        // Guard against a degenerate axis so the mapping below never divides by zero.
        if (b.max_x - b.min_x) < 1e-6 {
            b.min_x -= 0.5;
            b.max_x += 0.5;
        }
        if (b.max_y - b.min_y) < 1e-6 {
            b.min_y -= 0.5;
            b.max_y += 0.5;
        }
        b
    }
}

/// Plot layered point sets into an ASCII grid. Later layers draw over earlier
/// ones, so pass background samples first and the thing you care about last.
///
/// y increases upward, unlike the SFML demos this ports from.
pub fn ascii_scatter(layers: &[(char, &[(f32, f32)])], b: Bounds, w: usize, h: usize) -> String {
    let mut grid = vec![b' '; w * h];

    for &(mark, points) in layers {
        for &(x, y) in points {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let fx = (x - b.min_x) / (b.max_x - b.min_x);
            let fy = (y - b.min_y) / (b.max_y - b.min_y);
            if !(0.0..=1.0).contains(&fx) || !(0.0..=1.0).contains(&fy) {
                continue;
            }
            let px = ((fx * (w - 1) as f32) as usize).min(w - 1);
            let py = ((1.0 - fy) * (h - 1) as f32) as usize;
            grid[px + py.min(h - 1) * w] = mark as u8;
        }
    }

    let mut out = String::with_capacity((w + 1) * h);
    for row in grid.chunks(w) {
        for &c in row {
            out.push(c as char);
        }
        out.push('\n');
    }
    out
}

/// Format a confusion matrix with per-class recall. Rows are the true class,
/// columns the predicted one.
pub fn confusion_table(counts: &[Vec<u64>], labels: &[String]) -> String {
    let n = labels.len();
    let mut out = String::new();

    let _ = write!(out, "{:>10} |", "true\\pred");
    for l in labels {
        let _ = write!(out, " {l:>7}");
    }
    let _ = writeln!(out, " |  recall");
    let _ = writeln!(out, "{}", "-".repeat(12 + 8 * n + 11));

    for i in 0..n {
        let _ = write!(out, "{:>10} |", labels[i]);
        let total: u64 = counts[i].iter().sum();
        for j in 0..n {
            let _ = write!(out, " {:>7}", counts[i][j]);
        }
        let recall = if total == 0 { f64::NAN } else { counts[i][i] as f64 / total as f64 };
        let _ = writeln!(out, " |  {:>6.1}%", recall * 100.0);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_tracks_window_mean_and_drops_old_values() {
        let mut r = Rolling::new(3, 0.5);
        for v in [1.0, 2.0, 3.0, 4.0] {
            r.push(v);
        }
        // Window holds the last three: 2, 3, 4.
        assert!((r.mean() - 3.0).abs() < 1e-6);
        assert_eq!(r.len(), 3);
        assert_eq!(r.count(), 4);
    }

    #[test]
    fn sparkline_handles_flat_and_empty_series() {
        assert_eq!(sparkline(&[]), "");
        assert_eq!(sparkline(&[5.0, 5.0, 5.0]).chars().count(), 3);
        assert_eq!(sparkline(&[0.0, 1.0]).chars().count(), 2);
    }

    #[test]
    fn ascii_image_downsamples_to_the_requested_shape() {
        let pixels = vec![0u8; 64 * 64];
        let s = ascii_image(&pixels, 64, 64, 16, 8);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 8);
        assert!(lines.iter().all(|l| l.chars().count() == 16));
    }

    #[test]
    fn ascii_image_maps_black_and_white_to_ramp_ends() {
        let black = ascii_image(&[0u8; 4], 2, 2, 2, 2);
        let white = ascii_image(&[255u8; 4], 2, 2, 2, 2);
        assert!(black.trim_end().chars().all(|c| c == ' ' || c == '\n'));
        assert!(white.contains('@'));
    }

    #[test]
    fn ascii_scatter_draws_later_layers_on_top() {
        let a = [(0.0f32, 0.0f32)];
        let b = [(0.0f32, 0.0f32)];
        let s = ascii_scatter(&[('.', &a), ('#', &b)], Bounds::square(1.0), 9, 9);
        assert!(s.contains('#'));
        assert!(!s.contains('.'));
    }

    #[test]
    fn ascii_scatter_skips_out_of_bounds_points() {
        let pts = [(99.0f32, 99.0f32)];
        let s = ascii_scatter(&[('#', &pts)], Bounds::square(1.0), 5, 5);
        assert!(!s.contains('#'));
    }
}
