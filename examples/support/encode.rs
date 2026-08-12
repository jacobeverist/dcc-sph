// Scalar → CSDR encoding helpers shared by the demos.
//
// Upstream OgmaNeoDemos binds scalars into columns with one recurring idiom:
//
//     static_cast<int>(value_in_0_1 * (res - 1) + 0.5f)
//
// `static_cast<int>` truncates toward zero, so for non-negative inputs that is
// round-half-up. Rust's `as i32` truncates the same way, so the ports below are
// faithful — but only because the values are clamped non-negative first. Do not
// "simplify" these to `.round()`: that rounds half away from zero and would
// diverge on exact .5 boundaries.

use dcc_sph::helpers::sigmoidf;

/// Bin a value already in `[0, 1]` into `[0, res - 1]`. Out-of-range input is
/// clamped, which several upstream demos rely on (Pusher deliberately saturates
/// a `[-2, 2]` delta into a 16-level column).
pub fn bin_unit(v: f32, res: i32) -> i32 {
    let v = v.clamp(0.0, 1.0);
    ((v * (res - 1) as f32 + 0.5) as i32).clamp(0, res - 1)
}

/// Bin a value from `[lo, hi]` into `[0, res - 1]`.
pub fn bin_range(v: f32, lo: f32, hi: f32, res: i32) -> i32 {
    bin_unit((v - lo) / (hi - lo), res)
}

/// Inverse of [`bin_range`] — the bin's representative value.
pub fn unbin_range(ci: i32, lo: f32, hi: f32, res: i32) -> f32 {
    ci as f32 / (res - 1) as f32 * (hi - lo) + lo
}

/// Squash an unbounded value through a sigmoid, then bin it. This is upstream's
/// `binningEncoder` from `demos/csdrScalarEncoder.hpp`, and how `Runner_Run`
/// handles sensors with no natural range (joint torques, IMU deltas).
pub fn bin_sigmoid(v: f32, res: i32, squash: f32) -> i32 {
    bin_unit(sigmoidf(v * squash), res)
}

/// Encode a float in `[0, 1]` as two 4-bit nibbles (2 columns, 16 cells each).
///
/// Upstream `Unorm8ToCSDR`. Used by `wave_prediction`; the wavy demos use the
/// single-column [`bin_range`] path instead, matching upstream's
/// `SINGLE_COLUMN_ENCODER` default.
pub fn unorm8_to_csdr(x: f32) -> [i32; 2] {
    let i = (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8 as i32;
    [i & 0x0f, (i >> 4) & 0x0f]
}

/// Decode two 4-bit nibble indices back to a float in `[0, 1]`.
pub fn csdr_to_unorm8(csdr: &[i32]) -> f32 {
    (csdr[0] | (csdr[1] << 4)) as f32 / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_unit_spans_the_full_range_and_clamps() {
        assert_eq!(bin_unit(0.0, 16), 0);
        assert_eq!(bin_unit(1.0, 16), 15);
        assert_eq!(bin_unit(-5.0, 16), 0);
        assert_eq!(bin_unit(5.0, 16), 15);
    }

    #[test]
    fn bin_range_round_trips_within_half_a_bin() {
        let (lo, hi, res) = (-1.25f32, 1.25f32, 64);
        let half_bin = (hi - lo) / (res - 1) as f32 * 0.5;
        for i in 0..=100 {
            let v = lo + (hi - lo) * i as f32 / 100.0;
            let back = unbin_range(bin_range(v, lo, hi, res), lo, hi, res);
            assert!((back - v).abs() <= half_bin + 1e-5, "v={v} back={back}");
        }
    }

    #[test]
    fn unorm8_round_trips_to_byte_precision() {
        for i in 0..=255u32 {
            let x = i as f32 / 255.0;
            let back = csdr_to_unorm8(&unorm8_to_csdr(x));
            assert!((back - x).abs() < 1.0 / 255.0);
        }
    }
}
