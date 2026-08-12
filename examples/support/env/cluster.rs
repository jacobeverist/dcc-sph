// Two-dimensional data sources for the encoder visualisers.
//
// `gaussian_clusters` ports the synthetic data in `demos/Topo_Test_AON.cpp`.
// `DensityField` replaces `demos/EncVis.cpp`'s `resources/density_image5.png`,
// which is referenced by that demo but **absent from the upstream repository** —
// generating the field procedurally means the demo runs out of the box and is
// reproducible from `--seed`. See `doc/Demos.md`.

use crate::support::rng::Rng;
use dcc_sph::encoder::Encoder;
use dcc_sph::helpers::{Int3, VisibleLayerDesc};

const PI: f32 = std::f32::consts::PI;

/// The bare `Encoder` that `enc_vis` probes.
///
/// One visible layer of 1x2 columns: column 0 carries x, column 1 carries y. The
/// default radius of 2 means each hidden column's receptive field covers both, so
/// every cell sees a full (x, y) pair.
pub fn build_enc_vis_encoder(columns: i32, cells: i32, resolution: i32) -> Encoder {
    let mut e = Encoder::default();
    e.init_random(
        Int3::new(1, columns, cells),
        vec![VisibleLayerDesc { size: Int3::new(1, 2, resolution), radius: 2 }],
    );
    e
}

/// The bare `Encoder` that `topo_test` probes. Upstream: hidden 4x4x16 over a
/// 1x2x64 visible layer.
pub fn build_topo_encoder(hidden: Int3, resolution: i32) -> Encoder {
    let mut e = Encoder::default();
    e.init_random(
        hidden,
        vec![VisibleLayerDesc { size: Int3::new(1, 2, resolution), radius: 2 }],
    );
    e
}

/// Sample `num_clusters` affine-transformed Gaussian blobs, `points_per_cluster`
/// points each, returned with the cluster index alongside each point.
///
/// Upstream builds each cluster as
/// `translate(pos) * rotate(U(0, 2pi)) * scale(U(0,1), U(0,1))` applied to
/// `N(0,1)^2 * 0.1`, with `pos` uniform in `[-0.9, 0.9]^2`. The independent x and y
/// scales are what make the blobs elongated and the rotation is what makes their
/// orientation arbitrary — together they give the encoder something with real
/// structure to drape itself over, rather than isotropic dots.
pub fn gaussian_clusters(
    num_clusters: usize,
    points_per_cluster: usize,
    rng: &mut Rng,
) -> Vec<(f32, f32, usize)> {
    let mut out = Vec::with_capacity(num_clusters * points_per_cluster);

    for c in 0..num_clusters {
        let cx = rng.range(-1.0, 1.0) * 0.9;
        let cy = rng.range(-1.0, 1.0) * 0.9;
        let theta = rng.unit() * 2.0 * PI;
        let (sx, sy) = (rng.unit(), rng.unit());
        let (sin, cos) = theta.sin_cos();

        for _ in 0..points_per_cluster {
            let a = sx * rng.normal() * 0.1;
            let b = sy * rng.normal() * 0.1;
            out.push((cx + a * cos - b * sin, cy + a * sin + b * cos, c));
        }
    }

    out
}

/// A 2-D probability density on a pixel grid, sampled by inverse CDF.
///
/// Layout is `density[y + x * h]`, matching upstream's image indexing.
pub struct DensityField {
    pub w: usize,
    pub h: usize,
    density: Vec<f32>,
    /// Prefix sums over `density`, for binary-search sampling. Upstream rescans the
    /// whole image linearly for every sample; this is the same distribution at
    /// O(log n) instead of O(w*h) per draw, which matters when a demo takes
    /// hundreds of thousands of samples.
    cdf: Vec<f32>,
}

impl DensityField {
    /// Build a field with enough structure to be worth looking at: three Gaussian
    /// blobs of differing width plus an annulus, so the encoder has both compact
    /// modes and an extended curved manifold to cover.
    pub fn procedural(w: usize, h: usize) -> Self {
        let mut density = vec![0.0f32; w * h];

        // (centre x, centre y, sigma, weight), in normalised [0,1] coordinates.
        let blobs = [
            (0.25f32, 0.30f32, 0.07f32, 1.0f32),
            (0.72, 0.24, 0.05, 0.9),
            (0.50, 0.78, 0.10, 0.8),
        ];
        let (ring_x, ring_y, ring_r, ring_sigma, ring_weight) = (0.5f32, 0.5f32, 0.34f32, 0.035f32, 0.7f32);

        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            for y in 0..h {
                let v = (y as f32 + 0.5) / h as f32;

                let mut d = 0.0f32;

                for &(bx, by, sigma, weight) in &blobs {
                    let dx = u - bx;
                    let dy = v - by;
                    d += weight * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                }

                let rd = ((u - ring_x).powi(2) + (v - ring_y).powi(2)).sqrt() - ring_r;
                d += ring_weight * (-(rd * rd) / (2.0 * ring_sigma * ring_sigma)).exp();

                density[y + x * h] = d;
            }
        }

        let mut cdf = Vec::with_capacity(density.len());
        let mut running = 0.0f32;
        for &d in &density {
            running += d;
            cdf.push(running);
        }

        DensityField { w, h, density, cdf }
    }

    pub fn total(&self) -> f32 {
        *self.cdf.last().unwrap_or(&0.0)
    }

    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.density[y + x * self.h]
    }

    /// Draw one point, returned in normalised `[0, 1)^2`.
    pub fn sample(&self, rng: &mut Rng) -> (f32, f32) {
        let target = rng.unit() * self.total();

        // First index whose prefix sum reaches `target`.
        let idx = match self.cdf.binary_search_by(|probe| {
            probe.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(i) => i,
        }
        .min(self.cdf.len() - 1);

        let x = idx / self.h;
        let y = idx % self.h;

        (
            (x as f32 + 0.5) / self.w as f32,
            (y as f32 + 0.5) / self.h as f32,
        )
    }

    /// Draw `n` points, for plotting the field itself.
    pub fn points(&self, n: usize, rng: &mut Rng) -> Vec<(f32, f32)> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::Rng;

    #[test]
    fn clusters_are_the_requested_shape_and_stay_in_bounds() {
        let mut rng = Rng::new(11);
        let pts = gaussian_clusters(8, 100, &mut rng);
        assert_eq!(pts.len(), 800);
        assert!(pts.iter().all(|&(_, _, c)| c < 8));
        // pos is within 0.9 and the blobs have sigma <= 0.1, so a few sigma of slack.
        assert!(
            pts.iter().all(|&(x, y, _)| x.abs() < 1.5 && y.abs() < 1.5),
            "a cluster point escaped the plotting range"
        );
    }

    #[test]
    fn clusters_are_actually_separated() {
        let mut rng = Rng::new(12);
        let pts = gaussian_clusters(4, 100, &mut rng);
        let centroid = |c: usize| {
            let members: Vec<_> = pts.iter().filter(|p| p.2 == c).collect();
            let n = members.len() as f32;
            (
                members.iter().map(|p| p.0).sum::<f32>() / n,
                members.iter().map(|p| p.1).sum::<f32>() / n,
            )
        };
        // At least one pair should be clearly apart — they are placed at random, so
        // this is a sanity check on the generator, not on any particular seed.
        let mut max_sep: f32 = 0.0;
        for a in 0..4 {
            for b in (a + 1)..4 {
                let (ax, ay) = centroid(a);
                let (bx, by) = centroid(b);
                max_sep = max_sep.max(((ax - bx).powi(2) + (ay - by).powi(2)).sqrt());
            }
        }
        assert!(max_sep > 0.3, "clusters all landed on top of each other");
    }

    #[test]
    fn density_samples_land_in_range_and_follow_the_density() {
        let mut rng = Rng::new(13);
        let field = DensityField::procedural(64, 64);

        let mut high = 0usize;
        for _ in 0..4000 {
            let (u, v) = field.sample(&mut rng);
            assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));

            let x = ((u * 64.0) as usize).min(63);
            let y = ((v * 64.0) as usize).min(63);
            if field.at(x, y) > 0.1 {
                high += 1;
            }
        }
        // Sampling proportional to density should almost never land in dead space.
        assert!(high > 3600, "only {high}/4000 samples landed in dense regions");
    }

    #[test]
    fn density_field_is_not_uniform() {
        let field = DensityField::procedural(32, 32);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for x in 0..32 {
            for y in 0..32 {
                lo = lo.min(field.at(x, y));
                hi = hi.max(field.at(x, y));
            }
        }
        assert!(hi > lo * 10.0 + 0.1, "field is nearly flat: {lo}..{hi}");
    }
}
