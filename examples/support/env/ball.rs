// Bouncing-ball world and its software rasteriser, for `ball_physics`.
//
// Ported from `demos/Ball_Physics.cpp` (jacobeverist/OgmaNeoDemos @ aogmaneo),
// which drives a Box2D world and renders it to a 64x64 SFML texture. A single
// circle under gravity in an axis-aligned box needs no solver, so the physics is
// hand-written here and Box2D is not a dependency — see `doc/Demos.md`.
//
// The geometry constants are Box2D body definitions translated into the surfaces
// they produce:
//
//   ground   SetAsBox(2500, 2.5) at (0, -2.5)   -> occupies y in [-5, 0], top face y = 0
//   walls    SetAsBox(2.5, 2500) at (±10, 0)    -> inner faces at x = ±7.5
//   ball     radius 1.4, restitution 0.82, friction 0.01
//
// so the ball's centre is confined to x in [-6.1, 6.1] and y >= 1.4.

use crate::support::rng::Rng;
use dcc_sph::helpers::{Int3, VisibleLayerDesc};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use dcc_sph::image_encoder::ImageEncoder;

pub const FRAME_W: usize = 64;
pub const FRAME_H: usize = 64;

/// Frames per episode. Upstream's `simFrames`.
pub const EPISODE_FRAMES: usize = 90;

const GRAVITY: f32 = -9.81;
const BALL_RADIUS: f32 = 1.4;
const RESTITUTION: f32 = 0.82;
const FRICTION: f32 = 0.01;

const GROUND_TOP: f32 = 0.0;
const WALL_INNER: f32 = 7.5;

const START_X: f32 = 0.0;
const START_Y: f32 = 8.2;

/// Upstream steps Box2D three times per frame at 1/30 s, so a frame advances 0.1 s.
const SUBSTEPS: usize = 3;
const SUB_DT: f32 = 1.0 / 30.0;

// --- View transform ---
//
// Upstream draws into a 64x64 render texture at 64 px/m through a default-
// constructed `sf::View`, whose size is SFML's default 1000x1000 view-pixels, and
// only re-centres it. So 1000 view-pixels map onto 64 target pixels, and each
// output pixel covers 1000/64 view-pixels / 64 px-per-m of world.
//
// That is easy to misread as a 1 m window — it is not; it is 15.6 m, which is what
// makes the walls and ground visible at all.

/// Metres of world per output pixel.
const WORLD_PER_PX: f32 = 1000.0 / FRAME_W as f32 / 64.0;
const VIEW_CENTER_X: f32 = 0.0;
/// Upstream centres the view at `pixelsPerMeter * (0, -7.5)` and negates y when
/// positioning bodies, which puts the centre at world y = +7.5.
const VIEW_CENTER_Y: f32 = 7.5;

pub struct BallWorld {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    /// Frames elapsed in the current episode.
    pub frame: usize,
    pub episode: usize,
}

impl BallWorld {
    pub fn new() -> Self {
        BallWorld { x: START_X, y: START_Y, vx: 0.0, vy: 0.0, frame: 0, episode: 0 }
    }

    /// Return the ball to its start position with a fresh random velocity.
    /// Upstream draws both components from `U(-8, 8)`.
    pub fn reset(&mut self, rng: &mut Rng) {
        self.x = START_X;
        self.y = START_Y;
        self.vx = rng.range(-8.0, 8.0);
        self.vy = rng.range(-8.0, 8.0);
        self.frame = 0;
        self.episode += 1;
    }

    /// Advance one visible frame (0.1 s, in three substeps).
    pub fn step(&mut self) {
        for _ in 0..SUBSTEPS {
            self.vy += GRAVITY * SUB_DT;
            self.x += self.vx * SUB_DT;
            self.y += self.vy * SUB_DT;

            let min_x = -WALL_INNER + BALL_RADIUS;
            let max_x = WALL_INNER - BALL_RADIUS;
            let min_y = GROUND_TOP + BALL_RADIUS;

            if self.x < min_x {
                self.x = min_x;
                self.vx = -self.vx * RESTITUTION;
            } else if self.x > max_x {
                self.x = max_x;
                self.vx = -self.vx * RESTITUTION;
            }

            if self.y < min_y {
                self.y = min_y;
                self.vy = -self.vy * RESTITUTION;
                // Stand in for Box2D's tangential friction at the contact. Small
                // enough that it barely shows, but the ball would otherwise never
                // lose horizontal speed and the sequence would be less varied.
                self.vx *= 1.0 - FRICTION;
            }
        }

        self.frame += 1;
    }

    pub fn episode_over(&self) -> bool {
        self.frame >= EPISODE_FRAMES
    }

    /// Rasterise the scene into `out`, which must be `FRAME_W * FRAME_H` bytes.
    ///
    /// Indexed `y + x * FRAME_H` — the `address2` layout the ImageEncoder expects
    /// for a single-channel visible layer, and the same one upstream writes.
    ///
    /// Everything is drawn solid white on black, matching upstream's untinted SFML
    /// shapes on a black clear: the encoder sees a silhouette, not a shaded scene.
    pub fn rasterise(&self, out: &mut [u8]) {
        assert_eq!(out.len(), FRAME_W * FRAME_H);
        out.fill(0);

        let half_w = FRAME_W as f32 * 0.5 * WORLD_PER_PX;
        let half_h = FRAME_H as f32 * 0.5 * WORLD_PER_PX;

        for px in 0..FRAME_W {
            let wx = VIEW_CENTER_X - half_w + (px as f32 + 0.5) * WORLD_PER_PX;

            for py in 0..FRAME_H {
                // Output rows run downward; world y runs upward.
                let wy = VIEW_CENTER_Y + half_h - (py as f32 + 0.5) * WORLD_PER_PX;

                let in_ground = wy <= GROUND_TOP;
                let in_wall = wx.abs() >= WALL_INNER;
                let dx = wx - self.x;
                let dy = wy - self.y;
                let in_ball = dx * dx + dy * dy <= BALL_RADIUS * BALL_RADIUS;

                if in_ground || in_wall || in_ball {
                    out[py + px * FRAME_H] = 255;
                }
            }
        }
    }
}

impl Default for BallWorld {
    fn default() -> Self {
        Self::new()
    }
}

/// The `ImageEncoder` and `Hierarchy` that `ball_physics` uses.
///
/// Both are returned together because the hierarchy's IO port is sized to the
/// encoder's hidden layer, so the two configurations cannot be chosen apart.
///
/// Upstream: encoder hidden 20x20x16 over one 64x64x1 visible layer at radius 6;
/// hierarchy 2 layers of 10x10x32 with `up_radius: 4` — a radius-2 field would see
/// too little of a 20x20 CSDR per column.
pub fn build() -> (ImageEncoder, Hierarchy) {
    let enc_hidden = Int3::new(20, 20, 16);

    let mut enc = ImageEncoder::default();
    enc.init_random(
        enc_hidden,
        vec![VisibleLayerDesc { size: Int3::new(FRAME_W as i32, FRAME_H as i32, 1), radius: 6 }],
    );

    let io_descs = vec![IoDesc {
        size: enc_hidden,
        io_type: IoType::Prediction,
        up_radius: 4,
        ..Default::default()
    }];

    let layer_descs: Vec<LayerDesc> = (0..2)
        .map(|_| LayerDesc {
            hidden_size: Int3::new(10, 10, 32),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: 0,
            down_radius: 2,
            ticks_per_update: 1,
        })
        .collect();

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    (enc, h)
}

/// World coordinates at the centre of output pixel `(px, py)`.
pub fn pixel_to_world(px: usize, py: usize) -> (f32, f32) {
    let half_w = FRAME_W as f32 * 0.5 * WORLD_PER_PX;
    let half_h = FRAME_H as f32 * 0.5 * WORLD_PER_PX;
    (
        VIEW_CENTER_X - half_w + (px as f32 + 0.5) * WORLD_PER_PX,
        VIEW_CENTER_Y + half_h - (py as f32 + 0.5) * WORLD_PER_PX,
    )
}

/// The static scene — ground and walls, no ball. Subtracting this isolates the
/// only thing that moves.
pub fn background_frame() -> Vec<u8> {
    let mut w = BallWorld::new();
    // Park the ball far outside the view so only the static geometry is drawn.
    w.x = 1.0e6;
    w.y = 1.0e6;
    let mut buf = vec![0u8; FRAME_W * FRAME_H];
    w.rasterise(&mut buf);
    buf
}

/// Minimum lit non-background pixels before a blob counts as a ball. The ball
/// covers a few hundred pixels when fully drawn, so this only rejects specks.
const MIN_BALL_PIXELS: usize = 12;

/// Recover the ball's world position from a frame by taking the intensity-weighted
/// centroid of everything that is not static background.
///
/// This is what makes a generated sequence judgeable. Frame MSE cannot do it:
/// predicting a *blank* frame scores better than predicting a ball slightly out of
/// place, because a misplaced ball is wrong twice over — once where it is drawn and
/// once where it should have been. An undertrained model that has given up and
/// emits an empty frame therefore beats a model that has genuinely learned the
/// dynamics but is a little off. Position error has no such perverse optimum.
pub fn detect_ball(frame: &[u8], background: &[u8]) -> Option<(f32, f32)> {
    assert_eq!(frame.len(), background.len());

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut weight = 0.0f64;
    let mut count = 0usize;

    for px in 0..FRAME_W {
        for py in 0..FRAME_H {
            let i = py + px * FRAME_H;
            if background[i] > 0 {
                continue;
            }
            let v = frame[i] as f64;
            if v <= 0.0 {
                continue;
            }
            let (wx, wy) = pixel_to_world(px, py);
            sum_x += wx as f64 * v;
            sum_y += wy as f64 * v;
            weight += v;
            count += 1;
        }
    }

    if count < MIN_BALL_PIXELS || weight <= 0.0 {
        return None;
    }

    Some(((sum_x / weight) as f32, (sum_y / weight) as f32))
}

/// Mean squared error between two frames, in units of normalised intensity.
pub fn frame_mse(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = (x as f32 - y as f32) / 255.0;
            (d * d) as f64
        })
        .sum();
    (sum / a.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::Rng;

    #[test]
    fn ball_stays_within_its_walls_and_above_the_ground() {
        let mut rng = Rng::new(4);
        let mut w = BallWorld::new();
        for _ in 0..50 {
            w.reset(&mut rng);
            for _ in 0..EPISODE_FRAMES {
                w.step();
                assert!(w.y >= GROUND_TOP + BALL_RADIUS - 1e-3, "sank to y={}", w.y);
                assert!(w.x.abs() <= WALL_INNER - BALL_RADIUS + 1e-3, "escaped to x={}", w.x);
            }
        }
    }

    #[test]
    fn ball_loses_energy_and_settles_rather_than_bouncing_forever() {
        let mut w = BallWorld::new();
        w.vx = 0.0;
        w.vy = 0.0;
        for _ in 0..2000 {
            w.step();
        }
        // With restitution 0.82 the bounce height decays geometrically.
        assert!(w.y < START_Y, "never lost height: y={}", w.y);
    }

    #[test]
    fn rasterise_draws_ground_walls_and_ball() {
        let w = BallWorld::new();
        let mut buf = vec![0u8; FRAME_W * FRAME_H];
        w.rasterise(&mut buf);

        let lit = buf.iter().filter(|&&v| v > 0).count();
        assert!(lit > 0, "frame is empty");
        assert!(lit < buf.len(), "frame is entirely filled");

        // The ball starts at (0, 8.2), which is the view centre column.
        let centre_col = FRAME_W / 2;
        assert!(
            (0..FRAME_H).any(|py| buf[py + centre_col * FRAME_H] > 0),
            "no ball in the centre column"
        );
    }

    #[test]
    fn detect_ball_recovers_the_true_position() {
        let bg = background_frame();
        let mut w = BallWorld::new();
        let mut buf = vec![0u8; FRAME_W * FRAME_H];

        for &(x, y) in &[(0.0f32, 8.2f32), (-4.0, 5.0), (3.5, 11.0)] {
            w.x = x;
            w.y = y;
            w.rasterise(&mut buf);
            let found = detect_ball(&buf, &bg).expect("ball not detected");
            // One output pixel is ~0.24 m, so half a pixel of slack.
            assert!((found.0 - x).abs() < 0.15, "x {} vs {}", found.0, x);
            assert!((found.1 - y).abs() < 0.15, "y {} vs {}", found.1, y);
        }
    }

    #[test]
    fn detect_ball_returns_none_on_a_frame_with_no_ball() {
        let bg = background_frame();
        assert!(detect_ball(&bg, &bg).is_none());
    }

    #[test]
    fn frame_mse_is_zero_for_identical_frames_and_one_for_inverted() {
        let a = vec![0u8; 16];
        let b = vec![255u8; 16];
        assert!(frame_mse(&a, &a) < 1e-9);
        assert!((frame_mse(&a, &b) - 1.0).abs() < 1e-6);
    }
}
