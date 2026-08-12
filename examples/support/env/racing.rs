// The race track and car, for `car_racing`.
//
// Ported from `demos/Car_Racing.cpp` (jacobeverist/OgmaNeoDemos @ aogmaneo).
//
// Two of upstream's five PNGs are vendored into `assets/`: the collision mask and
// the checkpoint map, 26 KB together. The other three — background, foreground and
// the car sprite — are 590 KB of artwork that only ever gets drawn, and nothing in
// the simulation reads them. See `doc/Demos.md`.

use crate::support::rng::Rng;
use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use std::path::Path;

pub const NUM_SENSORS: usize = 12;
/// Sensors are packed into the smallest square grid that holds them; the spare
/// columns stay at 0, as upstream leaves them.
pub const SENSOR_GRID: usize = 4;
pub const SENSOR_RANGE: f32 = 70.0;
const SENSOR_ANGLE: f32 = 0.16;
const CAST_INCREMENT: f32 = 2.0;

pub const STEER_RES: i32 = 7;
const MAX_SPEED: f32 = 30.0;
const ACCEL: f32 = 0.1;
const SPIN_RATE: f32 = 0.16;
const DRAG: f32 = 0.95;

/// The collision mask: white pixels are wall.
pub struct Track {
    pub w: usize,
    pub h: usize,
    wall: Vec<bool>,
    /// Checkpoints in lap order. Upstream stores the ordinal in a pixel's red
    /// channel and marks presence with a non-zero alpha.
    pub checkpoints: Vec<(f32, f32)>,
    /// Total lap length, summed over checkpoint segments.
    pub lap_length: f32,
}

impl Track {
    /// Load the collision mask and checkpoint map from a directory of PNGs.
    pub fn load(dir: &Path) -> Result<Track, String> {
        let (cw, ch, collision) = read_rgba(&dir.join("racingCollision.png"))?;
        let (kw, kh, checks) = read_rgba(&dir.join("racingCheckpoints.png"))?;

        // Upstream treats "white" as wall; the mask is not perfectly binary after
        // any resampling, so threshold rather than testing for exact equality.
        let wall: Vec<bool> = (0..cw * ch)
            .map(|i| {
                let p = &collision[i * 4..i * 4 + 4];
                p[0] > 200 && p[1] > 200 && p[2] > 200
            })
            .collect();

        // Checkpoint ordinal lives in the red channel; alpha marks presence.
        let mut indexed: Vec<Option<(f32, f32)>> = Vec::new();
        for y in 0..kh {
            for x in 0..kw {
                let p = &checks[(x + y * kw) * 4..(x + y * kw) * 4 + 4];
                if p[3] == 0 {
                    continue;
                }
                let ord = p[0] as usize;
                if indexed.len() <= ord {
                    indexed.resize(ord + 1, None);
                }
                indexed[ord] = Some((x as f32, y as f32));
            }
        }

        let checkpoints: Vec<(f32, f32)> = indexed.into_iter().flatten().collect();
        if checkpoints.len() < 2 {
            return Err(format!(
                "racingCheckpoints.png yielded {} checkpoints; expected at least 2",
                checkpoints.len()
            ));
        }

        let mut lap_length = 0.0;
        for i in 0..checkpoints.len() {
            let a = checkpoints[i];
            let b = checkpoints[(i + 1) % checkpoints.len()];
            lap_length += (b.0 - a.0).hypot(b.1 - a.1);
        }

        Ok(Track { w: cw, h: ch, wall, checkpoints, lap_length })
    }

    pub fn is_wall(&self, x: f32, y: f32) -> bool {
        if x < 0.0 || y < 0.0 {
            return true;
        }
        let (xi, yi) = (x as usize, y as usize);
        if xi >= self.w || yi >= self.h {
            return true;
        }
        self.wall[xi + yi * self.w]
    }

    /// March along a ray in fixed increments until a wall is hit, as upstream does.
    pub fn raycast(&self, from: (f32, f32), dir: (f32, f32), range: f32) -> f32 {
        let steps = (range / CAST_INCREMENT).ceil() as usize;
        for s in 1..=steps {
            let d = CAST_INCREMENT * s as f32;
            if self.is_wall(from.0 + dir.0 * d, from.1 + dir.1 * d) {
                return d.min(range);
            }
        }
        range
    }
}

fn read_rgba(path: &Path) -> Result<(usize, usize, Vec<u8>), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().map_err(|e| format!("{}: {e}", path.display()))?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    buf.truncate(info.buffer_size());

    let (w, h) = (info.width as usize, info.height as usize);

    // Normalise whatever the file happens to be into RGBA8.
    let rgba = match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgba, png::BitDepth::Eight) => buf,
        (png::ColorType::Rgb, png::BitDepth::Eight) => buf
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            buf.iter().flat_map(|&g| [g, g, g, 255]).collect()
        }
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => buf
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        (c, d) => return Err(format!("{}: unsupported {c:?} at {d:?}", path.display())),
    };

    Ok((w, h, rgba))
}

pub struct Car {
    pub position: (f32, f32),
    pub rotation: f32,
    pub speed: f32,
}

pub struct Racing {
    pub track: Track,
    pub car: Car,
    pub checkpoint: usize,
    pub laps: i64,
    pub distance: f32,
    pub sensors: [f32; NUM_SENSORS],
    prev_position: (f32, f32),
}

impl Racing {
    pub fn new(track: Track) -> Self {
        let start = track.checkpoints[0];
        let next = track.checkpoints[1];
        let mut r = Racing {
            car: Car {
                position: start,
                rotation: (next.1 - start.1).atan2(next.0 - start.0),
                speed: 0.0,
            },
            checkpoint: 0,
            laps: 0,
            distance: 0.0,
            sensors: [1.0; NUM_SENSORS],
            prev_position: start,
            track,
        };
        r.cast_sensors();
        r
    }

    /// Put the car back on the first checkpoint, pointing at the second.
    pub fn reset(&mut self) {
        let start = self.track.checkpoints[0];
        let next = self.track.checkpoints[1];
        self.car.position = start;
        self.car.rotation = (next.1 - start.1).atan2(next.0 - start.0);
        self.car.speed = 0.0;
        self.checkpoint = 0;
        self.laps = 0;
        self.prev_position = start;
        self.cast_sensors();
    }

    fn cast_sensors(&mut self) {
        for s in 0..NUM_SENSORS {
            let a = SENSOR_ANGLE * (s as f32 - NUM_SENSORS as f32 * 0.5) + self.car.rotation;
            let (sin, cos) = a.sin_cos();
            let hit = self.track.raycast(self.car.position, (cos, sin), SENSOR_RANGE);
            self.sensors[s] = hit / SENSOR_RANGE;
        }
    }

    /// Advance one frame with the given steering column index. Returns the reward
    /// and whether the car crashed.
    pub fn step(&mut self, steer_ci: i32) -> (f32, bool) {
        self.prev_position = self.car.position;

        let steer = steer_ci as f32 / (STEER_RES - 1) as f32 * 2.0 - 1.0;

        // The car accelerates unconditionally; the policy only steers.
        let (sin, cos) = self.car.rotation.sin_cos();
        self.car.position.0 += cos * self.car.speed;
        self.car.position.1 += sin * self.car.speed;
        self.car.speed *= DRAG;
        self.car.speed = (self.car.speed + ACCEL).clamp(-MAX_SPEED, MAX_SPEED);
        self.car.rotation =
            (self.car.rotation + steer * SPIN_RATE) % (std::f32::consts::PI * 2.0);

        let crashed = self.track.is_wall(self.car.position.0, self.car.position.1);

        // Track progress along the current checkpoint segment.
        let track_dir = self.advance_checkpoints();

        let moved = (
            self.car.position.0 - self.prev_position.0,
            self.car.position.1 - self.prev_position.1,
        );
        let moved_len = moved.0.hypot(moved.1).max(0.00001);
        let car_dir = (moved.0 / moved_len, moved.1 / moved_len);

        // Reward: speed projected onto the direction of the track. Going fast the
        // wrong way scores negative, which is what stops the car simply spinning.
        let alignment = car_dir.0 * track_dir.0 + car_dir.1 * track_dir.1;
        let mut reward = 0.01 * self.car.speed.abs() * alignment;
        if crashed {
            reward -= 1.0;
        }
        reward *= 100.0;

        if crashed {
            self.reset();
        } else {
            self.cast_sensors();
        }

        (reward, crashed)
    }

    /// Move `checkpoint` forward or back as the car passes segment ends, update the
    /// cumulative distance, and return the unit direction of the current segment.
    fn advance_checkpoints(&mut self) -> (f32, f32) {
        let n = self.track.checkpoints.len();
        let a = self.track.checkpoints[self.checkpoint];
        let b = self.track.checkpoints[(self.checkpoint + 1) % n];

        let seg = (b.0 - a.0, b.1 - a.1);
        let seg_len = seg.0.hypot(seg.1).max(0.00001);
        let dir = (seg.0 / seg_len, seg.1 / seg_len);

        let rel = (self.car.position.0 - a.0, self.car.position.1 - a.1);
        let along = rel.0 * dir.0 + rel.1 * dir.1;

        if along > seg_len {
            self.checkpoint += 1;
            if self.checkpoint >= n {
                self.checkpoint = 0;
                self.laps += 1;
            }
        } else if along < 0.0 && self.checkpoint > 0 {
            self.checkpoint -= 1;
        }

        let mut completed = 0.0;
        for i in 0..self.checkpoint {
            let p = self.track.checkpoints[i];
            let q = self.track.checkpoints[(i + 1) % n];
            completed += (q.0 - p.0).hypot(q.1 - p.1);
        }
        self.distance =
            completed + along.clamp(0.0, seg_len) + self.laps as f32 * self.track.lap_length;

        dir
    }

    /// Sensor readings binned into the `SENSOR_GRID`-squared column layout.
    pub fn sensor_cis(&self, res: i32, out: &mut [i32]) {
        debug_assert_eq!(out.len(), SENSOR_GRID * SENSOR_GRID);
        out.fill(0);
        for s in 0..NUM_SENSORS {
            out[s] = crate::support::encode::bin_unit(self.sensors[s], res);
        }
    }
}

/// A steering index chosen uniformly at random — the baseline policy.
pub fn random_steer(rng: &mut Rng) -> i32 {
    rng.below(STEER_RES as usize) as i32
}


/// Sensor column resolution used by both the headless demo and the viewer.
pub const SENSOR_RES: i32 = 16;

/// The hierarchy `car_racing` uses.
///
/// Defined here rather than in the demo so the windowed viewer drives exactly the
/// same configuration; a second copy would drift.
pub fn build_hierarchy() -> Hierarchy {
    let io_descs = vec![
        IoDesc {
            size: Int3::new(SENSOR_GRID as i32, SENSOR_GRID as i32, SENSOR_RES),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 4,
            up_radius: 4,
            down_radius: 2,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, 1, STEER_RES),
            io_type: IoType::Action,
            num_dendrites_per_cell: 4,
            up_radius: 2,
            down_radius: 2,
            ..Default::default()
        },
    ];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(5, 5, 32),
        num_dendrites_per_cell: 4,
        up_radius: 2,
        recurrent_radius: 0,
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: false,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn assets() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
    }

    fn track() -> Track {
        Track::load(&assets()).expect("vendored track assets should load")
    }

    #[test]
    fn track_loads_with_walls_and_ordered_checkpoints() {
        let t = track();
        assert!(t.w > 0 && t.h > 0);
        assert!(t.checkpoints.len() >= 2, "expected a checkpoint ring");
        assert!(t.lap_length > 0.0);

        let walls = t.wall.iter().filter(|&&w| w).count();
        assert!(walls > 0, "collision mask has no walls");
        assert!(walls < t.wall.len(), "collision mask is entirely wall");
    }

    #[test]
    fn checkpoints_sit_on_open_track() {
        let t = track();
        let on_wall = t.checkpoints.iter().filter(|&&(x, y)| t.is_wall(x, y)).count();
        // A checkpoint inside a wall would make the car crash the moment it resets.
        assert!(
            on_wall * 4 < t.checkpoints.len(),
            "{on_wall} of {} checkpoints are inside walls",
            t.checkpoints.len()
        );
    }

    #[test]
    fn out_of_bounds_reads_as_wall() {
        let t = track();
        assert!(t.is_wall(-1.0, 10.0));
        assert!(t.is_wall(10.0, -1.0));
        assert!(t.is_wall(t.w as f32 + 5.0, 10.0));
    }

    #[test]
    fn raycast_is_bounded_by_its_range() {
        let t = track();
        let from = t.checkpoints[0];
        for i in 0..32 {
            let a = i as f32 / 32.0 * std::f32::consts::PI * 2.0;
            let d = t.raycast(from, (a.cos(), a.sin()), SENSOR_RANGE);
            assert!(d > 0.0 && d <= SENSOR_RANGE, "raycast returned {d}");
        }
    }

    #[test]
    fn car_makes_progress_when_steered_straight() {
        let mut r = Racing::new(track());
        let straight = STEER_RES / 2;
        let mut moved = false;
        for _ in 0..50 {
            let before = r.car.position;
            r.step(straight);
            if (r.car.position.0 - before.0).abs() + (r.car.position.1 - before.1).abs() > 0.0 {
                moved = true;
            }
        }
        assert!(moved, "car never moved");
    }

    #[test]
    fn sensors_stay_normalised_and_spare_columns_stay_zero() {
        let mut r = Racing::new(track());
        let mut cis = vec![0i32; SENSOR_GRID * SENSOR_GRID];
        for i in 0..200 {
            r.step((i % STEER_RES as usize) as i32);
            assert!(r.sensors.iter().all(|&s| (0.0..=1.0).contains(&s)));
            r.sensor_cis(16, &mut cis);
            for &c in &cis[NUM_SENSORS..] {
                assert_eq!(c, 0, "spare sensor column was written");
            }
            assert!(cis.iter().all(|&c| (0..16).contains(&c)));
        }
    }

    #[test]
    fn crashing_resets_the_car_to_the_start() {
        let mut r = Racing::new(track());
        // Drive it straight into out-of-bounds, which reads as wall.
        r.car.position = (-5.0, -5.0);
        let (_, crashed) = r.step(STEER_RES / 2);
        assert!(crashed);
        assert_eq!(r.checkpoint, 0);
        assert_eq!(r.laps, 0);
        assert_eq!(r.car.speed, 0.0);
    }
}
