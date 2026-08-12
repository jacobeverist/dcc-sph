// The cat-and-mouse world, for `cat_mouse`.
//
// Ported from `demos/catmouse/CatMouseEnv.cpp` (jacobeverist/OgmaNeoDemos @
// aogmaneo). Upstream already hand-writes this — it is grid raycasting against a
// bitmap, no physics engine.
//
// Upstream loads the map from `resources/map0.png`, which is **absent from the
// upstream repository** (only `map1.png` is present), so the map is generated here
// instead. `--seed` makes it reproducible. See `doc/Demos.md`.

use crate::support::rng::Rng;
use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

const PI: f32 = std::f32::consts::PI;

pub const FOV: f32 = PI * 0.7;
pub const SCAN_RAYS: usize = 30;
const ANGLE_SPEED: f32 = 8.0;
const ACCEL: f32 = 80.0;
const DECCEL: f32 = 8.0;
const RANGE: f32 = 20.0;
const DEPTH_STEP: f32 = 0.2;
pub const AGENT_RADIUS: f32 = 0.4;

/// Observation width: 30 depth rays, heading, a 2-component "I can see them"
/// vector, and normalised position. Upstream's `obsSize()`.
pub const OBS_SIZE: usize = SCAN_RAYS + 5;
/// Thrust, strafe, turn.
pub const ACTION_SIZE: usize = 3;

/// A wall bitmap. `true` is solid, matching upstream's black pixels.
pub struct Map {
    pub w: usize,
    pub h: usize,
    solid: Vec<bool>,
}

impl Map {
    pub fn is_solid(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return true;
        }
        self.solid[y as usize + x as usize * self.h]
    }

    /// Generate a maze by randomised depth-first search on a grid of cells, then
    /// knock out extra walls so the result has loops.
    ///
    /// A perfect maze — exactly one path between any two points — makes a poor
    /// chase: the mouse can always be cornered and the cat can never be dodged, so
    /// neither agent has anything interesting to learn. `braid` reopens some walls
    /// to create cycles, which is what gives the chase somewhere to go. Upstream's
    /// hand-drawn map has open rooms for the same reason.
    pub fn generate(cells_x: usize, cells_y: usize, braid: f32, rng: &mut Rng) -> Map {
        let w = cells_x * 2 + 1;
        let h = cells_y * 2 + 1;
        let mut solid = vec![true; w * h];

        let idx = |x: usize, y: usize| y + x * h;

        // Depth-first carve over odd-indexed cells.
        let mut visited = vec![false; cells_x * cells_y];
        let mut stack: Vec<(usize, usize)> = Vec::new();

        let start = (rng.below(cells_x), rng.below(cells_y));
        visited[start.0 + start.1 * cells_x] = true;
        solid[idx(start.0 * 2 + 1, start.1 * 2 + 1)] = false;
        stack.push(start);

        while let Some(&(cx, cy)) = stack.last() {
            let mut candidates: Vec<(usize, usize, usize, usize)> = Vec::new();
            // (neighbour x, neighbour y, wall x, wall y)
            if cx > 0 && !visited[(cx - 1) + cy * cells_x] {
                candidates.push((cx - 1, cy, cx * 2, cy * 2 + 1));
            }
            if cx + 1 < cells_x && !visited[(cx + 1) + cy * cells_x] {
                candidates.push((cx + 1, cy, cx * 2 + 2, cy * 2 + 1));
            }
            if cy > 0 && !visited[cx + (cy - 1) * cells_x] {
                candidates.push((cx, cy - 1, cx * 2 + 1, cy * 2));
            }
            if cy + 1 < cells_y && !visited[cx + (cy + 1) * cells_x] {
                candidates.push((cx, cy + 1, cx * 2 + 1, cy * 2 + 2));
            }

            if candidates.is_empty() {
                stack.pop();
                continue;
            }

            let (nx, ny, wx, wy) = candidates[rng.below(candidates.len())];
            visited[nx + ny * cells_x] = true;
            solid[idx(wx, wy)] = false;
            solid[idx(nx * 2 + 1, ny * 2 + 1)] = false;
            stack.push((nx, ny));
        }

        // Braid: reopen interior walls at random to introduce loops.
        for x in 1..w - 1 {
            for y in 1..h - 1 {
                // Only walls between two cells, never pillar corners.
                let between = (x % 2 == 0) ^ (y % 2 == 0);
                if between && solid[idx(x, y)] && rng.chance(braid) {
                    solid[idx(x, y)] = false;
                }
            }
        }

        Map { w, h, solid }
    }

    /// Every open cell, in raster order — the spawn candidates.
    pub fn open_cells(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for x in 0..self.w {
            for y in 0..self.h {
                if !self.solid[y + x * self.h] {
                    out.push((x, y));
                }
            }
        }
        out
    }

    /// Fixed-step raycast, returning the distance to the first wall or [`RANGE`].
    ///
    /// Upstream samples every 0.2 units rather than running a DDA, so results are
    /// quantised to that step; kept as-is because the observation encoding is
    /// calibrated around it.
    pub fn depth(&self, from: (f32, f32), angle: f32) -> f32 {
        let (sin, cos) = angle.sin_cos();
        let steps = (RANGE / DEPTH_STEP).ceil() as usize;

        for s in 1..=steps {
            let d = DEPTH_STEP * s as f32;
            let x = from.0 + cos * d;
            let y = from.1 + sin * d;
            if self.is_solid(x.floor() as i32, y.floor() as i32) {
                return d;
            }
        }

        RANGE
    }
}

#[derive(Clone, Copy, Default)]
pub struct Agent {
    pub pos: (f32, f32),
    pub vel: (f32, f32),
    pub angle: f32,
}

pub struct CatMouseEnv {
    pub map: Map,
    pub cat: Agent,
    pub mouse: Agent,
    pub distance: f32,
    done: bool,
}

impl CatMouseEnv {
    pub fn new(map: Map, rng: &mut Rng) -> Self {
        let mut env = CatMouseEnv {
            map,
            cat: Agent::default(),
            mouse: Agent::default(),
            distance: 0.0,
            done: false,
        };
        env.reset(rng);
        env
    }

    pub fn done(&self) -> bool {
        self.done
    }

    /// Respawn both agents far apart. Upstream picks the cat uniformly, then offsets
    /// the mouse by between a third and two thirds of the way around the open-cell
    /// list — a cheap way to guarantee they never start on top of each other.
    pub fn reset(&mut self, rng: &mut Rng) {
        let open = self.map.open_cells();
        assert!(open.len() >= 2, "map has fewer than two open cells");

        let cat_i = rng.below(open.len());
        let offset = open.len() / 3 + rng.below(open.len() / 3 + 1);
        let mouse_i = (cat_i + offset) % open.len();

        let centre = |c: (usize, usize)| (c.0 as f32 + 0.5, c.1 as f32 + 0.5);

        self.cat = Agent {
            pos: centre(open[cat_i]),
            vel: (0.0, 0.0),
            angle: rng.unit() * 2.0 * PI,
        };
        self.mouse = Agent {
            pos: centre(open[mouse_i]),
            vel: (0.0, 0.0),
            angle: rng.unit() * 2.0 * PI,
        };

        self.done = false;
        self.update_distance();
    }

    fn update_distance(&mut self) {
        let dx = self.cat.pos.0 - self.mouse.pos.0;
        let dy = self.cat.pos.1 - self.mouse.pos.1;
        self.distance = (dx * dx + dy * dy).sqrt();
    }

    /// Advance both agents by `dt`. Actions are three values in `[0, 1]`, remapped
    /// internally to `[-1, 1]` for thrust, strafe and turn.
    pub fn step(&mut self, cat_actions: &[f32], mouse_actions: &[f32], dt: f32) {
        Self::integrate(&mut self.cat, cat_actions, dt);
        Self::integrate(&mut self.mouse, mouse_actions, dt);

        Self::collide(&self.map, &mut self.cat);
        Self::collide(&self.map, &mut self.mouse);

        self.update_distance();
        if self.distance < 2.0 * AGENT_RADIUS {
            self.done = true;
        }
    }

    fn integrate(agent: &mut Agent, actions: &[f32], dt: f32) {
        // Turn rate is commanded directly — there is no angular inertia.
        agent.angle += (actions[2] * 2.0 - 1.0) * ANGLE_SPEED * dt;
        agent.angle %= 2.0 * PI;

        let (sin, cos) = agent.angle.sin_cos();
        let dir = (cos, sin);
        let strafe = (-sin, cos);

        let thrust = actions[0] * 2.0 - 1.0;
        let slide = actions[1] * 2.0 - 1.0;

        // Acceleration along the body frame, with linear drag: terminal speed is
        // ACCEL / DECCEL = 10 units/s per unit of command.
        agent.vel.0 += ((dir.0 * thrust + strafe.0 * slide) * ACCEL - agent.vel.0 * DECCEL) * dt;
        agent.vel.1 += ((dir.1 * thrust + strafe.1 * slide) * ACCEL - agent.vel.1 * DECCEL) * dt;

        agent.pos.0 += agent.vel.0 * dt;
        agent.pos.1 += agent.vel.1 * dt;
    }

    /// Axis-separated collision against the four orthogonal neighbours of the
    /// occupied cell. Diagonals are not tested — upstream's behaviour, and the
    /// reason an agent can clip a corner slightly.
    fn collide(map: &Map, agent: &mut Agent) {
        let xi = agent.pos.0.floor() as i32;
        let yi = agent.pos.1.floor() as i32;

        if map.is_solid(xi - 1, yi) && agent.pos.0 - AGENT_RADIUS < xi as f32 {
            agent.pos.0 = xi as f32 + AGENT_RADIUS;
            agent.vel.0 = 0.0;
        }
        if map.is_solid(xi + 1, yi) && agent.pos.0 + AGENT_RADIUS > (xi + 1) as f32 {
            agent.pos.0 = (xi + 1) as f32 - AGENT_RADIUS;
            agent.vel.0 = 0.0;
        }
        if map.is_solid(xi, yi - 1) && agent.pos.1 - AGENT_RADIUS < yi as f32 {
            agent.pos.1 = yi as f32 + AGENT_RADIUS;
            agent.vel.1 = 0.0;
        }
        if map.is_solid(xi, yi + 1) && agent.pos.1 + AGENT_RADIUS > (yi + 1) as f32 {
            agent.pos.1 = (yi + 1) as f32 - AGENT_RADIUS;
            agent.vel.1 = 0.0;
        }
    }

    /// Observations for both agents, each `OBS_SIZE` values in `[0, 1]`.
    pub fn observations(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.observe(&self.cat, &self.mouse),
            self.observe(&self.mouse, &self.cat),
        )
    }

    fn observe(&self, me: &Agent, other: &Agent) -> Vec<f32> {
        let mut obs = Vec::with_capacity(OBS_SIZE);

        // Depth fan across the field of view.
        let start_angle = me.angle - FOV * 0.5;
        let step = FOV / (SCAN_RAYS.max(2) - 1) as f32;
        for s in 0..SCAN_RAYS {
            obs.push(self.map.depth(me.pos, start_angle + s as f32 * step) / RANGE);
        }

        // Heading, wrapped into [0, 1).
        let mut heading = me.angle / (2.0 * PI);
        if heading < 0.0 {
            heading += 1.0;
        }
        obs.push(heading);

        // "Can I see them, and where" — a body-frame unit vector, or (0.5, 0.5)
        // when they are out of view or behind a wall.
        let to = (other.pos.0 - me.pos.0, other.pos.1 - me.pos.1);
        let to_dist = (to.0 * to.0 + to.1 * to.1).sqrt().max(0.0001);
        let (sin, cos) = me.angle.sin_cos();
        let dot = (cos * to.0 + sin * to.1) / to_dist;

        let mut sense = (0.0f32, 0.0f32);
        if dot.clamp(-1.0, 1.0).acos().abs() < FOV * 0.5 {
            let to_angle = to.1.atan2(to.0);
            if self.map.depth(me.pos, to_angle) >= to_dist {
                let delta = angle_wrap(to_angle - me.angle);
                sense = (delta.cos(), delta.sin());
            }
        }
        obs.push(sense.0 * 0.5 + 0.5);
        obs.push(sense.1 * 0.5 + 0.5);

        // Absolute position, normalised.
        obs.push(me.pos.0 / self.map.w as f32);
        obs.push(me.pos.1 / self.map.h as f32);

        obs
    }
}

/// Wrap an angle difference into `[-pi, pi)`.
fn angle_wrap(delta: f32) -> f32 {
    let offset = delta + PI;
    let m = offset - (offset / (2.0 * PI)).floor() * (2.0 * PI);
    m - PI
}


pub const OBS_RES: i32 = 16;
pub const ACTION_RES: i32 = 5;

/// One of the two identical hierarchies `cat_mouse` uses.
///
/// Defined here so the windowed viewer drives exactly the same configuration.
pub fn build_hierarchy() -> Hierarchy {
    let io_descs = vec![
        IoDesc {
            size: Int3::new(7, 5, OBS_RES),
            io_type: IoType::None,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, ACTION_SIZE as i32, ACTION_RES),
            io_type: IoType::Action,
            ..Default::default()
        },
    ];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(5, 5, 128),
        num_dendrites_per_cell: 4,
        up_radius: 2,
        recurrent_radius: 0,
        down_radius: 2,
        ticks_per_update: 1,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h.params.ios[1].importance = 0.1;
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::Rng;

    #[test]
    fn generated_map_is_bordered_and_connected_enough_to_spawn_in() {
        let mut rng = Rng::new(31);
        let map = Map::generate(8, 8, 0.15, &mut rng);
        assert_eq!(map.w, 17);
        assert_eq!(map.h, 17);

        for x in 0..map.w as i32 {
            assert!(map.is_solid(x, 0) && map.is_solid(x, map.h as i32 - 1));
        }
        for y in 0..map.h as i32 {
            assert!(map.is_solid(0, y) && map.is_solid(map.w as i32 - 1, y));
        }

        // Every cell of the DFS grid gets carved, so at minimum that many are open.
        assert!(map.open_cells().len() >= 64);
    }

    #[test]
    fn out_of_range_lookups_read_as_solid() {
        let mut rng = Rng::new(32);
        let map = Map::generate(4, 4, 0.0, &mut rng);
        assert!(map.is_solid(-1, 5));
        assert!(map.is_solid(5, -1));
        assert!(map.is_solid(9999, 9999));
    }

    #[test]
    fn depth_is_bounded_and_shortens_when_facing_a_wall() {
        let mut rng = Rng::new(33);
        let map = Map::generate(6, 6, 0.1, &mut rng);
        let open = map.open_cells();
        let from = (open[0].0 as f32 + 0.5, open[0].1 as f32 + 0.5);

        for i in 0..64 {
            let a = i as f32 / 64.0 * 2.0 * PI;
            let d = map.depth(from, a);
            assert!(d > 0.0 && d <= RANGE, "depth {d} out of range");
        }
    }

    #[test]
    fn agents_never_end_up_inside_walls() {
        let mut rng = Rng::new(34);
        let map = Map::generate(8, 8, 0.2, &mut rng);
        let mut env = CatMouseEnv::new(map, &mut rng);

        for _ in 0..20_000 {
            let a = [rng.unit(), rng.unit(), rng.unit()];
            let b = [rng.unit(), rng.unit(), rng.unit()];
            env.step(&a, &b, 1.0 / 120.0);

            for agent in [&env.cat, &env.mouse] {
                let cell = (agent.pos.0.floor() as i32, agent.pos.1.floor() as i32);
                assert!(
                    !env.map.is_solid(cell.0, cell.1),
                    "agent entered a wall at {:?}",
                    agent.pos
                );
            }

            if env.done() {
                env.reset(&mut rng);
            }
        }
    }

    #[test]
    fn observations_are_the_right_width_and_all_in_unit_range() {
        let mut rng = Rng::new(35);
        let map = Map::generate(6, 6, 0.15, &mut rng);
        let mut env = CatMouseEnv::new(map, &mut rng);

        for _ in 0..3_000 {
            let a = [rng.unit(), rng.unit(), rng.unit()];
            let b = [rng.unit(), rng.unit(), rng.unit()];
            env.step(&a, &b, 1.0 / 120.0);

            let (cat, mouse) = env.observations();
            assert_eq!(cat.len(), OBS_SIZE);
            assert_eq!(mouse.len(), OBS_SIZE);
            for v in cat.iter().chain(&mouse) {
                assert!((0.0..=1.0).contains(v), "observation {v} left [0, 1]");
            }

            if env.done() {
                env.reset(&mut rng);
            }
        }
    }

    #[test]
    fn agents_spawn_apart_and_capture_is_detected() {
        let mut rng = Rng::new(36);
        let map = Map::generate(8, 8, 0.15, &mut rng);
        let mut env = CatMouseEnv::new(map, &mut rng);
        assert!(!env.done(), "spawned already captured");
        assert!(env.distance > 2.0 * AGENT_RADIUS);

        // Force them together and confirm the capture fires.
        env.mouse.pos = env.cat.pos;
        env.step(&[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5], 1.0 / 120.0);
        assert!(env.done(), "capture not detected at zero separation");
    }
}
