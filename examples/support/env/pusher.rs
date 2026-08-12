// The pusher world, for `pusher`.
//
// Ported from `demos/Pusher.cpp` (jacobeverist/OgmaNeoDemos @ aogmaneo). Upstream
// already hand-writes this — there is no physics engine involved, and no
// integration either: the object has no inertia and only ever moves when the
// pusher is overlapping it.
//
// The agent controls a disc in a square arena and has to shove a second disc onto
// the origin. It is a genuinely awkward control problem because the pusher must
// position itself on the far side of the object before pushing, which means
// briefly moving away from the goal.

use crate::support::rng::Rng;
use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

pub const SENSOR_RES: i32 = 16;
pub const ACTION_RES: i32 = 5;

/// The hierarchy `pusher` uses.
///
/// The action port's `up_radius: 0` gives it a 1x1 receptive field — the port is
/// two columns wide and there is nothing next to them worth seeing — and
/// `importance = 0.0` keeps it out of the encoder's input entirely, so the
/// observation alone determines the state.
pub fn build_hierarchy() -> Hierarchy {
    let io_descs = vec![
        IoDesc {
            size: Int3::new(2, 2, SENSOR_RES),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 16,
            up_radius: 2,
            down_radius: 3,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, 2, ACTION_RES),
            io_type: IoType::Action,
            num_dendrites_per_cell: 16,
            up_radius: 0,
            down_radius: 3,
            ..Default::default()
        },
    ];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(7, 7, 32),
        num_dendrites_per_cell: 4,
        up_radius: 2,
        recurrent_radius: 0,
        down_radius: 2,
        ticks_per_update: 1,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h.params.ios[1].importance = 0.0;
    h
}

pub const OBJECT_RAD: f32 = 0.1;
pub const PUSHER_RAD: f32 = 0.1;
/// Maximum movement per axis per step. Note this is per-axis, not a normalised
/// vector, so diagonal moves are faster than axis-aligned ones — upstream's
/// behaviour, preserved.
pub const MAX_SPEED: f32 = 0.08;
/// The object is home when its centre is this close to the origin.
pub const GOAL_RADIUS: f32 = 0.08;

/// What ended the attempt, if anything.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Outcome {
    Ongoing,
    Goal,
    OutOfBounds,
    /// No terminal outcome within [`PusherWorld::timeout`] steps.
    Timeout,
}

pub struct PusherWorld {
    pub object: (f32, f32),
    pub pusher: (f32, f32),
    /// `-1.0` marks "just reset" so the first reward after a reset carries no
    /// shaping delta. Upstream's sentinel, kept because it matters: without it the
    /// agent is rewarded or punished for the teleport itself.
    dist_prev: f32,
    object_dist_prev: f32,
    steps_since_reset: usize,
    /// Steps allowed before the object is respawned regardless. `0` disables it,
    /// reproducing upstream, which has no episode limit at all.
    ///
    /// **This is an addition, and the demo does not work without it.** The object
    /// only ever moves while the pusher overlaps it, so a policy that stops moving
    /// freezes the world: no reward, no termination, nothing to learn from, for
    /// ever. Standing still is therefore a perfect local optimum worth exactly 0,
    /// and the actor finds it within about 50k steps and never leaves — goals and
    /// losses both drop to zero and stay there. A timeout makes idling
    /// unproductive and keeps fresh states arriving.
    pub timeout: usize,
}

impl PusherWorld {
    pub fn new() -> Self {
        PusherWorld {
            object: (0.3, 0.3),
            pusher: (0.0, 0.0),
            dist_prev: -1.0,
            object_dist_prev: -1.0,
            steps_since_reset: 0,
            timeout: 0,
        }
    }

    /// Advance one step given the two action column indices, and return the reward
    /// and outcome.
    ///
    /// The ordering is upstream's and matters: the overlap is resolved using *last*
    /// step's positions, before the pusher moves.
    pub fn step(&mut self, action: (i32, i32), action_res: i32, rng: &mut Rng) -> (f32, Outcome) {
        // 1. Resolve overlap. The pusher is immovable, so the object takes the
        //    whole correction.
        let dx = self.object.0 - self.pusher.0;
        let dy = self.object.1 - self.pusher.1;
        let dist_between = (dx * dx + dy * dy).sqrt();
        if dist_between < OBJECT_RAD + PUSHER_RAD && dist_between > 0.0 {
            let push = OBJECT_RAD + PUSHER_RAD - dist_between;
            self.object.0 += push * dx / dist_between;
            self.object.1 += push * dy / dist_between;
        }

        // 2. Decode the action into a per-axis delta and move the pusher.
        let decode = |ci: i32| MAX_SPEED * (ci as f32 / (action_res - 1) as f32 * 2.0 - 1.0);
        self.pusher.0 = (self.pusher.0 + decode(action.0)).clamp(-1.0, 1.0);
        self.pusher.1 = (self.pusher.1 + decode(action.1)).clamp(-1.0, 1.0);

        // 3. Shaped reward: progress toward the goal, and toward the object.
        let dist_to_centre = (self.object.0 * self.object.0 + self.object.1 * self.object.1).sqrt();
        let odx = self.object.0 - self.pusher.0;
        let ody = self.object.1 - self.pusher.1;
        let dist_to_object = (odx * odx + ody * ody).sqrt();

        if self.dist_prev == -1.0 {
            self.dist_prev = dist_to_centre;
        }
        if self.object_dist_prev == -1.0 {
            self.object_dist_prev = dist_to_object;
        }

        let mut reward = -5.0 * (dist_to_centre - self.dist_prev)
            - 2.0 * (dist_to_object - self.object_dist_prev);

        self.dist_prev = dist_to_centre;
        self.object_dist_prev = dist_to_object;

        // 4. Terminal conditions. Note the object is never clamped — leaving the
        //    arena is how an attempt fails.
        let out_of_bounds = self.object.0 < -1.0
            || self.object.0 > 1.0
            || self.object.1 < -1.0
            || self.object.1 > 1.0;

        self.steps_since_reset += 1;

        let outcome = if dist_to_centre < GOAL_RADIUS {
            Outcome::Goal
        } else if out_of_bounds {
            Outcome::OutOfBounds
        } else if self.timeout > 0 && self.steps_since_reset >= self.timeout {
            Outcome::Timeout
        } else {
            Outcome::Ongoing
        };

        if outcome != Outcome::Ongoing {
            // The terminal reward *replaces* the shaping term rather than adding
            // to it, as upstream does. A timeout is not the agent's fault and pays
            // nothing either way — its purpose is to move the object, not to teach.
            reward = match outcome {
                Outcome::Goal => 100.0,
                Outcome::OutOfBounds => -0.5,
                _ => 0.0,
            };

            self.object = (rng.range(-1.0, 1.0) * 0.6, rng.range(-1.0, 1.0) * 0.6);
            self.dist_prev = -1.0;
            self.object_dist_prev = -1.0;
            self.steps_since_reset = 0;
            // The pusher is deliberately *not* reset — there are no episodes here,
            // just one continuous stream with occasional teleports of the object.
        }

        (reward, outcome)
    }

    /// The four observed scalars: pusher position, and the object's offset from the
    /// pusher. Each is mapped to `[0, 1]` for binning.
    ///
    /// The offset spans `[-2, 2]` but is squeezed through the same `v * 0.5 + 0.5`
    /// mapping as the position, so it saturates outside `[-1, 1]`. That is
    /// upstream's deliberate lossy encoding: fine resolution near the object, none
    /// at all once it is far away.
    pub fn observation(&self) -> [f32; 4] {
        [
            self.pusher.0 * 0.5 + 0.5,
            self.pusher.1 * 0.5 + 0.5,
            (self.object.0 - self.pusher.0) * 0.5 + 0.5,
            (self.object.1 - self.pusher.1) * 0.5 + 0.5,
        ]
    }
}

impl Default for PusherWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::Rng;

    #[test]
    fn pusher_stays_inside_the_arena() {
        let mut rng = Rng::new(21);
        let mut w = PusherWorld::new();
        for _ in 0..20_000 {
            let a = (rng.below(5) as i32, rng.below(5) as i32);
            w.step(a, 5, &mut rng);
            assert!(w.pusher.0.abs() <= 1.0 + 1e-6);
            assert!(w.pusher.1.abs() <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn touching_the_object_pushes_it_clear() {
        let mut rng = Rng::new(22);
        let mut w = PusherWorld::new();
        w.pusher = (0.0, 0.0);
        w.object = (0.05, 0.0);
        // Action 2 of 5 is the zero-movement bin, so only the push resolves.
        w.step((2, 2), 5, &mut rng);
        let sep = (w.object.0 - w.pusher.0).hypot(w.object.1 - w.pusher.1);
        assert!(sep >= OBJECT_RAD + PUSHER_RAD - 1e-5, "still overlapping: {sep}");
    }

    #[test]
    fn reaching_the_goal_pays_out_and_respawns() {
        let mut rng = Rng::new(23);
        let mut w = PusherWorld::new();
        w.object = (0.0, 0.0);
        w.pusher = (0.9, 0.9);
        let (reward, outcome) = w.step((2, 2), 5, &mut rng);
        assert_eq!(outcome, Outcome::Goal);
        assert_eq!(reward, 100.0);
        // Respawned somewhere in [-0.6, 0.6]^2.
        assert!(w.object.0.abs() <= 0.6 && w.object.1.abs() <= 0.6);
    }

    #[test]
    fn leaving_the_arena_is_penalised() {
        let mut rng = Rng::new(24);
        let mut w = PusherWorld::new();
        w.object = (1.5, 0.0);
        w.pusher = (-0.9, -0.9);
        let (reward, outcome) = w.step((2, 2), 5, &mut rng);
        assert_eq!(outcome, Outcome::OutOfBounds);
        assert_eq!(reward, -0.5);
    }

    #[test]
    fn timeout_respawns_the_object_and_pays_nothing() {
        let mut rng = Rng::new(26);
        let mut w = PusherWorld::new();
        w.timeout = 10;
        // Park the pusher away from the object so nothing else can terminate.
        w.pusher = (-0.9, -0.9);
        w.object = (0.5, 0.5);

        let mut saw_timeout = false;
        for _ in 0..10 {
            let (reward, outcome) = w.step((2, 2), 5, &mut rng);
            if outcome == Outcome::Timeout {
                assert_eq!(reward, 0.0, "a timeout should not pay out");
                saw_timeout = true;
            }
        }
        assert!(saw_timeout, "timeout never fired after 10 steps with timeout = 10");
    }

    #[test]
    fn an_idle_policy_never_stalls_forever_with_a_timeout() {
        // The demo's whole reason for adding a timeout: a motionless policy must
        // still see the world change.
        let mut rng = Rng::new(27);
        let mut w = PusherWorld::new();
        w.timeout = 50;
        w.pusher = (-0.9, -0.9);

        let mut resets = 0;
        for _ in 0..500 {
            if w.step((2, 2), 5, &mut rng).1 != Outcome::Ongoing {
                resets += 1;
            }
        }
        assert!(resets >= 9, "only {resets} resets in 500 idle steps");
    }

    #[test]
    fn observations_stay_in_unit_range_for_in_arena_states() {
        let mut rng = Rng::new(25);
        let mut w = PusherWorld::new();
        for _ in 0..5_000 {
            let a = (rng.below(5) as i32, rng.below(5) as i32);
            w.step(a, 5, &mut rng);
            let obs = w.observation();
            // The first two are positions and are always in range; the offsets may
            // legitimately exceed it and are clamped by the encoder.
            assert!((0.0..=1.0).contains(&obs[0]));
            assert!((0.0..=1.0).contains(&obs[1]));
        }
    }
}
