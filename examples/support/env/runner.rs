// The articulated runner, for `runner`.
//
// Ported from `demos/Runner_Run.cpp` and `demos/runner/Runner.cpp`
// (jacobeverist/OgmaNeoDemos @ aogmaneo), which drive Box2D. This is the one demo
// whose physics is not hand-written: four limbs of two segments, eight revolute
// motors with angle limits and torque caps, contact sensing and world raycasts
// need a constraint solver. `rapier2d` stands in for Box2D — see `doc/Demos.md`.
//
// Every constant below is transcribed from upstream.

use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use rapier2d::prelude::*;

pub const SENSOR_RES: i32 = 21;
pub const ACTION_RES: i32 = 11;
/// 4x6 columns for 23 sensors; the last stays at zero, as upstream leaves it.
pub const SENSOR_COLUMNS: usize = 24;

/// The hierarchy `runner` uses.
///
/// The observation port is `IoType::None` — sensors are observed, never predicted —
/// and the action port is kept out of the encoder's input with `importance = 0.0`.
pub fn build_hierarchy() -> Hierarchy {
    let io_descs = vec![
        IoDesc {
            size: Int3::new(4, 6, SENSOR_RES),
            io_type: IoType::None,
            num_dendrites_per_cell: 16,
            up_radius: 6,
            down_radius: 5,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(2, 4, ACTION_RES),
            io_type: IoType::Action,
            num_dendrites_per_cell: 16,
            up_radius: 4,
            down_radius: 5,
            ..Default::default()
        },
    ];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(5, 5, 64),
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

pub const NUM_SEGMENTS: usize = 8;
pub const NUM_WHISKERS: usize = 6;
/// 8 joint angles, torso angle, 4 foot contacts, 6 whiskers, 3 IMU terms.
pub const STATE_SIZE: usize = 8 + 1 + 4 + NUM_WHISKERS + 3;
/// Plus the hurdle-distance sensor.
pub const SENSOR_COUNT: usize = STATE_SIZE + 1;

const GRAVITY: f32 = -9.81;
const GROUND_HALF_WIDTH: f32 = 10_000.0;
const GROUND_HALF_HEIGHT: f32 = 2.5;
/// The ground box is centred on the origin, so its top face is at +2.5.
pub const GROUND_TOP: f32 = GROUND_HALF_HEIGHT;

const NUM_HURDLES: usize = 100;
const HURDLE_HALF_WIDTH: f32 = 0.075;
const HURDLE_BASE_HEIGHT: f32 = 0.1;
const HURDLE_HEIGHT_INC: f32 = 0.02;
const HURDLE_OFFSET: f32 = 5.0;
const HURDLE_START: f32 = 10.0;

const TORSO_HALF: (f32, f32) = (0.225, 0.05);
const TORSO_DENSITY: f32 = 2.5;
const TORSO_FRICTION: f32 = 1.0;
const TORSO_RESTITUTION: f32 = 0.01;

const SEG_HALF: (f32, f32) = (0.075, 0.015);
const SEG_DENSITY: f32 = 2.0;
const SEG_FRICTION: f32 = 5.0;
const SEG_RESTITUTION: f32 = 0.001;
const SEG_MIN_ANGLE: f32 = -1.1;
const SEG_MAX_ANGLE: f32 = 1.1;
const SEG_MAX_TORQUE: f32 = 0.25;
const SEG_MAX_SPEED: f32 = 70.0;
/// `length / 2 - thickness / 2` for a 0.15-long, 0.03-thick segment.
const SEG_OFFSET: f32 = 0.06;

const LEG_INSET: f32 = 0.075;
const BODY_WIDTH: f32 = 0.45;
const WHISKER_LEN: f32 = 1.0;
const WHISKER_SPREAD: f32 = 0.25;
const WHISKER_OFFSET_X: f32 = 0.05;

pub const SPAWN_HEIGHT: f32 = 2.762;
pub const MAX_BODY_ANGLE: f32 = 2.2;
pub const STUCK_TIME: f32 = 5.0;
const STUCK_DT: f32 = 0.017;
const PHYSICS_DT: f32 = 1.0 / 60.0;

/// Collision groups. The runner's parts must collide with the world but never with
/// each other, or the co-located legs at each hip would fight constantly.
const GROUP_WORLD: Group = Group::GROUP_1;
const GROUP_RUNNER: Group = Group::GROUP_2;

/// Why an episode ended.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ResetReason {
    Flipped,
    HitWall,
    Stuck,
}

pub struct RunnerWorld {
    world: PhysicsWorld,

    torso: RigidBodyHandle,
    segments: Vec<RigidBodyHandle>,
    segment_joints: Vec<ImpulseJointHandle>,
    /// The relative angle each joint is *built* at.
    ///
    /// Box2D lets a revolute joint carry a reference frame — upstream sets
    /// `frameA.q = relativeAngle` — so its limits and reported angle are measured
    /// from the pose the limb was assembled in. rapier's 2D `RevoluteJointBuilder`
    /// has no such frame: limits apply to the raw relative rotation of the two
    /// bodies. Since the segments are assembled at -0.75pi and +0.5pi, applying
    /// [-1.1, 1.1] directly would construct every joint already outside its own
    /// limit, and the solver locks the whole body rigid — the runner then cannot
    /// move at all, at any torque, under any policy. Offsetting limits, motor
    /// targets and the reported angle by this rest angle restores upstream's
    /// semantics.
    rest_angles: Vec<f32>,
    foot_colliders: Vec<ColliderHandle>,
    hurdle_x: Vec<f32>,

    /// Low-passed motor target angles and speeds, one per joint.
    positions: Vec<f32>,
    speeds: Vec<f32>,

    whiskers: [f32; NUM_WHISKERS],
    prev_lin_vel: Vector,
    prev_ang_vel: f32,
    imu: (f32, f32, f32),

    average_vel: f32,
    stuck_timer: f32,
}

impl RunnerWorld {
    pub fn new() -> Self {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::new(0.0, GRAVITY);
        world.integration_parameters.dt = PHYSICS_DT;

        let mut w = RunnerWorld {
            world,
            torso: RigidBodyHandle::invalid(),
            segments: Vec::new(),
            segment_joints: Vec::new(),
            rest_angles: Vec::new(),
            foot_colliders: Vec::new(),
            hurdle_x: Vec::new(),
            positions: vec![0.0; NUM_SEGMENTS],
            speeds: vec![0.0; NUM_SEGMENTS],
            whiskers: [1.0; NUM_WHISKERS],
            prev_lin_vel: Vector::ZERO,
            prev_ang_vel: 0.0,
            imu: (0.0, 0.0, 0.0),
            average_vel: 0.0,
            stuck_timer: 0.0,
        };

        w.build_world();
        w.spawn_runner();
        w
    }

    fn build_world(&mut self) {
        let world_groups = InteractionGroups::new(GROUP_WORLD, GROUP_WORLD | GROUP_RUNNER, InteractionTestMode::And);

        self.world.insert(
            RigidBodyBuilder::fixed(),
            ColliderBuilder::cuboid(GROUND_HALF_WIDTH, GROUND_HALF_HEIGHT)
                .collision_groups(world_groups),
        );

        // Hurdles get taller the further along the track they are, so the runner
        // faces a curriculum rather than a single obstacle.
        for i in 0..NUM_HURDLES {
            let height = HURDLE_BASE_HEIGHT + HURDLE_HEIGHT_INC * i as f32;
            let x = i as f32 * HURDLE_OFFSET + HURDLE_START;
            self.world.insert(
                RigidBodyBuilder::fixed()
                    .translation(Vector::new(x, GROUND_TOP + height * 0.5)),
                ColliderBuilder::cuboid(HURDLE_HALF_WIDTH, height * 0.5)
                    .collision_groups(world_groups),
            );
            self.hurdle_x.push(x);
        }
    }

    /// Build the torso and its four two-segment limbs at the spawn point.
    fn spawn_runner(&mut self) {
        let runner_groups = InteractionGroups::new(GROUP_RUNNER, GROUP_WORLD, InteractionTestMode::And);

        let (torso, _) = self.world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, SPAWN_HEIGHT))
                .can_sleep(false),
            ColliderBuilder::cuboid(TORSO_HALF.0, TORSO_HALF.1)
                .density(TORSO_DENSITY)
                .friction(TORSO_FRICTION)
                .restitution(TORSO_RESTITUTION)
                .collision_groups(runner_groups),
        );
        self.torso = torso;

        // Upstream attaches both "back" and "front" limbs at the *same* hip point on
        // each side, so this is two co-located legs per hip rather than fore and aft
        // legs. Preserved, because it changes the gait the actor has to find.
        let hips = [
            Vector::new(-BODY_WIDTH * 0.5 + LEG_INSET, -TORSO_HALF.1),
            Vector::new(-BODY_WIDTH * 0.5 + LEG_INSET, -TORSO_HALF.1),
            Vector::new(BODY_WIDTH * 0.5 - LEG_INSET, -TORSO_HALF.1),
            Vector::new(BODY_WIDTH * 0.5 - LEG_INSET, -TORSO_HALF.1),
        ];
        let relative_angles = [
            std::f32::consts::PI * -0.75,
            std::f32::consts::PI * 0.5,
        ];

        for hip in hips {
            let mut parent = torso;
            let mut attach = hip;
            let mut parent_angle = 0.0f32;

            for (si, &rel) in relative_angles.iter().enumerate() {
                let angle = parent_angle + rel;

                let parent_pos = *self.world.bodies[parent].position();
                let world_attach = parent_pos.transform_point(attach);
                let translation = Vector::new(
                    world_attach.x + angle.cos() * SEG_OFFSET,
                    world_attach.y + angle.sin() * SEG_OFFSET,
                );

                let (seg, collider) = self.world.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(translation)
                        .rotation(angle)
                        .can_sleep(false),
                    ColliderBuilder::cuboid(SEG_HALF.0, SEG_HALF.1)
                        .density(SEG_DENSITY)
                        .friction(SEG_FRICTION)
                        .restitution(SEG_RESTITUTION)
                        .collision_groups(runner_groups),
                );

                // Limits are offset by the angle the joint is assembled at — see
                // `rest_angles` for why.
                let joint = RevoluteJointBuilder::new()
                    .local_anchor1(attach)
                    .local_anchor2(Vector::new(-SEG_OFFSET, 0.0))
                    .limits([rel + SEG_MIN_ANGLE, rel + SEG_MAX_ANGLE])
                    .motor_max_force(SEG_MAX_TORQUE)
                    // Box2D's `maxMotorTorque` is a torque, so the force-based
                    // model is the faithful one. `AccelerationBased` would scale
                    // the cap by each segment's inertia and let the motors
                    // overpower their own angle limits.
                    .motor_model(MotorModel::ForceBased);
                let jh = self.world.insert_impulse_joint(parent, seg, joint);

                self.segments.push(seg);
                self.segment_joints.push(jh);
                self.rest_angles.push(rel);
                // Only the distal segment of each limb is a "foot".
                if si == relative_angles.len() - 1 {
                    self.foot_colliders.push(collider);
                }

                parent = seg;
                parent_angle = angle;
                attach = Vector::new(SEG_OFFSET, 0.0);
            }
        }
    }

    /// Tear the runner down and rebuild it at the spawn point.
    pub fn reset(&mut self) {
        for seg in std::mem::take(&mut self.segments) {
            self.world.remove_body(seg);
        }
        self.world.remove_body(self.torso);

        self.segment_joints.clear();
        self.rest_angles.clear();
        self.foot_colliders.clear();
        self.positions.iter_mut().for_each(|p| *p = 0.0);
        self.speeds.iter_mut().for_each(|s| *s = 0.0);
        self.prev_lin_vel = Vector::ZERO;
        self.prev_ang_vel = 0.0;
        self.imu = (0.0, 0.0, 0.0);
        self.stuck_timer = 0.0;
        self.average_vel = 0.0;

        self.spawn_runner();
    }

    pub fn torso_position(&self) -> Vector {
        self.world.bodies[self.torso].translation()
    }

    pub fn torso_angle(&self) -> f32 {
        self.world.bodies[self.torso].rotation().angle()
    }

    pub fn forward_velocity(&self) -> f32 {
        self.world.bodies[self.torso].linvel().x
    }

    /// Drive the eight motors from actions in `[0, 1]`.
    ///
    /// A two-stage low-pass over a proportional position servo, exactly as
    /// upstream: the target angle is smoothed, the resulting error is smoothed
    /// again, and the result becomes a motor *speed* rather than a torque.
    pub fn motor_update(&mut self, actions: &[f32]) {
        debug_assert_eq!(actions.len(), NUM_SEGMENTS);

        for i in 0..NUM_SEGMENTS {
            let target = actions[i] * (SEG_MAX_ANGLE - SEG_MIN_ANGLE) + SEG_MIN_ANGLE;
            self.positions[i] += 0.5 * (target - self.positions[i]);

            let current = self.joint_angle(i);
            let error = self.positions[i] - current;
            self.speeds[i] += 0.5 * (error - self.speeds[i]);

            let handle = self.segment_joints[i];
            if let Some(joint) = self.world.impulse_joints.get_mut(handle, true) {
                if let Some(rev) = joint.data.as_revolute_mut() {
                    rev.set_motor_velocity(self.speeds[i] * SEG_MAX_SPEED, 1.0);
                }
            }
        }
    }

    /// The angle of joint `i`, as the difference between its two bodies' rotations.
    fn joint_angle(&self, i: usize) -> f32 {
        let handle = self.segment_joints[i];
        let Some(joint) = self.world.impulse_joints.get(handle) else {
            return 0.0;
        };
        let a = self.world.bodies[joint.body1()].rotation().angle();
        let b = self.world.bodies[joint.body2()].rotation().angle();
        // Measured from the pose the joint was assembled in, so this lands in
        // [-1.1, 1.1] and matches what Box2D's `b2RevoluteJoint_GetAngle` reports
        // upstream.
        angle_wrap(b - a - self.rest_angles[i])
    }

    /// The joint's raw relative angle, in the solver's own frame.
    fn joint_angle_raw(&self, i: usize) -> f32 {
        self.joint_angle(i) + self.rest_angles[i]
    }

    pub fn step_physics(&mut self) {
        self.world.step();

        // IMU: upstream reports per-frame velocity *differences*, not accelerations
        // — it never divides by dt. Preserved, since the sensor scaling depends on it.
        let lin = self.world.bodies[self.torso].linvel();
        let ang = self.world.bodies[self.torso].angvel();
        self.imu = (
            lin.x - self.prev_lin_vel.x,
            lin.y - self.prev_lin_vel.y,
            ang - self.prev_ang_vel,
        );
        self.prev_lin_vel = lin;
        self.prev_ang_vel = ang;

        self.average_vel = 0.99 * self.average_vel + 0.01 * lin.x;
        if self.average_vel.abs() < 0.1 {
            self.stuck_timer += STUCK_DT;
        } else {
            self.stuck_timer = (self.stuck_timer - STUCK_DT * 2.0).max(0.0);
        }
    }

    /// Whether this episode should end, and why.
    pub fn reset_reason(&self) -> Option<ResetReason> {
        if self.torso_angle().abs() > MAX_BODY_ANGLE {
            return Some(ResetReason::Flipped);
        }
        if self.whiskers[0] < 0.01 {
            return Some(ResetReason::HitWall);
        }
        if self.stuck_timer >= STUCK_TIME {
            return Some(ResetReason::Stuck);
        }
        None
    }

    /// Build the sensor vector. Values are unbounded; the demo squashes them.
    ///
    /// **Deviation from upstream, deliberate.** `runner/Runner.cpp` writes the four
    /// foot-contact flags with `state[si++] = 1.0f` *inside* the "is this foot
    /// touching" conditional, so the write index only advances when a contact is
    /// found. The whisker and IMU readings therefore shift position in the vector
    /// by however many feet happen to be on the ground, and the tail is left at
    /// zero. Here each flag goes to a fixed slot. Reproducing the original is
    /// possible but makes the sensor layout contact-count dependent for no benefit,
    /// and it changes the learning problem either way.
    pub fn state_vector(&mut self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), STATE_SIZE);
        out.fill(0.0);

        for i in 0..NUM_SEGMENTS {
            out[i] = self.joint_angle(i);
        }
        out[8] = self.torso_angle();

        // Fixed slots 9..13, one per foot.
        for (f, &collider) in self.foot_colliders.iter().enumerate() {
            let touching = self
                .world
                .contact_pairs_with(collider)
                .any(|pair| pair.has_any_active_contact());
            out[9 + f] = if touching { 1.0 } else { 0.0 };
        }

        self.cast_whiskers();
        out[13..13 + NUM_WHISKERS].copy_from_slice(&self.whiskers);

        out[19] = self.imu.0;
        out[20] = self.imu.1;
        out[21] = self.imu.2;
    }

    /// Six 1 m rays from the torso's leading edge, fanning downward.
    fn cast_whiskers(&mut self) {
        let pos = *self.world.bodies[self.torso].position();
        let origin = pos.transform_point(Vector::new(BODY_WIDTH * 0.5 + WHISKER_OFFSET_X, 0.0));
        let base_angle = self.world.bodies[self.torso].rotation().angle();

        // Whiskers must see the world but not the runner's own limbs.
        let filter = QueryFilter::default().groups(InteractionGroups::new(
            GROUP_RUNNER,
            GROUP_WORLD,
            InteractionTestMode::And,
        ));

        for i in 0..NUM_WHISKERS {
            let angle = base_angle - WHISKER_SPREAD * i as f32;
            let ray = Ray::new(origin, Vector::new(angle.cos(), angle.sin()));
            self.whiskers[i] = self
                .world
                .cast_ray(&ray, WHISKER_LEN, true, filter)
                .map(|(_, toi)| toi / WHISKER_LEN)
                .unwrap_or(1.0);
        }
    }

    /// Distance to the next hurdle ahead, normalised as upstream does:
    /// `min(1, dist / (2 * hurdle spacing))`, saturating past the last hurdle.
    pub fn hurdle_sensor(&self) -> f32 {
        let x = self.torso_position().x;
        match self.hurdle_x.iter().find(|&&hx| hx > x) {
            None => 1.0,
            Some(&hx) => (0.5 * (hx - x) / HURDLE_OFFSET).min(1.0),
        }
    }
}

impl Default for RunnerWorld {
    fn default() -> Self {
        Self::new()
    }
}

fn angle_wrap(a: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let offset = a + pi;
    let m = offset - (offset / (2.0 * pi)).floor() * (2.0 * pi);
    m - pi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_builds_with_eight_motors_and_four_feet() {
        let w = RunnerWorld::new();
        assert_eq!(w.segments.len(), NUM_SEGMENTS);
        assert_eq!(w.segment_joints.len(), NUM_SEGMENTS);
        assert_eq!(w.foot_colliders.len(), 4);
        assert_eq!(w.hurdle_x.len(), NUM_HURDLES);
    }

    #[test]
    fn runner_spawns_above_the_ground_and_falls_onto_it() {
        let mut w = RunnerWorld::new();
        assert!(w.torso_position().y > GROUND_TOP);

        for _ in 0..600 {
            w.motor_update(&[0.5; NUM_SEGMENTS]);
            w.step_physics();
        }

        // It must land on the ground, not sink through it.
        assert!(
            w.torso_position().y > GROUND_TOP - 0.5,
            "torso sank to {}",
            w.torso_position().y
        );
        assert!(w.torso_position().y.is_finite());
    }

    #[test]
    fn joints_start_inside_their_limits_and_stay_there() {
        // The limits are offset by each joint's assembly angle. Get that wrong and
        // every joint is born outside its own limit, the solver locks the body
        // rigid, and the runner cannot move at all under any policy — which is
        // silent, since nothing errors and the demo just reports zero distance.
        let mut w = RunnerWorld::new();

        for i in 0..NUM_SEGMENTS {
            let a = w.joint_angle(i);
            assert!(
                a.abs() <= SEG_MAX_ANGLE + 1e-3,
                "joint {i} was assembled at {a}, outside [{SEG_MIN_ANGLE}, {SEG_MAX_ANGLE}]"
            );
            // And the raw angle should be near the pose it was built in.
            let raw = w.joint_angle_raw(i);
            assert!((angle_wrap(raw - w.rest_angles[i])).abs() <= SEG_MAX_ANGLE + 1e-3);
        }

        for _ in 0..600 {
            w.motor_update(&[0.5; NUM_SEGMENTS]);
            w.step_physics();
        }
        for i in 0..NUM_SEGMENTS {
            let a = w.joint_angle(i);
            // Solvers allow a little overshoot; anything beyond this means the
            // limit is not being applied in the frame we think it is.
            assert!(a.abs() < SEG_MAX_ANGLE + 0.3, "joint {i} drifted to {a}");
        }
    }

    #[test]
    fn the_body_can_actually_move_when_driven() {
        // A locked-up body reports exactly zero travel, which is what a wrong joint
        // frame produces. Drive the motors through their range and check something
        // happens at all.
        let mut w = RunnerWorld::new();
        let start = w.torso_position();

        for t in 0..1200 {
            let phase = (t as f32 * 0.05).sin() * 0.5 + 0.5;
            let actions: Vec<f32> = (0..NUM_SEGMENTS)
                .map(|i| if i % 2 == 0 { phase } else { 1.0 - phase })
                .collect();
            w.motor_update(&actions);
            w.step_physics();
        }

        let moved = (w.torso_position() - start).length();
        assert!(moved > 0.05, "body barely moved ({moved} m) — joints may be locked");
    }

    #[test]
    fn state_vector_is_finite_and_the_right_width() {
        let mut w = RunnerWorld::new();
        let mut state = vec![0.0f32; STATE_SIZE];

        for i in 0..400 {
            let actions: Vec<f32> = (0..NUM_SEGMENTS)
                .map(|j| ((i + j) % 11) as f32 / 10.0)
                .collect();
            w.motor_update(&actions);
            w.step_physics();
            w.state_vector(&mut state);

            assert_eq!(state.len(), STATE_SIZE);
            assert!(state.iter().all(|v| v.is_finite()), "non-finite sensor: {state:?}");
        }
    }

    #[test]
    fn whiskers_and_contacts_stay_in_their_fixed_slots() {
        let mut w = RunnerWorld::new();
        let mut state = vec![0.0f32; STATE_SIZE];

        for _ in 0..900 {
            w.motor_update(&[0.5; NUM_SEGMENTS]);
            w.step_physics();
        }
        w.state_vector(&mut state);

        // Contacts are flags.
        for f in 0..4 {
            let v = state[9 + f];
            assert!(v == 0.0 || v == 1.0, "contact flag {f} was {v}");
        }
        // Whiskers are fractions.
        for i in 0..NUM_WHISKERS {
            let v = state[13 + i];
            assert!((0.0..=1.0).contains(&v), "whisker {i} was {v}");
        }
    }

    #[test]
    fn feet_touch_the_ground_after_landing() {
        let mut w = RunnerWorld::new();
        let mut state = vec![0.0f32; STATE_SIZE];

        let mut any_contact = false;
        for _ in 0..900 {
            w.motor_update(&[0.5; NUM_SEGMENTS]);
            w.step_physics();
            w.state_vector(&mut state);
            if state[9..13].iter().any(|&v| v > 0.5) {
                any_contact = true;
                break;
            }
        }
        assert!(any_contact, "no foot ever registered a contact after landing");
    }

    #[test]
    fn hurdle_sensor_saturates_far_away_and_shrinks_on_approach() {
        let w = RunnerWorld::new();
        // Spawn is at x = 0, the first hurdle at x = 10, spacing 5: 0.5*10/5 = 1.0.
        assert!((w.hurdle_sensor() - 1.0).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&w.hurdle_sensor()));
    }

    #[test]
    fn reset_rebuilds_the_runner_at_the_spawn_point() {
        let mut w = RunnerWorld::new();
        for _ in 0..300 {
            w.motor_update(&[0.9; NUM_SEGMENTS]);
            w.step_physics();
        }
        w.reset();

        assert_eq!(w.segments.len(), NUM_SEGMENTS);
        assert_eq!(w.foot_colliders.len(), 4);
        assert!((w.torso_position().x).abs() < 1e-3, "did not respawn at x = 0");
        assert!((w.torso_position().y - SPAWN_HEIGHT).abs() < 1e-3);
        assert_eq!(w.stuck_timer, 0.0);
    }

    #[test]
    fn repeated_resets_do_not_leak_bodies() {
        let mut w = RunnerWorld::new();
        let baseline = w.world.bodies.len();
        for _ in 0..10 {
            for _ in 0..30 {
                w.motor_update(&[0.5; NUM_SEGMENTS]);
                w.step_physics();
            }
            w.reset();
        }
        assert_eq!(w.world.bodies.len(), baseline, "reset leaked rigid bodies");
    }
}
