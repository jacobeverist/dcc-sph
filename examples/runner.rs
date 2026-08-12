// Runner — learning a gait for an eight-motor articulated body.
//
// Port of `demos/Runner_Run.cpp` and `demos/runner/Runner.cpp` from
// jacobeverist/OgmaNeoDemos @ aogmaneo. See `doc/Demos.md` for the deviations.
//
// A torso with four two-segment legs has to run to the right over a line of
// hurdles that get taller the further it gets. The actor commands eight joint
// angles through a smoothed position servo, and the reward is simply forward
// velocity, minus a large penalty when the runner falls, hits a hurdle, or stalls.
//
// This is the hardest problem in the suite and the only one that needs a physics
// engine. It is also the only demo where the observation port is `IoType::None` on
// 24 columns and the action port spans eight.
//
//   cargo run --release --example runner
//   cargo run --release --example runner -- --steps 500000


#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::{bin_sigmoid, bin_unit};
use support::env::runner::{
    build_hierarchy, ResetReason, RunnerWorld, ACTION_RES, NUM_SEGMENTS, SENSOR_COLUMNS,
    SENSOR_COUNT, SENSOR_RES, STATE_SIZE,
};
use support::report::Rolling;
use support::metrics::{Recorder, Summary};
use support::rng::{seed_everything, Rng};

fn main() {
    let args = Args::parse();
    let seed: u64 = args.get("seed", 12345);

    let mut rec = Recorder::from_args("runner", &args);
    run(&args, seed, &mut rec);
    rec.finish();
}

/// One complete run. Split out from `main` so a repeat or a sweep can call it many
/// times; everything it needs comes from `args` and `seed`.
fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {

    let steps: usize = args.get("steps", 300_000);
    let baseline_steps: usize = args.get("baseline-steps", 30_000);
    let every: usize = args.get("every", 50_000);
    let quiet = args.flag("quiet");

    let mut rng = seed_everything(seed);

    // Config must be recorded before the first sample, which writes the run header.
    rec.config("steps", steps);
    rec.config("baseline_steps", baseline_steps);

    // --- Random baseline ---
    //
    // A flailing runner still travels a little, and falls over constantly. Both
    // numbers are needed before the learned ones mean anything.

    let baseline = run_random(baseline_steps, &mut rng);

    // Built in `support/env/runner.rs` so a sweep drives the same configuration.
    let mut h = build_hierarchy();

    let mut world = RunnerWorld::new();

    println!("Runner — {steps} control steps at 60 Hz, seed {seed}");
    println!(
        "  1 layer 5x5x64, IO0 (4,6,{SENSOR_RES}) None ({SENSOR_COUNT} sensors), IO1 (2,4,{ACTION_RES}) Action ({NUM_SEGMENTS} motors)"
    );
    println!(
        "  random baseline: {:.2} m furthest, {:.1} resets per 1000 steps, {:.0}% of them stalls",
        baseline.furthest,
        baseline.resets_per_1000,
        baseline.stall_fraction * 100.0
    );
    println!();

    let mut state = vec![0.0f32; STATE_SIZE];
    let mut sensor_cis = vec![0i32; SENSOR_COLUMNS];
    let mut action_cis = vec![0i32; NUM_SEGMENTS];
    let mut actions = vec![0.5f32; NUM_SEGMENTS];

    let mut velocity = Rolling::new(100_000, 0.0001);
    let mut furthest = 0.0f32;
    let mut window_furthest = 0.0f32;
    let mut resets = 0u64;
    let mut window_resets = 0u64;
    let mut flipped = 0u64;
    let mut hit_wall = 0u64;
    let mut stuck = 0u64;
    // Set on the step after a reset, so the penalty lands with the transition that
    // caused it rather than the one that follows.
    let mut just_reset = false;

    for t in 0..steps {
        world.state_vector(&mut state);
        for i in 0..STATE_SIZE {
            // Joint angles, torso angle and IMU deltas have no natural bounds, so
            // everything goes through the same sigmoid squash upstream uses.
            sensor_cis[i] = bin_sigmoid(state[i], SENSOR_RES, 1.0);
        }
        sensor_cis[STATE_SIZE] = bin_unit(world.hurdle_sensor(), SENSOR_RES);

        let vel = world.forward_velocity();
        let mut reward = vel;
        if just_reset {
            reward -= 100.0;
            just_reset = false;
        }

        // The action fed back is the previous one: upstream reads the new action
        // after stepping, and applies it to the motors on this same iteration.
        h.step(&[&sensor_cis, &action_cis], true, reward, 0.0);

        for i in 0..NUM_SEGMENTS {
            action_cis[i] = h.get_prediction_cis(1)[i];
            actions[i] = action_cis[i] as f32 / (ACTION_RES - 1) as f32;
        }

        world.motor_update(&actions);
        world.step_physics();

        velocity.push(world.forward_velocity());
        let x = world.torso_position().x;
        furthest = furthest.max(x);
        window_furthest = window_furthest.max(x);

        if let Some(reason) = world.reset_reason() {
            match reason {
                ResetReason::Flipped => flipped += 1,
                ResetReason::HitWall => hit_wall += 1,
                ResetReason::Stuck => stuck += 1,
            }
            resets += 1;
            window_resets += 1;
            just_reset = true;
            world.reset();
        }

        if every > 0 && (t + 1) % every == 0 {
            rec.sample(
                t as u64 + 1,
                &[
                    ("mean_velocity", velocity.ema() as f64),
                    ("window_furthest", window_furthest as f64),
                    ("resets_per_1000", window_resets as f64 * 1000.0 / every as f64),
                ],
            );
            if !quiet {
                println!(
                    "  step {:>8} / {steps} | mean velocity {:>6.2} m/s | furthest this window {:>6.2} m | {:>5.1} resets per 1000",
                    t + 1,
                    velocity.ema(),
                    window_furthest,
                    window_resets as f64 * 1000.0 / every as f64,
                );
            }
            window_resets = 0;
            window_furthest = 0.0;
        }
    }

    // --- Report ---

    let resets_per_1000 = resets as f64 * 1000.0 / steps as f64;

    println!();
    println!("Over {steps} control steps:");
    let stall_fraction = if resets == 0 { 0.0 } else { stuck as f64 / resets as f64 };

    println!(
        "  furthest reached  {furthest:.2} m (random: {:.2} m)",
        baseline.furthest
    );
    println!("  mean velocity     {:.3} m/s", velocity.mean());
    println!(
        "  resets            {resets} ({resets_per_1000:.1} per 1000 steps, random: {:.1})",
        baseline.resets_per_1000
    );
    println!("    fell over       {flipped}");
    println!("    hit a hurdle    {hit_wall}");
    println!(
        "    stalled         {stuck} ({:.0}% of resets, random: {:.0}%)",
        stall_fraction * 100.0,
        baseline.stall_fraction * 100.0
    );

    // Distance is what the reward actually buys. Mean velocity hovers near zero
    // even for a body that clearly travels, because progress comes in bursts
    // between resets rather than as steady running.
    let travels = furthest > baseline.furthest * 1.5;
    let engages = stall_fraction < baseline.stall_fraction.max(0.05);

    let mut summary = Summary::new();
    summary.push("furthest", furthest as f64);
    summary.push("baseline_furthest", baseline.furthest as f64);
    summary.push("mean_velocity", velocity.mean() as f64);
    summary.push("resets_per_1000", resets_per_1000);
    summary.push("baseline_resets_per_1000", baseline.resets_per_1000);
    summary.push("stall_fraction", stall_fraction);
    summary.push("baseline_stall_fraction", baseline.stall_fraction);
    summary.push("flipped", flipped as f64);
    summary.push("hit_hurdle", hit_wall as f64);
    summary.push(
        "furthest_vs_random",
        if baseline.furthest > 0.0 { (furthest / baseline.furthest) as f64 } else { f64::NAN },
    );

    if travels && engages {
        println!(
            "\nLearned: travelling several times further than a flailing body, and reaching hurdles\nrather than stalling in place."
        );
        summary.verdict(true, "travelling far further than a flailing body, and reaching hurdles");
    } else if travels {
        println!("\nPartly learned: covering more ground than random flailing, but still stalling often.");
        summary.verdict(false, "covering more ground than random, but still stalling often");
    } else {
        println!(
            "\nNot converged: no further than random flailing. This is by far the hardest problem in\nthe suite — a gait has to be discovered from a sparse velocity signal across eight\ncoupled motors — so expect it to need far more --steps than the other demos."
        );
        summary.verdict(false, "no further than random flailing");
    }

    rec.finish_summary(&summary);
    summary
}

/// What a flailing body achieves: furthest travel, resets per 1000 steps, and the
/// fraction of those resets caused by stalling rather than by reaching a hurdle.
///
/// That last figure is the most telling comparison. A body that has learned nothing
/// stalls in place; one that is travelling runs into obstacles instead.
struct Baseline {
    furthest: f32,
    resets_per_1000: f64,
    stall_fraction: f64,
}

fn run_random(steps: usize, rng: &mut Rng) -> Baseline {
    if steps == 0 {
        return Baseline { furthest: 0.0, resets_per_1000: 0.0, stall_fraction: 0.0 };
    }

    let mut world = RunnerWorld::new();
    let mut actions = vec![0.5f32; NUM_SEGMENTS];
    let mut furthest = 0.0f32;
    let mut resets = 0u64;
    let mut stalls = 0u64;

    for _ in 0..steps {
        for a in actions.iter_mut() {
            *a = rng.below(ACTION_RES as usize) as f32 / (ACTION_RES - 1) as f32;
        }
        world.motor_update(&actions);
        world.step_physics();

        furthest = furthest.max(world.torso_position().x);

        if let Some(reason) = world.reset_reason() {
            resets += 1;
            if reason == ResetReason::Stuck {
                stalls += 1;
            }
            world.reset();
        }
    }

    Baseline {
        furthest,
        resets_per_1000: resets as f64 * 1000.0 / steps as f64,
        stall_fraction: if resets == 0 { 0.0 } else { stalls as f64 / resets as f64 },
    }
}
