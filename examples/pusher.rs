// Pusher — actor-critic control with a shaped reward.
//
// Port of `demos/Pusher.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// The agent drives a disc around a square arena and has to shove a second disc onto
// the origin. What makes it more than a homing task is that the pusher has to get
// on the *far* side of the object first, which means moving away from the goal
// before it can make progress.
//
// This is the crate's `Actor` on a multi-column action port: two columns of five
// cells, one per axis, so the policy emits a 5x5 grid of moves rather than a
// single choice.
//
// A random-action baseline runs first, so the learned numbers can be read against
// something rather than in a vacuum.
//
//   cargo run --release --example pusher
//   cargo run --release --example pusher -- --steps 500000


#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::bin_unit;
use support::env::pusher::{build_hierarchy, Outcome, PusherWorld, ACTION_RES, SENSOR_RES};
use support::report::Rolling;
use support::metrics::{Recorder, Summary};
use support::sweep;
use support::rng::{seed_everything, Rng};

fn main() {
    let args = Args::parse();

    let mut rec = Recorder::from_args("pusher", &args);
    // `drive` runs this once normally, or many times under --repeat / --sweep.
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

/// One complete run. Split out from `main` so a repeat or a sweep can call it many
/// times; everything it needs comes from `args` and `seed`.
fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {

    let steps: usize = args.get("steps", 300_000);
    let baseline_steps: usize = args.get("baseline-steps", 50_000);
    let every: usize = args.get("every", 50_000);
    // Upstream has an exploration hook wired up but leaves it at 0, relying on the
    // actor's own stochastic policy. Same default here.
    // Upstream leaves its exploration hook at 0, relying on the actor's own
    // stochastic policy. That is not enough here — see the timeout note below.
    let exploration: f32 = args.get("exploration", 0.05);
    // Steps before the object is respawned regardless. `--timeout 0` reproduces
    // upstream, which has no episode limit; see `PusherWorld::timeout` for why the
    // demo collapses without one.
    let timeout: usize = args.get("timeout", 500);
    // `--silent` is set by the sweep driver in matrix mode: it suppresses the
    // final report too, not just the periodic lines that `--quiet` covers.
    let silent = args.flag("silent");
    let quiet = silent || args.flag("quiet");

    // Everything this run prints goes through `say!`, which honours --silent. The
    // sweep driver sets that flag in matrix mode: twenty runs of scatter plots and
    // ASCII frames would bury the comparison table the sweep exists to produce.
    macro_rules! say {
        ($($arg:tt)*) => {
            if !silent {
                println!($($arg)*);
            }
        };
    }

    let mut rng = seed_everything(seed);

    // Config must be recorded before the first sample, which is what writes the
    // run header.
    rec.config("steps", steps);
    rec.config("timeout", timeout);
    rec.config("exploration", exploration);

    // --- Random baseline ---
    //
    // Goals reached per 100k steps means nothing on its own: the object respawns
    // in [-0.6, 0.6]^2 and can drift onto the origin unaided. This measures how
    // often that happens without a policy.

    let baseline = run_random(baseline_steps, timeout, &mut rng);

    // Built in `support/env/pusher.rs` so a sweep drives the same configuration.
    let mut h = build_hierarchy();

    let mut world = PusherWorld::new();
    world.timeout = timeout;

    say!("Pusher — {steps} steps, seed {seed}");
    say!("  1 layer 7x7x32, IO0 (2,2,{SENSOR_RES}) Prediction, IO1 (1,2,{ACTION_RES}) Action (importance 0)");
    say!(
        "  random baseline over {baseline_steps} steps: {:.1} goals and {:.1} losses per 100k steps",
        baseline.0, baseline.1
    );
    say!();

    let mut reward_ema = Rolling::new(100_000, 0.0001);
    let mut goals = 0u64;
    let mut losses = 0u64;
    let mut window_goals = 0u64;
    let mut window_losses = 0u64;

    let mut sensor_cis = vec![0i32; 4];
    let mut action_cis = vec![0i32; 2];

    for t in 0..steps {
        // Read the policy's action, apply it, and feed that same action back in —
        // upstream's ordering, and the one the Actor's history expects.
        action_cis[0] = h.get_prediction_cis(1)[0];
        action_cis[1] = h.get_prediction_cis(1)[1];

        for a in action_cis.iter_mut() {
            if exploration > 0.0 && rng.chance(exploration) {
                *a = rng.below(ACTION_RES as usize) as i32;
            }
        }

        let (reward, outcome) = world.step((action_cis[0], action_cis[1]), ACTION_RES, &mut rng);

        match outcome {
            Outcome::Goal => {
                goals += 1;
                window_goals += 1;
            }
            Outcome::OutOfBounds => {
                losses += 1;
                window_losses += 1;
            }
            Outcome::Ongoing | Outcome::Timeout => {}
        }

        let obs = world.observation();
        for (i, &v) in obs.iter().enumerate() {
            sensor_cis[i] = bin_unit(v, SENSOR_RES);
        }

        h.step(&[&sensor_cis, &action_cis], true, reward, 0.0);

        reward_ema.push(reward);

        if every > 0 && (t + 1) % every == 0 {
            let per100k = 100_000.0 / every as f64;
            rec.sample(
                t as u64 + 1,
                &[
                    ("reward_ema", reward_ema.ema() as f64),
                    ("goals_per_100k", window_goals as f64 * per100k),
                    ("losses_per_100k", window_losses as f64 * per100k),
                ],
            );
            if !quiet {
                say!(
                    "  step {:>8} / {steps} | reward EMA {:>8.4} | per 100k: {:.1} goals, {:.1} lost",
                    t + 1,
                    reward_ema.ema(),
                    window_goals as f64 * per100k,
                    window_losses as f64 * per100k,
                );
            }
            window_goals = 0;
            window_losses = 0;
        }
    }

    // --- Report ---

    let scale = 100_000.0 / steps as f64;
    let goals_per_100k = goals as f64 * scale;
    let losses_per_100k = losses as f64 * scale;

    say!();
    say!("Over {steps} steps:");
    say!("  goals reached   {goals} ({goals_per_100k:.1} per 100k steps)");
    say!("  object lost     {losses} ({losses_per_100k:.1} per 100k steps)");
    say!("  reward EMA      {:.4}", reward_ema.ema());
    say!(
        "  random baseline {:.1} goals, {:.1} lost per 100k steps",
        baseline.0, baseline.1
    );

    let mut summary = Summary::new();
    summary.push("goals_per_100k", goals_per_100k);
    summary.push("losses_per_100k", losses_per_100k);
    summary.push("baseline_goals_per_100k", baseline.0);
    summary.push("baseline_losses_per_100k", baseline.1);
    summary.push("reward_ema", reward_ema.ema() as f64);
    // The ratio is what a sweep should compare: absolute goal counts move with
    // --steps, but "how many times better than random" does not.
    summary.push(
        "goals_vs_random",
        if baseline.0 > 0.0 { goals_per_100k / baseline.0 } else { f64::NAN },
    );

    if goals_per_100k > baseline.0 * 1.5 {
        say!(
            "\nLearned: reaching the goal far more often than random action does ({goals_per_100k:.1} vs {:.1} per 100k).",
            baseline.0
        );
        summary.verdict(true, "reaching the goal far more often than random action");
    } else {
        say!(
            "\nNot converged: no clear improvement on the random baseline — try more --steps."
        );
        summary.verdict(false, "no clear improvement on the random baseline");
    }

    rec.finish_summary(&summary);
    summary
}

/// Run the world under uniformly random actions and return goals and losses per
/// 100k steps.
fn run_random(steps: usize, timeout: usize, rng: &mut Rng) -> (f64, f64) {
    if steps == 0 {
        return (0.0, 0.0);
    }

    let mut world = PusherWorld::new();
    world.timeout = timeout;
    let mut goals = 0u64;
    let mut losses = 0u64;

    for _ in 0..steps {
        let a = (
            rng.below(ACTION_RES as usize) as i32,
            rng.below(ACTION_RES as usize) as i32,
        );
        match world.step(a, ACTION_RES, rng).1 {
            Outcome::Goal => goals += 1,
            Outcome::OutOfBounds => losses += 1,
            Outcome::Ongoing | Outcome::Timeout => {}
        }
    }

    let scale = 100_000.0 / steps as f64;
    (goals as f64 * scale, losses as f64 * scale)
}
