// Stacking RL — build the configuration you are shown.
//
// Port of `demos/Stacking_RL.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// Three blocks, three columns, a grabber that can carry one block at a time. A
// target configuration is displayed; the reward is the fraction of grid cells that
// match it, cubed. Nothing tells the agent *how* to build anything — only how close
// it currently is.
//
// This is the suite's only goal-conditioned RL demo, and it exists to compare the
// two ways of saying "I want that configuration":
//
//   --goal-mode port   the target grid as a fourth input port (default)
//   --goal-mode top    the target distilled to a top-layer CSDR, via step_with_goal
//
// Upstream passes a per-IO-port goal array to a three-argument `step` that exists in
// no published AOgmaNeo revision. `port` is the closest mainline expression of it;
// `top` uses this crate's RUST-ONLY goal path, the one `stacking_prog` needs. Which
// is better is a measurement, so the demo makes it one:
//
//   cargo run --release --example stacking_rl
//   cargo run --release --example stacking_rl -- --sweep goal-mode=port,top --repeat 5

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::checkpoint;
use support::env::stacking::{
    build_rl_hierarchy, distil_program, fill_cis, match_fraction, random_stacks, GoalMode,
    StackingWorld, HEIGHT, STATE_SIZE, WIDTH,
};
use support::metrics::{Recorder, Summary};
use support::probe;
use support::report::{sparkline, Rolling};
use support::rng::{seed_everything, Rng};
use support::sweep;

fn main() {
    let args = Args::parse();

    let mut rec = Recorder::from_args("stacking_rl", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let steps: usize = args.get("steps", 300_000);
    let baseline_steps: usize = args.get("baseline-steps", 30_000);
    let every: usize = args.get("every", 50_000);
    // Upstream re-randomises the target with probability 0.1 per rendered frame,
    // which works out at roughly one change every few thousand simulation ticks. A
    // fixed episode length is the same idea on a schedule, and it is what makes
    // "how well was the target built" a per-episode number rather than a smear.
    let episode: usize = args.get("episode", 300);
    // Rollout length for `--goal-mode top`. Upstream's `stacking_prog` uses 32.
    let distil_iters: usize = args.get("distil-iters", 32);

    // Parsed explicitly rather than through `Args::get`, which falls back to the
    // default on a parse failure — a silently ignored `--goal-mode prot` would
    // quietly report the wrong thing.
    let mode: GoalMode = match args.str("goal-mode") {
        None => GoalMode::Port,
        Some(s) => s.parse().unwrap_or_else(|e| panic!("{e}")),
    };

    let silent = args.flag("silent");
    let quiet = silent || args.flag("quiet");

    macro_rules! say {
        ($($arg:tt)*) => { if !silent { println!($($arg)*); } };
    }

    let mut rng = seed_everything(seed);

    rec.config("steps", steps);
    rec.config("episode", episode);
    rec.config("goal_mode", format!("{mode:?}").to_lowercase());
    if mode == GoalMode::Top {
        rec.config("distil_iters", distil_iters);
    }

    // --- Baselines ---
    //
    // Random actions say what the score is worth nothing; the scripted solver says
    // what it is worth everything. A match fraction near 0.7 sounds respectable
    // until you notice random play scores that much, because most cells are empty
    // in both grids and agree by default.
    let random = run_reference(baseline_steps, episode, false, &mut rng);
    let scripted = run_reference(baseline_steps, episode, true, &mut rng);

    let mut h = build_rl_hierarchy(mode);
    checkpoint::maybe_load(&mut h, args);

    say!("Stacking RL — {steps} steps, seed {seed}");
    say!(
        "  {WIDTH} columns x {HEIGHT} high, 1 layer 5x5x32, goal mode {}",
        match mode {
            GoalMode::Port => "port (fourth IO port, input-only)",
            GoalMode::Top => "top (distilled top-layer CSDR via step_with_goal)",
        }
    );
    say!("  target re-drawn every {episode} steps");
    say!(
        "  random actions:  mean match {:.3}, built {:.1}% of targets",
        random.0,
        random.1 * 100.0
    );
    say!(
        "  scripted solver: mean match {:.3}, built {:.1}% of targets",
        scripted.0,
        scripted.1 * 100.0
    );
    say!();

    let mut world = StackingWorld::new(&mut rng);
    let mut target = random_stacks(&mut rng);

    let mut state_cis = vec![0i32; STATE_SIZE];
    let mut goal_cis = vec![0i32; STATE_SIZE];
    let mut action_cis = vec![0i32; 1];
    let mut position_cis = vec![0i32; 1];
    let mut program: Vec<i32> = Vec::new();

    fill_cis(&target, &mut goal_cis);
    if mode == GoalMode::Top {
        program = distil_program(&h, &goal_cis, distil_iters);
    }

    let mut match_ema = Rolling::new(20_000, 0.0002);
    let mut built = Rolling::new(256, 0.005);
    let mut episodes = 0u64;
    let mut trend: Vec<f32> = Vec::new();

    for t in 0..steps {
        // End of an episode: score it, then draw a new target. The world is left
        // where it is, as upstream leaves it — the agent has to rebuild from
        // wherever the last target left the blocks, which is the harder and more
        // honest version of the task.
        if t > 0 && t % episode == 0 {
            world.fill_state(&mut state_cis);
            built.push(if match_fraction(&state_cis, &goal_cis) == 1.0 { 1.0 } else { 0.0 });
            episodes += 1;

            target = random_stacks(&mut rng);
            fill_cis(&target, &mut goal_cis);
            // The program is distilled once per target rather than once per step:
            // it depends only on the goal grid and the current weights, and a clone
            // plus a 32-step rollout on every one of 300k steps would dominate the
            // run.
            if mode == GoalMode::Top {
                program = distil_program(&h, &goal_cis, distil_iters);
            }
        }

        world.fill_state(&mut state_cis);
        position_cis[0] = world.position as i32;

        let m = match_fraction(&state_cis, &goal_cis);
        match_ema.push(m);
        let reward = m * m * m; // upstream's `reward *= reward * reward`

        match mode {
            GoalMode::Port => h.step(
                &[&state_cis, &action_cis, &position_cis, &goal_cis],
                true,
                reward,
                0.0,
            ),
            GoalMode::Top => h.step_with_goal(
                &[&state_cis, &action_cis, &position_cis],
                &program,
                true,
                reward,
                0.0,
            ),
        }

        action_cis[0] = h.get_prediction_cis(1)[0];
        world.apply(action_cis[0]);

        if every > 0 && (t + 1) % every == 0 {
            trend.push(match_ema.mean());
            let actor = probe::actor_stats(&h, 1);
            let critic = actor.map(|a| a.mean_value as f64).unwrap_or(f64::NAN);
            let fill = actor.map(|a| a.history_fill() as f64).unwrap_or(f64::NAN);

            rec.sample(
                t as u64 + 1,
                &[
                    ("match", match_ema.mean() as f64),
                    ("built_rate", built.mean() as f64),
                    ("critic_value", critic),
                    ("history_fill", fill),
                    ("episodes", episodes as f64),
                ],
            );
            if quiet {
                continue;
            }
            say!(
                "  step {:>8} / {steps} | match {:.3} | built {:>5.1}% of {episodes} targets | critic {critic:>7.3}",
                t + 1,
                match_ema.mean(),
                built.mean() * 100.0,
            );
        }
    }

    // --- Report ---

    say!();
    say!("Over {steps} steps ({episodes} targets):");
    say!(
        "  mean match          {:.3}   (random {:.3}, scripted {:.3})",
        match_ema.mean(),
        random.0,
        scripted.0
    );
    say!(
        "  targets built       {:.1}%  (random {:.1}%, scripted {:.1}%)",
        built.mean() * 100.0,
        random.1 * 100.0,
        scripted.1 * 100.0
    );

    if trend.len() >= 3 {
        say!("\n  Match fraction over training:");
        say!("    {}", sparkline(&trend));
    }

    // How much of the gap between random play and a perfect solver was closed. The
    // raw match fraction is a poor headline on its own: random play already scores
    // around 0.7 because most cells are empty in both grids.
    let span = scripted.0 - random.0;
    let closed = if span.abs() < 1e-6 { 0.0 } else { (match_ema.mean() - random.0) / span };

    let mut summary = Summary::new();
    summary.push("match", match_ema.mean() as f64);
    summary.push("baseline_match", random.0 as f64);
    summary.push("scripted_match", scripted.0 as f64);
    summary.push("gap_closed", closed as f64);
    summary.push("built_rate", built.mean() as f64);
    summary.push("baseline_built_rate", random.1 as f64);
    summary.push("scripted_built_rate", scripted.1 as f64);
    summary.push("episodes", episodes as f64);

    say!(
        "  gap to scripted     {:.1}% closed",
        closed * 100.0
    );

    if closed > 0.15 {
        say!("\nThe goal is being used: the agent builds targets meaningfully better than random action.");
        summary.verdict(true, "closed a real fraction of the gap to a scripted solver");
    } else {
        say!("\nNo clear use of the goal yet — try more --steps, or compare --goal-mode port against top.");
        summary.verdict(false, "no better than random action on the goal-matching score");
    }

    checkpoint::maybe_save(&h, args);
    rec.finish_summary(&summary);
    summary
}

/// Mean match fraction and build rate under a reference policy — random actions, or
/// the scripted solver. Same episode schedule as the real run.
fn run_reference(steps: usize, episode: usize, scripted: bool, rng: &mut Rng) -> (f32, f32) {
    if steps == 0 {
        return (0.0, 0.0);
    }

    let mut world = StackingWorld::new(rng);
    let mut target = random_stacks(rng);
    let mut state_cis = vec![0i32; STATE_SIZE];
    let mut goal_cis = vec![0i32; STATE_SIZE];
    fill_cis(&target, &mut goal_cis);

    let mut total = 0.0f64;
    let mut built = 0u64;
    let mut episodes = 0u64;

    for t in 0..steps {
        if t > 0 && t % episode == 0 {
            world.fill_state(&mut state_cis);
            if match_fraction(&state_cis, &goal_cis) == 1.0 {
                built += 1;
            }
            episodes += 1;
            target = random_stacks(rng);
            fill_cis(&target, &mut goal_cis);
        }

        world.fill_state(&mut state_cis);
        total += match_fraction(&state_cis, &goal_cis) as f64;

        let action =
            if scripted { world.scripted_action(&target) } else { rng.below(4) as i32 };
        world.apply(action);
    }

    let rate = if episodes == 0 { 0.0 } else { built as f32 / episodes as f32 };
    ((total / steps as f64) as f32, rate)
}
