// Cat and Mouse — two hierarchies learning against each other.
//
// Port of `demos/Cat_Mouse.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// Two agents share a maze. Each has its own independent `Hierarchy` with its own
// actor, each sees only a 30-ray depth fan plus a "can I see the other one" sense,
// and the reward is exactly zero-sum: +100 to the cat on capture, -100 to the
// mouse, nothing otherwise. Neither is told what the other is doing.
//
// This is the only demo in the suite with two hierarchies in one environment, and
// the only one whose difficulty moves on its own: whatever either agent learns
// changes the problem the other one faces.
//
// Upstream loads `resources/map0.png`, which is missing from the repository, so the
// maze is generated from `--seed`.
//
//   cargo run --release --example cat_mouse
//   cargo run --release --example cat_mouse -- --steps 400000 --cells 10

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::bin_unit;
use support::env::catmouse::{
    build_hierarchy, CatMouseEnv, Map, ACTION_RES, ACTION_SIZE, OBS_RES, OBS_SIZE,
};
use support::report::{sparkline, Rolling};
use support::rng::{seed_everything, Rng};

/// Physics runs at 120 Hz, control and learning at 30 Hz — four physics substeps
/// per decision, with the action held across them. Upstream's `dt` and `aiDT`.
const SUBSTEPS: usize = 4;
const DT: f32 = 1.0 / 120.0;

fn main() {
    let args = Args::parse();

    // Counted in AI steps (decisions), not physics substeps.
    let steps: usize = args.get("steps", 200_000);
    let baseline_steps: usize = args.get("baseline-steps", 20_000);
    let seed: u64 = args.get("seed", 12345);
    let every: usize = args.get("every", 25_000);
    // A 5x5 cell maze is 11x11 tiles. Bigger mazes look better but make capture so
    // rare that nothing is measurable: at 8x8 a random cat catches the mouse in
    // about 9% of episodes and a 40k-step run produces two captures total, which is
    // not enough to tell learning from noise.
    let cells: usize = args.get("cells", 5);
    let braid: f32 = args.get("braid", 0.15);
    // Give up on an episode after this many AI steps. Upstream has no time limit;
    // without one a mouse that simply outruns the cat produces an episode that
    // never ends, and mean time-to-capture becomes unmeasurable.
    let timeout: usize = args.get("timeout", 600);
    let quiet = args.flag("quiet");

    let mut rng = seed_everything(seed);

    let map = Map::generate(cells, cells, braid, &mut rng);
    let (map_w, map_h) = (map.w, map.h);
    let mut env = CatMouseEnv::new(map, &mut rng);

    // --- Random baseline ---

    let baseline = run_random(baseline_steps, cells, braid, timeout, &mut rng);

    // --- Two hierarchies, identical descriptors ---
    //
    // The observation port is `IoType::None`: the agents never predict what they
    // will see, only act on it. Note that means `get_prediction_cis(0)` returns an
    // empty slice — upstream's commented-out "curiosity" reward reads exactly that
    // and would panic here, which is why it is not ported.
    //
    // The descriptors live in the environment module so the windowed viewer drives
    // the same configuration.

    let mut cat_h = build_hierarchy();
    let mut mouse_h = build_hierarchy();

    println!("Cat and Mouse — {steps} decisions at 30 Hz, seed {seed}");
    println!("  maze {map_w}x{map_h} (generated; upstream's map0.png is missing from the repo)");
    println!(
        "  two hierarchies, each 1 layer 5x5x128, IO0 (7,5,{OBS_RES}) None, IO1 (1,{ACTION_SIZE},{ACTION_RES}) Action"
    );
    println!(
        "  random baseline: {:.1}% of episodes end in capture, mean {:.0} steps to capture",
        baseline.0 * 100.0,
        baseline.1
    );
    println!();

    let mut cat_obs_cis = vec![0i32; OBS_SIZE];
    let mut mouse_obs_cis = vec![0i32; OBS_SIZE];
    let mut cat_action_cis = vec![0i32; ACTION_SIZE];
    let mut mouse_action_cis = vec![0i32; ACTION_SIZE];
    let mut cat_actions = vec![0.5f32; ACTION_SIZE];
    let mut mouse_actions = vec![0.5f32; ACTION_SIZE];

    let mut cat_reward = 0.0f32;
    let mut mouse_reward = 0.0f32;

    let mut captures = 0u64;
    let mut episodes = 0u64;
    let mut episode_steps = 0usize;
    let mut time_to_capture = Rolling::new(64, 0.02);
    let mut capture_rate = Rolling::new(64, 0.02);
    let mut separation = Rolling::new(20_000, 0.0005);
    let mut trend: Vec<f32> = Vec::new();

    for t in 0..steps {
        // Observe the current state and bin it.
        let (cat_obs, mouse_obs) = env.observations();
        for i in 0..OBS_SIZE {
            cat_obs_cis[i] = bin_unit(cat_obs[i], OBS_RES);
            mouse_obs_cis[i] = bin_unit(mouse_obs[i], OBS_RES);
        }

        // Step with the reward accumulated since the last decision, and with the
        // *previous* action fed back — upstream updates its action buffer after
        // calling step, so that is what the hierarchy sees.
        cat_h.step(&[&cat_obs_cis, &cat_action_cis], true, cat_reward, 0.0);
        mouse_h.step(&[&mouse_obs_cis, &mouse_action_cis], true, mouse_reward, 0.0);
        cat_reward = 0.0;
        mouse_reward = 0.0;

        for i in 0..ACTION_SIZE {
            cat_action_cis[i] = cat_h.get_prediction_cis(1)[i];
            mouse_action_cis[i] = mouse_h.get_prediction_cis(1)[i];
            cat_actions[i] = cat_action_cis[i] as f32 / (ACTION_RES - 1) as f32;
            mouse_actions[i] = mouse_action_cis[i] as f32 / (ACTION_RES - 1) as f32;
        }

        // Hold the decision across four physics substeps.
        for _ in 0..SUBSTEPS {
            env.step(&cat_actions, &mouse_actions, DT);
            separation.push(env.distance);

            if env.done() {
                cat_reward += 100.0;
                mouse_reward -= 100.0;
                captures += 1;
                episodes += 1;
                time_to_capture.push(episode_steps as f32);
                capture_rate.push(1.0);
                episode_steps = 0;
                env.reset(&mut rng);
            }
        }

        episode_steps += 1;

        if timeout > 0 && episode_steps >= timeout {
            episodes += 1;
            capture_rate.push(0.0);
            episode_steps = 0;
            env.reset(&mut rng);
        }

        if !quiet && every > 0 && (t + 1) % every == 0 {
            trend.push(time_to_capture.mean());
            println!(
                "  step {:>8} / {steps} | captures {captures:>5} | capture rate {:>5.1}% | mean steps to capture {:>6.0} | mean separation {:.1}",
                t + 1,
                capture_rate.mean() * 100.0,
                time_to_capture.mean(),
                separation.mean(),
            );
        }
    }

    // --- Report ---

    println!();
    println!("Over {steps} decisions ({episodes} episodes):");
    println!("  captures            {captures}");
    println!(
        "  capture rate        {:.1}% of the last {} episodes (random: {:.1}%)",
        capture_rate.mean() * 100.0,
        capture_rate.len(),
        baseline.0 * 100.0
    );
    println!(
        "  steps to capture    {:.0} (random: {:.0})",
        time_to_capture.mean(),
        baseline.1
    );
    println!("  mean separation     {:.2}", separation.mean());

    if trend.len() >= 3 {
        println!("\n  Steps-to-capture over training:");
        println!("    {}", sparkline(&trend));
        println!(
            "    (no direction is 'correct' here — the cat pulls it down, the mouse pushes it up,\n     and both are learning, so an arms race shows up as movement rather than convergence.)"
        );
    }

    if capture_rate.mean() > baseline.0 * 1.2 {
        println!("\nThe cat is ahead: it catches the mouse more often than random movement does.");
    } else if capture_rate.mean() < baseline.0 * 0.8 {
        println!("\nThe mouse is ahead: it evades better than random movement does.");
    } else {
        println!("\nEvenly matched so far, or neither has learned much — try more --steps.");
    }
}

/// Capture rate and mean steps-to-capture under uniformly random actions.
fn run_random(
    steps: usize,
    cells: usize,
    braid: f32,
    timeout: usize,
    rng: &mut Rng,
) -> (f32, f32) {
    if steps == 0 {
        return (0.0, 0.0);
    }

    let map = Map::generate(cells, cells, braid, rng);
    let mut env = CatMouseEnv::new(map, rng);

    let mut captures = 0u64;
    let mut episodes = 0u64;
    let mut total_steps = 0u64;
    let mut episode_steps = 0usize;

    let mut cat = vec![0.5f32; ACTION_SIZE];
    let mut mouse = vec![0.5f32; ACTION_SIZE];

    for _ in 0..steps {
        for i in 0..ACTION_SIZE {
            cat[i] = rng.below(ACTION_RES as usize) as f32 / (ACTION_RES - 1) as f32;
            mouse[i] = rng.below(ACTION_RES as usize) as f32 / (ACTION_RES - 1) as f32;
        }

        for _ in 0..SUBSTEPS {
            env.step(&cat, &mouse, DT);
            if env.done() {
                captures += 1;
                episodes += 1;
                total_steps += episode_steps as u64;
                episode_steps = 0;
                env.reset(rng);
            }
        }

        episode_steps += 1;

        if timeout > 0 && episode_steps >= timeout {
            episodes += 1;
            episode_steps = 0;
            env.reset(rng);
        }
    }

    let rate = if episodes == 0 { 0.0 } else { captures as f32 / episodes as f32 };
    let mean = if captures == 0 { f32::NAN } else { total_steps as f32 / captures as f32 };
    (rate, mean)
}
