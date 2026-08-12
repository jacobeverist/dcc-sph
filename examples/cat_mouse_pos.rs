// Cat and Mouse with a learned positional memory.
//
// Port of `demos/Cat_Mouse_Pos.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// The same chase as `cat_mouse`, with one addition: a third Prediction port whose
// output is never compared against anything. Instead each prediction nudges an
// accumulator, and what is fed back into the port next step is the *residual*
// between that accumulator and the prediction. The port therefore has to learn to
// emit whatever increment keeps the estimate consistent with what the agent sees —
// dead reckoning, learned end to end rather than supplied.
//
// This is the only demo in the suite where a port's own prediction becomes its next
// input, and the only one where the hierarchy maintains state outside itself. The
// question it answers is whether that helps: run it against plain `cat_mouse` at
// equal seeds and compare.
//
//   cargo run --release --example cat_mouse_pos
//   cargo run --release --example cat_mouse_pos -- --repeat 5

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::checkpoint;
use support::encode::bin_unit;
use support::env::catmouse::{
    build_pos_hierarchy, CatMouseEnv, Map, PositionalMemory, ACTION_RES, ACTION_SIZE,
    MEMORY_COLUMNS, MEMORY_RES, OBS_RES, OBS_SIZE,
};
use support::metrics::{Recorder, Summary};
use support::probe;
use support::report::Rolling;
use support::rng::{seed_everything, Rng};
use support::sweep;

/// Physics at 120 Hz, control and learning at 30 Hz, as in `cat_mouse`.
const SUBSTEPS: usize = 4;
const DT: f32 = 1.0 / 120.0;

fn main() {
    let args = Args::parse();

    let mut rec = Recorder::from_args("cat_mouse_pos", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let steps: usize = args.get("steps", 200_000);
    let baseline_steps: usize = args.get("baseline-steps", 20_000);
    let every: usize = args.get("every", 25_000);
    let cells: usize = args.get("cells", 5);
    let braid: f32 = args.get("braid", 0.15);
    let timeout: usize = args.get("timeout", 600);
    let num_layers: usize = args.get("layers", 5);
    let memory_rate: f32 = args.get("memory-rate", 0.1);
    let silent = args.flag("silent");
    let quiet = silent || args.flag("quiet");

    macro_rules! say {
        ($($arg:tt)*) => {
            if !silent {
                println!($($arg)*);
            }
        };
    }

    let mut rng = seed_everything(seed);

    rec.config("steps", steps);
    rec.config("cells", cells);
    rec.config("braid", braid);
    rec.config("timeout", timeout);
    rec.config("layers", num_layers);
    rec.config("memory_rate", memory_rate);

    let map = Map::generate(cells, cells, braid, &mut rng);
    let (map_w, map_h) = (map.w, map.h);
    let mut env = CatMouseEnv::new(map, &mut rng);

    let baseline = run_random(baseline_steps, cells, braid, timeout, &mut rng);

    // Two agents, each with its own hierarchy and its own positional memory.
    let mut cat_h = build_pos_hierarchy(num_layers);
    let mut mouse_h = build_pos_hierarchy(num_layers);
    checkpoint::maybe_load(&mut cat_h, args);

    let mut cat_mem = PositionalMemory::new(MEMORY_COLUMNS, MEMORY_RES);
    let mut mouse_mem = PositionalMemory::new(MEMORY_COLUMNS, MEMORY_RES);
    cat_mem.rate = memory_rate;
    mouse_mem.rate = memory_rate;

    say!("Cat and Mouse + positional memory — {steps} decisions at 30 Hz, seed {seed}");
    say!("  maze {map_w}x{map_h}, {num_layers} layers 5x5x32");
    say!(
        "  IO0 (7,5,{OBS_RES}) Prediction, IO1 (1,{ACTION_SIZE},{ACTION_RES}) Action, IO2 (2,2,{MEMORY_RES}) Prediction memory"
    );
    say!(
        "  random baseline: {:.1}% of episodes end in capture, mean {:.0} steps to capture",
        baseline.0 * 100.0,
        baseline.1
    );
    say!();

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
    // Whether the learned estimate actually tracks where the cat is. This is the
    // demo's own question, separate from whether the chase goes well.
    let mut memory_error = Rolling::new(20_000, 0.0005);

    for t in 0..steps {
        let (cat_obs, mouse_obs) = env.observations();
        for i in 0..OBS_SIZE {
            cat_obs_cis[i] = bin_unit(cat_obs[i], OBS_RES);
            mouse_obs_cis[i] = bin_unit(mouse_obs[i], OBS_RES);
        }

        cat_h.step(
            &[&cat_obs_cis, &cat_action_cis, &cat_mem.cis],
            true,
            cat_reward,
            0.0,
        );
        mouse_h.step(
            &[&mouse_obs_cis, &mouse_action_cis, &mouse_mem.cis],
            true,
            mouse_reward,
            0.0,
        );
        cat_reward = 0.0;
        mouse_reward = 0.0;

        // Fold each hierarchy's own memory-port prediction back into its estimate.
        cat_mem.update(cat_h.get_prediction_cis(2));
        mouse_mem.update(mouse_h.get_prediction_cis(2));

        for i in 0..ACTION_SIZE {
            cat_action_cis[i] = cat_h.get_prediction_cis(1)[i];
            mouse_action_cis[i] = mouse_h.get_prediction_cis(1)[i];
            cat_actions[i] = cat_action_cis[i] as f32 / (ACTION_RES - 1) as f32;
            mouse_actions[i] = mouse_action_cis[i] as f32 / (ACTION_RES - 1) as f32;
        }

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

        memory_error.push(cat_mem.position_error(
            env.cat.pos.0 / map_w as f32,
            env.cat.pos.1 / map_h as f32,
        ));

        episode_steps += 1;

        if timeout > 0 && episode_steps >= timeout {
            episodes += 1;
            capture_rate.push(0.0);
            episode_steps = 0;
            env.reset(&mut rng);
        }

        if every > 0 && (t + 1) % every == 0 {
            let actor = probe::actor_stats(&cat_h, 1);
            let critic = actor.map(|a| a.mean_value as f64).unwrap_or(f64::NAN);
            let fill = actor.map(|a| a.history_fill() as f64).unwrap_or(f64::NAN);

            rec.sample(
                t as u64 + 1,
                &[
                    ("capture_rate", capture_rate.mean() as f64),
                    ("critic_value", critic),
                    ("history_fill", fill),
                    ("steps_to_capture", time_to_capture.mean() as f64),
                    ("mean_separation", separation.mean() as f64),
                    ("memory_error", memory_error.mean() as f64),
                    ("captures", captures as f64),
                ],
            );
            if quiet {
                continue;
            }
            say!(
                "  step {:>8} / {steps} | captures {captures:>5} | capture rate {:>5.1}% | steps to capture {:>6.0} | memory error {:.3}",
                t + 1,
                capture_rate.mean() * 100.0,
                time_to_capture.mean(),
                memory_error.mean(),
            );
        }
    }

    say!();
    say!("Over {steps} decisions ({episodes} episodes):");
    say!("  captures            {captures}");
    say!(
        "  capture rate        {:.1}% (random: {:.1}%)",
        capture_rate.mean() * 100.0,
        baseline.0 * 100.0
    );
    say!(
        "  steps to capture    {:.0} (random: {:.0})",
        time_to_capture.mean(),
        baseline.1
    );
    say!("  mean separation     {:.2}", separation.mean());
    say!(
        "  memory error        {:.3} (0.383 is the distance between two random points on a wrapped unit square)",
        memory_error.mean()
    );

    let mut summary = Summary::new();
    summary.push("capture_rate", capture_rate.mean() as f64);
    summary.push("baseline_capture_rate", baseline.0 as f64);
    summary.push("steps_to_capture", time_to_capture.mean() as f64);
    summary.push("baseline_steps_to_capture", baseline.1 as f64);
    summary.push("mean_separation", separation.mean() as f64);
    summary.push("memory_error", memory_error.mean() as f64);
    summary.push("captures", captures as f64);
    summary.push("episodes", episodes as f64);

    // The distance between two independent uniform points on a wrapped unit square.
    // An estimate that has learned nothing sits here; one that tracks sits below.
    //
    // Each wrapped axis difference is uniform on [0, 0.5], so the expected distance
    // is `0.5 * E[hypot(u, v)]` for `u, v ~ U(0, 1)`, and that expectation has the
    // closed form `(sqrt(2) + ln(1 + sqrt(2))) / 3`. Together:
    //
    //     0.5 * (sqrt(2) + ln(1 + sqrt(2))) / 3 = 0.3826
    //
    // which a 400k-sample simulation agrees with to four places.
    const RANDOM_MEMORY_ERROR: f64 = 0.3826;
    let tracks = (memory_error.mean() as f64) < RANDOM_MEMORY_ERROR * 0.8;

    // The chase is the outcome measure that does not depend on any assumption about
    // what the memory port encodes, so it decides the verdict.
    let chases = (capture_rate.mean() as f64) > baseline.0 as f64 * 1.2;

    say!();
    if tracks {
        say!("The positional memory tracks world coordinates directly: its error is well below");
        say!("the random-guess distance.");
    } else {
        say!("The positional memory does not align with world coordinates — its error sits at");
        say!("the random-guess distance.");
        say!();
        say!("Read that narrowly. Nothing constrains the port to encode position in the frame");
        say!("this metric assumes: a rotated, permuted, reflected or offset encoding would be");
        say!("just as useful to the agent and would still score at chance here. The measurement");
        say!("shows the estimate is not a drop-in world position, not that it carries no");
        say!("positional information. Upstream does not measure this at all.");
    }

    say!();
    if chases {
        say!(
            "The chase itself works: {:.1}% capture rate against {:.1}% for random movement.",
            capture_rate.mean() * 100.0,
            baseline.0 * 100.0
        );
        summary.verdict(true, "the cat learns to hunt; see memory_error for the port itself");
    } else {
        say!("The chase has not pulled ahead of random movement yet — try more --steps.");
        summary.verdict(false, "the chase has not pulled ahead of random movement");
    }
    say!(
        "\nWhether the extra port helps is a separate question: run plain cat_mouse at the\nsame seeds and compare capture_rate."
    );

    checkpoint::maybe_save(&cat_h, args);

    rec.finish_summary(&summary);
    summary
}

/// Capture rate and mean steps-to-capture under uniformly random actions.
fn run_random(steps: usize, cells: usize, braid: f32, timeout: usize, rng: &mut Rng) -> (f32, f32) {
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
