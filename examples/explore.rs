// Explore — an agent rewarded purely for being surprised.
//
// Port of `demos/Explore.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// There is no goal and no external reward. The agent's reward *is* its own
// prediction error: the fraction of observation columns the hierarchy got wrong.
// Predicting your surroundings well earns nothing, so the only way to score is to
// keep going somewhere you have not modelled yet — curiosity as a reward function,
// and the only signal of its kind in this suite.
//
//     for (int i = 0; i < agentObsi.size(); i++)
//         if (agentObsi[i] != h.get_prediction_cis(0)[i]) agentCuriosity++;
//     agentCuriosity /= agentObsi.size();
//
// This works only because the observation port is `IoType::Prediction`. It is
// exactly the term that had to be dropped from `cat_mouse`, whose observation port
// is `IoType::None` and returns an empty slice.
//
// Success is measured as map coverage against a random walk — the thing curiosity
// is supposed to buy, and not something the reward mentions.
//
//   cargo run --release --example explore
//   cargo run --release --example explore -- --steps 300000 --cells 7

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::checkpoint;
use support::encode::bin_unit;
use support::env::catmouse::{
    build_explore_hierarchy, CatMouseEnv, Coverage, Map, ACTION_RES, ACTION_SIZE, OBS_RES, OBS_SIZE,
};
use support::metrics::{Recorder, Summary};
use support::probe;
use support::report::Rolling;
use support::rng::{seed_everything, Rng};
use support::sweep;
use support::vec::{Bundle, SegVec};

const SUBSTEPS: usize = 4;
const DT: f32 = 1.0 / 120.0;

/// The hypervector map: 64 segments of 8, big enough to bundle a few hundred
/// place-observations without saturating.
const S: usize = 64;
const L: usize = 8;
type V = SegVec<S, L>;

fn main() {
    let args = Args::parse();

    let mut rec = Recorder::from_args("explore", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let steps: usize = args.get("steps", 150_000);
    // 0 means "match --steps", which is the fair setting and the default.
    let baseline_steps: usize = args.get("baseline-steps", 0);
    let every: usize = args.get("every", 25_000);
    let cells: usize = args.get("cells", 6);
    let braid: f32 = args.get("braid", 0.15);
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

    let map = Map::generate(cells, cells, braid, &mut rng);
    let (map_w, map_h) = (map.w, map.h);

    // The random-walk baseline runs on the same map *for the same number of
    // decisions*. Coverage is monotonic — it only ever goes up — so a shorter
    // baseline would flatter the agent for no reason other than having had less
    // time. `--baseline-steps` can shorten it, but the default matches.
    let baseline_steps = if baseline_steps == 0 { steps } else { baseline_steps };
    let baseline = run_random(&map, baseline_steps, &mut rng);

    let mut coverage = Coverage::new(&map);
    let mut env = CatMouseEnv::new(map, &mut rng);

    let mut h = build_explore_hierarchy();
    checkpoint::maybe_load(&mut h, args);

    say!("Explore — {steps} decisions, curiosity reward only, seed {seed}");
    say!("  maze {map_w}x{map_h} (generated; upstream's map_test.png is missing from the repo)");
    say!("  IO0 (7,5,{OBS_RES}) Prediction — the reward is this port's own error");
    say!(
        "  random-walk baseline over the same {baseline_steps} decisions: {:.1}% covered",
        baseline.fraction() * 100.0
    );
    say!();

    let mut obs_cis = vec![0i32; OBS_SIZE];
    let mut action_cis = vec![0i32; ACTION_SIZE];
    let mut actions = vec![0.5f32; ACTION_SIZE];
    // The second agent never moves; only the explorer matters here.
    let still = vec![0.5f32; ACTION_SIZE];

    let mut curiosity = Rolling::new(20_000, 0.0005);
    let mut reward_acc = 0.0f32;

    // Upstream also accumulates a hypervector "map": the top layer's code bound to
    // a positional vector, bundled into one `Bundle` that stands for the whole
    // space. Now that `support/vec.rs` exists it is nearly free, and it is a second
    // read on whether the agent is building a place representation at all.
    let mut place_vectors: Vec<V> = Vec::new();
    for _ in 0..16 {
        place_vectors.push(V::randomized(&mut rng));
    }
    let mut space = Bundle::<S, L>::zero();

    for t in 0..steps {
        let (obs, _) = env.observations();
        for i in 0..OBS_SIZE {
            obs_cis[i] = bin_unit(obs[i], OBS_RES);
        }

        // Curiosity: how much of what we just saw the hierarchy failed to predict.
        // Read *before* stepping, so it scores the standing prediction.
        let predicted = h.get_prediction_cis(0);
        let mismatches = if predicted.len() == OBS_SIZE {
            (0..OBS_SIZE).filter(|&i| obs_cis[i] != predicted[i]).count()
        } else {
            0
        };
        let surprise = mismatches as f32 / OBS_SIZE as f32;
        curiosity.push(surprise);
        reward_acc += surprise;

        h.step(&[&obs_cis, &action_cis], true, reward_acc, 0.0);
        reward_acc = 0.0;

        for i in 0..ACTION_SIZE {
            action_cis[i] = h.get_prediction_cis(1)[i];
            actions[i] = action_cis[i] as f32 / (ACTION_RES - 1) as f32;
        }

        for _ in 0..SUBSTEPS {
            env.step(&actions, &still, DT);
        }

        coverage.visit(&env.map, env.cat.pos);

        // Bundle "where I am" bound to "what I see" into the running map vector.
        let px = ((env.cat.pos.0 / map_w as f32) * 4.0).clamp(0.0, 3.0) as usize;
        let py = ((env.cat.pos.1 / map_h as f32) * 4.0).clamp(0.0, 3.0) as usize;
        let place = place_vectors[px + py * 4];

        // The observation is 35 columns; the vector is 64 segments. Pad the rest.
        let mut code_cis = vec![0i32; S];
        code_cis[..OBS_SIZE.min(S)].copy_from_slice(&obs_cis[..OBS_SIZE.min(S)]);
        let code = V::from_cis(&code_cis);

        space.blend_vec(&place.bind(&code), 0.0005);

        if every > 0 && (t + 1) % every == 0 {
            let actor = probe::actor_stats(&h, 1);
            let critic = actor.map(|a| a.mean_value as f64).unwrap_or(f64::NAN);

            rec.sample(
                t as u64 + 1,
                &[
                    ("coverage", coverage.fraction() as f64),
                    ("curiosity", curiosity.mean() as f64),
                    ("critic_value", critic),
                    ("visited_cells", coverage.visited() as f64),
                ],
            );
            if quiet {
                continue;
            }
            say!(
                "  step {:>8} / {steps} | coverage {:>5.1}% ({} cells) | curiosity {:.3}",
                t + 1,
                coverage.fraction() * 100.0,
                coverage.visited(),
                curiosity.mean(),
            );
        }
    }

    // --- Report ---

    say!();
    say!("Over {steps} decisions:");
    say!(
        "  coverage            {:.1}% of the maze ({} cells) — random walk: {:.1}%",
        coverage.fraction() * 100.0,
        coverage.visited(),
        baseline.fraction() * 100.0
    );
    say!(
        "  curiosity           {:.3}  (mean fraction of observation columns mispredicted)",
        curiosity.mean()
    );
    say!(
        "  place code          thinned map vector settles to {} distinct segments",
        space.thin().as_slice().iter().collect::<std::collections::HashSet<_>>().len()
    );

    let mut summary = Summary::new();
    summary.push("coverage", coverage.fraction() as f64);
    summary.push("baseline_coverage", baseline.fraction() as f64);
    summary.push("curiosity", curiosity.mean() as f64);
    summary.push("visited_cells", coverage.visited() as f64);

    // Time-to-coverage discriminates where final coverage cannot: on a small maze
    // a random walk eventually reaches nearly every cell, so the totals converge
    // while the *speed* of getting there still separates them.
    let pace = |c: &Coverage, f: f32| c.steps_to(f).map(|s| s as f64).unwrap_or(f64::NAN);
    let agent_90 = pace(&coverage, 0.9);
    let random_90 = pace(&baseline, 0.9);

    say!(
        "  steps to 50%        {:.0} vs {:.0} for the random walk",
        pace(&coverage, 0.5),
        pace(&baseline, 0.5)
    );
    say!(
        "  steps to 90%        {} vs {}",
        if agent_90.is_finite() { format!("{agent_90:.0}") } else { "never reached".into() },
        if random_90.is_finite() { format!("{random_90:.0}") } else { "never reached".into() }
    );

    summary.push("steps_to_50", pace(&coverage, 0.5));
    summary.push("steps_to_90", agent_90);
    summary.push("baseline_steps_to_50", pace(&baseline, 0.5));
    summary.push("baseline_steps_to_90", random_90);

    // Faster to the same coverage is the claim curiosity can actually support.
    //
    // A milestone the baseline never reaches counts as a win rather than as missing
    // data — reaching 90% at all when a random walk finishes at 89.5% is strictly
    // better, and treating that NaN as a failure would report the opposite.
    let faster = match (agent_90.is_finite(), random_90.is_finite()) {
        (true, true) => agent_90 < random_90 * 0.9,
        (true, false) => true,
        _ => false,
    };

    say!();
    if faster {
        say!("Learned: curiosity reached 90% of the maze faster than a random walk, with no");
        say!("goal and no external reward.");
        summary.verdict(true, "reached 90% coverage faster than a random walk");
    } else {
        say!("Not converged: no faster to 90% coverage than a random walk.");
        say!();
        say!("Two things make this hard to show on a small maze. Curiosity is a *vanishing*");
        say!("signal — it pays only while the model is still wrong, so it fades exactly where");
        say!("the agent has already been — and a random walk in a small space eventually");
        say!("covers nearly everything anyway, which is why final coverage saturates and only");
        say!("time-to-coverage discriminates at all. Try --cells 10 and more --steps.");
        summary.verdict(false, "no faster to 90% coverage than a random walk");
    }

    checkpoint::maybe_save(&h, args);

    rec.finish_summary(&summary);
    summary
}

/// Fraction of the maze a random walk covers in `steps` decisions.
fn run_random(map: &Map, steps: usize, rng: &mut Rng) -> Coverage {
    let mut coverage = Coverage::new(map);
    if steps == 0 {
        return coverage;
    }

    // The walk must explore the *same* maze for the comparison to mean anything.
    let mut env = CatMouseEnv::new(map.clone(), rng);

    let mut actions = vec![0.5f32; ACTION_SIZE];
    let still = vec![0.5f32; ACTION_SIZE];

    for _ in 0..steps {
        for a in actions.iter_mut() {
            *a = rng.below(ACTION_RES as usize) as f32 / (ACTION_RES - 1) as f32;
        }
        for _ in 0..SUBSTEPS {
            env.step(&actions, &still, DT);
        }
        coverage.visit(&env.map, env.cat.pos);
    }

    coverage
}
