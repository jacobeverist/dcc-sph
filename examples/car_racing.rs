// Car Racing — steering a track from whisker sensors alone.
//
// Port of `demos/Car_Racing.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// The car accelerates on its own; the policy only steers. It sees twelve raycast
// distances to the track edge, fanned around its heading, and nothing else — no
// position, no map, no heading. The reward is its speed projected onto the
// direction of the track, so going fast the wrong way scores negative.
//
// This is the only demo that reads a real asset: the track's collision mask and
// checkpoint ring are the upstream PNGs, vendored into `assets/`.
//
//   cargo run --release --example car_racing
//   cargo run --release --example car_racing -- --steps 500000

use std::path::PathBuf;

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::env::racing::{
    build_hierarchy, random_steer, Racing, Track, NUM_SENSORS, SENSOR_GRID, SENSOR_RES, STEER_RES,
};
use support::report::Rolling;
use support::metrics::{Recorder, Summary};
use support::rng::{seed_everything, Rng};

/// Frames held straight at the start, before the policy takes over. Upstream's
/// `actionSetCounter`, which stops the car spinning on the spot from frame one.
const WARMUP_FRAMES: usize = 10;

fn main() {
    let args = Args::parse();
    let seed: u64 = args.get("seed", 12345);

    let mut rec = Recorder::from_args("car_racing", &args);
    run(&args, seed, &mut rec);
    rec.finish();
}

/// One complete run. Split out from `main` so a repeat or a sweep can call it many
/// times; everything it needs comes from `args` and `seed`.
fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {

    let steps: usize = args.get("steps", 300_000);
    let baseline_steps: usize = args.get("baseline-steps", 50_000);
    let every: usize = args.get("every", 50_000);
    // Upstream layers 2% uniform-random steering on top of the actor's own
    // stochastic policy.
    let exploration: f32 = args.get("exploration", 0.02);
    let assets: String = args.get(
        "assets",
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .to_string_lossy()
            .into_owned(),
    );
    let quiet = args.flag("quiet");

    let mut rng = seed_everything(seed);

    // Config must be recorded before the first sample, which writes the run header.
    rec.config("steps", steps);
    rec.config("exploration", exploration);

    let track = match Track::load(std::path::Path::new(&assets)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Could not load the track from {assets}: {e}");
            eprintln!("Expected racingCollision.png and racingCheckpoints.png there.");
            std::process::exit(1);
        }
    };

    let track_desc = format!(
        "{}x{} mask, {} checkpoints, lap length {:.0}",
        track.w,
        track.h,
        track.checkpoints.len(),
        track.lap_length
    );

    // --- Random baseline ---

    let baseline = run_random(baseline_steps, &assets, &mut rng);

    // --- Hierarchy ---
    //
    // Defined in the environment module so the windowed viewer drives exactly the
    // same configuration.

    let mut h = build_hierarchy();

    let mut env = Racing::new(track);

    println!("Car Racing — {steps} frames, seed {seed}");
    println!("  track: {track_desc}");
    println!(
        "  1 layer 5x5x32, IO0 ({SENSOR_GRID},{SENSOR_GRID},{SENSOR_RES}) Prediction ({NUM_SENSORS} sensors), IO1 (1,1,{STEER_RES}) Action"
    );
    println!(
        "  random baseline: {:.1} crashes and {:.0} distance per 1000 frames",
        baseline.0, baseline.1
    );
    println!();

    let mut sensor_cis = vec![0i32; SENSOR_GRID * SENSOR_GRID];
    let mut action_cis = vec![0i32; 1];

    let mut reward_ema = Rolling::new(100_000, 0.0001);
    let mut crashes = 0u64;
    let mut window_crashes = 0u64;
    let mut best_lap_distance = 0.0f32;
    let mut total_distance = 0.0f64;
    let mut window_distance = 0.0f64;
    let mut last_distance = env.distance;
    let mut laps = 0i64;

    for t in 0..steps {
        // Choose the steering angle.
        let mut steer = if t < WARMUP_FRAMES {
            STEER_RES / 2
        } else {
            h.get_prediction_cis(1)[0]
        };
        if exploration > 0.0 && rng.chance(exploration) {
            steer = random_steer(&mut rng);
        }

        let (reward, crashed) = env.step(steer);

        if crashed {
            crashes += 1;
            window_crashes += 1;
            last_distance = env.distance;
        } else {
            // Distance is cumulative and resets with the car, so accumulate deltas.
            let delta = env.distance - last_distance;
            if delta > 0.0 {
                total_distance += delta as f64;
                window_distance += delta as f64;
            }
            last_distance = env.distance;
            best_lap_distance = best_lap_distance.max(env.distance);
        }
        laps = laps.max(env.laps);

        env.sensor_cis(SENSOR_RES, &mut sensor_cis);
        // Feed back the steering that was actually executed, including any random
        // override, so the actor learns from what happened rather than what it asked for.
        action_cis[0] = steer;

        h.step(&[&sensor_cis, &action_cis], true, reward, 0.0);
        reward_ema.push(reward);

        if every > 0 && (t + 1) % every == 0 {
            let per1000 = 1000.0 / every as f64;
            rec.sample(
                t as u64 + 1,
                &[
                    ("reward_ema", reward_ema.ema() as f64),
                    ("crashes_per_1000", window_crashes as f64 * per1000),
                    ("distance_per_1000", window_distance * per1000),
                    ("laps", laps as f64),
                ],
            );
            if quiet {
                window_crashes = 0;
                window_distance = 0.0;
                continue;
            }
            println!(
                "  frame {:>8} / {steps} | reward EMA {:>8.3} | per 1000: {:>5.1} crashes, {:>7.0} distance | best run {:.0}",
                t + 1,
                reward_ema.ema(),
                window_crashes as f64 * per1000,
                window_distance * per1000,
                best_lap_distance,
            );
            window_crashes = 0;
            window_distance = 0.0;
        }
    }

    // --- Report ---

    let per1000 = 1000.0 / steps as f64;
    let crashes_per_1000 = crashes as f64 * per1000;
    let distance_per_1000 = total_distance * per1000;

    println!();
    println!("Over {steps} frames:");
    println!("  crashes          {crashes} ({crashes_per_1000:.1} per 1000 frames)");
    println!("  distance         {distance_per_1000:.0} per 1000 frames");
    println!("  furthest run     {best_lap_distance:.0} (lap length {:.0})", env.track.lap_length);
    println!("  laps completed   {laps}");
    println!("  reward EMA       {:.3}", reward_ema.ema());
    println!(
        "  random baseline  {:.1} crashes, {:.0} distance per 1000 frames",
        baseline.0, baseline.1
    );

    let mut summary = Summary::new();
    summary.push("crashes_per_1000", crashes_per_1000);
    summary.push("baseline_crashes_per_1000", baseline.0);
    summary.push("distance_per_1000", distance_per_1000);
    summary.push("baseline_distance_per_1000", baseline.1);
    summary.push("laps", laps as f64);
    summary.push("furthest_run", best_lap_distance as f64);
    summary.push("reward_ema", reward_ema.ema() as f64);

    if crashes_per_1000 < baseline.0 * 0.8 && distance_per_1000 > baseline.1 * 1.2 {
        println!("\nLearned: crashing less and covering more track than random steering.");
        summary.verdict(true, "crashing less and covering more track than random steering");
    } else if distance_per_1000 > baseline.1 * 1.2 {
        println!("\nPartly learned: covering more track than random steering, but crashing as often.");
        summary.verdict(false, "covering more track than random, but crashing as often");
    } else {
        println!("\nNot converged: no clear gain on the random baseline — try more --steps.");
        summary.verdict(false, "no clear gain on the random baseline");
    }

    rec.finish_summary(&summary);
    summary
}

/// Crashes and distance per 1000 frames under uniformly random steering.
fn run_random(steps: usize, assets: &str, rng: &mut Rng) -> (f64, f64) {
    if steps == 0 {
        return (0.0, 0.0);
    }

    let track = match Track::load(std::path::Path::new(assets)) {
        Ok(t) => t,
        Err(_) => return (0.0, 0.0),
    };
    let mut env = Racing::new(track);

    let mut crashes = 0u64;
    let mut distance = 0.0f64;
    let mut last = env.distance;

    for _ in 0..steps {
        let (_, crashed) = env.step(random_steer(rng));
        if crashed {
            crashes += 1;
            last = env.distance;
        } else {
            let delta = env.distance - last;
            if delta > 0.0 {
                distance += delta as f64;
            }
            last = env.distance;
        }
    }

    let per1000 = 1000.0 / steps as f64;
    (crashes as f64 * per1000, distance * per1000)
}
