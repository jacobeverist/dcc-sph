// Ball Physics — next-frame video prediction, then generating a trajectory from
// nothing but the model's own output.
//
// Port of `demos/Ball_Physics.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// A ball bounces around a box. The scene is rasterised to a 64x64 silhouette, an
// `ImageEncoder` compresses it to a 20x20x16 CSDR, and a hierarchy learns to
// predict the next frame's CSDR. Nothing is ever told the ball's position or
// velocity — the physics has to be inferred from pixels.
//
// The demo is the generation phase. After five seed frames of real input, the
// hierarchy is fed *its own predictions* with learning off, and `reconstruct()`
// turns each prediction back into pixels. If it has learned the dynamics the ball
// keeps moving and bouncing off the walls with no input at all.
//
//   cargo run --release --example ball_physics
//   cargo run --release --example ball_physics -- --train-episodes 400 --gen-episodes 3

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::env::ball::{
    background_frame, build, detect_ball, frame_mse, BallWorld, EPISODE_FRAMES, FRAME_H, FRAME_W,
};
use support::metrics::{Recorder, Summary};
use support::report::{ascii_image, side_by_side, Rolling};
use support::rng::seed_everything;

/// Frames of real input at the start of a generation episode, before the loop
/// closes. Upstream's `simFrame > 5` gate.
const SEED_FRAMES: usize = 5;

fn main() {
    let args = Args::parse();
    let seed: u64 = args.get("seed", 12345);

    let mut rec = Recorder::from_args("ball_physics", &args);
    run(&args, seed, &mut rec);
    rec.finish();
}

/// One complete run. Split out from `main` so a repeat or a sweep can call it many
/// times; everything it needs comes from `args` and `seed`.
fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {

    let train_episodes: usize = args.get("train-episodes", 250);
    let gen_episodes: usize = args.get("gen-episodes", 3);
    let every: usize = args.get("every", 50);
    let quiet = args.flag("quiet");
    // Rows of ASCII per rendered frame. Terminal cells are about twice as tall as
    // they are wide, so the width is doubled to keep the picture square.
    let art_rows: usize = args.get("art-rows", 20);
    let art_every: usize = args.get("art-every", 15);

    let mut rng = seed_everything(seed);

    rec.config("train_episodes", train_episodes);
    rec.config("gen_episodes", gen_episodes);
    rec.config("seed_frames", SEED_FRAMES);

    // Encoder and hierarchy are built together in `support/env/ball.rs`: the
    // hierarchy's IO port is sized to the encoder's hidden layer, so the two
    // configurations cannot be chosen apart.
    let (mut enc, mut h) = build();

    let mut world = BallWorld::new();
    let mut frame = vec![0u8; FRAME_W * FRAME_H];

    println!(
        "Ball Physics — {train_episodes} training + {gen_episodes} generation episodes of {EPISODE_FRAMES} frames, seed {seed}"
    );
    println!("  ImageEncoder {FRAME_W}x{FRAME_H}x1 radius 6 -> 20x20x16, hierarchy 2 layers 10x10x32");
    println!("  generation closes the loop after {SEED_FRAMES} seed frames");
    println!();

    // --- Training ---

    let mut train_mse = Rolling::new(2_000, 0.001);

    for ep in 0..train_episodes {
        world.reset(&mut rng);
        // Only the hierarchy carries state across frames; the encoder recomputes
        // its hidden CIs from scratch each step and has nothing to reset.
        h.clear_state();

        // Prediction of the *next* frame, held until that frame arrives.
        let mut pending: Option<Vec<u8>> = None;

        for _ in 0..EPISODE_FRAMES {
            world.rasterise(&mut frame);

            if let Some(p) = pending.take() {
                train_mse.push(frame_mse(&p, &frame));
            }

            enc.step(&[&frame], true, true);
            let hidden = enc.get_hidden_cis().to_vec();
            h.step(&[&hidden], true, 0.0, 0.0);

            enc.reconstruct(h.get_prediction_cis(0));
            pending = Some(enc.get_reconstruction(0).to_vec());

            world.step();
        }

        if every > 0 && (ep + 1) % every == 0 {
            rec.sample(ep as u64 + 1, &[("train_mse", train_mse.mean() as f64)]);
            if !quiet {
                println!(
                    "  training episode {:>5} / {train_episodes} | next-frame MSE {:.5}",
                    ep + 1,
                    train_mse.mean()
                );
            }
        }
    }

    println!("\nTraining done — next-frame MSE {:.5}\n", train_mse.mean());

    // --- Generation ---
    //
    // The encoder is not stepped once the loop closes: the real image is ignored
    // entirely and the hierarchy runs on its own predictions. The ground-truth
    // world keeps advancing purely so there is something to score against.

    let background = background_frame();

    let mut gen_mse = Rolling::new(10_000, 0.01);
    let mut frozen_mse = Rolling::new(10_000, 0.01);
    // The metrics that actually say whether the dynamics were learned: does a ball
    // still exist in the generated frame, and is it anywhere near the real one?
    let mut gen_pos_err = Rolling::new(10_000, 0.01);
    let mut frozen_pos_err = Rolling::new(10_000, 0.01);
    let mut ball_present = Rolling::new(10_000, 0.01);
    // Position error bucketed by how far the loop has been running unaided. A
    // single average hides the whole shape of the result: closed-loop rollout
    // tracks well at first and drifts as errors compound, and where that turn
    // happens is the interesting number.
    const HORIZON_BUCKET: usize = 10;
    let num_buckets = EPISODE_FRAMES.div_ceil(HORIZON_BUCKET);
    let mut by_horizon: Vec<Rolling> = (0..num_buckets).map(|_| Rolling::new(10_000, 0.01)).collect();

    for ep in 0..gen_episodes {
        world.reset(&mut rng);
        // Only the hierarchy carries state across frames; the encoder recomputes
        // its hidden CIs from scratch each step and has nothing to reset.
        h.clear_state();

        println!("Generation episode {} / {gen_episodes}", ep + 1);

        let mut pending: Option<Vec<u8>> = None;
        // The baseline to beat: hold the last real frame and never update it.
        let mut frozen: Option<Vec<u8>> = None;

        for f in 0..EPISODE_FRAMES {
            world.rasterise(&mut frame);

            let closed_loop = f >= SEED_FRAMES;

            if let Some(p) = pending.take() {
                let mse = frame_mse(&p, &frame);
                if closed_loop {
                    gen_mse.push(mse);

                    let truth = detect_ball(&frame, &background);
                    let guess = detect_ball(&p, &background);
                    ball_present.push(if guess.is_some() { 1.0 } else { 0.0 });

                    if let (Some(t), Some(g)) = (truth, guess) {
                        let err = ((t.0 - g.0).powi(2) + (t.1 - g.1).powi(2)).sqrt();
                        gen_pos_err.push(err);
                        let bucket = (f - SEED_FRAMES) / HORIZON_BUCKET;
                        if let Some(b) = by_horizon.get_mut(bucket) {
                            b.push(err);
                        }
                    }

                    if let Some(fr) = &frozen {
                        frozen_mse.push(frame_mse(fr, &frame));
                        if let (Some(t), Some(f)) = (truth, detect_ball(fr, &background)) {
                            frozen_pos_err
                                .push(((t.0 - f.0).powi(2) + (t.1 - f.1).powi(2)).sqrt());
                        }
                    }
                }

                if !quiet && art_every > 0 && closed_loop && f % art_every == 0 {
                    let pred_art = ascii_image(&p, FRAME_W, FRAME_H, art_rows * 2, art_rows);
                    let real_art = ascii_image(&frame, FRAME_W, FRAME_H, art_rows * 2, art_rows);
                    println!("  frame {f:>3}  predicted (closed loop)          actual                                MSE {mse:.5}");
                    println!("{}", side_by_side(&pred_art, &real_art, 4));
                }
            }

            if closed_loop {
                // Feed the hierarchy its own prediction; the encoder sits idle.
                let fed = h.get_prediction_cis(0).to_vec();
                h.step(&[&fed], false, 0.0, 0.0);
            } else {
                enc.step(&[&frame], true, true);
                let hidden = enc.get_hidden_cis().to_vec();
                h.step(&[&hidden], true, 0.0, 0.0);
                frozen = Some(frame.clone());
            }

            enc.reconstruct(h.get_prediction_cis(0));
            pending = Some(enc.get_reconstruction(0).to_vec());

            world.step();
        }
    }

    // --- Report ---

    println!();
    println!("Closed-loop generation over {} scored frames:", gen_mse.len());
    println!("  ball still drawn   {:.1}% of frames", ball_present.mean() * 100.0);
    println!(
        "  position error     {:.2} m over {} frames where both balls were found",
        gen_pos_err.mean(),
        gen_pos_err.len()
    );
    println!(
        "  frozen baseline    {:.2} m  (hold the last real frame forever)",
        frozen_pos_err.mean()
    );
    println!(
        "  frame MSE          {:.5} generated vs {:.5} frozen",
        gen_mse.mean(),
        frozen_mse.mean()
    );
    println!(
        "\n  (frame MSE is reported for completeness only — it prefers a blank frame to a\n   slightly misplaced ball, so position error is the metric that means something.)"
    );

    println!("\n  Position error by how long the loop has run unaided:");
    for (b, r) in by_horizon.iter().enumerate() {
        if r.is_empty() {
            continue;
        }
        let lo = b * HORIZON_BUCKET;
        let hi = lo + HORIZON_BUCKET - 1;
        // ~0.5 m per bar keeps the whole 15.6 m view on one line.
        let bar = "#".repeat(((r.mean() * 2.0) as usize).min(40));
        println!("    frames {lo:>3}-{hi:<3}  {:>5.2} m  {bar}", r.mean());
    }

    let learned = ball_present.mean() > 0.5 && gen_pos_err.mean() < frozen_pos_err.mean();

    let mut summary = Summary::new();
    summary.push("ball_present", ball_present.mean() as f64);
    summary.push("position_error", gen_pos_err.mean() as f64);
    summary.push("frozen_position_error", frozen_pos_err.mean() as f64);
    summary.push("generated_mse", gen_mse.mean() as f64);
    summary.push("frozen_mse", frozen_mse.mean() as f64);
    summary.push("train_mse", train_mse.mean() as f64);
    // The rollout curve, so a sweep can see where tracking breaks down rather than
    // only that the average moved.
    for (b, r) in by_horizon.iter().enumerate() {
        if !r.is_empty() {
            summary.push(&format!("position_error_h{}", b * HORIZON_BUCKET), r.mean() as f64);
        }
    }

    if learned {
        println!(
            "\nLearned: the generated ball persists and tracks the real one better than a frozen frame."
        );
        summary.verdict(true, "the generated ball persists and tracks better than a frozen frame");
    } else if ball_present.mean() <= 0.5 {
        println!(
            "\nNot converged: the generated ball fades out — the model collapsed to predicting empty space. Try more --train-episodes."
        );
        summary.verdict(false, "the generated ball fades out");
    } else {
        println!(
            "\nNot converged: the generated ball persists but drifts worse than a frozen frame. Try more --train-episodes."
        );
        summary.verdict(false, "the generated ball drifts worse than a frozen frame");
    }

    rec.finish_summary(&summary);
    summary
}
