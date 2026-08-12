// Video Prediction — learning to predict the next frame, then dreaming the clip.
//
// Port of `demos/Video_Prediction.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// An `ImageEncoder` compresses each RGB frame to a CSDR, a hierarchy learns to
// predict the next one, and `reconstruct()` turns predictions back into pixels.
// After several training passes over the clip the loop closes: the hierarchy is fed
// its own predictions with learning off and generates the rest unaided.
//
// Upstream reads frames with OpenCV, but *only* `cv::VideoCapture` — no `resize`,
// no `cvtColor`. What the demo needs is a sequence of RGB frames, which is not a
// reason to take a video-decoding dependency, so the default source is procedural
// and `--frames <dir>` points at real ones. See `support/env/video.rs`.
//
//   cargo run --release --example video_prediction
//   cargo run --release --example video_prediction -- --passes 20 --frames frames/

#[path = "support/mod.rs"]
mod support;

use dcc_sph::helpers::{Int3, VisibleLayerDesc};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use dcc_sph::image_encoder::ImageEncoder;

use support::args::Args;
use support::checkpoint;
use support::env::video::{frame_detail, frame_mse, to_gray, FrameSource};
use support::metrics::{Recorder, Summary};
use support::report::{ascii_image, side_by_side, Rolling};
use support::rng::seed_everything;
use support::sweep;

/// Frames of real input before the loop closes in the generation phase.
const SEED_FRAMES: usize = 5;

fn main() {
    let args = Args::parse();

    let mut rec = Recorder::from_args("video_prediction", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    // Upstream makes 20 passes over the clip; the default here is lower so the demo
    // finishes in a sensible time on the procedural source.
    let passes: usize = args.get("passes", 8);
    let gen_frames: usize = args.get("gen-frames", 60);
    let every: usize = args.get("every", 2);
    let art_rows: usize = args.get("art-rows", 16);
    let art_every: usize = args.get("art-every", 20);
    let silent = args.flag("silent");
    let quiet = silent || args.flag("quiet");

    macro_rules! say {
        ($($arg:tt)*) => {
            if !silent {
                println!($($arg)*);
            }
        };
    }

    let _rng = seed_everything(seed);

    let mut source = FrameSource::from_args(args);
    let (w, h) = source.dims();
    let frame_count = source.len();

    rec.config("passes", passes);
    rec.config("gen_frames", gen_frames);
    rec.config("frame_width", w);
    rec.config("frame_height", h);
    rec.config("frame_count", frame_count);

    // --- Encoder and hierarchy ---
    //
    // Upstream: encoder hidden 32x32x32 over one (w, h, 3) visible layer at radius
    // 8; hierarchy 2 layers of 16x16x32 over a single Prediction port sized to the
    // encoder's hidden layer. Note the visible layer's z is 3 — the RGB channels —
    // so the buffer layout is `channel + 3 * (y + h * x)`, unlike the single-channel
    // `y + x * h` that `ball_physics` uses.

    let enc_hidden = Int3::new(32, 32, 32);

    let mut enc = ImageEncoder::default();
    enc.init_random(
        enc_hidden,
        vec![VisibleLayerDesc { size: Int3::new(w as i32, h as i32, 3), radius: 8 }],
    );

    let io_descs = vec![IoDesc {
        size: enc_hidden,
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 4,
        up_radius: 2,
        ..Default::default()
    }];

    let layer_descs: Vec<LayerDesc> = (0..2)
        .map(|_| LayerDesc {
            hidden_size: Int3::new(16, 16, 32),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: 0,
            down_radius: 2,
            ticks_per_update: 1,
        })
        .collect();

    let mut h_net = Hierarchy::new();
    h_net.init_random(&io_descs, &layer_descs);
    checkpoint::maybe_load(&mut h_net, args);

    say!("Video Prediction — {}, seed {seed}", source.describe());
    say!("  ImageEncoder {w}x{h}x3 radius 8 -> 32x32x32, hierarchy 2 layers 16x16x32");
    say!("  {passes} training passes, then {gen_frames} generated frames");
    say!();

    // --- Training ---

    let mut train_mse = Rolling::new(4_000, 0.001);

    for pass in 0..passes {
        h_net.clear_state();
        let mut pending: Option<Vec<u8>> = None;

        for i in 0..frame_count {
            let frame = source.frame(i).to_vec();

            // Score last step's prediction against the frame that just arrived.
            if let Some(p) = pending.take() {
                train_mse.push(frame_mse(&p, &frame));
            }

            enc.step(&[&frame], true, true);
            let hidden = enc.get_hidden_cis().to_vec();
            h_net.step(&[&hidden], true, 0.0, 0.0);

            enc.reconstruct(h_net.get_prediction_cis(0));
            pending = Some(enc.get_reconstruction(0).to_vec());
        }

        if every > 0 && (pass + 1) % every == 0 {
            rec.sample(pass as u64 + 1, &[("train_mse", train_mse.mean() as f64)]);
            if !quiet {
                say!(
                    "  pass {:>3} / {passes} | next-frame MSE {:.5}",
                    pass + 1,
                    train_mse.mean()
                );
            }
        }
    }

    say!("\nTraining done — next-frame MSE {:.5}\n", train_mse.mean());

    // --- Generation ---
    //
    // After the seed frames the encoder is not stepped at all: the real clip is
    // ignored and the hierarchy runs on its own predictions. The clip keeps
    // advancing only so there is something to score against.

    h_net.clear_state();

    let mut gen_mse = Rolling::new(10_000, 0.01);
    let mut frozen_mse = Rolling::new(10_000, 0.01);
    // Detail is the check MSE cannot be trusted on — see `env::video::frame_detail`.
    let mut gen_detail = Rolling::new(10_000, 0.01);
    let mut real_detail = Rolling::new(10_000, 0.01);
    let mut pending: Option<Vec<u8>> = None;
    let mut frozen: Option<Vec<u8>> = None;

    for f in 0..gen_frames {
        let frame = source.frame(f).to_vec();
        let closed_loop = f >= SEED_FRAMES;

        if let Some(p) = pending.take() {
            if closed_loop {
                let mse = frame_mse(&p, &frame);
                gen_mse.push(mse);
                gen_detail.push(frame_detail(&p));
                real_detail.push(frame_detail(&frame));
                if let Some(fr) = &frozen {
                    frozen_mse.push(frame_mse(fr, &frame));
                }

                if !quiet && art_every > 0 && f % art_every == 0 {
                    let pred = ascii_image(&to_gray(&p, w, h), w, h, art_rows * 2, art_rows);
                    let real = ascii_image(&to_gray(&frame, w, h), w, h, art_rows * 2, art_rows);
                    say!("  frame {f:>3}  generated                        actual                          MSE {mse:.5}");
                    say!("{}", side_by_side(&pred, &real, 4));
                }
            }
        }

        if closed_loop {
            let fed = h_net.get_prediction_cis(0).to_vec();
            h_net.step(&[&fed], false, 0.0, 0.0);
        } else {
            enc.step(&[&frame], true, true);
            let hidden = enc.get_hidden_cis().to_vec();
            h_net.step(&[&hidden], true, 0.0, 0.0);
            frozen = Some(frame.clone());
        }

        enc.reconstruct(h_net.get_prediction_cis(0));
        pending = Some(enc.get_reconstruction(0).to_vec());
    }

    // --- Report ---

    // How much of the real frame's variance the generated one retains. A model
    // hedging its bets — emitting the blurry average of everywhere the shapes might
    // be — scores well on MSE while driving this toward zero.
    let detail_ratio = if real_detail.mean() > 0.0 {
        (gen_detail.mean() / real_detail.mean()) as f64
    } else {
        f64::NAN
    };

    say!();
    say!("Closed-loop generation over {} scored frames:", gen_mse.len());
    say!("  generated MSE   {:.5}", gen_mse.mean());
    say!(
        "  frozen-frame    {:.5}  (baseline: hold the last real frame forever)",
        frozen_mse.mean()
    );
    say!("  training MSE    {:.5}  (open loop, for reference)", train_mse.mean());
    say!(
        "  detail          {:.4} generated vs {:.4} actual  ({:.0}% retained)",
        gen_detail.mean(),
        real_detail.mean(),
        detail_ratio * 100.0
    );

    let mut summary = Summary::new();
    summary.push("train_mse", train_mse.mean() as f64);
    summary.push("generated_detail", gen_detail.mean() as f64);
    summary.push("actual_detail", real_detail.mean() as f64);
    summary.push("detail_ratio", detail_ratio);
    summary.push("generated_mse", gen_mse.mean() as f64);
    summary.push("frozen_mse", frozen_mse.mean() as f64);
    summary.push("frames", frame_count as f64);

    // Both checks are needed, and neither is sufficient. Beating the frozen frame on
    // MSE can be achieved by hedging — emitting the blurry average of everywhere the
    // shapes might be — which is why detail is checked too. Retaining detail alone
    // would be satisfied by echoing the last frame verbatim, which is what the
    // frozen baseline already does.
    let beats_frozen = gen_mse.mean() < frozen_mse.mean();
    let keeps_detail = detail_ratio > 0.5;

    say!();
    match (beats_frozen, keeps_detail) {
        (true, true) => {
            say!("Learned: generating from its own predictions beats holding the last real frame,");
            say!("and the generated frames keep {:.0}% of the real detail rather than hedging.", detail_ratio * 100.0);
            summary.verdict(true, "generation beats a frozen frame and keeps its detail");
        }
        (true, false) => {
            say!("Hedging: the generated frames beat a frozen frame on MSE, but retain only");
            say!("{:.0}% of the real detail — the model is emitting the blurry average of where", detail_ratio * 100.0);
            say!("the scene might be rather than predicting where it is. MSE rewards exactly");
            say!("that, which is why it is not the whole story here.");
            summary.verdict(false, "beats frozen on MSE but by hedging");
        }
        (false, _) => {
            say!("Not converged: generation is no better than a frozen frame — try more --passes.");
            summary.verdict(false, "generation no better than a frozen frame");
        }
    }

    checkpoint::maybe_save(&h_net, args);

    rec.finish_summary(&summary);
    summary
}
