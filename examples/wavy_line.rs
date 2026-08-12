// Wavy Line — online multi-channel sequence prediction with N-step lookahead.
//
// Port of `demos/Wavy_Line.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// Several sine waves are combined into signals that are fed to the hierarchy one
// sample at a time. It learns online to predict each signal's next value. On top
// of that, the demo does an N-step-ahead rollout every step: snapshot the
// hierarchy's state, feed its own predictions back to itself N-1 times, read the
// N-step prediction, then restore the snapshot. That round trip through
// `write_state`/`read_state` is the part of the library this demo exists to
// exercise — the rollout must leave the hierarchy exactly as it found it, and
// `--check-state` verifies that every step rather than assuming it.
//
// Runs headless by default:
//   cargo run --release --example wavy_line
//   cargo run --release --example wavy_line -- --steps 40000 --ahead 8 --noise 0.01

use dcc_sph::helpers::{Int3, SliceReader, StreamReader, StreamWriter, VecWriter};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::{bin_range, unbin_range};
use support::env::wavy::{WavyLine, LINE_MAX, LINE_MIN};
use support::report::{sparkline, Rolling};
use support::rng::seed_everything;

/// Cells per input column. Upstream's `inputColumnSize` under
/// `USE_SIMPLE_FLOAT_ENCODER_`, which is the branch that actually compiles.
const COLUMN_SIZE: i32 = 64;

fn main() {
    let args = Args::parse();

    let steps: usize = args.get("steps", 20_000);
    let ahead: usize = args.get("ahead", 5);
    let num_inputs: usize = args.get("inputs", 2);
    let seed: u64 = args.get("seed", 12345);
    let every: usize = args.get("every", 2_000);
    let noise: f32 = args.get("noise", 0.0);
    let quiet = args.flag("quiet");
    // Verifying the snapshot round trip costs one comparison per step, so it is
    // on by default; `--no-check-state` turns it off.
    let check_state = !args.flag("no-check-state");

    assert!(ahead >= 1, "--ahead must be at least 1");
    assert!(num_inputs >= 1, "--inputs must be at least 1");

    let mut rng = seed_everything(seed);

    // --- Hierarchy ---
    //
    // Upstream: one layer of 5x5x32, one prediction IO per signal, each a single
    // column of 64 cells. `ticks_per_update: 1` keeps this crate's tick-gating
    // addition out of the way so the behaviour matches upstream, which has no
    // such mechanism (see doc/Divergences.md).

    let io_descs: Vec<IoDesc> = (0..num_inputs)
        .map(|_| IoDesc {
            size: Int3::new(1, 1, COLUMN_SIZE),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 4,
            up_radius: 2,
            down_radius: 2,
            ..Default::default()
        })
        .collect();

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(5, 5, 32),
        num_dendrites_per_cell: 4,
        up_radius: 2,
        recurrent_radius: 0,
        down_radius: 2,
        ticks_per_update: 1,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let mut env = WavyLine::new(num_inputs);
    env.noise = noise;

    println!("Wavy Line — {num_inputs} signals, {steps} steps, {ahead}-step lookahead, seed {seed}");
    println!(
        "  1 layer 5x5x32, IO {num_inputs}x(1,1,{COLUMN_SIZE}) Prediction, range [{LINE_MIN}, {LINE_MAX}], noise {noise}"
    );
    println!();

    // --- Metric accumulators ---
    //
    // A prediction made at step t is about step t+1 (or t+ahead), so each is held
    // in a delay queue and scored when the value it predicted actually arrives.

    let mut mae_1 = Rolling::new(2_000, 0.001);
    let mut mae_n = Rolling::new(2_000, 0.001);
    // Persistence — "the value will not change" — is the baseline to beat, and it
    // must be measured at the *same* horizon as the prediction it judges. These
    // signals are smooth and heavily oversampled, so 1-step persistence is very
    // strong and near the encoder's quantisation floor; at N steps it falls apart,
    // which is where a temporal model has something to prove.
    let mut mae_persist_1 = Rolling::new(2_000, 0.001);
    let mut mae_persist_n = Rolling::new(2_000, 0.001);

    let mut trace_actual = Rolling::new(64, 1.0);
    let mut trace_pred = Rolling::new(64, 1.0);

    let mut pending_1: Option<Vec<f32>> = None;
    let mut pending_n: std::collections::VecDeque<Vec<f32>> = std::collections::VecDeque::new();
    let mut prev_actual: Option<Vec<f32>> = None;
    let mut actual_history: std::collections::VecDeque<Vec<f32>> = std::collections::VecDeque::new();

    let mut state_mismatches: u64 = 0;

    let mut input_cis: Vec<Vec<i32>> = vec![vec![0i32]; num_inputs];

    for t in 0..steps {
        let values = env.advance(&mut rng);

        // Score the predictions that were made for this timestep.
        if let Some(p) = pending_1.take() {
            mae_1.push(mean_abs_error(&p, &values));
        }
        if pending_n.len() == ahead {
            if let Some(p) = pending_n.pop_front() {
                mae_n.push(mean_abs_error(&p, &values));
            }
        }
        if let Some(prev) = &prev_actual {
            mae_persist_1.push(mean_abs_error(prev, &values));
        }
        if actual_history.len() == ahead {
            if let Some(old) = actual_history.pop_front() {
                mae_persist_n.push(mean_abs_error(&old, &values));
            }
        }
        actual_history.push_back(values.clone());

        // --- Encode and step ---

        for (i, &v) in values.iter().enumerate() {
            input_cis[i][0] = bin_range(v, LINE_MIN, LINE_MAX, COLUMN_SIZE);
        }
        let refs: Vec<&[i32]> = input_cis.iter().map(|v| v.as_slice()).collect();
        h.step(&refs, true, 0.0, 0.0);

        // One-step prediction: what the hierarchy expects at t+1.
        let pred_1: Vec<f32> = (0..num_inputs)
            .map(|i| decode(h.get_prediction_cis(i)[0]))
            .collect();

        // --- N-step rollout, sandwiched between a state snapshot and restore ---

        if ahead > 1 {
            let before: Vec<Vec<i32>> = (0..num_inputs)
                .map(|i| h.get_prediction_cis(i).to_vec())
                .collect();

            let mut writer = VecWriter::new();
            h.write_state(&mut writer as &mut dyn StreamWriter);
            let snapshot = writer.data;

            for _ in 0..(ahead - 1) {
                let fed: Vec<Vec<i32>> = (0..num_inputs)
                    .map(|i| h.get_prediction_cis(i).to_vec())
                    .collect();
                let fed_refs: Vec<&[i32]> = fed.iter().map(|v| v.as_slice()).collect();
                h.step(&fed_refs, false, 0.0, 0.0);
            }

            let pred_n: Vec<f32> = (0..num_inputs)
                .map(|i| decode(h.get_prediction_cis(i)[0]))
                .collect();

            let mut reader = SliceReader::new(&snapshot);
            h.read_state(&mut reader as &mut dyn StreamReader);

            if check_state {
                let after: Vec<Vec<i32>> = (0..num_inputs)
                    .map(|i| h.get_prediction_cis(i).to_vec())
                    .collect();
                if before != after {
                    state_mismatches += 1;
                }
            }

            pending_n.push_back(pred_n);
        } else {
            pending_n.push_back(pred_1.clone());
        }

        trace_actual.push(values[0]);
        trace_pred.push(pred_1[0]);

        pending_1 = Some(pred_1);
        prev_actual = Some(values);

        // --- Periodic report ---

        if !quiet && every > 0 && (t + 1) % every == 0 {
            println!(
                "step {:>7} | MAE 1-step {:.4} (persist {:.4})  {}-step {:.4} (persist {:.4}) | jumps {}",
                t + 1,
                mae_1.mean(),
                mae_persist_1.mean(),
                ahead,
                mae_n.mean(),
                mae_persist_n.mean(),
                env.jumps(),
            );
            println!("  signal 0 actual    {}", sparkline(&trace_actual.as_slice()));
            println!("  signal 0 predicted {}", sparkline(&trace_pred.as_slice()));
        }
    }

    // --- Summary ---

    println!();
    println!("Final over the last {} scored steps:", mae_1.len());
    println!(
        "  1-step      MAE {:.4}   vs persistence {:.4}",
        mae_1.mean(),
        mae_persist_1.mean()
    );
    println!(
        "  {ahead}-step      MAE {:.4}   vs persistence {:.4}",
        mae_n.mean(),
        mae_persist_n.mean()
    );

    // The encoder discretises to `COLUMN_SIZE` levels, so even a perfect predictor
    // carries the round-trip error of that binning. For values spread uniformly
    // inside a bin, the expected absolute error is a quarter of the bin width —
    // that is the number a converged 1-step MAE should approach, not go under by much.
    let bin_width = (LINE_MAX - LINE_MIN) / (COLUMN_SIZE - 1) as f32;
    println!(
        "  encoder limit   {:.4}   (mean quantisation error; bin width {:.4} over {COLUMN_SIZE} levels)",
        bin_width * 0.25,
        bin_width
    );

    if ahead > 1 && check_state {
        if state_mismatches == 0 {
            println!(
                "\nState round trip: OK — {steps} write_state/read_state cycles left predictions identical."
            );
        } else {
            println!(
                "\nState round trip: FAILED — {state_mismatches}/{steps} cycles changed the prediction."
            );
        }
    }

    // Judge on the N-step horizon: that is where a temporal model has to actually
    // model the signal rather than lean on smoothness.
    if ahead > 1 {
        if mae_n.mean() < mae_persist_n.mean() {
            println!(
                "\nLearned: the {ahead}-step prediction beats {ahead}-step persistence ({:.4} < {:.4}).",
                mae_n.mean(),
                mae_persist_n.mean()
            );
        } else {
            println!(
                "\nNot converged: the {ahead}-step prediction has not beaten persistence yet — try more --steps."
            );
        }
    } else if mae_1.mean() < mae_persist_1.mean() {
        println!("\nLearned: the 1-step prediction beats 1-step persistence.");
    } else {
        println!("\nNot converged: try more --steps, or --ahead > 1 for a less trivial baseline.");
    }
}

fn decode(ci: i32) -> f32 {
    unbin_range(ci, LINE_MIN, LINE_MAX, COLUMN_SIZE)
}

fn mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / n as f32
}
