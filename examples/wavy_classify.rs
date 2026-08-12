// Wavy Classify — streaming classification of which signal is playing.
//
// Port of `demos/Wavy_Classify.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// One of five signals plays at a time, each a different mixture of the same three
// sines. The hierarchy gets two Prediction ports: the signal value, and the class
// label. During training both are fed in. During inference the label is withheld
// — its port is fed the hierarchy's own previous label prediction — so the class
// has to be recovered from the signal alone.
//
// The classes are genuinely confusable: 3 is `in0 + in1` and 4 is `in0 + in1 +
// in2`, so a single sample carries almost no information and the model has to
// integrate over time. That is what makes this a test of temporal memory rather
// than of a lookup table.
//
// Upstream never measures accuracy at all — it draws the true and predicted class
// as two overlaid curves and leaves it to the eye. This reports a confusion
// matrix, which is the main thing the port adds.
//
//   cargo run --release --example wavy_classify
//   cargo run --release --example wavy_classify -- --train-steps 120000 --hold 3000

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::bin_range;
use support::env::wavy::{
    build_classify_hierarchy, WavyClassify, CLASS_COLUMN_SIZE as SIGNAL_COLUMN_SIZE, CLASS_MAX,
    CLASS_MIN, NUM_CLASSES,
};
use support::metrics::{Recorder, Summary};
use support::sweep;
use support::report::{confusion_table, Rolling};
use support::rng::seed_everything;

fn main() {
    let args = Args::parse();

    let mut rec = Recorder::from_args("wavy_classify", &args);
    // `drive` runs this once normally, or many times under --repeat / --sweep.
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

/// One complete run. Split out from `main` so a repeat or a sweep can call it many
/// times; everything it needs comes from `args` and `seed`.
fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {

    let train_steps: usize = args.get("train-steps", 80_000);
    let test_steps: usize = args.get("test-steps", 20_000);
    // How long each class plays before switching. The model needs a few cycles of
    // the slowest component to tell the classes apart, so very short holds make
    // the task unwinnable regardless of training.
    let hold: usize = args.get("hold", 2_000);
    // Steps after a class switch excluded from the "settled" accuracy figure.
    let settle: usize = args.get("settle", 200);
    let every: usize = args.get("every", 20_000);
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
    // Encoder-side weight of the label port. See the note below on why this
    // defaults to 0.0 rather than upstream's 0.1; `--label-importance 0.1`
    // reproduces upstream's configuration and its collapse.
    let label_importance: f32 = args.get("label-importance", 0.0);
    let num_layers: usize = args.get("layers", 4);

    assert!(hold > settle, "--hold must exceed --settle");

    let mut rng = seed_everything(seed);

    rec.config("train_steps", train_steps);
    rec.config("test_steps", test_steps);
    rec.config("hold", hold);
    rec.config("settle", settle);
    rec.config("layers", num_layers);
    rec.config("label_importance", label_importance);

    // Built in `support/env/wavy.rs`. Two choices there are load-bearing and argued
    // in doc/Demos.md: the exponential tick stack, and label importance defaulting
    // to 0.0 rather than upstream's 0.1.
    let mut h = build_classify_hierarchy(num_layers, label_importance);

    let mut env = WavyClassify::new();

    say!(
        "Wavy Classify — {NUM_CLASSES} classes, {train_steps} train + {test_steps} test steps, seed {seed}"
    );
    say!(
        "  {num_layers} layers 5x5x64 (ticks 1..{}), IO0 (1,1,{SIGNAL_COLUMN_SIZE}) Prediction signal, IO1 (1,1,{NUM_CLASSES}) Prediction label (importance {label_importance})",
        1usize << (num_layers - 1)
    );
    say!("  class held for {hold} steps; first {settle} after a switch excluded from settled accuracy");
    say!();

    // --- Training: both ports observed ---

    let mut class = rng.below(NUM_CLASSES);
    let mut since_switch = 0usize;

    let mut signal_cis = vec![0i32; 1];
    let mut label_cis = vec![0i32; 1];

    // Optimistic by construction, and reported only as a diagnostic: the decoder
    // is updated toward the current target immediately before the activation this
    // reads, so online learning leaks into the measurement and it runs near 100%
    // even when the model has learned nothing generalisable. The held-out figure
    // below is the real one.
    let mut train_hits = Rolling::new(4_000, 0.001);

    for t in 0..train_steps {
        env.advance();

        if since_switch >= hold {
            class = next_class(class, &mut rng);
            since_switch = 0;
        }
        since_switch += 1;

        // Read the standing prediction before stepping: it was made last step and
        // is about *this* one.
        let predicted_now = h.get_prediction_cis(1)[0];
        if since_switch > settle {
            train_hits.push(if predicted_now == class as i32 { 1.0 } else { 0.0 });
        }

        signal_cis[0] = bin_range(env.value(class), CLASS_MIN, CLASS_MAX, SIGNAL_COLUMN_SIZE);
        // The label column index *is* the class — no binning, one cell per class.
        label_cis[0] = class as i32;

        h.step(&[&signal_cis, &label_cis], true, 0.0, 0.0);

        if every > 0 && (t + 1) % every == 0 {
            rec.sample(
                t as u64 + 1,
                &[("online_train_accuracy", train_hits.mean() as f64)],
            );
            if !quiet {
                say!(
                    "  training step {:>7} / {train_steps} | settled train accuracy {:.1}%",
                    t + 1,
                    train_hits.mean() * 100.0
                );
            }
        }
    }

    say!(
        "\nOnline training accuracy over the last {} settled steps: {:.1}% (optimistic — see comment)",
        train_hits.len(),
        train_hits.mean() * 100.0
    );

    // --- Inference: the label is withheld ---
    //
    // The label port is fed the hierarchy's own previous prediction, so nothing
    // about the true class reaches it except through the signal. Learning is off,
    // matching upstream's `P` key.

    say!("\nInference — label withheld, learning off:\n");

    let mut confusion = vec![vec![0u64; NUM_CLASSES]; NUM_CLASSES];
    let mut settled_confusion = vec![vec![0u64; NUM_CLASSES]; NUM_CLASSES];

    for _ in 0..test_steps {
        env.advance();

        if since_switch >= hold {
            class = next_class(class, &mut rng);
            since_switch = 0;
        }
        since_switch += 1;

        signal_cis[0] = bin_range(env.value(class), CLASS_MIN, CLASS_MAX, SIGNAL_COLUMN_SIZE);

        let fed_label = h.get_prediction_cis(1)[0];
        label_cis[0] = fed_label;

        h.step(&[&signal_cis, &label_cis], false, 0.0, 0.0);

        let predicted = h.get_prediction_cis(1)[0].clamp(0, NUM_CLASSES as i32 - 1) as usize;

        confusion[class][predicted] += 1;
        if since_switch > settle {
            settled_confusion[class][predicted] += 1;
        }
    }

    // --- Report ---

    let labels: Vec<String> = (0..NUM_CLASSES).map(|c| format!("c{c}")).collect();

    say!("Confusion over all {test_steps} inference steps (rows true, columns predicted):");
    say!("{}", confusion_table(&confusion, &labels));
    say!("  overall accuracy  {:.1}%", accuracy(&confusion) * 100.0);

    say!(
        "\nExcluding the first {settle} steps after each class switch (settled):"
    );
    say!("{}", confusion_table(&settled_confusion, &labels));
    say!("  settled accuracy  {:.1}%", accuracy(&settled_confusion) * 100.0);

    let chance = 1.0 / NUM_CLASSES as f64;
    say!("  chance            {:.1}%", chance * 100.0);

    let mut summary = Summary::new();
    summary.push("accuracy", accuracy(&confusion));
    summary.push("settled_accuracy", accuracy(&settled_confusion));
    summary.push("chance", chance);
    summary.push("online_train_accuracy", train_hits.mean() as f64);
    // Per-class recall, so a sweep can see *which* classes a configuration confuses
    // rather than only that it scored lower overall.
    for (c, row) in settled_confusion.iter().enumerate() {
        let total: u64 = row.iter().sum();
        let recall = if total == 0 { f64::NAN } else { row[c] as f64 / total as f64 };
        summary.push(&format!("recall_c{c}"), recall);
    }

    if accuracy(&settled_confusion) > chance * 2.0 {
        say!("\nLearned: settled accuracy is well above chance.");
        summary.verdict(true, "settled accuracy is well above chance");
    } else {
        say!(
            "\nNot converged: settled accuracy is near chance — try more --train-steps or a longer --hold."
        );
        summary.verdict(false, "settled accuracy is near chance");
    }

    rec.finish_summary(&summary);
    summary
}

/// Pick a class different from the current one, so every switch is a real change.
fn next_class(current: usize, rng: &mut support::rng::Rng) -> usize {
    let offset = 1 + rng.below(NUM_CLASSES - 1);
    (current + offset) % NUM_CLASSES
}

fn accuracy(confusion: &[Vec<u64>]) -> f64 {
    let total: u64 = confusion.iter().flatten().sum();
    if total == 0 {
        return f64::NAN;
    }
    let correct: u64 = (0..confusion.len()).map(|i| confusion[i][i]).sum();
    correct as f64 / total as f64
}
