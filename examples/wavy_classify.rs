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

use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::bin_range;
use support::env::wavy::{WavyClassify, CLASS_MAX, CLASS_MIN, NUM_CLASSES};
use support::report::{confusion_table, Rolling};
use support::rng::seed_everything;

/// Cells per signal column. Upstream's `inputColumnSize`.
const SIGNAL_COLUMN_SIZE: i32 = 32;

fn main() {
    let args = Args::parse();

    let train_steps: usize = args.get("train-steps", 80_000);
    let test_steps: usize = args.get("test-steps", 20_000);
    // How long each class plays before switching. The model needs a few cycles of
    // the slowest component to tell the classes apart, so very short holds make
    // the task unwinnable regardless of training.
    let hold: usize = args.get("hold", 2_000);
    // Steps after a class switch excluded from the "settled" accuracy figure.
    let settle: usize = args.get("settle", 200);
    let seed: u64 = args.get("seed", 12345);
    let every: usize = args.get("every", 20_000);
    let quiet = args.flag("quiet");
    // Encoder-side weight of the label port. See the note below on why this
    // defaults to 0.0 rather than upstream's 0.1; `--label-importance 0.1`
    // reproduces upstream's configuration and its collapse.
    let label_importance: f32 = args.get("label-importance", 0.0);
    let num_layers: usize = args.get("layers", 4);

    assert!(hold > settle, "--hold must exceed --settle");

    let mut rng = seed_everything(seed);

    // --- Hierarchy ---
    //
    // Two layers of 5x5x64, a signal port and a label port, both Prediction.
    //
    // `ios[1].importance` weights the label port on the *encoder* input side only
    // (hierarchy.rs sets it on the layer-0 encoder's visible layer); the label
    // decoder predicts from the hidden state regardless. That distinction decides
    // whether this demo works.
    //
    // Upstream uses 0.1, which lets the true label into the hidden state during
    // training. The label is then constant for `hold` steps at a time, so
    // "predict the next label" is solved by the identity — copy the label just
    // given — and the decoder never learns to infer class from the signal. At
    // inference, when the label is withheld and the port is fed the model's own
    // prediction, that identity latches onto whatever it happened to emit first
    // and the confusion matrix collapses into a single column.
    //
    // At 0.0 the label cannot reach the hidden state at all, the representation is
    // driven purely by the signal, and the decoder has to do real classification.
    // Upstream never measured accuracy, so its 0.1 was never contradicted by
    // anything. `--label-importance 0.1` reproduces it.

    let io_descs = vec![
        IoDesc {
            size: Int3::new(1, 1, SIGNAL_COLUMN_SIZE),
            io_type: IoType::Prediction,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, 1, NUM_CLASSES as i32),
            io_type: IoType::Prediction,
            ..Default::default()
        },
    ];

    // Layer l updates once every 2^l bottom-layer steps, so the stack spans an
    // exponentially growing window of the signal's history.
    //
    // This is the one place the demos depart from "ticks_per_update: 1 everywhere".
    // Telling these classes apart means measuring frequency, which needs tens of
    // samples of context: class 1 has a period of ~22 steps, class 0 of 80, and
    // classes 3 and 4 differ only by a 40-step component. A flat stack has only
    // self-recurrence to work with and sits near chance no matter how long it
    // trains. Upstream's `//lds[i].ticks_per_update = 2;` is commented out in
    // Wavy_Classify.cpp — the mechanism did not exist in that AOgmaNeo revision.
    let layer_descs: Vec<LayerDesc> = (0..num_layers)
        .map(|l| LayerDesc {
            hidden_size: Int3::new(5, 5, 64),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: 0,
            down_radius: 2,
            ticks_per_update: 1usize << l,
        })
        .collect();

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h.params.ios[1].importance = label_importance;

    let mut env = WavyClassify::new();

    println!(
        "Wavy Classify — {NUM_CLASSES} classes, {train_steps} train + {test_steps} test steps, seed {seed}"
    );
    println!(
        "  {num_layers} layers 5x5x64 (ticks 1..{}), IO0 (1,1,{SIGNAL_COLUMN_SIZE}) Prediction signal, IO1 (1,1,{NUM_CLASSES}) Prediction label (importance {label_importance})",
        1usize << (num_layers - 1)
    );
    println!("  class held for {hold} steps; first {settle} after a switch excluded from settled accuracy");
    println!();

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

        if !quiet && every > 0 && (t + 1) % every == 0 {
            println!(
                "  training step {:>7} / {train_steps} | settled train accuracy {:.1}%",
                t + 1,
                train_hits.mean() * 100.0
            );
        }
    }

    println!(
        "\nOnline training accuracy over the last {} settled steps: {:.1}% (optimistic — see comment)",
        train_hits.len(),
        train_hits.mean() * 100.0
    );

    // --- Inference: the label is withheld ---
    //
    // The label port is fed the hierarchy's own previous prediction, so nothing
    // about the true class reaches it except through the signal. Learning is off,
    // matching upstream's `P` key.

    println!("\nInference — label withheld, learning off:\n");

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

    println!("Confusion over all {test_steps} inference steps (rows true, columns predicted):");
    println!("{}", confusion_table(&confusion, &labels));
    println!("  overall accuracy  {:.1}%", accuracy(&confusion) * 100.0);

    println!(
        "\nExcluding the first {settle} steps after each class switch (settled):"
    );
    println!("{}", confusion_table(&settled_confusion, &labels));
    println!("  settled accuracy  {:.1}%", accuracy(&settled_confusion) * 100.0);

    let chance = 1.0 / NUM_CLASSES as f64;
    println!("  chance            {:.1}%", chance * 100.0);

    if accuracy(&settled_confusion) > chance * 2.0 {
        println!("\nLearned: settled accuracy is well above chance.");
    } else {
        println!(
            "\nNot converged: settled accuracy is near chance — try more --train-steps or a longer --hold."
        );
    }
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
