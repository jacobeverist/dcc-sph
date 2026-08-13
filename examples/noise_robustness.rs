// Noise Robustness — how much input corruption a learned representation survives.
//
// RUST-ONLY: this demo has no counterpart in OgmaNeoDemos. It exists so that
// `dcc-sph`, `dcc-sparsey` and `dcc-htm` each answer the same question in the same
// shape. See `doc/Demos.md` for the cross-repo demo contract.
//
// A book of random patterns is memorised, one per class: each is held for a few
// steps while both the pattern and its label are observed. At test time the label
// is withheld — the label port is fed the hierarchy's own previous prediction — and
// a fixed fraction of observation columns is corrupted. The question is not whether
// the model can classify, but how the accuracy *decays* as corruption rises.
//
// The control is exact-match lookup, which is perfect at zero noise and blind one
// column off it. It is not a competitor to beat; it is the null hypothesis for the
// shape of the curve. A system that has memorised produces the lookup curve, and a
// system that has generalised does not.
//
// The interesting invocation is the sweep, because one point on a decay curve is
// not a decay curve:
//
//   cargo run --release --example noise_robustness
//   cargo run --release --example noise_robustness -- --sweep noise=0,0.1,0.2,0.3,0.4,0.5 --repeat 5

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::checkpoint;
use support::env::noise::{
    build_hierarchy, Corruptor, LookupTable, PatternBook, COLUMNS, COLUMN_SIZE, GRID_H, GRID_W,
};
use support::metrics::{Recorder, Summary};
use support::report::{confusion_table, Rolling};
use support::rng::{seed_everything, Rng};
use support::sweep;

use dcc_sph::hierarchy::Hierarchy;

fn main() {
    let args = Args::parse();

    let mut rec = Recorder::from_args("noise_robustness", &args);
    // `drive` runs this once normally, or many times under --repeat / --sweep.
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

/// One complete run. Split out from `main` so a repeat or a sweep can call it many
/// times; everything it needs comes from `args` and `seed`.
fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let classes: usize = args.get("classes", 8);
    let num_layers: usize = args.get("layers", 2);
    // How long one pattern is presented. A next-step predictor cannot classify a
    // pattern it is seeing for the first time this tick, so a presentation has to
    // last long enough for the label decoder to have something to predict from.
    let hold: usize = args.get("hold", 16);
    // Steps after a switch excluded from scoring, for the same reason.
    let settle: usize = args.get("settle", 4);
    let train_steps: usize = args.get("train-steps", 60_000);
    let test_steps: usize = args.get("test-steps", 20_000);
    // The corruption level under test. This is the axis worth sweeping.
    let noise: f32 = args.get("noise", 0.25);
    let every: usize = args.get("every", 20_000);
    // `--silent` is set by the sweep driver in matrix mode: it suppresses the
    // final report too, not just the periodic lines that `--quiet` covers.
    let silent = args.flag("silent");
    let quiet = silent || args.flag("quiet");

    macro_rules! say {
        ($($arg:tt)*) => {
            if !silent {
                println!($($arg)*);
            }
        };
    }

    assert!(hold > settle, "--hold must exceed --settle");
    assert!(classes >= 2, "--classes must be at least 2");
    assert!((0.0..=1.0).contains(&noise), "--noise must be in [0, 1]");

    let mut rng = seed_everything(seed);

    rec.config("classes", classes);
    rec.config("layers", num_layers);
    rec.config("hold", hold);
    rec.config("settle", settle);
    rec.config("train_steps", train_steps);
    rec.config("test_steps", test_steps);
    rec.config("noise", noise);

    let book = PatternBook::generate(classes, &mut rng);
    let separation = book.min_separation();

    // The control memorises the clean book exactly, which is all it can do.
    let mut lookup = LookupTable::new();
    for c in 0..book.len() {
        lookup.learn(book.get(c), c);
    }

    let mut h = build_hierarchy(classes, num_layers);
    // Resume from a checkpoint if one was given, before any training.
    checkpoint::maybe_load(&mut h, args);

    say!("Noise Robustness — {classes} patterns, {train_steps} train + {test_steps} test steps, seed {seed}");
    say!(
        "  observation ({GRID_W},{GRID_H},{COLUMN_SIZE}) Prediction = {COLUMNS} columns, label (1,1,{classes}) Prediction, {num_layers} layers 5x5x64 (ticks 1..{})",
        1usize << (num_layers - 1)
    );
    say!("  pattern held for {hold} steps; first {settle} after a switch excluded from scoring");
    say!("  closest two patterns differ in {separation} of {COLUMNS} columns");
    say!("  testing at noise {:.2} = {} corrupted columns", noise, (noise * COLUMNS as f32 + 0.5) as usize);
    say!();

    // --- Training: clean patterns, label observed ---

    let mut class = rng.below(classes);
    let mut since_switch = 0usize;
    let mut label_cis = vec![0i32; 1];

    // Optimistic by construction and reported only as a diagnostic: the decoder is
    // updated toward the current target immediately before the activation this
    // reads, so online learning leaks into it. The held-out passes below are real.
    let mut train_hits = Rolling::new(4_000, 0.001);

    for t in 0..train_steps {
        if since_switch >= hold {
            class = next_class(class, classes, &mut rng);
            since_switch = 0;
        }
        since_switch += 1;

        // Read the standing prediction before stepping: it was made last step and
        // is about *this* one.
        let predicted_now = h.get_prediction_cis(1)[0];
        if since_switch > settle {
            train_hits.push(if predicted_now == class as i32 { 1.0 } else { 0.0 });
        }

        label_cis[0] = class as i32;
        h.step(&[book.get(class), &label_cis], true, 0.0, 0.0);

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

    // --- Evaluation: label withheld, learning off ---
    //
    // Two passes, clean and corrupted. Both are needed from a single run: without
    // the clean pass there is no way to tell "corruption hurt" from "it never
    // learned", which is exactly the confusion a sweep over --noise would otherwise
    // produce at its high end.

    say!("\nEvaluation — label withheld, learning off:\n");

    let clean = evaluate(&mut h, &book, 0.0, test_steps, hold, settle, &mut rng, &lookup);
    let noisy = evaluate(&mut h, &book, noise, test_steps, hold, settle, &mut rng, &lookup);

    // --- Report ---

    let labels: Vec<String> = (0..classes).map(|c| format!("p{c}")).collect();
    let chance = 1.0 / classes as f64;

    say!("Clean (noise 0.00), rows true and columns predicted:");
    say!("{}", confusion_table(&clean.confusion, &labels));
    say!("  accuracy          {:.1}%", clean.accuracy() * 100.0);
    say!("  exact-match       {:.1}%", clean.lookup_accuracy() * 100.0);

    say!("\nCorrupted (noise {noise:.2}):");
    say!("{}", confusion_table(&noisy.confusion, &labels));
    say!("  accuracy          {:.1}%", noisy.accuracy() * 100.0);
    say!("  exact-match       {:.1}%", noisy.lookup_accuracy() * 100.0);
    say!("  chance            {:.1}%", chance * 100.0);

    let retention = if clean.accuracy() > 0.0 {
        noisy.accuracy() / clean.accuracy()
    } else {
        f64::NAN
    };
    say!("\n  retention         {:.1}% of clean accuracy survives noise {noise:.2}", retention * 100.0);

    let mut summary = Summary::new();
    summary.push("noise", noise as f64);
    summary.push("accuracy", noisy.accuracy());
    summary.push("clean_accuracy", clean.accuracy());
    summary.push("retention", retention);
    summary.push("baseline_lookup_accuracy", noisy.lookup_accuracy());
    summary.push("baseline_chance", chance);
    summary.push(
        "accuracy_vs_chance",
        if chance > 0.0 { noisy.accuracy() / chance } else { f64::NAN },
    );
    // There is deliberately no `accuracy_vs_lookup` ratio. Exact match is zero at
    // every noise level above zero *by construction*, so the ratio is either 0/0 or
    // x/0 and carries no information the two raw numbers do not. `baseline_chance`
    // is the twin that `accuracy_vs_chance` is a ratio of.
    summary.push("min_separation", separation as f64);
    summary.push("online_train_accuracy", train_hits.mean() as f64);

    // The verdict is about the clean pass on purpose. Failing to classify at high
    // corruption is the correct answer, not a failure of the run, so a verdict
    // keyed on the noisy pass would report "not converged" for a demo working
    // exactly as intended — and would do it for most rows of the sweep this demo
    // exists to produce.
    if clean.accuracy() > chance * 2.0 {
        let note = format!(
            "patterns learned (clean {:.0}% vs chance {:.0}%); {:.0}% of that survives noise {noise:.2}, against {:.0}% for exact match",
            clean.accuracy() * 100.0,
            chance * 100.0,
            retention * 100.0,
            noisy.lookup_accuracy() * 100.0
        );
        say!("\nLearned: {note}.");
        summary.verdict(true, note);
    } else {
        say!("\nNot converged: clean accuracy is near chance — the patterns were never learned, so the noise result says nothing. Try more --train-steps or a longer --hold.");
        summary.verdict(false, "clean accuracy is near chance, so the noise result is uninformative");
    }

    checkpoint::maybe_save(&h, args);

    rec.finish_summary(&summary);
    summary
}

/// One held-out pass. Learning is off and the label port is fed the hierarchy's own
/// previous prediction, so nothing about the true class reaches the model except
/// through the (possibly corrupted) observation.
struct Eval {
    confusion: Vec<Vec<u64>>,
    lookup_hits: u64,
    scored: u64,
}

impl Eval {
    fn accuracy(&self) -> f64 {
        if self.scored == 0 {
            return f64::NAN;
        }
        let correct: u64 = (0..self.confusion.len()).map(|i| self.confusion[i][i]).sum();
        correct as f64 / self.scored as f64
    }

    fn lookup_accuracy(&self) -> f64 {
        if self.scored == 0 {
            return f64::NAN;
        }
        self.lookup_hits as f64 / self.scored as f64
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    h: &mut Hierarchy,
    book: &PatternBook,
    fraction: f32,
    steps: usize,
    hold: usize,
    settle: usize,
    rng: &mut Rng,
    lookup: &LookupTable,
) -> Eval {
    let classes = book.len();
    let mut eval = Eval {
        confusion: vec![vec![0u64; classes]; classes],
        lookup_hits: 0,
        scored: 0,
    };

    let mut corruptor = Corruptor::new();
    let mut observation = vec![0i32; COLUMNS];
    let mut label_cis = vec![0i32; 1];

    let mut class = rng.below(classes);
    let mut since_switch = 0usize;

    for _ in 0..steps {
        if since_switch >= hold {
            class = next_class(class, classes, rng);
            since_switch = 0;
        }
        since_switch += 1;

        corruptor.apply(book.get(class), fraction, rng, &mut observation);

        // The label port sees only what the model itself predicted.
        label_cis[0] = h.get_prediction_cis(1)[0];
        h.step(&[&observation, &label_cis], false, 0.0, 0.0);

        if since_switch > settle {
            let predicted = h.get_prediction_cis(1)[0].clamp(0, classes as i32 - 1) as usize;
            eval.confusion[class][predicted] += 1;
            // Scored on exactly the same corrupted input, in the same pass, so the
            // two numbers are comparable rather than merely adjacent.
            if lookup.classify(&observation) == Some(class) {
                eval.lookup_hits += 1;
            }
            eval.scored += 1;
        }
    }

    eval
}

/// Pick a class different from the current one, so every switch is a real change.
fn next_class(current: usize, classes: usize, rng: &mut Rng) -> usize {
    let offset = 1 + rng.below(classes - 1);
    (current + offset) % classes
}
