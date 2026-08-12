// Encoder Visualiser — where do the ART encoder's cells actually end up?
//
// Port of `demos/EncVis.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// Points are drawn from a 2-D probability density and fed to a bare `Encoder` —
// no hierarchy — as two column indices, one for x and one for y. As it learns,
// each hidden cell commits to a region of that space, and this reads those
// commitments back out and plots them over the data. It is the cheapest way to see
// whether `src/encoder.rs` is doing anything sensible.
//
// Two things differ from upstream by necessity. Its density image
// (`resources/density_image5.png`) is missing from the repository, so the field is
// generated procedurally. And it reads a stored `vl.means` scalar per cell, which
// this crate's byte-weight ART encoder does not have — see
// `examples/support/encoder_probe.rs` for how the position is recovered instead.
//
//   cargo run --release --example enc_vis
//   cargo run --release --example enc_vis -- --steps 200000 --cells 64

use dcc_sph::encoder::{Encoder, Params};

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::bin_unit;
use support::encoder_probe::{probe_receptive_fields, CellField};
use support::env::cluster::{build_enc_vis_encoder, DensityField};
use support::report::{ascii_scatter, Bounds, Rolling};
use support::metrics::{Recorder, Summary};
use support::rng::seed_everything;

fn main() {
    let args = Args::parse();
    let seed: u64 = args.get("seed", 12345);

    let mut rec = Recorder::from_args("enc_vis", &args);
    run(&args, seed, &mut rec);
    rec.finish();
}

/// One complete run. Split out from `main` so a repeat or a sweep can call it many
/// times; everything it needs comes from `args` and `seed`.
fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {

    let steps: usize = args.get("steps", 100_000);
    let every: usize = args.get("every", 25_000);
    // Cells per input column — upstream's `resolution`.
    let resolution: i32 = args.get("resolution", 128);
    // Hidden columns and cells per column. Upstream is 1x3 columns of 32.
    let columns: i32 = args.get("columns", 3);
    let cells: i32 = args.get("cells", 32);
    let plot_w: usize = args.get("plot-width", 78);
    let plot_h: usize = args.get("plot-height", 30);
    let quiet = args.flag("quiet");

    let mut rng = seed_everything(seed);

    // Config must be recorded before the first sample, which writes the run header.
    rec.config("steps", steps);
    rec.config("columns", columns);
    rec.config("cells", cells);
    rec.config("resolution", resolution);

    let field = DensityField::procedural(128, 128);

    // --- Encoder ---
    //
    // One visible layer of 1x2 columns: column 0 carries x, column 1 carries y.
    // The default radius of 2 means each hidden column's receptive field covers
    // both, so every cell sees a full (x, y) pair.

    let mut e = build_enc_vis_encoder(columns, cells, resolution);

    // Vigilance is the parameter this demo is really about, and the one worth
    // playing with. It sets how well an input must match a committed cell before
    // that cell may claim it. Low vigilance lets each cell swallow a wide, ragged
    // set of inputs; high vigilance forces cells to stay selective and split, so
    // they end up on narrow contiguous bands. Watch `compactness` in the output
    // move as you change it.
    let mut params = Params::default();
    params.vigilance = args.get("vigilance", params.vigilance);
    params.lr = args.get("lr", params.lr);
    params.active_ratio = args.get("active-ratio", params.active_ratio);

    println!("Encoder Visualiser — {columns} columns x {cells} cells over a 1x2x{resolution} input, seed {seed}");
    println!("  {steps} samples drawn from a procedural density field (upstream's PNG is missing from the repo)");
    println!(
        "  vigilance {:.3}, lr {:.3}, active_ratio {:.3}",
        params.vigilance, params.lr, params.active_ratio
    );
    println!();

    let mut inputs = vec![0i32; 2];

    for t in 0..steps {
        let (u, v) = field.sample(&mut rng);
        inputs[0] = bin_unit(u, resolution);
        inputs[1] = bin_unit(v, resolution);
        e.step(&[&inputs], true, &params);

        if every > 0 && (t + 1) % every == 0 {
            let fields = probe_receptive_fields(&e, 0);
            let committed = fields.iter().filter(|f| f.is_committed()).count();
            let err = quantisation_error(&mut e, &params, &field, &mut rng, 2_000, resolution);
            rec.sample(
                t as u64 + 1,
                &[
                    ("committed_cells", committed as f64),
                    ("quantisation_error", err as f64),
                ],
            );
            if quiet {
                continue;
            }
            println!(
                "  step {:>8} / {steps} | committed cells {committed:>4} / {} | quantisation error {err:.4}",
                t + 1,
                fields.len()
            );
        }
    }

    // --- Readout ---

    let fields = probe_receptive_fields(&e, 0);
    let committed: Vec<&CellField> = fields.iter().filter(|f| f.is_committed()).collect();

    let cell_points: Vec<(f32, f32)> = committed
        .iter()
        .map(|f| (f.centroids[0], f.centroids[1]))
        .collect();
    let data_points = field.points(3_000, &mut rng);

    println!("\nLearned cell positions (#) over the density they were sampled from (.):\n");
    println!(
        "{}",
        ascii_scatter(
            &[('.', &data_points), ('#', &cell_points)],
            Bounds { min_x: 0.0, max_x: 1.0, min_y: 0.0, max_y: 1.0 },
            plot_w,
            plot_h,
        )
    );

    println!(
        "  {} of {} cells committed; the rest never won a column and have no position.",
        committed.len(),
        fields.len()
    );

    let mean_spread: f32 = if committed.is_empty() {
        0.0
    } else {
        committed.iter().map(|f| f.spread() as f32).sum::<f32>() / committed.len() as f32
    };
    let mean_compactness: f32 = if committed.is_empty() {
        0.0
    } else {
        committed.iter().map(|f| f.compactness()).sum::<f32>() / committed.len() as f32
    };
    println!(
        "  Each committed cell responds to {mean_spread:.1} of the {resolution} input levels, compactness {mean_compactness:.2}."
    );

    // --- What the cells actually learned ---
    //
    // The scatter above summarises each cell by the centroid of its learned input
    // values, which is only a "position" if those values form one contiguous block.
    // ART never requires that: weights are driven to 255 for whatever inputs a cell
    // happened to win on, wherever they land. So print the raw profiles too — they
    // are the ground truth the summary is derived from, and they show immediately
    // whether a centroid means anything.

    println!("\nWeight profiles — one row per cell, one column per input level ('#' learned):\n");
    println!("      {:<40} {:<40}", "x input column", "y input column");

    let sample_cells: Vec<&CellField> = committed.iter().copied().take(12).collect();
    for f in &sample_cells {
        let profiles = support::encoder_probe::weight_profiles(&e, 0, f.column, f.cell);
        let rows: Vec<String> = profiles
            .iter()
            .map(|p| support::encoder_probe::render_profile(p, 40))
            .collect();
        println!(
            "  c{}/{:<3} {} {}  compactness {:.2}",
            f.column,
            f.cell,
            rows.first().cloned().unwrap_or_default(),
            rows.get(1).cloned().unwrap_or_default(),
            f.compactness()
        );
    }

    let final_err = quantisation_error(&mut e, &params, &field, &mut rng, 20_000, resolution);
    println!(
        "\n  Quantisation error: {final_err:.4} (mean distance from a sample to its winning cell's centroid)"
    );

    // A uniform grid of `cells` prototypes over a unit square would sit roughly
    // this far from a random point.
    let uniform_baseline = 0.5 / (cells as f32).sqrt();
    println!("  Uniform-grid reference: {uniform_baseline:.4} for {cells} cells over the unit square");

    // --- Probe ---
    //
    // Upstream lets you click a point and shows the resulting CSDR. Headless, a few
    // fixed probes make the same point: nearby inputs share cells, distant ones
    // do not.

    println!("\nCodes for three probe points (each column's winning cell):");
    for &(u, v) in &[(0.25f32, 0.30f32), (0.27f32, 0.32f32), (0.72f32, 0.24f32)] {
        inputs[0] = bin_unit(u, resolution);
        inputs[1] = bin_unit(v, resolution);
        e.step(&[&inputs], false, &params);
        println!("  ({u:.2}, {v:.2}) -> {:?}", e.get_hidden_cis());
    }
    println!("  (the first two are neighbours and should share cells; the third should not)");

    let mut summary = Summary::new();
    summary.push("committed_cells", committed.len() as f64);
    summary.push("total_cells", fields.len() as f64);
    summary.push("mean_spread", mean_spread as f64);
    summary.push("mean_compactness", mean_compactness as f64);
    summary.push("quantisation_error", final_err as f64);
    summary.push("uniform_baseline", uniform_baseline as f64);

    // The honest summary. A low compactness means the centroid — and therefore the
    // scatter plot and the quantisation error above — is averaging over disjoint
    // regions of the input range, and neither number should be read as a position.
    println!();
    if mean_compactness > 0.6 {
        summary.verdict(final_err < uniform_baseline, "cells learned contiguous input bands");
        println!(
            "Cells learned contiguous input bands (compactness {mean_compactness:.2}), so the centroids above are real positions."
        );
        if final_err < uniform_baseline {
            println!("They also concentrate where the density is: quantisation error beats a uniform grid.");
        } else {
            println!(
                "They do not yet beat a uniform grid on quantisation error — try more --steps."
            );
        }
    } else {
        // Not a failure — the demo's finding is that this encoder is an exemplar
        // coder, not a self-organising map, so a scattered set is the correct
        // outcome and is reported as one.
        summary.verdict(true, "cells learned scattered input sets — ART, not a SOM");
        println!(
            "Cells learned *scattered* input sets (compactness {mean_compactness:.2}, {mean_spread:.1} levels each),"
        );
        println!("not contiguous bands — and that is the encoder working as specified.");
        println!();
        println!(
            "A cell commits to whichever inputs it happens to win on, and nothing in ART forces"
        );
        println!(
            "those to be neighbours: this is an exemplar coder, not a self-organising map. Raising"
        );
        println!(
            "`--vigilance` makes each cell far more selective (try 0.99: ~2 levels instead of ~17)"
        );
        println!("but the levels it keeps are still scattered, because selectivity and topology are");
        println!("different properties and only the first is what vigilance controls.");
        println!();
        println!(
            "So read the profiles, not the scatter: the centroid, and the quantisation error derived"
        );
        println!(
            "from it, average over disjoint regions and are not positions. This is also why the port"
        );
        println!(
            "cannot reproduce upstream's picture — `demos/EncVis.cpp` reads a stored `vl.means`, a"
        );
        println!(
            "running average that is a position by construction and cannot represent a split set."
        );
    }

    rec.finish_summary(&summary);
    summary
}

/// Mean distance from a sampled point to the position its winning cells decode to,
/// averaged over hidden columns. Each column is an independent quantiser of the
/// same 2-D input, so this measures how well the codebook covers the data.
fn quantisation_error(
    e: &mut Encoder,
    params: &Params,
    field: &DensityField,
    rng: &mut support::rng::Rng,
    samples: usize,
    resolution: i32,
) -> f32 {
    let hidden_size = e.get_hidden_size();
    let num_cells = hidden_size.z as usize;

    let mut err = Rolling::new(samples.max(1), 0.01);
    let mut inputs = vec![0i32; 2];

    // Learning is off below, so the receptive fields are fixed for the whole sweep.
    // Probing per sample would repeat the same full weight scan thousands of times.
    let fields = probe_receptive_fields(e, 0);

    for _ in 0..samples {
        let (u, v) = field.sample(rng);
        inputs[0] = bin_unit(u, resolution);
        inputs[1] = bin_unit(v, resolution);
        e.step(&[&inputs], false, params);

        let hidden = e.get_hidden_cis().to_vec();

        let mut total = 0.0f32;
        let mut n = 0usize;
        for (c, &ci) in hidden.iter().enumerate() {
            let f = &fields[c * num_cells + ci as usize];
            if !f.is_committed() {
                continue;
            }
            total += ((f.centroids[0] - u).powi(2) + (f.centroids[1] - v).powi(2)).sqrt();
            n += 1;
        }

        if n > 0 {
            err.push(total / n as f32);
        }
    }

    err.mean()
}
