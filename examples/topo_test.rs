// Topology Test — does the encoder preserve neighbourhood structure?
//
// Port of `demos/Topo_Test_AON.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// Eight Gaussian clusters are scattered across a 2-D plane and fed to a bare
// `Encoder` as (x, y) column indices. Each hidden column then holds a chain of
// cells, and the question is whether cells that are *adjacent within a column*
// have landed on points that are *adjacent in input space*. If so the column has
// organised itself into something like a curve draped over the data, and the code
// it emits degrades gracefully with distance rather than arbitrarily.
//
// That is the property worth auditing in a port: it is easy to write an encoder
// that quantises correctly and destroys topology, and nothing in the smoke tests
// would notice.
//
// Two deviations. Upstream reads a stored `vl.means` scalar, which this crate's
// byte-weight ART encoder does not have — see `examples/support/encoder_probe.rs`.
// And upstream feeds points only from the cluster you select with the number keys;
// headless, all clusters are sampled uniformly, which is what makes the topology
// figure meaningful across the whole dataset rather than one blob at a time.
//
//   cargo run --release --example topo_test
//   cargo run --release --example topo_test -- --steps 200000 --clusters 4

use dcc_sph::encoder::{Encoder, Params};
use dcc_sph::helpers::{Int3, VisibleLayerDesc};

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::bin_unit;
use support::encoder_probe::{probe_receptive_fields, CellField};
use support::env::cluster::gaussian_clusters;
use support::report::{ascii_scatter, Bounds};
use support::rng::seed_everything;

fn main() {
    let args = Args::parse();

    let steps: usize = args.get("steps", 100_000);
    let seed: u64 = args.get("seed", 12345);
    let every: usize = args.get("every", 25_000);
    let num_clusters: usize = args.get("clusters", 8);
    let points_per_cluster: usize = args.get("points", 100);
    let resolution: i32 = args.get("resolution", 64);
    let plot_w: usize = args.get("plot-width", 78);
    let plot_h: usize = args.get("plot-height", 30);
    let quiet = args.flag("quiet");

    let mut rng = seed_everything(seed);

    let data = gaussian_clusters(num_clusters, points_per_cluster, &mut rng);

    // --- Encoder ---
    //
    // Upstream: hidden 4x4x16 over a 1x2x64 visible layer.

    let hidden = Int3::new(4, 4, 16);
    let mut e = Encoder::default();
    e.init_random(
        hidden,
        vec![VisibleLayerDesc { size: Int3::new(1, 2, resolution), radius: 2 }],
    );
    let params = Params::default();

    println!(
        "Topology Test — hidden {}x{}x{} over a 1x2x{resolution} input, seed {seed}",
        hidden.x, hidden.y, hidden.z
    );
    println!(
        "  {num_clusters} Gaussian clusters x {points_per_cluster} points, {steps} training samples"
    );
    println!();

    let mut inputs = vec![0i32; 2];

    for t in 0..steps {
        let (x, y, _) = data[rng.below(data.len())];
        // Cluster coordinates live in [-1, 1]; the encoder wants [0, 1].
        inputs[0] = bin_unit(x * 0.5 + 0.5, resolution);
        inputs[1] = bin_unit(y * 0.5 + 0.5, resolution);
        e.step(&[&inputs], true, &params);

        if !quiet && every > 0 && (t + 1) % every == 0 {
            let fields = probe_receptive_fields(&e, 0);
            let (topo, chains) = topology_score(&fields, hidden.z as usize);
            let committed = fields.iter().filter(|f| f.is_committed()).count();
            println!(
                "  step {:>8} / {steps} | committed {committed:>4} / {} | neighbour distance {topo:.4} over {chains} chains",
                t + 1,
                fields.len()
            );
        }
    }

    // --- Readout ---

    let fields = probe_receptive_fields(&e, 0);
    let committed: Vec<&CellField> = fields.iter().filter(|f| f.is_committed()).collect();

    // Back into cluster coordinates for plotting.
    let cell_points: Vec<(f32, f32)> = committed
        .iter()
        .map(|f| (f.centroids[0] * 2.0 - 1.0, f.centroids[1] * 2.0 - 1.0))
        .collect();
    let data_points: Vec<(f32, f32)> = data.iter().map(|&(x, y, _)| (x, y)).collect();

    println!("\nLearned cell positions (#) over the cluster data (.):\n");
    println!(
        "{}",
        ascii_scatter(
            &[('.', &data_points), ('#', &cell_points)],
            Bounds::square(1.2),
            plot_w,
            plot_h,
        )
    );

    println!(
        "  {} of {} cells committed.",
        committed.len(),
        fields.len()
    );

    let (topo, chains) = topology_score(&fields, hidden.z as usize);
    let shuffled = shuffled_baseline(&committed);

    println!("\nTopology:");
    println!("  neighbour distance  {topo:.4}  (mean gap between cells adjacent within a column)");
    println!("  scrambled baseline  {shuffled:.4}  (mean gap between randomly paired cells)");
    println!("  measured over {chains} adjacent pairs");

    println!();
    if topo < shuffled * 0.8 {
        println!(
            "Organised: cells adjacent within a column sit closer together than chance, so the"
        );
        println!("columns have laid themselves out along the data rather than scattering.");
    } else {
        println!(
            "Not organised: adjacent cells are no closer than randomly paired ones. This is the"
        );
        println!("expected answer, and more training will not change it.");
        println!();
        println!(
            "`Encoder` has no topology-forming mechanism. Its only neighbourhood parameter,"
        );
        println!(
            "`Params::l_radius`, drives *lateral inhibition* — it decides whether a column is"
        );
        println!(
            "allowed to learn at all by counting how many neighbours scored higher. It never"
        );
        println!(
            "updates a neighbour's weights. Learning touches the winning cell and nothing else, so"
        );
        println!("cell index within a column carries no spatial meaning.");
        println!();
        println!(
            "Contrast `ImageEncoder`, which is a genuine SOM: it carries `Params::falloff` and"
        );
        println!(
            "`Params::n_radius`, and updates cells at distance d from the winner at `rate *"
        );
        println!(
            "falloff^d`. Topology is a property of that encoder, not of this one — which is worth"
        );
        println!("knowing before reaching for `Encoder` expecting a map.");
        println!();
        println!(
            "Upstream's `demos/Topo_Test_AON.cpp` probes an AOgmaNeo revision whose encoder stored"
        );
        println!(
            "`vl.means`, a different formulation; the difference is algorithmic, not a port defect."
        );
    }
}

/// Mean distance between the decoded positions of cells that are adjacent within a
/// column, and how many such pairs contributed.
///
/// Only pairs where *both* cells have committed count — an uncommitted cell has no
/// position, and treating its default as one would drag the figure toward zero and
/// make an untrained encoder look perfectly organised.
fn topology_score(fields: &[CellField], cells_per_column: usize) -> (f32, usize) {
    let mut total = 0.0f32;
    let mut pairs = 0usize;

    for chunk in fields.chunks(cells_per_column) {
        for w in chunk.windows(2) {
            if !w[0].is_committed() || !w[1].is_committed() {
                continue;
            }
            let dx = w[0].centroids[0] - w[1].centroids[0];
            let dy = w[0].centroids[1] - w[1].centroids[1];
            total += (dx * dx + dy * dy).sqrt();
            pairs += 1;
        }
    }

    if pairs == 0 {
        (f32::NAN, 0)
    } else {
        (total / pairs as f32, pairs)
    }
}

/// Mean distance between all unordered pairs of committed cells — what the
/// neighbour distance would look like if position had nothing to do with adjacency.
fn shuffled_baseline(committed: &[&CellField]) -> f32 {
    if committed.len() < 2 {
        return f32::NAN;
    }

    let mut total = 0.0f64;
    let mut pairs = 0usize;

    for i in 0..committed.len() {
        for j in (i + 1)..committed.len() {
            let dx = committed[i].centroids[0] - committed[j].centroids[0];
            let dy = committed[i].centroids[1] - committed[j].centroids[1];
            total += (dx * dx + dy * dy).sqrt() as f64;
            pairs += 1;
        }
    }

    (total / pairs as f64) as f32
}
