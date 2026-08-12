//! Performance baseline for the SPH `Hierarchy::step` hot path.
//!
//! Establishes the reference numbers for the efficiency work (parallelizing the learn
//! pass, hotspot fixes). Every optimization is measured against these. Run
//! single-threaded via `RAYON_NUM_THREADS=1 cargo bench -p dcc_sph` for a stable
//! serial baseline, or without it to see the rayon-parallel numbers.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dcc_sph::helpers::{rand_get_state, set_global_state, Int3};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

/// Build a hierarchy: one prediction IO (`io_cols × io_cells`) feeding `num_layers`
/// encoder layers of `hidden` size. Deterministic (RNG reset to the default stream).
fn build(io_cols: i32, io_cells: i32, hidden: i32, hidden_cells: i32, num_layers: usize) -> Hierarchy {
    set_global_state(rand_get_state(12345));
    let io_descs = vec![IoDesc {
        size: Int3::new(io_cols, io_cols, io_cells),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 4,
        up_radius: 2,
        down_radius: 2,
        value_size: 64,
        value_num_dendrites_per_cell: 4,
        history_capacity: 64,
    }];
    let layer_descs: Vec<LayerDesc> = (0..num_layers)
        .map(|_| LayerDesc {
            hidden_size: Int3::new(hidden, hidden, hidden_cells),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 1,
        })
        .collect();
    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h
}

/// A deterministic stream of input CSDRs (`n_cols` column indices in `0..cells`).
fn input_stream(steps: usize, n_cols: usize, cells: i32) -> Vec<Vec<i32>> {
    let mut s: u64 = 0x243f6a8885a308d3;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as i32
    };
    (0..steps)
        .map(|_| (0..n_cols).map(|_| next().rem_euclid(cells)).collect())
        .collect()
}

fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("step");
    // (label, io_cols, io_cells, hidden, hidden_cells, layers)
    let configs = [
        ("small_2L", 4i32, 16i32, 5i32, 32i32, 2usize),
        ("mid_3L", 6, 16, 8, 64, 3),
    ];
    for (label, io_cols, io_cells, hidden, hidden_cells, layers) in configs {
        let n_cols = (io_cols * io_cols) as usize;
        let stream = input_stream(64, n_cols, io_cells);

        group.bench_with_input(BenchmarkId::new("learn", label), &(), |b, _| {
            let mut h = build(io_cols, io_cells, hidden, hidden_cells, layers);
            let mut i = 0usize;
            b.iter(|| {
                let inp = &stream[i % stream.len()];
                h.step(black_box(&[inp]), true, 0.0, 0.0);
                i += 1;
                black_box(h.get_prediction_cis(0));
            });
        });

        group.bench_with_input(BenchmarkId::new("infer", label), &(), |b, _| {
            let mut h = build(io_cols, io_cells, hidden, hidden_cells, layers);
            // Warm the weights so inference is representative.
            for s in &stream {
                h.step(&[s], true, 0.0, 0.0);
            }
            let mut i = 0usize;
            b.iter(|| {
                let inp = &stream[i % stream.len()];
                h.step(black_box(&[inp]), false, 0.0, 0.0);
                i += 1;
                black_box(h.get_prediction_cis(0));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_step);
criterion_main!(benches);
