// ----------------------------------------------------------------------------
//  AOgmaNeo functional-fidelity golden-vector generator.
//
//  Links DIRECTLY against upstream AOgmaNeo (reference commit 645a54a), runs the
//  same deterministic scenario as the Rust side
//  (tests/support/fidelity_scenario.rs), and emits the result as JSON on
//  stdout. That JSON is committed as tests/fixtures/wave_fidelity_golden.json
//  and diffed by `cargo test -p dcc_sph --test fidelity`.
//
//  Determinism / parity contract (must mirror the Rust scenario exactly):
//    * global RNG reset to rand_get_state(12345) before init_random (same PCG32
//      stream as Rust);
//    * built with -DUSE_STD_MATH (exact std transcendentals, not AOgmaNeo's custom
//      fixed-iteration approximations);
//    * built WITHOUT -fopenmp, so the unconditional `#pragma omp parallel for`
//      (PARALLEL_FOR) becomes a no-op → fully sequential → reproducible RNG order;
//    * upstream 645a54a has no `leak` (plain softplus == Rust leak=0) and no
//      `ticks_per_update` (every layer every step == Rust ticks=1).
//
//  Build+run via fidelity/build_and_generate.sh.
// ----------------------------------------------------------------------------

#include "hierarchy.h"

#include <cstdio>
#include <vector>

// Must match tests/support/fidelity_scenario.rs.
static const int STEPS = 200;
static const int NUM_LAYERS = 2;

// Target waveform: 1.0 whenever t is divisible by 20 or 7, else 0.0.
static float wave(int t) {
    return (t % 20 == 0 || t % 7 == 0) ? 1.0f : 0.0f;
}

// Encode a float in [0,1] as two 4-bit nibbles (2 columns x 16 cells).
static void unorm8_to_csdr(float x, int out[2]) {
    int i = static_cast<int>(static_cast<unsigned char>(x * 255.0f + 0.5f));
    out[0] = i & 0x0f;
    out[1] = (i >> 4) & 0x0f;
}

static void print_int_array(const aon::Int_Buffer &a) {
    putchar('[');
    for (int i = 0; i < a.size(); i++) {
        if (i) putchar(',');
        printf("%d", a[i]);
    }
    putchar(']');
}

static void print_int_vec(const int *a, int n) {
    putchar('[');
    for (int i = 0; i < n; i++) {
        if (i) putchar(',');
        printf("%d", a[i]);
    }
    putchar(']');
}

int main() {
    // Deterministic RNG: same default stream as the Rust side.
    aon::global_state = aon::rand_get_state(12345);

    aon::Array<aon::Hierarchy::IO_Desc> io_descs(1);
    io_descs[0].size = aon::Int3(1, 2, 16);
    io_descs[0].type = aon::prediction;
    io_descs[0].num_dendrites_per_cell = 4;
    io_descs[0].up_radius = 2;
    io_descs[0].down_radius = 2;
    io_descs[0].value_size = 64;
    io_descs[0].value_num_dendrites_per_cell = 4;
    io_descs[0].history_capacity = 64;

    aon::Array<aon::Hierarchy::Layer_Desc> layer_descs(NUM_LAYERS);
    for (int l = 0; l < NUM_LAYERS; l++) {
        layer_descs[l].hidden_size = aon::Int3(5, 5, 64);
        layer_descs[l].num_dendrites_per_cell = 4;
        layer_descs[l].up_radius = 2;
        layer_descs[l].recurrent_radius = -1; // recurrence off
        layer_descs[l].down_radius = 2;
    }

    aon::Hierarchy h;
    h.init_random(io_descs, layer_descs);

    printf("{\n  \"steps\": [\n");

    aon::Int_Buffer input_cis(2);
    for (int t = 0; t < STEPS; t++) {
        int csdr[2];
        unorm8_to_csdr(wave(t), csdr);
        input_cis[0] = csdr[0];
        input_cis[1] = csdr[1];

        aon::Array<aon::Int_Buffer_View> input_views(1);
        input_views[0] = aon::Int_Buffer_View(input_cis);

        h.step(input_views, true, 0.0f, 0.0f);

        const aon::Int_Buffer &pred = h.get_prediction_cis(0);
        const aon::Int_Buffer &hidden = h.get_encoder(NUM_LAYERS - 1).get_hidden_cis();

        printf("    {\"input_cis\": ");
        print_int_vec(csdr, 2);
        printf(", \"prediction_cis\": ");
        print_int_array(pred);
        printf(", \"hidden_cis\": ");
        print_int_array(hidden);
        printf("}%s\n", (t + 1 < STEPS) ? "," : "");
    }

    printf("  ],\n  \"final_prediction_acts\": [");
    const aon::Float_Buffer &acts = h.get_prediction_acts(0);
    for (int i = 0; i < acts.size(); i++) {
        if (i) putchar(',');
        printf("%.9g", acts[i]);
    }
    printf("]\n}\n");

    return 0;
}
