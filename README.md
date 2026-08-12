# dcc_sph — Sparse Predictive Hierarchies in Rust

A Rust port of the [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo) library by [Ogma Intelligent Systems Corp](https://ogma.ai), pinned to upstream commit [`645a54a`](https://github.com/ogmacorp/AOgmaNeo/tree/645a54ace656b0ac2476a56a0dac19faacbd87ab).

**Sparse Predictive Hierarchies (SPH)** is an online machine learning system with a low compute footprint that learns from streaming data without forgetting. Data moves as **CSDRs** — flat `Vec<i32>` arrays where each integer selects the active cell in a column of a 2-D grid. A stack of sparse coding layers encodes inputs upward; decoders reconstruct predictions back downward. Higher layers clock less often, which buys exponential temporal memory for near-constant added compute per layer.

This is a standalone, self-contained crate. It was extracted from the `dcc-core` workspace and now builds and versions independently, with **no dependency on dcc-core**.

> **This crate is NonCommercial.** CC BY-NC-SA 4.0, matching upstream, and that restriction travels into anything linking it. See [License](#license-and-attribution).

---

## Status

**Ported and in use:** the full forward stack — `Encoder` (ART sparse coder), `Decoder` (multi-dendrite perceptrons), `Actor` (actor-critic RL with eligibility traces), `ImageEncoder` (SOM with reconstruction), and `Hierarchy` as the top-level orchestrator, plus save/load of weights and state.

**Divergences are catalogued, not hidden.** The prioritized audit — including deferred and added items — is in [`doc/Divergences.md`](doc/Divergences.md).

---

## Features

- **Online learning** — learns from a data stream one sample at a time, in order, without forgetting
- **Sparse representations** — CSDRs keep compute proportional to active content, not total network size
- **Short-term memory** — exponential memory via a clockwork hierarchy of layers
- **Reinforcement learning** — built-in actor-critic with eligibility traces
- **Image encoding** — self-organizing map pre-encoder with reconstruction
- **Serialization** — save/load weights, state, or both
- **Parallel forward pass** — encoder and decoder columns computed in parallel via `rayon`

---

## Quick start

Requires a Rust toolchain (stable).

```bash
git clone https://github.com/jacobeverist/dcc-sph
cd dcc-sph
cargo build --release  # build the library
cargo test             # unit + integration tests
cargo run --release --example wave_prediction
```

To use it from another crate:

```toml
[dependencies]
dcc_sph = { git = "https://github.com/jacobeverist/dcc-sph" }
```

---

## Concepts in 30 seconds

| Term | Meaning |
| --- | --- |
| **CSDR** | Columnar Sparse Distributed Representation — a flat `Vec<i32>`, one entry per column, each naming the active cell in that column. The currency between every component. |
| **Column** | A cell stack at one grid position. `Int3(x, y, z)` sizes a layer: an `x × y` grid of columns, `z` cells deep. |
| **Encoder** | Sparse coder mapping visible CSDRs up to a hidden CSDR (ART-style). |
| **Decoder** | Multi-dendrite perceptrons reconstructing a prediction of the next input from hidden state. |
| **Layer / tick** | Each hierarchy level clocks on its own period. Higher = slower = longer effective memory. |
| **Actor** | Actor-critic head with eligibility traces, for reinforcement learning against a reward signal. |
| **`leak`** | Decoder leak rate. The crate default is `0.01`; set it to `0.0` for the upstream-faithful regime (see Fidelity). |

Full glossary for reading the source: [`doc/NameReference.md`](doc/NameReference.md).

---

## Minimal example

```rust
use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

// Configure the hierarchy
let io_descs = vec![IoDesc {
    size: Int3::new(4, 4, 16), // 4×4 grid, 16 cells per column
    io_type: IoType::Prediction,
    ..IoDesc::default()
}];
let layer_descs = vec![LayerDesc {
    hidden_size: Int3::new(4, 4, 16),
    ..LayerDesc::default()
}];

let mut h = Hierarchy::new();
h.init_random(&io_descs, &layer_descs);

// Step with your input CSDR (Vec<i32>, values in [0, column_size))
let input_cis: Vec<i32> = vec![0i32; 4 * 4];
h.step(&[&input_cis], true, 0.0, 0.0);

// Read the next-step prediction
let prediction: &[i32] = h.get_prediction_cis(0);
```

---

## Demos

Runnable examples in [`examples/`](examples/). Everything below runs **headless and text-only with no features enabled**, so it works over SSH and in CI.

Start here:

| Demo | Command | Shows |
| --- | --- | --- |
| **Wave prediction** | `cargo run --release --example wave_prediction` | Wavy-line sequence prediction with ASCII recall output. The "hello world". |
| **CartPole** | `cargo run --release --example cartpole` | Balancing via the built-in actor-critic, with built-in physics — no Python. |

### Ported from OgmaNeoDemos

Thirteen demos ported from [OgmaNeoDemos](https://github.com/jacobeverist/OgmaNeoDemos/tree/aogmaneo), Ogma's own demo repository. Each doubles as an end-to-end check that a feature of the library works on a real problem. [`doc/Demos.md`](doc/Demos.md) records what each one does and where it departs from its source.

| Demo | Command | Shows |
| --- | --- | --- |
| **Wavy Line** | `cargo run --release --example wavy_line` | Multi-channel prediction with N-step lookahead; verifies the `write_state`/`read_state` round trip every step. |
| **Wavy Classify** | `cargo run --release --example wavy_classify` | Streaming five-way classification with the label withheld at inference. Reports a confusion matrix. |
| **Ball Physics** | `cargo run --release --example ball_physics` | `ImageEncoder` + `reconstruct()`: after five seed frames it generates the bouncing ball from its own predictions. |
| **Video Prediction** | `cargo run --release --example video_prediction` | RGB frames through an `ImageEncoder`, then a generated clip. Procedural source; `--frames` for real ones. |
| **VSA Char** | `cargo run --release --example vsa_char` | A whole word compressed into one hypervector — which *is* a CSDR, so there is no encoder. |
| **Pusher** | `cargo run --release --example pusher` | `Actor` on a multi-column action port with a dense shaped reward. |
| **Cat and Mouse** | `cargo run --release --example cat_mouse` | **Two hierarchies** competing in one maze, zero-sum reward. |
| **Cat and Mouse + memory** | `cargo run --release --example cat_mouse_pos` | A port's own prediction fed back as its next input: dead reckoning learned end to end. |
| **Explore** | `cargo run --release --example explore` | Curiosity as the entire reward — the agent is paid for its own prediction error. |
| **Car Racing** | `cargo run --release --example car_racing` | Steering a real track from twelve raycast sensors. Completes laps. |
| **Runner** | `cargo run --release --example runner` | An eight-motor articulated body learning a gait. The hardest of the set. |
| **Encoder Visualiser** | `cargo run --release --example enc_vis` | What the ART encoder's cells actually learn, as weight profiles and a scatter. |
| **Topology Test** | `cargo run --release --example topo_test` | Whether `Encoder` preserves neighbourhood structure. (It does not — see `doc/Demos.md`.) |

Each takes `--steps`, `--seed`, `--every` and `--quiet`; `--seed` fully determines a run. The RL demos measure themselves against a random-action baseline on the same world and seed, so the numbers mean something.

A windowed viewer lives in the separate `examples-viz` crate — `cargo run --release -p dcc_sph_viz_examples --example viewer` — for the demos where motion is the point. It is a way to eyeball a demo, not instrumentation, and it changes nothing about the simulation or the reported numbers. It is a separate crate so that no graphics dependency appears in the library's manifest; see [`doc/Conformance.md`](doc/Conformance.md).

### Running the Gymnasium demos

These drive [Gymnasium](https://gymnasium.farama.org/) environments via PyO3, so they need a Python environment as well as the crate feature. On Apple Silicon, use the Homebrew Python explicitly so the interpreter architecture matches the Rust binary:

```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install maturin 'gymnasium[box2d]'
deactivate

PYO3_PYTHON=`pwd`/.venv/bin/python3 cargo build --release --features gymnasium-examples
```

The runners detect the `.venv` directory themselves, so no activation is needed to run them.

---

## Public API at a glance

- **Build:** `Hierarchy::new()` then `init_random(&io_descs, &layer_descs)`, described by `IoDesc` / `LayerDesc` / `IoType`.
- **Drive a step:** `h.step(&[&input_cis], learn_enabled, reward, mimic)`.
- **Read output:** `get_prediction_cis(io_index)`, and the per-layer hidden CSDRs.
- **Reinforcement learning:** pass `IoType::Action` and supply reward to `step`; the `Actor` head handles credit assignment.
- **Images:** `ImageEncoder` — encode to a CSDR, and `reconstruct()` back.
- **Persist:** save/load of weights, state, or both.
- **Determinism:** `helpers::set_global_state` / `get_global_state` expose the PCG32 stream, mirroring the C++ globals.

---

## Documentation

| Document | Description |
|---|---|
| [`doc/UserGuide.md`](doc/UserGuide.md) | Full user guide: concepts, API reference, RL example |
| [`doc/Architecture.md`](doc/Architecture.md) | Component structure and how the pieces compose |
| [`doc/Tuning.md`](doc/Tuning.md) | Parameter descriptions and tuning advice |
| [`doc/NameReference.md`](doc/NameReference.md) | Variable naming glossary for reading the source |
| [`doc/PortNotes.md`](doc/PortNotes.md) | Mapping from C++ names/types to Rust equivalents |
| [`doc/MethodFidelity.md`](doc/MethodFidelity.md) | Method-by-method correspondence with AOgmaNeo |
| [`doc/Divergences.md`](doc/Divergences.md) | Where this port intentionally differs, and why |

---

## Source layout

```
src/
  lib.rs           — crate root and public re-exports
  helpers.rs       — Int2/Int3, PCG32 RNG, VecWriter/SliceReader, CircleBuffer
  encoder.rs       — ART sparse coder (parallel)
  decoder.rs       — multi-dendrite perceptrons (parallel)
  actor.rs         — actor-critic RL with eligibility traces
  image_encoder.rs — SOM for images with reconstruct()
  hierarchy.rs     — top-level orchestrator
tests/
  smoke_test.rs    — integration tests
  fidelity.rs      — C++ parity comparison against committed fixtures
  demos_support.rs — runs the unit tests inside examples/support/
  fixtures/        — committed golden data
examples/
  support/         — shared demo scaffolding: args, encoding, reporting, environments
  *.rs             — the demos themselves (see Demos)
assets/            — the two track bitmaps car_racing needs
benches/           — criterion benchmarks
fidelity/          — harness for regenerating fixtures (needs an AOgmaNeo checkout)
doc/               — documentation
```

**This is a transliteration, not an idiomatic rewrite.** Module names, function names and loop structure deliberately mirror the C++ so `doc/MethodFidelity.md` can be audited method by method — which is why `manual_is_multiple_of` and `needless_range_loop` are allowed in `Cargo.toml`. `t % 20 == 0` reads the same in both languages; `t.is_multiple_of(20)` reads better as Rust and worse as a comparison against the source it mirrors.

---

## Fidelity

In the **faithful regime** — `leak = 0`, `ticks = 1` — the integer coding output is **bit-exact** with upstream: over 200 learning steps, `prediction_cis` and `hidden_cis` matched 200/200. Only internal float activations drift, at ~8.4e-3 (softmax and transcendental op-ordering), never enough to change an arg-max.

Parity holds because both implementations use the identical PCG32 RNG. It is **expected to break outside that regime** (`leak ≠ 0`, `ticks > 1`) *by design*, not by defect — those are deliberate additions. Note the crate default is `leak = 0.01`, so the faithful regime is something you opt into.

`cargo test` runs the unit and integration suites plus the parity comparison against committed fixtures. Regenerating those fixtures needs an out-of-tree AOgmaNeo checkout — see [`fidelity/README.md`](fidelity/README.md) and [`doc/MethodFidelity.md`](doc/MethodFidelity.md).

CI pins `RAYON_NUM_THREADS=1`. That is not tidiness: the parallel paths use deterministic per-column subseeds so results *are* order-independent, but pinning the thread count keeps a failure meaning "the algorithm changed" rather than "the scheduler differed".

---

## Optional features

- `gymnasium-examples` — builds the two PyO3-backed example runners. Off by default, and deliberately not part of `--all-features` in CI, because it pulls `pyo3` with `auto-initialize` and needs Python development headers. With it off, no consumer of this crate pulls any Python.

---

## Building the C++ reference (macOS)

The original C++ can be built alongside to verify algorithm correctness. On Apple Silicon it needs CMake, OpenMP and LLVM:

```bash
brew install cmake llvm libomp

export OpenMP_ROOT=$(brew --prefix)/opt/libomp
export CPPFLAGS="-I/opt/homebrew/include -I${OpenMP_ROOT}/include"
export LDFLAGS="-L/opt/homebrew/lib -L${OpenMP_ROOT}/lib"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++

mkdir build && cd build
cmake .. && make
```

Key abbreviations shared by the C++ and the Rust: `vl` = visible layer, `hc` = hidden column, `ci` = column index, `cis` = column indices, `wi` = weight index, `diam` = diameter (2×radius+1). Further correspondence notes are in [`doc/PortNotes.md`](doc/PortNotes.md).

---

## License and attribution

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/), the same license as upstream [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo) — ShareAlike requires it of adapted material, and a port is adapted material. Full text in [`LICENSE.md`](LICENSE.md); upstream's own copy is preserved in [`AOGMANEO_LICENSE.md`](AOGMANEO_LICENSE.md).

**It is NonCommercial.** That restriction travels into anything linking this crate.

Copyright (c) 2026 Jacob Everist, for the Rust port.

AOgmaNeo is © 2020-2025 [Ogma Intelligent Systems Corp](https://ogma.ai). The attribution CC BY-NC-SA 4.0 §3(a) requires — and which §6(a) terminates the grant for omitting — is in [`PROVENANCE.md`](PROVENANCE.md). For uses outside this grant, Ogma asks that you contact `licenses@ogmacorp.com`.
