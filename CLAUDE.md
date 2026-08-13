# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`dcc_sph` implements **Sparse Predictive Hierarchies (SPH)** — an online learning architecture with a low compute footprint that learns from streaming data without catastrophic forgetting. It is a clean-room Rust port of [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo/tree/645a54ace656b0ac2476a56a0dac19faacbd87ab) (C++, Ogma Intelligent Systems Corp) at commit `645a54a`.

**Licensing is not boilerplate here.** This crate is CC BY-NC-SA 4.0 — the same license as upstream, as ShareAlike requires — and it is **NonCommercial**. That restriction travels into anything linking this crate. Read [`PROVENANCE.md`](PROVENANCE.md) before making distribution decisions; the attribution it carries is a condition of the grant, not a courtesy.

**Module names deliberately mirror the C++** so `doc/MethodFidelity.md` can be audited function by function. Do not rename them for idiom.
## Rust Codebase (Primary)

### Build & Test

```bash
# Build
cargo build --release

# Run all tests
cargo test

# Learning-outcome assertions, deliberately excluded from `cargo test` and from CI
# (`test = false` in Cargo.toml). Whether a configuration learns is an offline
# experiment, not a build gate. CI compiles and lints this target but never runs it.
cargo test --release --test learning -- --nocapture

# Run a single test
cargo test test_hierarchy_create_and_step

# Run with output visible
cargo test -- --nocapture

# Lint (must be clean, 0 warnings)
cargo clippy

# Pure-Rust examples (no extra deps)
cargo run --release --example cartpole
cargo run --release --example wave_prediction

# Gymnasium examples — requires Python venv (see below)
PYO3_PYTHON=`pwd`/.venv/bin/python3 cargo build --release -p dcc_sph_gym_examples
cargo run --release -p dcc_sph_gym_examples --example cartpole_env_runner
cargo run --release -p dcc_sph_gym_examples --example lunarlander
```

**Python venv setup (Apple Silicon — one-time):**

```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install 'gymnasium[box2d]'
deactivate
```

The gymnasium examples auto-detect `.venv` at runtime; no activation needed when running.

### Source Layout

```
src/
  lib.rs           — module declarations only
  helpers.rs       — Int2/Int3/Float2, PCG32 RNG, VecWriter/SliceReader, CircleBuffer
  encoder.rs       — ART sparse coder (parallel via rayon)
  decoder.rs       — multi-dendrite perceptrons (parallel via rayon)
  actor.rs         — actor-critic RL with eligibility traces (sequential)
  image_encoder.rs — SOM for image data, supports reconstruct()
  hierarchy.rs     — top-level orchestrator
tests/
  smoke_test.rs    — 10 integration tests
  demos_support.rs — compiles examples/support/ in test config so its unit tests run
examples/
  cartpole.rs             — CartPole RL (pure Rust physics)
  wave_prediction.rs      — wavy-line sequence prediction (pure Rust)
  support/                — shared demo scaffolding (see below)
  wavy_line.rs, wavy_classify.rs, ball_physics.rs, video_prediction.rs,
  vsa_char.rs, pusher.rs, cat_mouse.rs, cat_mouse_pos.rs, explore.rs,
  car_racing.rs, runner.rs, enc_vis.rs, topo_test.rs,
  stacking_rl.rs, stacking_prog.rs, noise_robustness.rs
                          — ported from OgmaNeoDemos; see doc/Demos.md
examples-viz/             — SEPARATE CRATE; macroquad lives here, not in the library
  examples/
    viewer.rs             — the only windowed target
examples-gym/             — SEPARATE CRATE; pyo3 lives here, not in the library
  examples/
    cartpole_env_runner.rs — CartPole-v1 via gymnasium
    lunarlander.rs         — LunarLander-v3 via gymnasium
assets/                   — the two track bitmaps car_racing needs
```

### Demos

Sixteen demos: fifteen ported from [OgmaNeoDemos](https://github.com/jacobeverist/OgmaNeoDemos/tree/aogmaneo) (Ogma's own, same CC BY-NC-SA 4.0 licence — attribution is in `PROVENANCE.md`), plus `noise_robustness`, which is RUST-ONLY. **[`doc/Demos.md`](doc/Demos.md) is the reference**: it records each demo's upstream source and every deviation from it, and several of those deviations are load-bearing rather than cosmetic.

`doc/Demos.md` also states the **cross-repo demo contract** — the flags, record schema, baseline rule and verdict wording shared with `dcc-sparsey` and `dcc-htm`. The three repositories share no code and cannot (CC BY-NC-SA and AGPL do not mix), so the contract is prose that each repository answers separately, exactly as each answers R1–R16 in its own `doc/Conformance.md`. `noise_robustness` is the one demo all three implement.

They **run headless and text-only with no features enabled**, which is the default path and the one CI runs. Keep it that way — the windowed viewer exists so a demo can be eyeballed, not as instrumentation. Real visualisation is dcc-dashboard's job, and `examples/support/viz.rs` should stay at five functions.

**The viewer is a separate crate, and that is a conformance requirement rather than a preference.** R16 of dcc-core's import contract says local applications belong in a separate crate, not behind an optional feature: an optional dependency is still a real `[dependencies]` entry that lands in the lockfile and constrains resolution for every consumer. `r10_runtime_dependencies_stay_minimal` in `tests/conformance.rs` fails the build if anything but `rayon` (and `pyo3`, pending its own move) appears there. Read [`doc/Conformance.md`](doc/Conformance.md) before adding any dependency.

Shared code lives in `examples/support/`, pulled into each demo with `#[path = "support/mod.rs"] mod support;` — the idiom `fidelity_dump.rs` already uses for `tests/support/`. Cargo does not auto-discover a directory under `examples/` without a `main.rs`, so it is not itself a target. Example targets default to `test = false`, so `tests/demos_support.rs` includes the same tree to get its unit tests run.

`png` and `rapier2d` are plain **dev-dependencies**, not optional dependencies: cargo builds them only for examples, tests and benches, so no consumer of the crate ever pulls them, and `getrandom` stays out of the lockfile so R12 keeps holding. `macroquad` is not a dependency of this package at all — it belongs to `examples-viz`.

When adding a demo, give it an explicit `[[example]]` block, keep the environment in `support/env/`, and have it report against a baseline — a bare number with nothing to compare it to cannot distinguish learning from noise, which is how several real bugs in this suite were found.

### Architecture

**Hierarchy** is the user-facing entry point. It holds a stack of Encoder layers with associated Decoders and optional Actors.

Configuration is split:
- **Structural** (`IoDesc`, `LayerDesc`) — set at `init_random()` time; cannot change afterwards.
- **Runtime** (`LayerParams` containing `DecoderParams`, `encoder::Params`) — adjustable anytime via `hierarchy.params`.

`IoType` on each `IoDesc` determines behavior:
- `None` — input-only (no decoder/actor)
- `Prediction` — encoder + decoder predicts the IO's next state
- `Action` — encoder + actor (RL); uses reward signal passed to `step()`

`step(&[input_cis], learn, reward, mimic)` drives one timestep. `mimic` is `f32` (not `bool`).

**Data representation**: all inputs/outputs are `Vec<i32>` of column indices (`cis`). A column index selects one active cell within each spatial column. Layer sizes are `Int3 { x, y, z }` where `x*y` is the number of columns and `z` is the column size (number of cells per column).

**Parallelism**: Encoder and Decoder forward passes use `rayon` (`into_par_iter()`). Actor and ImageEncoder are sequential. Parallel kernels collect per-column results into `Vec<ForwardResult>` to avoid shared mutable state.

**Serialization**: `VecWriter` / `SliceReader` do field-by-field little-endian I/O (not raw struct binary). Access the written bytes as `writer.data` (not `.into_bytes()`).

**RNG**: thread-local PCG32 via `global_rand()`. Column kernels seed a local RNG with `rand_get_state(seed)` for deterministic per-column randomness.

### Key Clippy Suppressions

`#![allow(clippy::needless_range_loop)]` is at the top of every compute module. Private column kernels carry `#[allow(clippy::too_many_arguments)]`. These are intentional and should be preserved.






