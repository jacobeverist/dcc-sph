# dcc_sph — Architecture (Rust implementation)

How the `dcc_sph` crate is built in Rust — a port of **AOgmaNeo** (Sparse Predictive Hierarchies). Terms: [NameReference.md](NameReference.md); upstream → Rust mapping: [PortNotes.md](PortNotes.md); parameters: [Tuning.md](Tuning.md).

## The model in one paragraph

An SPH **Hierarchy** is a vertical stack of **layers** driven one timestep at a time. Each step runs an **up pass** — bottom-to-top **Encoders** (ART sparse coders) turn inputs into hidden **CSDRs** — then a **down pass** — top-to-bottom **Decoders** predict each layer's next input (or **Actors** emit RL actions). All data is discrete: a **CSDR** is a flat array of column indices, one active cell per column. Learning is fully **online** (one sample per step, no replay shuffle).

## Data representation — columnar SDRs of `i32`

A **CSDR** is an `IntBuffer = Vec<i32>` (`helpers.rs`), one entry per column holding that column's active cell index `ci ∈ [0, z)`. A layer's shape is `Int3 { x, y, z }`: `x·y` = column count, `z` = cells per column. Flat, row-major addressing: `address2(pos, {x,y})` for columns, `address3` (`z + dimz·(y + dimy·x)`) for cells. Receptive fields are `(2·radius+1)²` patches projected between layers by `project()` using scale `visible_size / hidden_size`.

Weight buffer types (`helpers.rs`): `ByteBuffer = Vec<u8>` (encoder/image weights, `[0,255]`), `SByteBuffer = Vec<i8>` (decoder weights, `[-127,127]`), `FloatBuffer = Vec<f32>` (activations + **actor** weights), `IntBuffer` (the CI arrays).

## Top-level structure — `Hierarchy` (`hierarchy.rs`)

`Hierarchy` owns every compute module by value:

- `encoders: Vec<Encoder>` — one per layer.
- `decoders: Vec<Vec<Decoder>>` — `decoders[layer][d]`; layer 0 has one per Prediction IO, higher layers exactly one.
- `actors: Vec<Actor>` — one per Action IO, all at layer 0.
- `hidden_cis_prev: Vec<IntBuffer>` — per-layer previous encoder output (the **recurrence** source).
- `feedback_cis_prev: Vec<IntBuffer>` — per-layer top-down decoder feedback from the layer above.
- `i_indices` / `d_indices` — the IO ↔ decoder/actor wiring maps (`-1` = none).
- `io_sizes`, `io_types`, and the clockwork gating `ticks`, `updates`, `ticks_per_update`.
- `params: Params` — the runtime-adjustable knobs.

Wiring is built once in **`init_random(&io_descs, &layer_descs)`**: each layer's encoder gets one `VisibleLayerDesc` per IO (layer 0) or one from the layer below (higher layers), plus an optional recurrent self-VLD; each decoder/actor gets a VLD onto its own layer's hidden state plus (unless top layer) a second VLD for feedback from above.

## The compute modules

### Encoder — ART sparse coder (`encoder.rs`)
`VisibleLayer { weights: ByteBuffer, hidden_totals: IntBuffer, importance }`. Hidden state adds `hidden_committed_flags` (per-cell "committed" bit), `hidden_learn_flags`, `hidden_comparisons`. Forward (`compute_forward`, rayon-parallel over columns): ART **choice** `activation = complemented/(choice + count − total)` and **match** `match_val = complemented/count_except`; a committed cell wins only if `match_val ≥ vigilance`, else an uncommitted cell is allocated. Learn (sequential): **lateral inhibition** rejects a winner if too many higher-scoring neighbors sit within `l_radius` (beyond `active_ratio`); the winner moves its byte weights toward 255 at `lr` (or 1.0 fast-commit if uncommitted) and sets its committed flag. `Params`: `choice` (0.01), `vigilance` (0.9), `lr` (0.5), `active_ratio` (0.1), `l_radius` (2).

### Decoder — multi-dendrite perceptron (`decoder.rs`)
`VisibleLayer { weights: SByteBuffer }`. Each cell has `num_dendrites_per_cell` dendrites in **push-pull**: the first half contribute `−1`, the second `+1`; `activation = softplus(x) − leak·softplus(−x)`, summed with sign then **softmax over `z`**; argmax = the predicted CI. Learn (`learn_column`): error `= lr·127·(onehot(target) − hidden_act)`, applied as stochastically-rounded `i8` deltas clamped to `[-127,127]`. `Params`: `scale` (8.0), `lr` (0.1), `leak` (0.01). Forward is rayon-parallel.

### Actor — actor-critic RL (`actor.rs`)
`VisibleLayer { value_weights: FloatBuffer, policy_weights: FloatBuffer }` (float, not byte). A **value head** (softmax over `value_size` bins → scalar via `symexp`) and a **policy head** (softmax → action sampled by cumulative draw), both multi-dendrite perceptrons. A `CircleBuffer<HistorySample>` (`input_cis`, targets, `hidden_values`, `reward`) drives replay: each step, once history exceeds `min_steps`, it replays `history_iters` random past steps, computing a smoothed multi-step **TD return** with per-column adaptive normalization (`hidden_td_scales`, decayed by `td_scale_decay`). `Params`: `vlr` (0.1), `plr` (0.01), `discount` (0.99), `smoothing`, `td_scale_decay`, `value_range`, `min_steps`, `history_iters`. **Sequential** (not rayon-parallel).

### ImageEncoder — SOM (`image_encoder.rs`)
A self-organizing map for raw byte-pixel images with `reconstruct()` (reverse-projection weights, `falloff`/`n_radius` neighborhood). A pre-encoder utility, **not** on the core predictive path.

## Module map (`src/`)

| File | Role |
|---|---|
| `hierarchy.rs` | Top-level `Hierarchy` orchestrator — stacks encoders, wires decoders/actors, runs the up/down passes, serialization. |
| `encoder.rs` | ART sparse coder (rayon-parallel forward). |
| `decoder.rs` | Multi-dendrite predictive perceptron (rayon-parallel forward). |
| `actor.rs` | Actor-critic RL with history replay (sequential). |
| `image_encoder.rs` | SOM image encoder + reconstruction. |
| `helpers.rs` | Foundation — buffer typedefs, `Int2/3/4`/`Float*`, math (`symlogf`/`symexpf`, `softplusf`), addressing/projection, PCG32 RNG, `CircleBuffer`, `StreamReader/Writer` (+ `VecWriter`/`SliceReader`/`FileWriter`/`FileReader`), rayon thread control. |
| `lib.rs` | Module declarations only. |

## The step loop — `Hierarchy::step(&[input_cis], learn, reward, mimic)`

1. **Tick/update gating** — layer 0 always updates; for `l ≥ 1`, `ticks[l] += 1`, and when it reaches `ticks_per_update[l]` the layer updates (reset to 0) else it is skipped in the up pass. (Non-updating layers' decoders still `activate` in the down pass so predictions stay fresh.)
2. Push IO importances into the layer-0 encoder's visible layers.
3. **Up pass** (bottom → top): for each updating layer, snapshot `hidden_cis_prev[l]` and `feedback_cis_prev[l]` (from the decoder above); assemble encoder inputs (IO inputs at layer 0, else the lower encoder's hidden CIs) plus, if recurrent, `hidden_cis_prev[l]` weighted by `recurrent_importance`; call `encoder.step`.
4. **Down pass** (top → bottom): if `learn`, decoders learn on `(hidden_cis_prev[l], feedback_cis_prev[l])` targeting the true input (layer 0) or the lower encoder's CIs; with `anticipation`, also re-activate + learn on the *current* encoder output. Then always `activate` every decoder on `(current encoder CIs, decoder-above CIs)`. **Actors** run only at layer 0, using `reward` + `mimic`.

## Temporal memory — two mechanisms coexist

- **Recurrence** (upstream's newer scheme): `LayerDesc.recurrent_radius` (default 0; `-1` disables) appends a self-referential encoder visible layer onto the layer's own hidden grid; the up pass feeds `hidden_cis_prev[l]` back weighted by `recurrent_importance` (default 0.5). `is_layer_recurrent(l)` detects it.
- **Ticked clockwork** (older "exponential memory"): `ticks_per_update` runs higher layers only every N bottom-layer steps.

Both are wired independently and can be combined. ⚠ **The referenced upstream (`645a54a`) has recurrence only — no `ticks_per_update`; this crate adds ticks on top.** See [Divergences.md](Divergences.md).

## Learning (all online, per step)

Encoder ART (vigilance-gated commit + lateral inhibition, byte weights → 255); Decoder (softmax-error, stochastically-rounded `i8` deltas); Actor (value `vlr` / policy `plr` / `discount` / `smoothing`, float additive updates, history replay). `anticipation` adds extra decoder training on the current (not just previous) encoder output.

## Config surface

- **Structural** (fixed after `init_random`): `IoDesc` (`size`, `io_type`, `num_dendrites_per_cell`, `up_radius`, `down_radius`, `value_size`, `value_num_dendrites_per_cell`, `history_capacity`) and `LayerDesc` (`hidden_size`, `num_dendrites_per_cell`, `up_radius`, `recurrent_radius`, `down_radius`, `ticks_per_update`).
- **Runtime** (mutable via `hierarchy.params`): `LayerParams` (decoder + encoder params + `recurrent_importance`), `IoParams` (decoder + actor params + `importance`), top-level `Params` (`layers`, `ios`, `anticipation`).
- `IoType ∈ {None, Prediction, Action}`. Construction: `Hierarchy::new()` → `init_random(&io_descs, &layer_descs)`.
- **Serialization**: field-by-field little-endian via `StreamReader/Writer`, magic `0x4d474f41` ("AOGM") + version; each module implements `write`/`read` (+ transient-only `write_state`/`read_state` and `write_weights`/`read_weights`).

## Rust-port specifics

- **rayon parallelism**: encoder / decoder / image-encoder *forward* passes run `into_par_iter` over columns; learn passes and the whole actor are sequential. Per-column kernels collect into `Vec<_>` to avoid shared mutable state; a custom PCG32 RNG with deterministic per-column subseeds keeps results reproducible under parallelism.
- **Both temporal mechanisms** (ticks + recurrence) present — see above.
- **Rust-only knobs**: `anticipation`, `recurrent_importance`, and the actor's `smoothing` + per-column adaptive `hidden_td_scales` / `td_scale_decay`.

## Consumers

This crate is the algorithm and nothing else. [dcc-core](https://github.com/jacobeverist/dcc-core/tree/main) adapts these types onto its own `Node` interface in [`engine/src/nodes/ports/sph/`](https://github.com/jacobeverist/dcc-core/tree/main/engine/src/nodes/ports/sph) — sparse coder, decoder, actor, hierarchy and image encoder — so SPH can be compared against other architectures in one network. That adaptation lives entirely on the consumer's side; nothing here reaches back across it.
