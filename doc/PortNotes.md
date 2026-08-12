# AOgmaNeo: C++ to Rust Reference

This document maps the original C++ AOgmaNeo library (preserved in  [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo/tree/645a54ace656b0ac2476a56a0dac19faacbd87ab)) to the Rust port (in `src/`). Use it when reading C++ documentation or source and working in the Rust codebase.

---

## Naming Conventions

The C++ and Python docs use different casing conventions than the Rust port:

| C++ / Python docs | Rust port | Notes |
|---|---|---|
| `Layer_Desc` | `LayerDesc` | Pascal-case, underscores dropped |
| `IO_Desc` | `IoDesc` | |
| `Image_Encoder` | `ImageEncoder` | |
| `Circle_Buffer` | `CircleBuffer` | |
| `Stream_Writer` | `StreamWriter` | |
| `Stream_Reader` | `StreamReader` | |
| `Byte_Buffer` | `ByteBuffer` | Type alias for `Vec<u8>` |
| `S_Byte_Buffer` | `SByteBuffer` | Type alias for `Vec<i8>` |
| `Int_Buffer` | `IntBuffer` | Type alias for `Vec<i32>` |
| `Float_Buffer` | `FloatBuffer` | Type alias for `Vec<f32>` |
| `Vec2<int>` / `Int2` | `Int2` | `{x, y}` fields |
| `Vec3<int>` / `Int3` | `Int3` | `{x, y, z}` fields |
| snake_case fields/methods | snake_case | Same convention |

---

## Type Buffer Mappings

| C++ type | Rust type | Storage |
|---|---|---|
| `Byte_Buffer` | `ByteBuffer = Vec<u8>` | unsigned 8-bit; encoder weights |
| `S_Byte_Buffer` | `SByteBuffer = Vec<i8>` | signed 8-bit; decoder/actor weights |
| `Int_Buffer` | `IntBuffer = Vec<i32>` | column indices, totals |
| `Float_Buffer` | `FloatBuffer = Vec<f32>` | activations, probabilities |
| `UShort_Buffer` | `UShortBuffer = Vec<u16>` | (defined but unused in current port) |

---

## Class / Struct Mappings

### Hierarchy (`hierarchy.rs`)

| C++ | Rust | Notes |
|---|---|---|
| `Hierarchy` | `Hierarchy` | |
| `IO_Desc` | `IoDesc` | Structural config per IO port |
| `Layer_Desc` | `LayerDesc` | Structural config per hidden layer |
| `Hierarchy::Params` | `Params` | Contains `layers: Vec<LayerParams>`, `ios: Vec<IoParams>` |
| `IO_Desc::type` | `IoDesc::io_type` | `IoType` enum: `None`, `Prediction`, `Action` |
| `IO_Desc::num_dendrites_per_cell` | `IoDesc::num_dendrites_per_cell` | Policy head dendrites |
| `IO_Desc::value_num_dendrites_per_cell` | `IoDesc::value_num_dendrites_per_cell` | Value head dendrites (RL only) |
| `IO_Desc::up_radius` | `IoDesc::up_radius` | IO → encoder receptive field |
| `IO_Desc::down_radius` | `IoDesc::down_radius` | Encoder → decoder/actor receptive field |
| `IO_Desc::history_capacity` | `IoDesc::history_capacity` | Credit-assignment horizon |
| `Layer_Desc::hidden_size` | `LayerDesc::hidden_size` | `Int3 {x, y, z}` |
| `Layer_Desc::ticks_per_update` | *(not present)* | Temporal stride; not yet ported |
| `Layer_Desc::temporal_horizon` | *(not present)* | Memory window; not yet ported |
| `Layer_Desc::recurrent_radius` | `LayerDesc::recurrent_radius` | `-1` disables recurrence |
| `h.params.layers[i]` | `h.params.layers[i]` (a `LayerParams`) | `.encoder`, `.decoder`, `.recurrent_importance` |
| `h.params.ios[i]` | `h.params.ios[i]` (an `IoParams`) | `.decoder`, `.actor`, `.importance` |
| `hierarchy.step(input_cis, reward, mimic)` | `hierarchy.step(&[&cis], learn, reward, mimic)` | `mimic` is `f32` (not `bool`) |

**Index mapping fields** (same names in Rust):

| Name | Meaning |
|---|---|
| `i_indices` | Maps decoder index → IO index |
| `d_indices` | Maps IO index → decoder index (`-1` if no decoder) |
| `io_sizes` | `Int3` size per IO port |
| `io_types` | `u8` enum tag per IO port |

---

### Encoder (`encoder.rs`)

| C++ | Rust | Notes |
|---|---|---|
| `Encoder` | `Encoder` | |
| `Encoder::Visible_Layer_Desc` | `encoder::VisibleLayerDesc` | `.size`, `.radius` |
| `Encoder::Visible_Layer` | `encoder::VisibleLayer` | `.weights` (`ByteBuffer`), `.hidden_totals`, `.importance` |
| `Encoder::Params` | `encoder::Params` | |
| `params.choice` | `Params::choice` | ART choice param (alpha) |
| `params.vigilance` | `Params::vigilance` | ART vigilance in `[0, 1]` |
| `params.lr` | `Params::lr` | Learning rate |
| `params.active_ratio` | `Params::active_ratio` | Lateral inhibition activity ratio |
| `params.l_radius` | `Params::l_radius` | Lateral inhibition radius |
| `encoder.step(input_cis, learn, params)` | `encoder.step(&[&cis], learn_enabled, &params)` | |

**C++ parallelism**: OpenMP `PARALLEL_FOR` over hidden columns.
**Rust equivalent**: `rayon::into_par_iter()` over columns, collecting `Vec<ForwardResult>`. The snapshot trick (`hidden_totals_snapshot`) avoids shared mutable state.

---

### Decoder (`decoder.rs`)

| C++ | Rust | Notes |
|---|---|---|
| `Decoder` | `Decoder` | |
| `Decoder::Visible_Layer_Desc` | `decoder::VisibleLayerDesc` | `.size`, `.radius` |
| `Decoder::Visible_Layer` | `decoder::VisibleLayer` | `.weights` (`SByteBuffer`) |
| `Decoder::Params` | `decoder::Params` | |
| `params.scale` | `Params::scale` | byte-weight range scale |
| `params.lr` | `Params::lr` | Learning rate |
| `params.leak` | *(not present)* | Leaky ReLU; not in current port |
| `decoder.activate(input_cis, params)` | `decoder.activate(&[&cis], &params)` | Forward pass |
| `decoder.learn(input_cis, target_cis, params)` | `decoder.learn(&[&cis], &target, &params)` | Weight update |

---

### Actor (`actor.rs`)

| C++ | Rust | Notes |
|---|---|---|
| `Actor` | `Actor` | |
| `Actor::Visible_Layer_Desc` | `actor::VisibleLayerDesc` | `.size`, `.radius` |
| `Actor::Visible_Layer` | `actor::VisibleLayer` | `.value_weights`, `.policy_weights` (both `FloatBuffer`) |
| `Actor::History_Sample` | `actor::HistorySample` | `.input_cis`, `.hidden_target_cis_prev`, `.hidden_values`, `.reward` |
| `Actor::Params` | `actor::Params` | |
| `params.vlr` | `Params::vlr` | Value function learning rate |
| `params.plr` | `Params::plr` | Policy learning rate |
| `params.discount` | `Params::discount` | RL discount (lambda) |
| `params.trace_decay` | *(replaced)* | `td_scale_decay` in Rust |
| `params.policy_clip` | *(not present)* | |
| `params.value_clip` | *(not present)* | |
| `params.leak` | *(not present)* | |
| Circle buffer of history | `CircleBuffer<HistorySample>` | `history_samples` field |

**Note on borrow conflict**: In Rust, `input_cis_owned: Vec<Vec<i32>>` is cloned from the history buffer before mutable borrows of the actor to avoid Rust's aliasing rules. No equivalent exists in C++.

---

### ImageEncoder (`image_encoder.rs`)

| C++ | Rust | Notes |
|---|---|---|
| `Image_Encoder` | `ImageEncoder` | |
| `Image_Encoder::Visible_Layer_Desc` | `image_encoder::VisibleLayerDesc` | `.size` (Int3 with z = channels), `.radius` |
| `Image_Encoder::Visible_Layer` | `image_encoder::VisibleLayer` | `.weights`, `.recon_weights`, `.reconstruction` (all `ByteBuffer`) |
| `Image_Encoder::Params` | `image_encoder::Params` | |
| `params.lr` | `Params::lr` | SOM learning rate |
| `params.falloff` | `Params::falloff` | Neighbourhood falloff |
| `params.n_radius` | `Params::n_radius` | Neighbourhood radius |
| `image_encoder.step(inputs, learn, learn_recon)` | `image_encoder.step(&[&bytes], learn, learn_recon)` | Inputs are `&[u8]` (raw bytes), not CSDRs |
| `image_encoder.reconstruct(hidden_cis)` | `image_encoder.reconstruct(&hidden_cis)` | Clone `get_hidden_cis()` first to avoid borrow conflict |

**C++ parallelism**: OpenMP.
**Rust port**: fully sequential (no `rayon`).

---

## Variable Name Glossary (from `doc/NameReference.md`)

These abbreviations appear in both C++ and Rust source identically:

| Abbreviation | Meaning |
|---|---|
| `vl` | visible layer (the input-side of an encoder/decoder) |
| `vld` | visible layer descriptor |
| `hc` | index within a hidden column, range `[0, hidden_size.z)` |
| `vc` | index within a visible column, range `[0, vld.size.z)` |
| `ci` / `in_ci` | column index (the active cell chosen within a column) |
| `cis` | column indices (a full CSDR buffer, one `ci` per column) |
| `wi` | weight index (final flat index into a weight buffer) |
| `wi_start` / `wi_offset` | partially computed weight index |
| `diam` | receptive field diameter: `2 * radius + 1` |
| `area` | receptive field area: `diam * diam` |
| `h_to_v` | `Float2` scale factors to project hidden positions → visible positions |
| `visible_center` | center of the receptive field on the visible layer |
| `field_lower_bound` | lower bound before edge clamping |
| `iter_lower_bound` / `iter_upper_bound` | clamped iteration bounds |
| `hidden_stride` | weight index stride when hidden cell changes by 1 |
| `offset` | 2D position within the receptive field `[0, diam)` |
| `max_index` | index of winning (highest activation) cell |
| `max_activation` | the winning activation value |
| `delta` | weight increment |
| `count` | normalization counter (number of receptive field inputs) |
| `total` | float normalization accumulator (softmax denominator) |
| `state` | local PCG32 RNG state for a column |
| `base` / `base_state` | seed for deriving per-column RNG sub-seeds |
| `ticks` | exponential-memory clock counter per layer |
| `updates` | whether a layer activated this timestep |
| `i` | IO index |
| `d` | decoder index |
| `e` | encoder |
| `a` | actor |
| `t` | time index |
| `prev` suffix | value from previous timestep |
| `next` suffix | value for next timestep |
| `acts` suffix | activations buffer |
| `probs` suffix | probabilities buffer |
| `importance` | relative input scaling factor (default `1.0`) |

---

## Serialization

| C++ | Rust |
|---|---|
| Abstract `Stream_Writer` / `Stream_Reader` | Traits `StreamWriter` / `StreamReader` in `helpers.rs` |
| `File_Write_Stream` / `File_Read_Stream` | Not provided; use `VecWriter` + file I/O |
| `VecWriter` | `VecWriter { data: Vec<u8> }` — access bytes as `.data` (not `.into_bytes()`) |
| `SliceReader` | `SliceReader<'a> { data: &[u8], pos: usize }` |
| `write()` / `read()` | `write(&mut dyn StreamWriter)` / `read(&mut dyn StreamReader)` — full state + weights |
| `write_state()` / `read_state()` | Same names — runtime state only (no weights) |
| `write_weights()` / `read_weights()` | Same names — weights only |

All values are serialized **field-by-field in little-endian** — not raw struct memory. Binary format differs from C++.

---

## Tuning Guide Applied to Rust

The `doc/Tuning.md` parameter names match the Rust `Params` structs directly. Access pattern:

```rust
// C++ / Python:  h.params.ios[0].decoder.lr = 0.1
// Rust:
h.params.ios[0].decoder.lr = 0.1;

// C++ / Python:  h.params.layers[0].encoder.vigilance = 0.95
// Rust:
h.params.layers[0].encoder.vigilance = 0.95;

// C++ / Python:  h.params.ios[0].actor.vlr = 0.01
// Rust:
h.params.ios[0].actor.vlr = 0.01;
```

**Parameters not yet in the Rust port** (present in Tuning but absent from Rust `Params` structs):
- `LayerDesc::ticks_per_update` and `temporal_horizon` — exponential memory stride/window
- `DecoderParams::leak` — leaky ReLU coefficient
- `ActorParams::policy_clip`, `value_clip`, `trace_decay`

**Rust-only parameters** (no C++ equivalent in docs):
- `Params::anticipation` — on `Hierarchy::Params`
- `ActorParams::smoothing`, `td_scale_decay`, `value_range`, `min_steps`, `history_iters`
- `LayerParams::recurrent_importance`
- `ImageEncoder::Params::scale`, `rr`
