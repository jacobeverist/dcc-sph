# dcc_sph — Method-Level Fidelity

A method-by-method comparison of the Rust `dcc_sph` crate against upstream **AOgmaNeo** (C++) @ [`645a54a`](https://github.com/ogmacorp/AOgmaNeo/tree/645a54ace656b0ac2476a56a0dac19faacbd87ab), so we know exactly how our algorithms are structured versus the original. Companion to [Divergences.md](Divergences.md) and [Architecture.md](Architecture.md). Produced by reading both sources method-by-method.

**Legend:** **FAITHFUL** (same algorithm) · **DIVERGENT** (ported but differs) · **MISSING** (no Rust equivalent) · **RUST-ONLY** (no C++ counterpart). Byte-accounting methods (`size`/`state_size`/`weights_size`) are **MISSING** across all modules and not re-listed each time.

## Headline findings

- **Core algorithms are faithfully ported** across every module — ART encoder (choice/vigilance/match, committed flags, lateral inhibition), push-pull multi-dendrite decoder (softmax, stochastic-rounded i8 deltas), actor-critic (TD(λ) return, `smoothing`, `discount`, `td_scale_decay`, symlog value bins, `history` replay), and the SOM ImageEncoder (+ reconstruction).
- **Two genuine numeric divergences from `645a54a`:**
  1. **Leaky-softplus `leak` term.** The Rust **decoder and actor** replace plain `softplus(act)` with `softplus(act) − leak·softplus(−act)` under a `Params::leak` **not present in the referenced C++ commit**. At `leak = 0.0` it reduces exactly to the C++; **the crate default is `0.01`, so default-constructed `Params` do NOT match upstream.** Set `params.leak = 0.0` for parity. (A consumer may default it to `0.0` on its own side — dcc-core does — but that is a fact about that consumer, not about this crate. See [Divergences.md](Divergences.md).)
  2. **Approximate vs exact transcendental math.** C++ `645a54a` ships **custom fixed-iteration `expf`/`logf`/`sqrtf`/`powf`/`sinf`/`cosf`** by default (Taylor series + Quake `rsqrt`); Rust uses exact std math. Results diverge unless the C++ is built with `USE_STD_MATH`.
- **Structural divergences (behaviorally equivalent unless noted):**
  - **Hierarchy adds clockwork tick-gating** (`ticks`/`ticks_per_update`/`updates`) that C++ `645a54a` **does not have** — confirmed: its `Hierarchy::step` runs every layer every step and gets temporal memory *only* from recurrence. The Rust port runs a **hybrid** (recurrence + ticks). See [Divergences.md](Divergences.md).
  - **Learn kernels are sequential** where C++ uses `PARALLEL_FOR`; identical per-column RNG seeds make outputs **bit-equal** (only forward passes use rayon).
  - **Serialization is field-wise little-endian**, not C++ raw-struct `memcpy` (which includes `Int3`/`Vec3` padding) → **binary formats are NOT interchangeable**. Rust also prepends a magic+version header to `Hierarchy`.
  - **RNG is thread-local** (per-thread PCG32 streams) vs C++'s single global → different draw ordering under parallelism; init-weight streams differ.
- **`array.h` (`Array`/`Array_View`) → Rust `Vec`/slices** (type substitution, not a functional gap).

---

## Encoder (`encoder.rs`)
| C++ method | Rust | Verdict | Note |
|---|---|---|---|
| `forward` | `compute_forward` | FAITHFUL | ART math identical (choice/vigilance/match, committed-flag gate, lateral inhibition). Rust adds a defensive `count_except > 0` guard (fires only on empty field). |
| `learn` | inlined in `step` | DIVERGENT (parallelism) | math identical (inhibition, `rate = committed ? lr : 1`, weights→255, `hidden_totals += Δ`); **C++ parallelizes learn, Rust sequential**; numerically identical. |
| `init_random` / `step` / `clear_state` | same | FAITHFUL / DIVERGENT | forward parallel both sides; learn parallelism differs; `hidden_totals` snapshotted for the borrow checker (equal). |
| `write`/`read` | same | DIVERGENT (format) | same fields; `Visible_Layer_Desc` written field-wise vs C++ raw blob. |
| `write_state`/`read_state`, `write_weights`/`read_weights`, accessors | same | FAITHFUL | 1:1. |
| — | `get_visible_layer_mut`, static `compute_forward` | RUST-ONLY | rayon-friendly refactor + explicit mutable accessor. |

## Decoder (`decoder.rs`)
| C++ method | Rust | Verdict | Note |
|---|---|---|---|
| `forward` | `compute_forward` | **DIVERGENT (added `leak`)** | push-pull signs, scales, softmax, sigmoid-derivative caching all faithful — **except** plain `softplus` → **leaky softplus** with new `leak` (default 0.01). The one genuine math divergence. |
| `learn` (kernel) / `learn` (driver) | `learn_column` / `learn` | DIVERGENT (parallelism) | delta math, stochastic i8 rounding, per-column RNG seeds identical; **C++ parallel, Rust sequential** → bit-equal output. |
| `init_random` / `activate` / `clear_state` | same | FAITHFUL | weight init, parallel forward, buffer zeroing match. |
| `write`/`read` | same | DIVERGENT (format) | raw-struct vs field-wise desc. |
| state/weights serialization, accessors | same | FAITHFUL | 1:1 (Rust exposes only `&`, no `_mut`). |
| — | `leak` Params field | RUST-ONLY | drives the leaky softplus. |

## Actor (`actor.rs`)
| C++ method | Rust | Verdict | Note |
|---|---|---|---|
| `forward` | `forward_column` | DIVERGENT (added `leak`) | dendrite/softmax/action-sampling faithful; same leaky-softplus divergence. |
| `learn` | `learn_column` | DIVERGENT | TD(λ) return, `td_scale_decay`/scaled-TD, symlog value-bin target, policy CE error all **faithful**; same `leak` divergence. |
| `init_random` | same | DIVERGENT | layout/dims match; weight init uses thread-local RNG (`(randf*2−1)*NOISE`) vs C++ shared-state `randf(-noise,noise)` → different stream. |
| `step` / `clear_state` | same | FAITHFUL | forward-all, history push/cap, `history_iters` replay from `t∈[min_steps,history_size)` match (Rust loop sequential). |
| `write`/`read`, `write_state`/`read_state` | same | DIVERGENT (format) | field-wise LE vs struct memcpy (incl. `history_samples.start`). |
| `write_weights`/`read_weights`, accessors | same | FAITHFUL | value+policy weights per visible layer. |
| — | `Params::leak` | RUST-ONLY | as decoder. |

## Hierarchy (`hierarchy.rs`)
| C++ method | Rust | Verdict | Note |
|---|---|---|---|
| `init_random` | same | DIVERGENT | encoder/decoder/actor wiring, `recurrent_radius` VL, `i_indices`/`d_indices` faithful; Rust **also** builds `ticks`/`updates`/`ticks_per_update` (no C++ counterpart). |
| `step` | same | **DIVERGENT ⚠** | **The headline divergence.** C++ `645a54a` runs every layer every step (`for l …` unconditional) with temporal memory **only** from recurrence; Rust **adds clockwork tick-gating** (`if !updates[l] continue`) gating encoder-forward + decoder-learn while **also keeping recurrence** — a hybrid matching neither classic OgmaNeo nor `645a54a`. Anticipation/recurrence/routing otherwise faithful. |
| `clear_state` | same | DIVERGENT | also resets tick state. |
| `is_layer_recurrent` / `io_layer_exists` / `get_prediction_*` | same | FAITHFUL | 1:1 predicates/getters. |
| `write`/`read` | same | DIVERGENT | Rust prepends magic+version and serializes tick state; C++ `memcpy`s `Layer_Params`/`IO_Params` whole; formats incompatible. |
| `write_state`/`read_state`, `write_weights`/`read_weights` | same | FAITHFUL | same traversal order. |
| — | tick state, `get_update`, `_mut` accessors | RUST-ONLY | the added clockwork + mutable accessors. |
| `get_num_encoder_visible_layers`, `get_num_decoders` | — | MISSING | minor accessors. |

## ImageEncoder (`image_encoder.rs`)
| C++ method | Rust | Verdict | Note |
|---|---|---|---|
| `forward` (fused forward+SOM-learn kernel) | `compute_forward_column` + `learn_column` | DIVERGENT (structure) | same 2-pass center/SOM/argmax + neighborhood learning; Rust **splits** parallel-forward from sequential-learn (recomputes center identically). Algorithm faithful. |
| `learn_reconstruction` / `reconstruct` | same | FAITHFUL | reverse-projection, `rand_roundf`, `scale`-stretched recon match. |
| `init_random` / `step` | same | DIVERGENT (parallelism/RNG) | same control flow; SOM+recon learn sequential; thread-local RNG. |
| `write`/`read` | same | DIVERGENT (format) | `Params` struct memcpy vs explicit fields. |
| state/weights serialization, accessors | same | FAITHFUL | 1:1. |

## helpers / array (`helpers.rs`)
| C++ item | Rust | Verdict | Note |
|---|---|---|---|
| `Array` / `Array_View` / buffers | `Vec` / slices / `*Buffer` aliases | (type substitution) | std containers replace the hand-rolled one; `Vec3.pad` (C++) contributes to the serialization-layout divergence. |
| `Int2/3/4`, `Float2/3/4`, `Circle_Buffer` | same | FAITHFUL | Rust `push_front` guards empty (C++ doesn't). |
| `address2/3/4`, `project`, `in_bounds`, `min_overhang` | same | FAITHFUL | identical formulas. |
| `sigmoidf`, `softplusf`, `symlogf`, `symexpf`, `logitf`, rounding (`roundf2i/b/sb`, `ceil_divide`, `rand_roundf`) | same | FAITHFUL | closed forms match; `ceilf`→`ceilf_to_i32` (std `.ceil()` — differs only for negative x, never passed). |
| `expf`/`logf`/`sqrtf`/`powf`/`sinf`/`cosf` (custom approximations) | std math (`.exp/.ln/.sqrt/…`) | **DIVERGENT / MISSING** | C++ default = fixed-iteration Taylor + Quake `rsqrt`; Rust = exact std → numeric drift unless C++ built `USE_STD_MATH`. `modf` (only for C++ `sinf`) MISSING (unneeded). |
| PCG32 (`rand_get_state`/`rand`/`randf`/`rand_normalf`/`rotr32`) | `*_step` fns | FAITHFUL | same constants/algorithm. |
| `global_state` (single global) | `GLOBAL_STATE` (thread-local) + `global_rand/f` | DIVERGENT | same seed/stream, but thread-local → different draw order under parallelism. |
| `Stream_Writer/Reader` (raw `void*`) / `PARALLEL_FOR` (OpenMP) / `set_num_threads` | typed-LE `StreamWriter/Reader` traits / rayon `into_par_iter` / rayon pool | DIVERGENT | typed little-endian I/O (formats not interchangeable); OpenMP → rayon. |
| — | `VecWriter`/`SliceReader`/`FileWriter`/`FileReader` | RUST-ONLY | concrete stream impls. |

---

## Functional-fidelity harness (executed cross-check)

The verdicts above are corroborated by an **executed differential test**, not just code
reading. `fidelity/` runs the same deterministic scenario on the Rust crate
and on upstream AOgmaNeo C++ (`645a54a`, built `-DUSE_STD_MATH`, single-threaded) and
diffs the output; `cargo test -p dcc_sph --test fidelity` asserts the Rust in-process
run against the committed golden vector (`tests/fixtures/wave_fidelity_golden.json`).

Result over 200 learning steps (single IO, two `(5,5,64)` layers, `leak=0`, `ticks=1`):

| Surface | Comparison | Result |
|---|---|---|
| `prediction_cis`, `hidden_cis` (integer CSDRs) | exact per step | **200/200 identical** |
| `final_prediction_acts` (softmax floats) | tolerance | max abs diff **~8.4e-3** |

So in the faithful regime the integer coding output is **bit-exact** with upstream —
direct evidence for the encoder/decoder "FAITHFUL" forward verdicts — and only internal
float activations drift at the ~1e-2 level (softmax/transcendental op-ordering; never
changes the arg-max). Parity holds because both use the identical PCG32 RNG; it is
*expected to break* outside this regime (`leak≠0`, `ticks>1`) by design. The harness
added `helpers::set_global_state`/`get_global_state` (mirroring the C++ globals) so the
RNG stream can be aligned. See [`fidelity/README.md`](../fidelity/README.md).

---

*See [Divergences.md](Divergences.md) for the prioritized audit (incl. the deferred/added items). Upstream: AOgmaNeo C++ @ `645a54a`.*
