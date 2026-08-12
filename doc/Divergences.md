# dcc_sph — Port Divergences

Architectural & structural differences between the upstream source and this dcc-core port — the **fidelity audit**: what we deliberately changed, what we added, and what to reconcile. Name/type mapping: [PortNotes.md](PortNotes.md); implementation: [Architecture.md](Architecture.md); method-by-method: [MethodFidelity.md](MethodFidelity.md).

**Upstream:** `AOgmaNeo` (C++) @ [`645a54a`](https://github.com/ogmacorp/AOgmaNeo/tree/645a54ace656b0ac2476a56a0dac19faacbd87ab) — the crate's stated reference commit (and the checkout these findings were verified against).

## Deliberate structural changes

- **Parallelism via `rayon`.** Encoder / Decoder / ImageEncoder **forward** passes use `into_par_iter` over columns; **`decoder.learn` is also parallel** (per-column disjoint weight/delta chunks via `chunks_mut`, above a 64-column threshold). Per-column deterministic RNG subseeds keep all parallel paths order-independent (bit-exact, validated). A structural change from the C++ OpenMP threading model, but same disjoint-column decomposition.
  - **`encoder.learn` stays sequential — measured, not assumed.** It is column-disjoint (same `chunks_mut` decomposition applies and is bit-exact), but its per-column work is far lighter than the decoder's (small receptive field, no dendrites), so the rayon fan-out + per-column work-list allocation *regressed* the learn hot path (mid_3L 1.27 ms → 1.42 ms, +12 %). Left sequential deliberately. The earlier "clean follow-on" note was optimistic; the measurement corrected it.
  - **The Actor stays sequential.** `actor.learn_column` is likewise column-disjoint (reads only its own column's state + the immutable replay history), so the same technique *would* apply — but it touches six mutable per-column arrays over a TD replay loop, runs only for **Action** IOs (off the prediction hot path), and has no benchmark/golden to prove a parallel port stays bit-exact or actually wins. Not worth the risk until RL throughput is an explicit goal.
- **Serialization.** Field-by-field little-endian via `VecWriter` / `SliceReader` (magic `"AOGM"` + version), **not** raw C++ struct binary.
- **RNG.** Custom thread-local PCG32 with deterministic per-column subseeds — a reproducibility layer over the parallelism.

## Numeric divergences (from the method-level audit)

Surfaced by [MethodFidelity.md](MethodFidelity.md) — the two genuine math differences from `645a54a`:

- **Leaky-softplus `leak` term.** The Rust **decoder** and **actor** use `softplus(act) − leak·softplus(−act)` (`Params::leak`); upstream uses plain `softplus`, which is the `leak = 0.0` case.

  ⚠️ **This crate's default is `leak = 0.01`, so out of the box it does NOT match upstream numerically.** Set it to `0.0` explicitly for parity — `params.leak = 0.0;`.

  An earlier version of this note said the difference was "reconciled at the dcc boundary", because dcc-core's node wrapper overrides the default to `0.0`. That override is real, but it lives in *dcc-core*, not here — so for every other consumer the sentence was false, and false in the one document a reader opens precisely to find out whether they can trust the numbers. What a consumer chooses to default is its business; the crate default is stated plainly above.
- **Exact vs approximate transcendental math.** C++ `645a54a` ships custom fixed-iteration `expf` / `logf` / `sqrtf` / `powf` / `sinf` / `cosf` by default (Taylor series + Quake `rsqrt`); Rust uses exact std math. Results diverge unless the C++ is built with `USE_STD_MATH`.

## Additions NOT in the referenced upstream ⚠

- **Ticked clockwork temporal memory (`ticks_per_update`).** The crate implements per-layer tick gating (`hierarchy.rs` — `ticks` / `ticks_per_update` / `updates`), but upstream `645a54a` has **no `ticks_per_update` or `temporal_horizon`** (0 references) — it carries **recurrence only** (`recurrent_radius` + `feedback_cis_prev`, present in *both*). So on the temporal axis the crate is a **superset**: recurrence (faithful) **plus** ticks (extra).
  - **Status: DEFERRED** (2026-07-04). Kept as-is for now — both mechanisms stay live and can be combined. The keep-as-intentional-extension vs remove-to-match-`645a54a` decision is postponed; **not** an active task.

- **Goal-conditioned step (`LayerDesc::top_feedback`, `Hierarchy::step_with_goal`).** Every layer but the top takes a second visible layer on its decoder carrying feedback from the layer above. The top layer has nothing above it, so `645a54a` gives its decoder a single visible layer — the `else` arm commented "top layer: no feedback second input". Setting `top_feedback` fills that slot from outside instead, with a goal CSDR over the top encoder's hidden columns, which is what makes "get the hierarchy into *this* state" expressible at all. `get_top_hidden_cis` returns a buffer in exactly that form, so a state the hierarchy has actually been in can be replayed as a goal.

  The nearest published upstream is AOgmaNeo's **`ubl3_recurrent`** branch, whose `step` takes `Int_Buffer_View top_feedback_cis` — the same shape. It is not on mainline, and `645a54a` has no such path.

  Three details are decisions rather than transcriptions, because there is nothing to transcribe from:

  - **Learning pairs the goal with the *previous* step's**, matching how feedback from the layer above is paired everywhere else: the decoder is corrected against the goal that was current when it made the prediction. The incoming goal is used for the activate pass only.
  - **The anticipation pass runs at the top layer too** once the slot exists, substituting the layer's own current hidden CIs for the goal, exactly as it does at every other layer. Uniformity was chosen over a special case; there is no reference for the combination either way.
  - **The flag lives on `LayerDesc`, not `Params`,** because it changes decoder arity and so is fixed at `init_random`. It is only meaningful on the topmost layer, and `init_random` asserts rather than ignoring it elsewhere — a silently-ignored structural flag is worse than a panic.

  **Status: RUST-ONLY, and the fidelity harness cannot check it.** The golden fixture is generated from C++ at `645a54a`, which has no goal path, so there is nothing to diff against; `tests/goal_conditioned.rs` carries the whole verification burden. Default is `false`, and `tests/goal_conditioned.rs::default_path_is_bit_identical` pins that claim to two hashes measured against the tree immediately before the feature landed — a real before/after comparison, not a snapshot of current behaviour. `SERIAL_VERSION` went 1 → 2: the flag is structural, cannot be defaulted on read, and version 1 files are rejected rather than guessed at.

## Faithfully ported — confirmed present in upstream `645a54a` (NOT divergences)

Verified in the C++ source: `recurrent_radius`, `recurrent_importance`, `anticipation`, `smoothing`, `td_scale_decay`, `value_range`, `min_steps`, `history_iters`. (These were *not* Rust-only additions — earlier catalogs inferred that from stale upstream **docs**, not the source.)

## Correctly absent — matching upstream

`temporal_horizon`, `policy_clip`, `value_clip`, `trace_decay` — 0 references in upstream `645a54a` (they survive only in stale upstream docs). The crate correctly omits them.

## Not ported / missing vs upstream

- **Module coverage is complete.** Every upstream compute module is ported — `encoder`, `decoder`, `actor`, `image_encoder`, `hierarchy` (+ `helpers`); no upstream module is omitted.
- **`array.h` (`Array` / `Array_View` / `Vec`)** — the upstream's custom container/tensor types are **not ported as types**; they are replaced by Rust-native `Vec<_>` / slices plus `Int2/3/4` and `Float2/3/4` in `helpers.rs`. A structural substitution, not a functional gap.
- ⚠ **Method-level fidelity not exhaustively audited.** The above is a module/type-level comparison against `645a54a`; a per-method pass — does every upstream method / branch have a Rust equivalent, with the same numerics? — has **not** been done. Record any gaps found here.

## Open questions / to verify

- **Ticks provenance + decision** — **deferred** (recorded above); not an active item.
- [x] **Validated** — numeric parity of the ART / decoder / actor math (`u8`/`i8` quantization, stochastic rounding, softplus/symlog) is confirmed: the fidelity harness matches the C++ golden **bit-exact** over 200 steps (integer CSDRs; float acts within ~8e-3). See `fidelity/README.md`.
- [x] **Validated / resolved** — `rayon` column-parallelism does **not** alter results: the fidelity test is bit-identical single- and multi-threaded. The forward pass and `decoder.learn` are parallel (per-column deterministic subseeds make them order-independent). `encoder.learn` was measured under the same technique and left sequential (net-negative — see above); the Actor stays sequential (RL-only, unmeasured). Benchmarks: `benches/sph_benchmarks.rs` (mid learn 2.64ms → 1.27ms, via `decoder.learn`).
