# Conformance with the dcc-core import contract

dcc-core imports this crate as a rev-pinned git dependency and wraps it as a `Node`. That imposes sixteen requirements, `R1`–`R16`.

**The requirements are defined in dcc-core**, in `docs/claude/third-party-import-pattern.md` under "What the crate itself must satisfy". This file is **this crate's answers**, not a second copy of the questions.

**What is mechanically checked lives in [`../tests/conformance.rs`](../tests/conformance.rs)** and runs on every `cargo test`. The rest is recorded here with rationale.

**This crate is the one that fails a requirement.** R9 is not satisfied and cannot be satisfied without restructuring; what follows is an honest account rather than a clean bill.

## Status

| | Requirement | Verdict | Checked by |
|---|---|---|---|
| R1 | Library-only package | ✅ | `r1_no_binary_targets` |
| R2 | dcc-agnostic — no dcc-core dependency | ✅ | `r2_no_dcc_dependency` |
| R3 | Own workspace root, SPDX license, committed lockfile | ✅ | `r3_standalone_workspace_root_with_spdx_license` |
| R4 | Algorithm object is `Send + Sync` | ✅ | `r4_algorithm_objects_are_send_and_sync` |
| R5 | State expressible as grouped CSDR or sparse index list | ✅ | CSDRs; plus one documented exception, see note |
| R6 | Config types are serde `Serialize + Deserialize + Clone` | n/a | configs are engine-owned; see note |
| R7 | Learned state serializes to bytes | ✅ | `VecWriter` / `SliceReader` |
| R8 | Forward pass separable from update | ⚠️ | mixed; see note |
| R9 | RNG per-object and seed-parameterised, no globals | ❌ | **fails**; mitigation pinned by `r9_global_rng_mitigation_api_is_intact` |
| R10 | Behavior-critical majors match dcc-core | ✅ | `r10_runtime_dependencies_stay_minimal`; see note |
| R11 | No `pyo3` | ⚠️ | `r11_pyo3_stays_optional_and_off_by_default` |
| R12 | `getrandom` absent; wasm32 clean | ✅ | `r12_getrandom_is_absent_from_the_graph`, plus a CI wasm32 build |
| R13 | `json-schema` feature for owned config types | n/a | no owned config types |
| R14 | Builds in isolation under a single feature | ✅ | dcc-core's CI |
| R15 | Node type tag is prefix-identifiable | ✅ | `SPH*`, wrapper-side |
| R16 | Local apps kept out of a consumer's graph | ⚠️ | `r10_runtime_dependencies_stay_minimal`; partly resolved, see note |

## R9 — the failure

Randomness here is a **thread-local process global** (`helpers::GLOBAL_STATE`), seeded from a hardcoded `12345`. `init_random` takes no seed parameter, so although `SPHActorNode` accepts a seed, stores it, and round-trips it through config, **that seed never reaches the algorithm**.

Two consequences, both silent:

- Runs sharing a thread contaminate one another.
- **Variant order changes the answer.** An early RL run reported 439 and 377 for two variants that were the same configuration.

There is no failing test and no compile error. This is why the requirement exists.

**The mitigation is dcc-core's, not this crate's.** `rl-lab/src/isolate.rs` spawns a fresh thread per run — never a pool, because pooled threads are reused and the contaminated state is thread-local — and seeds the stream explicitly through `rand_get_state` / `set_global_state`. A `RunToken` with a private constructor makes the discipline a compile error rather than a convention. For the same reason the driver calls `Network::execute` and never `execute_optimized`.

Those three functions are therefore **load-bearing public API of this crate**, and `r9_global_rng_mitigation_api_is_intact` pins them. Renaming one would break reproducibility of every published RL number, and the downstream failure would read as a missing function rather than as what it is.

It is also why dcc-core declares this crate in `[workspace.dependencies]` rather than per-crate: two consumers with drifting `rev` strings would build two copies, and the harness would seed one global while the engine drew from the other.

**Fixing it properly** means threading a seed through `Encoder`/`Decoder`/`Actor`/`Hierarchy` construction so each object owns its stream — a substantial change that would diverge from AOgmaNeo's structure, which this port deliberately mirrors so `doc/MethodFidelity.md` can be audited method by method. That trade has not been made. A downstream consequence worth knowing: bit-exact parity between the decomposed SPH stack and the monolithic `SPHHierarchyNode` is blocked by exactly this.

## Notes on the other rows that are not a plain yes

**R5 — representation, with a sanctioned exception.** The normal path is CSDRs: flat `Vec<i32>`, one active cell index per column, bridged to a `BitField` by dcc-core's shared grouped-SDR helper. `ImageEncoder` is the exception the requirement explicitly allows — it consumes raw `u8` pixels and its wrapper has no `NodeInput` at all, taking bytes through a `set_image` setter instead.

**R6 / R13 — configs are engine-owned.** Unlike the sibling ports, this crate's parameter types are not embedded in dcc-core's adapter configs; dcc-core defines its own config structs and converts at the boundary. So nothing here needs serde derives for config purposes, and there is no `json-schema` feature to expose. Both routes are sanctioned; this is the deliberate one.

**R8 — forward/update separation: mixed, and this crate has both cases.** `Encoder` and `Decoder` separate a re-runnable deterministic forward pass from the update, so their wrappers use the ordinary `compute()`/`learn()` split — at the cost of a second forward pass per learning tick. `Hierarchy` and `Actor` do forward+learn+tick in one `step` call, so theirs must use the monolithic-step recipe: no-op `compute()`, and override **both** `execute` and `execute_in_thread`. `Decoder` is the awkward one: its `learn()` has to re-activate on the *previous* step's features to restore the dendrite activations the weight update needs, so the wrapper caches the prediction explicitly.

**R10 — no shared runtime dependencies to skew.** This crate's only runtime dependency is `rayon`. It has no `rand`, `serde`, `schemars` or `thiserror` in `[dependencies]` — serde is dev-only, for the fidelity fixtures — so there is nothing to keep in step with dcc-core's majors.

That was prose until the demo suite landed, and prose is not a guard: adding a demo is exactly the kind of change that quietly adds a dependency here. `r10_runtime_dependencies_stay_minimal` now allowlists `[dependencies]` to `rayon` and `pyo3`, so the claim above fails the build rather than the documentation when it stops being true. It is a different shape from the siblings' `r10_*` version tests, because the property being defended is different — they keep majors in step, this keeps the list empty.

**R11 — `pyo3`, off by default.** The two Gymnasium example runners need it, behind the `gymnasium-examples` feature. `pyo3-ffi` sets `links = "python"` and cargo permits exactly one such package per graph, so two majors are *unresolvable*, not merely duplicated — and dcc-core's Python binding links this crate. It is survivable only because the version happens to match, which is luck rather than design. Under R16 these examples belong in a separate crate; that move has not been made.

**R12 — satisfied, and not by accident.** This crate has no `rand` dependency: its randomness is a PCG32 in `helpers`, which is precisely what makes bit-exact integer parity with the AOgmaNeo C++ achievable. So `getrandom` never enters the graph and `cargo check --target wasm32-unknown-unknown` passes standalone. Note `rayon` compiles for wasm32 but has no threads there, degrading to serial — fine for correctness, and CI pins `RAYON_NUM_THREADS=1` anyway so a fidelity failure means "the algorithm changed" rather than "the scheduler differed".

**R16 — half resolved, and the remaining half is `pyo3`.** `examples/` now holds thirteen headless demo runners plus the two Gymnasium ones. Examples themselves are free — a consumer never resolves an external package's dev-dependencies — so `png` and `rapier2d`, which several demos need, are ordinary dev-dependencies and cost nothing. `getrandom` stays out of the lockfile with them present, so R12 is unaffected.

The windowed demo viewer was the case that had to be decided rather than inherited. It needs `macroquad`, a windowing and GL stack, and the obvious route was an optional dependency behind a feature — exactly the shape `pyo3` has. That is what R16 forbids, and the reasoning does not depend on `links`: an optional dependency is still a real `[dependencies]` entry, it lands in the lockfile, it constrains resolution for every consumer, and it is one `default = [...]` edit away from being unavoidable. `macroquad` is milder than `pyo3` only in that two majors would duplicate rather than fail to resolve.

So the viewer lives in **`examples-viz/`**, a workspace member with its own manifest, and `macroquad` appears nowhere in this package. `[workspace] default-members = ["."]` keeps plain `cargo build` and `cargo test` from building it anyway — without that line, listing a member would hand every developer a GL stack on the default path, which is most of what the requirement is trying to prevent.

`pyo3` remains. Moving `cartpole_env_runner` and `lunarlander` into an `examples-gym` crate the same way would remove the last cross-repo version constraint in the whole import; `examples-viz` is a worked example of the move.
