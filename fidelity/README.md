# dcc_sph functional-fidelity harness

Proves the Rust `dcc_sph` crate reproduces upstream **AOgmaNeo C++** (reference
commit `645a54a`) by running the *same* deterministic scenario on both and diffing
the output. Because both use the **identical PCG32 RNG** (same constants, same default
seed `rand_get_state(12345)`) and the scenario pins the *faithful regime*, the integer
CSDR streams are expected to match **exactly**.

## Result (as committed)

Diffing 200 learning steps of the `wave(t)` scenario:

| Surface | Comparison | Result |
|---|---|---|
| `prediction_cis` (IO-0 predicted CSDR) | exact per step | **200/200 identical** |
| `hidden_cis` (top encoder CSDR) | exact per step | **200/200 identical** |
| `final_prediction_acts` (softmax floats) | tolerance | max abs diff **~8.4e-3** |

The integer coding output is bit-exact with upstream; only the internal float
activations drift at the ~1e-2 level (softmax/transcendental op-ordering), which never
changes the arg-max. This is the strongest fidelity evidence available and confirms
the `MethodFidelity.md` "FAITHFUL" verdicts for the encoder/decoder forward paths in
this regime.

## How it works

Two halves run the identical scenario (single IO `Int3(1,2,16)` prediction, two
`(5,5,64)` layers, the `wave(t) = (t%20==0 || t%7==0)` waveform, `unorm8_to_csdr`
encoding, 200 learn steps):

- **Rust side** — `tests/support/fidelity_scenario.rs` (shared by the
  `fidelity_dump` example and the `fidelity` test).
- **C++ side** — `cpp/generate_golden.cpp`, linked **directly against AOgmaNeo
  `645a54a`**, emits the golden JSON.

The committed golden vector is `tests/fixtures/wave_fidelity_golden.json`;
`cargo test -p dcc_sph --test fidelity` diffs the in-process Rust run against it. The
test **skips gracefully** (prints `SKIP`, passes) if the fixture is absent, so CI needs
no C++ toolchain — the fixture is committed.

## The parity contract (why it's bit-exact)

The scenario deliberately stays in the regime where the Rust port matches `645a54a`
(see `../doc/Divergences.md`):

| Knob | Rust side | C++ side | Why |
|---|---|---|---|
| `leak` | `0.0` (set after `init_random`) | absent in `645a54a` (plain softplus) | crate default 0.01 is a genuine divergence |
| `ticks_per_update` | `1` on every layer | absent in `645a54a` | tick-gating is the deferred Rust-only divergence |
| threading | `RAYON_NUM_THREADS=1` | built **without** `-fopenmp` | C++ learn kernels share one global RNG under OpenMP → not bit-reproducible otherwise |
| transcendentals | std math | `-DUSE_STD_MATH` | `645a54a` ships custom fixed-iteration `expf/logf/...` by default |
| global RNG | `set_global_state(rand_get_state(12345))` | `global_state = rand_get_state(12345)` | same PCG32 stream for weight init |

Note: `cpp/generate_golden.cpp` compiles the AOgmaNeo `.cpp` sources **directly**
(bypassing AOgmaNeo's OpenMP-`REQUIRED` CMake). `PARALLEL_FOR` in `helpers.h` is an
*unconditional* `#pragma omp parallel for`, so omitting `-fopenmp` turns it into a
no-op → sequential → reproducible RNG order.

## Regenerating the golden fixture

Requires a C++14 compiler and an AOgmaNeo checkout at `645a54a`:

```bash
fidelity/build_and_generate.sh /path/to/AOgmaNeo   # or set $AOGMANEO
RAYON_NUM_THREADS=1 cargo test --test fidelity
```

`build_and_generate.sh` compiles the AOgmaNeo sources directly and needs no CMake, so
the above is all you need for the golden fixture.

### Building AOgmaNeo's own CMake project on macOS

You only need this if you want to run the **upstream** project (its examples, or its
CMake build) rather than just generate the fixture. It is recorded because it is not
obvious and cost real time to work out: Apple's clang ships no OpenMP, and AOgmaNeo's
CMake marks OpenMP `REQUIRED`, so the stock toolchain fails.

```bash
# macOS (Apple Silicon)
brew install cmake llvm libomp

export OpenMP_ROOT=$(brew --prefix)/opt/libomp
export CPPFLAGS="-I/opt/homebrew/include -I${OpenMP_ROOT}/include"
export LDFLAGS="-L/opt/homebrew/lib -L${OpenMP_ROOT}/lib"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++

mkdir -p build && cd build && cmake .. && make
```

Note this builds it **with** OpenMP, which is the opposite of what the fidelity
harness wants — parallel C++ learn kernels share one global RNG and stop being
bit-reproducible. Use `build_and_generate.sh` for parity work, and this only for
running upstream on its own terms.

The script warns if the checkout isn't at `645a54a` — the golden vectors are only valid
at that commit. If you need an isolated build, a git worktree of the AOgmaNeo repo works
fine; pass its path as the argument.

## Why not PyAOgmaNeo?

PyAOgmaNeo exists (pybind11, with `set_global_state`/`set_num_threads`) but its local
build is pinned to a *different* commit (`906c958`), and its binding layer was written
against that API — rebuilding it against `645a54a` risks binding-compile breakage
(the version-skew crux). The direct C++ driver links the reference library at the
reference commit with no intermediary, which is both simpler and more faithful.
