#!/usr/bin/env bash
# Build the AOgmaNeo C++ golden-vector generator against the upstream reference
# commit and write the fixture consumed by `cargo test -p dcc_sph --test fidelity`.
#
# Usage:
#   fidelity/build_and_generate.sh [path-to-AOgmaNeo]
#
# The AOgmaNeo checkout is passed as $1 or via $AOGMANEO. There is deliberately NO
# default: a path that resolves on one machine makes the script look portable when
# it is not.
#
# The upstream checkout MUST be at the reference commit 645a54a (the crate's
# provenance pin). We compile the AOgmaNeo .cpp sources directly with -DUSE_STD_MATH
# and WITHOUT -fopenmp: the `#pragma omp parallel for` (PARALLEL_FOR) is unconditional
# in helpers.h, so omitting -fopenmp turns it into a no-op → fully sequential →
# reproducible RNG order (required for parity with the single-threaded Rust run).
set -euo pipefail

REF_COMMIT="645a54a"
AOGMANEO="${1:-${AOGMANEO:?pass the AOgmaNeo checkout as \$1 or set AOGMANEO}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$AOGMANEO/source/aogmaneo"
OUT_FIXTURE="$HERE/../tests/fixtures/wave_fidelity_golden.json"
BIN="$HERE/cpp/generate_golden"

if [[ ! -d "$SRC" ]]; then
    echo "error: AOgmaNeo source not found at $SRC" >&2
    echo "       pass the AOgmaNeo checkout path as arg 1." >&2
    exit 1
fi

# Verify the reference commit (warn, don't hard-fail, so a detached/worktree checkout
# at the same tree still works).
if command -v git >/dev/null && git -C "$AOGMANEO" rev-parse --short HEAD >/dev/null 2>&1; then
    HAVE="$(git -C "$AOGMANEO" rev-parse --short HEAD)"
    if [[ "$HAVE" != "$REF_COMMIT"* ]]; then
        echo "WARNING: AOgmaNeo is at $HAVE, expected reference $REF_COMMIT." >&2
        echo "         The golden vector is only valid at the reference commit." >&2
    fi
fi

CXX="${CXX:-c++}"
echo "Compiling generator ($CXX, -DUSE_STD_MATH, no OpenMP)..." >&2
"$CXX" -std=c++14 -O2 -DUSE_STD_MATH -Wno-unknown-pragmas \
    -I"$SRC" \
    "$SRC"/*.cpp \
    "$HERE/cpp/generate_golden.cpp" \
    -o "$BIN"

echo "Generating golden fixture → $OUT_FIXTURE" >&2
mkdir -p "$(dirname "$OUT_FIXTURE")"
"$BIN" > "$OUT_FIXTURE"

echo "Done. $(wc -l < "$OUT_FIXTURE") lines written." >&2
echo "Now run: RAYON_NUM_THREADS=1 cargo test -p dcc_sph --test fidelity" >&2
