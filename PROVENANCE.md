# Provenance and attribution

**Attribution here is a condition of the license, not a courtesy.** CC BY-NC-SA 4.0 §3(a) makes the notices below required, and §6(a) terminates the grant automatically if they are not met. Do not trim this file.

## Attribution (CC BY-NC-SA 4.0 §3(a))

- **Creator:** Ogma Intelligent Systems Corp — <https://ogma.ai>
- **Copyright notice:** `AOgmaNeo - Copyright (c) 2020-2025 Ogma Intelligent Systems Corp`
- **License:** this material is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**. Full text in [`LICENSE.md`](LICENSE.md); upstream's own notice in [`AOGMANEO_LICENSE.md`](AOGMANEO_LICENSE.md); canonical URI <https://creativecommons.org/licenses/by-nc-sa/4.0/>.
- **Disclaimer of warranties:** the material is provided **WITHOUT ANY WARRANTY**; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See §5 of the license.
- **URI to the licensed material:** <https://github.com/ogmacorp/AOgmaNeo>
- **Modified?** **Yes.** This crate is **Adapted Material**: a Rust port of the AOgmaNeo C++ source at commit `645a54ace656b0ac2476a56a0dac19faacbd87ab`. Structure, module names and algorithm order deliberately mirror the C++ so the two can be read side by side; the differences that remain are recorded in [`doc/Divergences.md`](doc/Divergences.md) and [`doc/MethodFidelity.md`](doc/MethodFidelity.md).
- **Adapter's license (§3(b) ShareAlike):** `CC-BY-NC-SA-4.0` — the same license, as ShareAlike requires.

For any use outside this grant, Ogma asks that you contact them: **licenses@ogmacorp.com**.

## Attribution for the demos (CC BY-NC-SA 4.0 §3(a))

The demos in [`examples/`](examples/) are ported from Ogma's demo repository, and two of its image assets are redistributed verbatim. Both are Adapted Material and Licensed Material respectively, so the same notices apply.

- **Creator:** Ogma Intelligent Systems Corp — <https://ogma.ai>
- **Copyright notice:** `OgmaNeoDemos - Copyright (c) Ogma Intelligent Systems Corp`
- **License:** **CC BY-NC-SA 4.0**, the same as this crate.
- **URI to the licensed material:** <https://github.com/ogmacorp/OgmaNeoDemos>, via the fork and branch actually ported from: <https://github.com/jacobeverist/OgmaNeoDemos/tree/aogmaneo>
- **Modified?** **Yes.** Nine C++/SFML demos were rewritten in Rust against this crate's API and made headless. Each demo's deviations from its source are recorded in [`doc/Demos.md`](doc/Demos.md).
- **Redistributed unmodified:** `assets/racingCollision.png` and `assets/racingCheckpoints.png`, copied from `resources/` in that repository and used by the `car_racing` demo.
- **Adapter's license (§3(b) ShareAlike):** `CC-BY-NC-SA-4.0`.

## What NonCommercial means for you

CC BY-NC-SA 4.0 permits use **for NonCommercial purposes only** — §1(k) defines that as "not primarily intended for or directed towards commercial advantage or monetary compensation".

**This restriction travels.** Anything that links `dcc_sph` inherits it for as long as it does so. That is precisely why dcc-core makes this crate an *optional* dependency behind its `sph` feature: `cargo build --no-default-features` there produces a build that does not include this code and does not carry this restriction.

> Whether ShareAlike reaches a *separate crate that merely depends on* this one turns on the Collection-vs-Adaptation distinction in §1(a)/§1(l), which is genuinely unsettled for software — Creative Commons themselves [discourage using CC licenses for code](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software) for this reason. Nothing here resolves that; it is flagged so a decision to distribute commercially is taken deliberately and with advice, rather than by accident.

## What this crate is

`dcc_sph` implements **Sparse Predictive Hierarchies** — an online machine-learning system with a low compute footprint that learns from streaming data without catastrophic forgetting.

| | |
|---|---|
| **Upstream** | [ogmacorp/AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo) (C++) |
| **Pinned commit** | `645a54ace656b0ac2476a56a0dac19faacbd87ab` |
| **Creator** | Ogma Intelligent Systems Corp |
| **This crate** | Rust port, © 2026 Jacob Everist, licensed CC BY-NC-SA 4.0 as ShareAlike requires |

## Fidelity

`fidelity/` holds the cross-language harness: a C++ generator that emits golden vectors from AOgmaNeo, compared against this port's output. It needs an out-of-tree AOgmaNeo checkout and is not required to build or test the crate — the committed fixture under `tests/fixtures/` is what the parity test reads, and its absence makes that test skip rather than fail. See [`fidelity/README.md`](fidelity/README.md).

## Relationship to dcc-core

Extracted from the [dcc-core](https://github.com/jacobeverist/dcc-core) workspace, which wraps these algorithms as nodes ([`engine/src/nodes/ports/sph/`](https://github.com/jacobeverist/dcc-core/tree/main/engine/src/nodes/ports/sph)) so SPH can be compared head-to-head against other architectures in one network. That wrapping lives entirely on the dcc-core side; this crate knows nothing about it, and must not.
