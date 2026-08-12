# dcc_sph — documentation index

Docs for the `dcc_sph` crate — a Rust port of **SPH** (Sparse Predictive Hierarchies); upstream [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo) (C++, Ogma Intelligent Systems Corp). Attribution and license terms: [`../PROVENANCE.md`](../PROVENANCE.md). The doc-set shape follows dcc-core's [third-party import pattern](https://github.com/jacobeverist/dcc-core/blob/main/docs/claude/third-party-import-pattern.md) → "Crate documentation standard":

| Doc | Purpose |
|-----|---------|
| [Architecture.md](Architecture.md) | The **Rust implementation** architecture (`Hierarchy`; encoder/decoder/actor internals; step loop; recurrence + ticks; rayon). |
| [NameReference.md](NameReference.md) | Terminology glossary. dcc-core maps these onto its canonical vocabulary in the [nomenclature crosswalk](https://github.com/jacobeverist/dcc-core/blob/main/docs/canonical/vocabulary/nomenclature-crosswalk.md). |
| [PortNotes.md](PortNotes.md) | Upstream (C++ `AOgmaNeo`) → Rust name/type mapping. |
| [Divergences.md](Divergences.md) | Architectural/structural differences vs upstream `645a54a` — the fidelity audit (incl. the extra ticks mechanism). |
| [MethodFidelity.md](MethodFidelity.md) | Method-by-method comparison vs upstream C++ (faithful / divergent / missing / added). |
| [UserGuide.md](UserGuide.md) | Concepts, public API, and the RL example. |
| [Tuning.md](Tuning.md) | Parameter descriptions + tuning advice. |
| [Demos.md](Demos.md) | The fifteen demos ported from Ogma's OgmaNeoDemos: what each one exercises, and every deviation from its source. |

See also [`../README.md`](../README.md), [`../CLAUDE.md`](../CLAUDE.md) and [`../PROVENANCE.md`](../PROVENANCE.md). The dcc-core node wrappers live at [`engine/src/nodes/ports/sph/`](https://github.com/jacobeverist/dcc-core/tree/main/engine/src/nodes/ports/sph).
