//! Local conformance checks for the dcc-core import contract.
//!
//! dcc-core imports this crate as a rev-pinned git dependency and wraps it as a `Node`.
//! That imposes requirements `R1`–`R16`, defined in dcc-core's
//! `docs/claude/third-party-import-pattern.md`; this crate's verdict on each — including
//! the one it **fails** — is in `doc/Conformance.md`.
//!
//! **Why the checks live here rather than in dcc-core.** A violation should fail in the
//! repository that can fix it, at the moment it is introduced. The sibling port
//! `dcc_sparsey` shipped a wasm32 breakage from extraction until 2026-08-12 because
//! nothing here looked, and dcc-core's own build masked it.
//!
//! These checks read `Cargo.toml` and `Cargo.lock` as text via `include_str!`, which
//! keeps the test dependency-free — and is why R3's "commit `Cargo.lock`" matters for
//! more than CI caching.

const CARGO_TOML: &str = include_str!("../Cargo.toml");
const CARGO_LOCK: &str = include_str!("../Cargo.lock");

/// Everything inside `[dependencies]`, stopping at the next top-level table.
fn dependencies_section() -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in CARGO_TOML.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            inside = t == "[dependencies]";
            continue;
        }
        if inside {
            out.push_str(line);
            out.push('\n');
        }
    }
    assert!(!out.is_empty(), "no [dependencies] section found in Cargo.toml");
    out
}

/// R4 — the long-lived algorithm objects must be `Send + Sync`.
///
/// dcc-core's `Node` trait requires it, because the engine executes nodes across
/// dependency levels in parallel. These are the types a wrapper stores as fields.
#[test]
fn r4_algorithm_objects_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<dcc_sph::hierarchy::Hierarchy>();
    assert_send_sync::<dcc_sph::encoder::Encoder>();
    assert_send_sync::<dcc_sph::decoder::Decoder>();
    assert_send_sync::<dcc_sph::actor::Actor>();
    assert_send_sync::<dcc_sph::image_encoder::ImageEncoder>();
}

/// R9 — **this crate FAILS the requirement**, so the mitigation's API is contractual.
///
/// Randomness here is a thread-local process global (`helpers::GLOBAL_STATE`), seeded
/// from a hardcoded constant, and `init_random` takes no seed parameter — so a node's
/// seed never reaches the algorithm. Two consequences, both silent: runs sharing a
/// thread contaminate each other, and variant *order* changes results. An early RL run
/// reported 439 and 377 for two variants that were the same configuration.
///
/// dcc-core works around it in `rl-lab/src/isolate.rs`: a freshly spawned thread per
/// run, never a pool, with the global stream seeded explicitly through the three
/// functions below. **Removing or renaming any of them breaks reproducibility of every
/// published RL number**, and the failure downstream would look like a compile error
/// about a missing function rather than what it is.
///
/// So this test pins them as load-bearing public API. It does not — cannot — assert
/// that the global state is gone; see `doc/Conformance.md` for what fixing R9 properly
/// would take.
#[test]
fn r9_global_rng_mitigation_api_is_intact() {
    // Referenced as function items, so a rename or signature change fails to compile.
    let _seed_from: fn(u64) -> u64 = dcc_sph::helpers::rand_get_state;
    let _set: fn(u64) = dcc_sph::helpers::set_global_state;
    let _get: fn() -> u64 = dcc_sph::helpers::get_global_state;

    // And the round-trip actually works, since the harness depends on it.
    let state = dcc_sph::helpers::rand_get_state(42);
    dcc_sph::helpers::set_global_state(state);
    assert_eq!(
        dcc_sph::helpers::get_global_state(),
        state,
        "R9 mitigation: the global RNG stream must be settable and readable, or \
         dcc-core's rl-lab/src/isolate.rs cannot make a run reproducible."
    );
}

/// R1 — the package dcc-core depends on is library-only.
#[test]
fn r1_no_binary_targets() {
    assert!(
        !CARGO_TOML.contains("[[bin]]"),
        "R1: this package must be library-only. Put applications in a separate crate \
         under a workspace `members` list instead — see doc/Conformance.md."
    );
}

/// R2 — dcc-agnostic: this crate must not depend on dcc-core.
#[test]
fn r2_no_dcc_dependency() {
    for line in dependencies_section().lines() {
        let name = line.split(['=', ' ']).next().unwrap_or("").trim();
        assert!(
            !name.starts_with("dcc_") && !name.starts_with("dcc-"),
            "R2: this crate must not depend on dcc-core or any dcc_* crate, found: {line}"
        );
    }
}

/// R3 — its own workspace root, with a real SPDX license.
///
/// The licence is not incidental here: CC BY-NC-SA 4.0 is **NonCommercial**, matching
/// upstream AOgmaNeo, and that restriction travels into anything linking this crate.
/// ShareAlike requires it of adapted material, and a port is adapted material — so this
/// value is not ours to relax.
#[test]
fn r3_standalone_workspace_root_with_spdx_license() {
    assert!(
        CARGO_TOML.contains("[workspace]"),
        "R3: a [workspace] table (empty, or with `members`) is required so cargo treats \
         this manifest as its own root."
    );

    let license = CARGO_TOML
        .lines()
        .find(|l| l.trim_start().starts_with("license"))
        .expect("R3: [package] must declare a license");
    assert!(
        license.contains("CC-BY-NC-SA-4.0"),
        "R3: license must stay CC-BY-NC-SA-4.0 — ShareAlike requires adapted material to \
         carry upstream AOgmaNeo's licence, and it is NonCommercial. Found: {license}"
    );
}

/// R11 — `pyo3` must not reach a consumer.
///
/// `pyo3-ffi` sets `links = "python"`, and cargo permits at most one package with a
/// given `links` value in an entire graph — so two pyo3 majors are *unresolvable*, not
/// merely duplicated. dcc-core's Python binding links the whole engine graph including
/// this crate.
///
/// This is now an assertion of **absence**, which is the strong form. Until 2026-08-12
/// `pyo3` was an optional dependency here for the two Gymnasium runners, and the test
/// could only check that it stayed optional and off by default — a weaker property,
/// because optionality does not help if any consumer enables the feature, and dcc-core's
/// Python binding builds with every port on.
///
/// The runners moved to `examples-gym/`, a workspace member that nothing depends on. A
/// sibling member never enters a consumer's graph, so the constraint is gone rather than
/// merely dormant, and this test can demand that `pyo3` not appear at all.
///
/// Do not "restore" it as an optional dependency to make something build. Anything
/// needing Python belongs in `examples-gym/` or another member crate.
#[test]
fn r11_pyo3_is_absent_from_the_library() {
    let offending: Vec<&str> = dependencies_section()
        .lines()
        .filter(|l| l.trim_start().starts_with("pyo3"))
        .map(|l| Box::leak(l.to_string().into_boxed_str()) as &str)
        .collect();

    assert!(
        offending.is_empty(),
        "R11: `pyo3` must not be a dependency of the library, optional or otherwise. \
         `pyo3-ffi` sets links = \"python\" and cargo permits exactly one such package \
         per dependency graph, so two majors are UNRESOLVABLE — and dcc-core's Python \
         binding links this crate. Put anything needing Python in `examples-gym/` (a \
         workspace member nothing depends on) instead. Found: {offending:?}"
    );
}

/// R12 — `getrandom` must not be in the dependency graph at all.
///
/// This crate has no `rand` dependency: its randomness is a PCG32 in `helpers`, which
/// is what makes bit-exact integer parity with the AOgmaNeo C++ possible in the first
/// place. So R12 holds trivially today and this test keeps it that way — adding `rand`
/// with default features would pull `rand_core`'s `os_rng` and therefore `getrandom`,
/// which cannot compile for wasm32-unknown-unknown without a backend.
#[test]
fn r12_getrandom_is_absent_from_the_graph() {
    assert!(
        !CARGO_LOCK.contains("name = \"getrandom\""),
        "R12: `getrandom` is in Cargo.lock, so this crate no longer builds for \
         wasm32-unknown-unknown standalone. If a `rand` dependency was just added, give \
         it `default-features = false` — do NOT select a getrandom backend here, which \
         is a binary's decision and would impose one host on every consumer."
    );
}
