// Reading learned receptive fields back out of an `Encoder`.
//
// Shared by `enc_vis` and `topo_test`, which both visualise where each hidden
// cell has planted itself in input space.
//
// **This is where the port departs from upstream.** `demos/EncVis.cpp` and
// `demos/Topo_Test_AON.cpp` both do:
//
//     int wi = cc + hidden_size.z * (offset.y + diam * (offset.x + diam * c));
//     float p = vl.means[wi];
//
// — one stored scalar per (cell, receptive-field offset), already a normalised
// position. This crate's `encoder::VisibleLayer` has no `means`. It ports a
// byte-weight ART formulation whose weights are indexed by the *input cell* too:
//
//     wi = cc + hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + size.z * column)))
//
// Weights start near zero (`global_rand() % INIT_WEIGHT_NOISEI`) and only ever
// grow toward 255, for the `in_ci` values a cell has actually won on. So instead
// of reading a scalar there is a whole learned histogram over the input column's
// cells, and the position is its weighted centroid. That carries strictly more
// information than a stored mean — `mass` below distinguishes a cell that has
// genuinely committed from one that has never fired, which upstream cannot tell.

use dcc_sph::encoder::Encoder;
use dcc_sph::helpers::{project, Float2, Int2};
use dcc_sph::hierarchy::{Hierarchy, IoType};

// --- Actor and decoder introspection ---
//
// Everything in this section reads library surface that nothing else in the
// repository reaches. `Hierarchy::get_actor` was never called, so the critic's
// value estimate and the replay buffer's fill level — the two numbers that explain
// *why* an RL run is or is not learning — could not be observed at all. The RL
// demos now report them.

/// What the actor on an Action port currently believes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActorStats {
    /// Mean critic value across the port's columns. Rises as the policy finds
    /// reward and flattens when it stops improving.
    pub mean_value: f32,
    pub max_value: f32,
    pub min_value: f32,
    /// How full the credit-assignment history is. Learning only begins once this
    /// exceeds `actor::Params::min_steps`, so a run that looks dead early is often
    /// just waiting for this to fill.
    pub history_size: usize,
    pub history_capacity: usize,
}

impl ActorStats {
    /// Fraction of the replay buffer in use.
    pub fn history_fill(&self) -> f32 {
        if self.history_capacity == 0 {
            0.0
        } else {
            self.history_size as f32 / self.history_capacity as f32
        }
    }
}

/// Read the actor behind IO port `io`, or `None` if that port has no actor.
///
/// The guard matters: `Hierarchy::get_actor` indexes through `d_indices`, which
/// holds a *decoder* index for a Prediction port and `-1` for a `None` port. Called
/// on either, it would silently read the wrong actor or panic.
pub fn actor_stats(h: &Hierarchy, io: usize) -> Option<ActorStats> {
    if h.get_io_type(io) != IoType::Action {
        return None;
    }

    let actor = h.get_actor(io);
    let values = actor.get_hidden_values();

    if values.is_empty() {
        return None;
    }

    let sum: f32 = values.iter().sum();

    Some(ActorStats {
        mean_value: sum / values.len() as f32,
        max_value: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        min_value: values.iter().copied().fold(f32::INFINITY, f32::min),
        history_size: actor.get_history_size(),
        history_capacity: actor.get_history_capacity(),
    })
}

/// Mean peak softmax probability over a port's columns: how *confident* the
/// prediction is, as opposed to whether it is right.
///
/// At initialisation this sits near `1/z` for a column of `z` cells. It rising is
/// the earliest visible sign that a decoder is learning something, well before any
/// accuracy metric moves.
///
/// Returns `None` for a `None` port, whose activations are empty.
pub fn prediction_confidence(h: &Hierarchy, io: usize) -> Option<f32> {
    let acts = h.get_prediction_acts(io);
    if acts.is_empty() {
        return None;
    }

    let z = h.get_io_size(io).z as usize;
    if z == 0 || acts.len() % z != 0 {
        return None;
    }

    let mut total = 0.0f32;
    let columns = acts.len() / z;
    for c in 0..columns {
        let column = &acts[c * z..(c + 1) * z];
        total += column.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    Some(total / columns as f32)
}

/// The critic's per-column value estimates for an Action port.
///
/// `Hierarchy::get_prediction_values` panics on anything but an Action port, so
/// this guards rather than letting a caller find that out at runtime.
pub fn action_values(h: &Hierarchy, io: usize) -> Option<&[f32]> {
    if h.get_io_type(io) != IoType::Action {
        return None;
    }
    Some(h.get_prediction_values(io))
}

/// Which layers ticked on the most recent step, and whether each is recurrent.
///
/// Only interesting when a demo sets `ticks_per_update > 1`: it shows the clockwork
/// actually gating, which is this crate's own addition over upstream.
pub fn layer_updates(h: &Hierarchy) -> Vec<(bool, bool)> {
    (0..h.get_num_layers())
        .map(|l| (h.get_update(l), h.is_layer_recurrent(l)))
        .collect()
}

/// Weight level at which an input cell counts as learned rather than noise.
///
/// `init_random` seeds every weight from `global_rand() % INIT_WEIGHT_NOISEI`, so
/// untrained weights sit in `0..8`; learning drives the winners to 255. Anything
/// halfway between is unambiguous.
///
/// Thresholding rather than taking a plain weighted centroid matters more than it
/// looks. A 64-cell input column carries ~224 units of uniform init noise spread
/// evenly across it against a single 255-unit spike, so a raw centroid is dragged
/// almost halfway to the middle of the range — it reports roughly 0.36 for a cell
/// trained solely at 0.25. The learned signal has to be separated from the floor
/// before averaging.
const COMMIT_THRESHOLD: u8 = 128;

/// One hidden cell's learned position in input space.
pub struct CellField {
    pub column: usize,
    pub cell: usize,
    /// Weighted centroid over the *learned* cells of each visited visible column,
    /// normalised to `[0, 1]`. For the 1x2 visible layouts these demos use,
    /// `centroids[0]` is x and `centroids[1]` is y. Zero where nothing is learned —
    /// check [`is_committed`](Self::is_committed) before reading.
    pub centroids: Vec<f32>,
    /// Fewest learned input cells found in any visited visible column. Zero means
    /// this hidden cell never won and its centroids are meaningless.
    pub support: usize,
    /// For each visited visible column, the index span from the lowest learned
    /// input cell to the highest, inclusive.
    ///
    /// ART does not require a cell's committed set to be contiguous — weights are
    /// driven to 255 for whatever inputs the cell happened to win on, wherever they
    /// fall. If `span` is much larger than the count of learned cells, the set is
    /// scattered and the centroid is an average over disconnected regions rather
    /// than a position. [`compactness`](Self::compactness) is that ratio.
    pub spans: Vec<usize>,
    /// Learned-cell counts per visited visible column, parallel to `spans`.
    pub counts: Vec<usize>,
}

impl CellField {
    /// True if every visible column in this cell's receptive field has at least one
    /// learned input cell, so the centroids mean something.
    pub fn is_committed(&self) -> bool {
        self.support > 0
    }

    /// How wide a span of input values this cell responds to, as a fraction of the
    /// input column. A cell that has generalised over a range reads higher than one
    /// pinned to a single value.
    pub fn spread(&self) -> usize {
        self.support
    }

    /// Fraction of each cell's index span that is actually learned, averaged over
    /// visible columns: 1.0 for a perfectly contiguous block, near 0 for a set
    /// scattered across the whole input range.
    ///
    /// Read this before trusting [`centroids`](Self::centroids) as a position.
    pub fn compactness(&self) -> f32 {
        if self.spans.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f32;
        for (&span, &count) in self.spans.iter().zip(&self.counts) {
            if span > 0 {
                total += count as f32 / span as f32;
            }
        }
        total / self.spans.len() as f32
    }
}

/// The clamped receptive-field bounds and lower corner for one hidden column,
/// re-deriving exactly what `Encoder`'s kernels compute.
fn field_bounds(e: &Encoder, vli: usize, column: usize) -> (Int2, Int2, Int2) {
    let hidden_size = e.get_hidden_size();
    let vld = e.get_visible_layer_desc(vli);

    let column_pos = Int2::new(
        (column / hidden_size.y as usize) as i32,
        (column % hidden_size.y as usize) as i32,
    );

    let h_to_v = Float2::new(
        vld.size.x as f32 / hidden_size.x as f32,
        vld.size.y as f32 / hidden_size.y as f32,
    );
    let visible_center = project(column_pos, h_to_v);
    let field_lower_bound =
        Int2::new(visible_center.x - vld.radius, visible_center.y - vld.radius);

    (
        field_lower_bound,
        Int2::new(field_lower_bound.x.max(0), field_lower_bound.y.max(0)),
        Int2::new(
            (visible_center.x + vld.radius).min(vld.size.x - 1),
            (visible_center.y + vld.radius).min(vld.size.y - 1),
        ),
    )
}

/// The raw weight profile of one hidden cell over each visible column in its
/// receptive field — one `Vec<u8>` of length `vld.size.z` per visited column.
///
/// This is the uninterpreted truth about what a cell has learned: which input
/// values drove its weights to 255. Everything else in this module is a summary of
/// these numbers, so when a summary looks surprising, print the profile.
pub fn weight_profiles(e: &Encoder, vli: usize, column: usize, cell: usize) -> Vec<Vec<u8>> {
    let hidden_size = e.get_hidden_size();
    let vl = e.get_visible_layer(vli);
    let vld = e.get_visible_layer_desc(vli);

    let diam = vld.radius * 2 + 1;
    let vsize_z = vld.size.z as usize;
    let (field_lower_bound, iter_lower_bound, iter_upper_bound) = field_bounds(e, vli, column);

    let mut out = Vec::new();

    for ix in iter_lower_bound.x..=iter_upper_bound.x {
        for iy in iter_lower_bound.y..=iter_upper_bound.y {
            let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);
            let mut profile = Vec::with_capacity(vsize_z);

            for in_ci in 0..vsize_z {
                let wi = cell
                    + hidden_size.z as usize
                        * (offset.y as usize
                            + diam as usize
                                * (offset.x as usize + diam as usize * (in_ci + vsize_z * column)));
                profile.push(vl.weights[wi]);
            }

            out.push(profile);
        }
    }

    out
}

/// Render a weight profile as a row of ASCII, one character per input cell.
///
/// `#` marks a learned weight, `-` an intermediate one, `.` the untrained floor.
pub fn render_profile(profile: &[u8], width: usize) -> String {
    (0..width)
        .map(|i| {
            let lo = i * profile.len() / width;
            let hi = (((i + 1) * profile.len()).div_ceil(width)).max(lo + 1).min(profile.len());
            let peak = profile[lo..hi].iter().copied().max().unwrap_or(0);
            if peak >= COMMIT_THRESHOLD {
                '#'
            } else if peak >= 32 {
                '-'
            } else {
                '.'
            }
        })
        .collect()
}

/// Decode every hidden cell's receptive field on visible layer `vli`.
///
/// The receptive-field geometry below re-derives exactly what `Encoder`'s own
/// kernels compute — `project` and the clamped field bounds come from
/// `dcc_sph::helpers`, so this cannot drift from the real indexing.
pub fn probe_receptive_fields(e: &Encoder, vli: usize) -> Vec<CellField> {
    let hidden_size = e.get_hidden_size();
    let vl = e.get_visible_layer(vli);
    let vld = e.get_visible_layer_desc(vli);

    let diam = vld.radius * 2 + 1;
    let h_to_v = Float2::new(
        vld.size.x as f32 / hidden_size.x as f32,
        vld.size.y as f32 / hidden_size.y as f32,
    );

    let num_columns = (hidden_size.x * hidden_size.y) as usize;
    let num_cells = hidden_size.z as usize;
    let vsize_z = vld.size.z as usize;

    let mut out = Vec::with_capacity(num_columns * num_cells);

    for c in 0..num_columns {
        let column_pos = Int2::new(
            (c / hidden_size.y as usize) as i32,
            (c % hidden_size.y as usize) as i32,
        );

        let visible_center = project(column_pos, h_to_v);
        let field_lower_bound =
            Int2::new(visible_center.x - vld.radius, visible_center.y - vld.radius);
        let iter_lower_bound = Int2::new(field_lower_bound.x.max(0), field_lower_bound.y.max(0));
        let iter_upper_bound = Int2::new(
            (visible_center.x + vld.radius).min(vld.size.x - 1),
            (visible_center.y + vld.radius).min(vld.size.y - 1),
        );

        for cc in 0..num_cells {
            let mut centroids = Vec::new();
            let mut spans = Vec::new();
            let mut counts = Vec::new();
            let mut support = usize::MAX;

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let offset =
                        Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    // Walk this visible column's weight profile and take the centre
                    // of mass of the cells that have actually been learned.
                    let mut num = 0.0f64;
                    let mut den = 0.0f64;
                    let mut learned = 0usize;
                    let mut lo = usize::MAX;
                    let mut hi = 0usize;

                    for in_ci in 0..vsize_z {
                        let wi = cc
                            + hidden_size.z as usize
                                * (offset.y as usize
                                    + diam as usize
                                        * (offset.x as usize
                                            + diam as usize * (in_ci + vsize_z * c)));

                        let w = vl.weights[wi];
                        if w >= COMMIT_THRESHOLD {
                            num += in_ci as f64 * w as f64;
                            den += w as f64;
                            learned += 1;
                            lo = lo.min(in_ci);
                            hi = hi.max(in_ci);
                        }
                    }

                    support = support.min(learned);
                    counts.push(learned);
                    spans.push(if learned == 0 { 0 } else { hi - lo + 1 });

                    let centroid = if den > 0.0 { (num / den) as f32 } else { 0.0 };
                    centroids.push(centroid / (vsize_z - 1) as f32);
                }
            }

            out.push(CellField {
                column: c,
                cell: cc,
                centroids,
                spans,
                counts,
                support: if support == usize::MAX { 0 } else { support },
            });
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcc_sph::helpers::{rand_get_state, set_global_state, Int3, VisibleLayerDesc};
    use dcc_sph::hierarchy::{IoDesc, LayerDesc};

    /// A hierarchy with one Prediction port, one Action port and one None port, so
    /// every guard below has all three cases to fail on.
    fn mixed_hierarchy() -> Hierarchy {
        set_global_state(rand_get_state(99));
        let io_descs = vec![
            IoDesc { size: Int3::new(1, 1, 8), io_type: IoType::Prediction, ..Default::default() },
            IoDesc { size: Int3::new(1, 1, 4), io_type: IoType::Action, ..Default::default() },
            IoDesc { size: Int3::new(1, 1, 8), io_type: IoType::None, ..Default::default() },
        ];
        let layer_descs = vec![LayerDesc {
            hidden_size: Int3::new(4, 4, 16),
            ..Default::default()
        }];
        let mut h = Hierarchy::new();
        h.init_random(&io_descs, &layer_descs);
        h
    }

    fn step_n(h: &mut Hierarchy, n: usize) {
        let a = vec![0i32; 1];
        for _ in 0..n {
            h.step(&[&a, &a, &a], true, 1.0, 0.0);
        }
    }

    #[test]
    fn actor_stats_only_answers_for_action_ports() {
        let mut h = mixed_hierarchy();
        step_n(&mut h, 4);

        assert!(actor_stats(&h, 0).is_none(), "a Prediction port has no actor");
        assert!(actor_stats(&h, 2).is_none(), "a None port has no actor");
        assert!(actor_stats(&h, 1).is_some(), "the Action port should have one");
    }

    #[test]
    fn history_fills_toward_capacity_and_stops() {
        let mut h = mixed_hierarchy();

        let before = actor_stats(&h, 1).unwrap();
        assert_eq!(before.history_size, 0);
        assert!(before.history_capacity > 0);

        step_n(&mut h, 20);
        let after = actor_stats(&h, 1).unwrap();
        assert!(after.history_size > 0, "history never filled");
        assert!(after.history_size <= after.history_capacity);
        assert!((0.0..=1.0).contains(&after.history_fill()));
    }

    #[test]
    fn actor_values_are_finite_and_ordered() {
        let mut h = mixed_hierarchy();
        step_n(&mut h, 30);
        let s = actor_stats(&h, 1).unwrap();
        assert!(s.mean_value.is_finite() && s.max_value.is_finite() && s.min_value.is_finite());
        assert!(s.min_value <= s.mean_value && s.mean_value <= s.max_value);
    }

    #[test]
    fn prediction_confidence_starts_near_chance_for_the_column_size() {
        let mut h = mixed_hierarchy();
        step_n(&mut h, 1);

        // Port 0 has 8 cells per column, so an untrained softmax sits near 1/8.
        let c = prediction_confidence(&h, 0).expect("Prediction port has activations");
        assert!((0.0..=1.0).contains(&c), "confidence {c} out of range");
        assert!(c > 1.0 / 8.0 * 0.5, "confidence {c} implausibly below chance");

        // A None port produces no activations at all.
        assert!(prediction_confidence(&h, 2).is_none());
    }

    #[test]
    fn action_values_are_guarded_to_action_ports() {
        let mut h = mixed_hierarchy();
        step_n(&mut h, 4);

        assert!(action_values(&h, 1).is_some());
        // Unguarded, `get_prediction_values` panics on these two.
        assert!(action_values(&h, 0).is_none());
        assert!(action_values(&h, 2).is_none());
    }

    #[test]
    fn layer_updates_reports_one_entry_per_layer() {
        let mut h = mixed_hierarchy();
        step_n(&mut h, 2);
        let u = layer_updates(&h);
        assert_eq!(u.len(), h.get_num_layers());
    }

    fn train(steps: usize, target: (f32, f32)) -> Encoder {
        set_global_state(rand_get_state(1));
        let mut e = Encoder::default();
        e.init_random(
            Int3::new(1, 2, 16),
            vec![VisibleLayerDesc { size: Int3::new(1, 2, 64), radius: 2 }],
        );
        let params = dcc_sph::encoder::Params::default();

        // Drive it with one fixed point so the committed cells must land there.
        let inputs = vec![
            (target.0 * 63.0) as i32,
            (target.1 * 63.0) as i32,
        ];
        for _ in 0..steps {
            e.step(&[&inputs], true, &params);
        }
        e
    }

    #[test]
    fn probe_returns_one_field_per_hidden_cell() {
        let e = train(1, (0.5, 0.5));
        let fields = probe_receptive_fields(&e, 0);
        // hidden 1x2x16 -> 2 columns of 16 cells.
        assert_eq!(fields.len(), 2 * 16);
        // Visible layer is 1x2, so each field sees two visible columns.
        assert!(fields.iter().all(|f| f.centroids.len() == 2));
    }

    #[test]
    fn committed_cells_land_on_the_input_they_were_trained_on() {
        let target = (0.25f32, 0.75f32);
        let e = train(200, target);
        let fields = probe_receptive_fields(&e, 0);

        let committed: Vec<&CellField> = fields.iter().filter(|f| f.is_committed()).collect();
        assert!(!committed.is_empty(), "nothing committed after 200 steps");

        for f in committed {
            assert!(
                (f.centroids[0] - target.0).abs() < 0.05,
                "x centroid {} strayed from {}",
                f.centroids[0],
                target.0
            );
            assert!(
                (f.centroids[1] - target.1).abs() < 0.05,
                "y centroid {} strayed from {}",
                f.centroids[1],
                target.1
            );
        }
    }

    #[test]
    fn untrained_cells_are_not_reported_as_committed() {
        let e = train(0, (0.5, 0.5));
        let fields = probe_receptive_fields(&e, 0);
        assert!(
            fields.iter().all(|f| !f.is_committed()),
            "an untrained encoder reported committed cells"
        );
    }
}
