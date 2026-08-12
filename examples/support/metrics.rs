// Machine-readable output from a demo run.
//
// Every demo prints a human-readable report to stdout, and that stays exactly as
// it is — this module is strictly additive. What it adds is a way to get the same
// numbers out *programmatically*, because otherwise the only route is a regex over
// stdout, and CI can then do no better than check the process exited 0. A demo that
// trains to pure noise still exits 0.
//
// The intended consumer is dcc-dashboard, so JSONL is the primary format: one
// self-describing object per line, appendable, tailable, and parseable without
// knowing which demo produced it.
//
//     {"kind":"run","demo":"pusher","seed":12345,"config":{"steps":300000}}
//     {"kind":"sample","step":50000,"metrics":{"reward_ema":0.021}}
//     {"kind":"summary","metrics":{"goals_per_100k":16.0}}
//     {"kind":"verdict","learned":true,"note":"..."}
//
// `--metrics-format csv` emits long format instead — `demo,seed,run,kind,step,
// metric,value`, one row per number — which is what a spreadsheet or a quick
// `awk` wants.
//
// With no `--metrics` flag the recorder is inert: no file, no buffer, and
// `sample()` returns before it touches its arguments.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use crate::support::args::Args;

/// A configuration value attached to a run record.
///
/// Config keys are whatever the demo chose to record; they are written in sorted
/// order so two runs of the same demo produce byte-identical headers.
#[derive(Clone, Debug, PartialEq)]
pub enum MetricValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

impl MetricValue {
    fn to_json(&self) -> String {
        match self {
            MetricValue::Int(v) => v.to_string(),
            MetricValue::Float(v) => json_number(*v),
            MetricValue::Bool(v) => v.to_string(),
            MetricValue::Text(v) => json_string(v),
        }
    }

    fn to_plain(&self) -> String {
        match self {
            MetricValue::Int(v) => v.to_string(),
            MetricValue::Float(v) => format!("{v}"),
            MetricValue::Bool(v) => v.to_string(),
            MetricValue::Text(v) => v.clone(),
        }
    }
}

impl From<i64> for MetricValue {
    fn from(v: i64) -> Self {
        MetricValue::Int(v)
    }
}
impl From<usize> for MetricValue {
    fn from(v: usize) -> Self {
        MetricValue::Int(v as i64)
    }
}
impl From<u64> for MetricValue {
    fn from(v: u64) -> Self {
        MetricValue::Int(v as i64)
    }
}
impl From<i32> for MetricValue {
    fn from(v: i32) -> Self {
        MetricValue::Int(v as i64)
    }
}
impl From<f32> for MetricValue {
    fn from(v: f32) -> Self {
        MetricValue::Float(v as f64)
    }
}
impl From<f64> for MetricValue {
    fn from(v: f64) -> Self {
        MetricValue::Float(v)
    }
}
impl From<bool> for MetricValue {
    fn from(v: bool) -> Self {
        MetricValue::Bool(v)
    }
}
impl From<&str> for MetricValue {
    fn from(v: &str) -> Self {
        MetricValue::Text(v.to_string())
    }
}
impl From<String> for MetricValue {
    fn from(v: String) -> Self {
        MetricValue::Text(v)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Format {
    Jsonl,
    Csv,
}

/// What a demo's `run()` hands back, so a caller can aggregate across runs without
/// knowing what the demo measures.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub metrics: Vec<(String, f64)>,
    pub learned: bool,
    pub note: String,
}

impl Summary {
    pub fn new() -> Self {
        Summary::default()
    }

    pub fn push(&mut self, name: &str, value: f64) {
        self.metrics.push((name.to_string(), value));
    }

    pub fn verdict(&mut self, learned: bool, note: impl Into<String>) {
        self.learned = learned;
        self.note = note.into();
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics.iter().find(|(k, _)| k == name).map(|(_, v)| *v)
    }
}

enum Sink {
    None,
    File(BufWriter<File>),
    /// Used by tests so the schema can be checked without touching the filesystem.
    Buffer(Vec<u8>),
}

impl Sink {
    fn is_none(&self) -> bool {
        matches!(self, Sink::None)
    }

    fn write_line(&mut self, line: &str) {
        match self {
            Sink::None => {}
            Sink::File(f) => {
                // A metrics file that fails to write is worth complaining about, but
                // never worth killing a long training run over.
                if let Err(e) = writeln!(f, "{line}") {
                    eprintln!("warning: could not write metrics: {e}");
                }
            }
            Sink::Buffer(b) => {
                b.extend_from_slice(line.as_bytes());
                b.push(b'\n');
            }
        }
    }
}

/// Writes one run's metric records.
pub struct Recorder {
    demo: &'static str,
    seed: u64,
    /// Distinguishes runs within one process, for `--repeat` and `--sweep`.
    run_index: usize,
    config: BTreeMap<String, MetricValue>,
    sink: Sink,
    format: Format,
    header_written: bool,
}

impl Recorder {
    /// Build a recorder from `--metrics <path>` and `--metrics-format {jsonl,csv}`.
    ///
    /// Without `--metrics` this is inert and costs nothing per call.
    pub fn from_args(demo: &'static str, args: &Args) -> Self {
        let format = match args.str("metrics-format") {
            None | Some("jsonl") => Format::Jsonl,
            Some("csv") => Format::Csv,
            Some(other) => {
                eprintln!("warning: unknown --metrics-format {other:?}; using jsonl");
                Format::Jsonl
            }
        };

        let sink = match args.str("metrics") {
            None => Sink::None,
            Some(path) => match File::create(PathBuf::from(path)) {
                Ok(f) => Sink::File(BufWriter::new(f)),
                Err(e) => {
                    eprintln!("warning: could not open {path} for metrics: {e}");
                    Sink::None
                }
            },
        };

        Recorder {
            demo,
            seed: args.get("seed", 12345u64),
            run_index: 0,
            config: BTreeMap::new(),
            sink,
            format,
            header_written: false,
        }
    }

    /// An inert recorder, for callers that never want output.
    pub fn disabled(demo: &'static str) -> Self {
        Recorder {
            demo,
            seed: 0,
            run_index: 0,
            config: BTreeMap::new(),
            sink: Sink::None,
            format: Format::Jsonl,
            header_written: false,
        }
    }

    /// A recorder writing into memory, for tests.
    pub fn to_buffer(demo: &'static str, seed: u64, format: Format) -> Self {
        Recorder {
            demo,
            seed,
            run_index: 0,
            config: BTreeMap::new(),
            sink: Sink::Buffer(Vec::new()),
            format,
            header_written: false,
        }
    }

    pub fn is_enabled(&self) -> bool {
        !self.sink.is_none()
    }

    /// Begin a new run within this recorder, resetting the per-run header.
    ///
    /// `--repeat` and `--sweep` call this between runs so every record carries the
    /// seed and run index it belongs to.
    pub fn begin_run(&mut self, run_index: usize, seed: u64) {
        self.run_index = run_index;
        self.seed = seed;
        self.config.clear();
        self.header_written = false;
    }

    /// Record a configuration value. Must be called before the first `sample`.
    pub fn config(&mut self, key: &str, value: impl Into<MetricValue>) {
        if self.sink.is_none() {
            return;
        }
        self.config.insert(key.to_string(), value.into());
    }

    fn ensure_header(&mut self) {
        if self.header_written || self.sink.is_none() {
            return;
        }
        self.header_written = true;

        match self.format {
            Format::Jsonl => {
                let mut line = String::from("{\"kind\":\"run\",\"demo\":");
                line.push_str(&json_string(self.demo));
                let _ = write!(line, ",\"seed\":{},\"run\":{},\"config\":{{", self.seed, self.run_index);
                for (i, (k, v)) in self.config.iter().enumerate() {
                    if i > 0 {
                        line.push(',');
                    }
                    line.push_str(&json_string(k));
                    line.push(':');
                    line.push_str(&v.to_json());
                }
                line.push_str("}}");
                self.sink.write_line(&line);
            }
            Format::Csv => {
                if self.run_index == 0 {
                    self.sink.write_line("demo,seed,run,kind,step,metric,value");
                }
                let rows: Vec<String> = self
                    .config
                    .iter()
                    .map(|(k, v)| {
                        format!(
                            "{},{},{},config,,{},{}",
                            csv_field(self.demo),
                            self.seed,
                            self.run_index,
                            csv_field(k),
                            csv_field(&v.to_plain())
                        )
                    })
                    .collect();
                for row in rows {
                    self.sink.write_line(&row);
                }
            }
        }
    }

    /// A periodic observation partway through a run.
    pub fn sample(&mut self, step: u64, metrics: &[(&str, f64)]) {
        if self.sink.is_none() {
            return;
        }
        self.ensure_header();
        self.write_metrics("sample", Some(step), metrics);
    }

    /// The final numbers for a run.
    pub fn summary(&mut self, metrics: &[(&str, f64)]) {
        if self.sink.is_none() {
            return;
        }
        self.ensure_header();
        self.write_metrics("summary", None, metrics);
    }

    /// Record a `Summary` wholesale — its metrics and then its verdict.
    pub fn finish_summary(&mut self, summary: &Summary) {
        if self.sink.is_none() {
            return;
        }
        let pairs: Vec<(&str, f64)> =
            summary.metrics.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        self.summary(&pairs);
        self.verdict(summary.learned, &summary.note);
    }

    /// The demo's own judgement on whether the run learned anything.
    pub fn verdict(&mut self, learned: bool, note: &str) {
        if self.sink.is_none() {
            return;
        }
        self.ensure_header();

        match self.format {
            Format::Jsonl => {
                let line = format!(
                    "{{\"kind\":\"verdict\",\"demo\":{},\"seed\":{},\"run\":{},\"learned\":{},\"note\":{}}}",
                    json_string(self.demo),
                    self.seed,
                    self.run_index,
                    learned,
                    json_string(note)
                );
                self.sink.write_line(&line);
            }
            Format::Csv => {
                let row = format!(
                    "{},{},{},verdict,,learned,{}",
                    csv_field(self.demo),
                    self.seed,
                    self.run_index,
                    if learned { 1 } else { 0 }
                );
                self.sink.write_line(&row);
            }
        }
    }

    fn write_metrics(&mut self, kind: &str, step: Option<u64>, metrics: &[(&str, f64)]) {
        match self.format {
            Format::Jsonl => {
                let mut line = format!(
                    "{{\"kind\":\"{kind}\",\"demo\":{},\"seed\":{},\"run\":{}",
                    json_string(self.demo),
                    self.seed,
                    self.run_index
                );
                if let Some(s) = step {
                    let _ = write!(line, ",\"step\":{s}");
                }
                line.push_str(",\"metrics\":{");
                for (i, (k, v)) in metrics.iter().enumerate() {
                    if i > 0 {
                        line.push(',');
                    }
                    line.push_str(&json_string(k));
                    line.push(':');
                    line.push_str(&json_number(*v));
                }
                line.push_str("}}");
                self.sink.write_line(&line);
            }
            Format::Csv => {
                let step_field = step.map(|s| s.to_string()).unwrap_or_default();
                let rows: Vec<String> = metrics
                    .iter()
                    .map(|(k, v)| {
                        format!(
                            "{},{},{},{kind},{step_field},{},{}",
                            csv_field(self.demo),
                            self.seed,
                            self.run_index,
                            csv_field(k),
                            json_number(*v)
                        )
                    })
                    .collect();
                for row in rows {
                    self.sink.write_line(&row);
                }
            }
        }
    }

    /// Flush and close. Dropping without calling this still flushes, via `BufWriter`.
    pub fn finish(mut self) {
        if let Sink::File(f) = &mut self.sink {
            if let Err(e) = f.flush() {
                eprintln!("warning: could not flush metrics: {e}");
            }
        }
    }

    /// The bytes written so far, for a buffer-backed recorder.
    pub fn buffer(&self) -> Option<&[u8]> {
        match &self.sink {
            Sink::Buffer(b) => Some(b),
            _ => None,
        }
    }
}

/// Format a float as JSON. JSON has no NaN or Infinity, so non-finite values become
/// `null` rather than producing a file that will not parse.
fn json_number(v: f64) -> String {
    if v.is_finite() {
        // `{:?}` round-trips an f64 exactly and never emits an exponent-less
        // integer that would lose the distinction between 1 and 1.0.
        format!("{v:?}")
    } else {
        "null".to_string()
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(rec: &Recorder) -> Vec<String> {
        String::from_utf8(rec.buffer().unwrap().to_vec())
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn disabled_recorder_writes_nothing() {
        let mut rec = Recorder::disabled("demo");
        rec.config("steps", 10usize);
        rec.sample(1, &[("a", 1.0)]);
        rec.summary(&[("a", 1.0)]);
        rec.verdict(true, "fine");
        assert!(!rec.is_enabled());
        assert!(rec.buffer().is_none());
    }

    #[test]
    fn jsonl_emits_run_sample_summary_verdict_in_order() {
        let mut rec = Recorder::to_buffer("pusher", 7, Format::Jsonl);
        rec.config("steps", 100usize);
        rec.sample(50, &[("reward", 0.5)]);
        rec.summary(&[("goals", 3.0)]);
        rec.verdict(true, "learned");

        let l = lines(&rec);
        assert_eq!(l.len(), 4);
        assert!(l[0].starts_with(r#"{"kind":"run","demo":"pusher","seed":7,"run":0"#), "{}", l[0]);
        assert!(l[0].contains(r#""steps":100"#));
        assert!(l[1].contains(r#""kind":"sample""#) && l[1].contains(r#""step":50"#));
        assert!(l[2].contains(r#""kind":"summary""#));
        assert!(l[3].contains(r#""kind":"verdict""#) && l[3].contains(r#""learned":true"#));
    }

    #[test]
    fn header_is_written_once_even_with_many_samples() {
        let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
        rec.config("k", 1usize);
        for i in 0..5 {
            rec.sample(i, &[("m", i as f64)]);
        }
        let runs = lines(&rec).iter().filter(|l| l.contains(r#""kind":"run""#)).count();
        assert_eq!(runs, 1);
    }

    #[test]
    fn begin_run_starts_a_fresh_header_per_run() {
        let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
        rec.config("k", 1usize);
        rec.sample(1, &[("m", 1.0)]);
        rec.begin_run(1, 2);
        rec.config("k", 2usize);
        rec.sample(1, &[("m", 2.0)]);

        let l = lines(&rec);
        let runs: Vec<&String> = l.iter().filter(|s| s.contains(r#""kind":"run""#)).collect();
        assert_eq!(runs.len(), 2);
        assert!(runs[0].contains(r#""seed":1"#) && runs[0].contains(r#""run":0"#));
        assert!(runs[1].contains(r#""seed":2"#) && runs[1].contains(r#""run":1"#));
    }

    #[test]
    fn non_finite_metrics_become_null_so_the_file_still_parses() {
        let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
        rec.sample(0, &[("nan", f64::NAN), ("inf", f64::INFINITY)]);
        let l = lines(&rec);
        assert!(l[1].contains(r#""nan":null"#), "{}", l[1]);
        assert!(l[1].contains(r#""inf":null"#), "{}", l[1]);
    }

    #[test]
    fn strings_are_escaped() {
        assert_eq!(json_string(r#"a"b\c"#), r#""a\"b\\c""#);
        assert_eq!(json_string("tab\there"), r#""tab\there""#);
    }

    #[test]
    fn csv_is_long_format_with_one_header() {
        let mut rec = Recorder::to_buffer("d", 3, Format::Csv);
        rec.config("steps", 5usize);
        rec.sample(1, &[("a", 1.0), ("b", 2.0)]);
        rec.summary(&[("a", 9.0)]);

        let l = lines(&rec);
        assert_eq!(l[0], "demo,seed,run,kind,step,metric,value");
        assert_eq!(l[1], "d,3,0,config,,steps,5");
        assert_eq!(l[2], "d,3,0,sample,1,a,1.0");
        assert_eq!(l[3], "d,3,0,sample,1,b,2.0");
        assert_eq!(l[4], "d,3,0,summary,,a,9.0");
    }

    #[test]
    fn csv_quotes_fields_containing_separators() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn summary_round_trips_through_finish_summary() {
        let mut s = Summary::new();
        s.push("score", 4.5);
        s.verdict(false, "not converged");
        assert_eq!(s.get("score"), Some(4.5));
        assert_eq!(s.get("missing"), None);

        let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
        rec.finish_summary(&s);
        let l = lines(&rec);
        assert!(l.iter().any(|x| x.contains(r#""score":4.5"#)));
        assert!(l.iter().any(|x| x.contains(r#""learned":false"#)));
    }

    #[test]
    fn config_is_sorted_so_two_runs_agree_byte_for_byte() {
        let mut a = Recorder::to_buffer("d", 1, Format::Jsonl);
        a.config("zebra", 1usize);
        a.config("alpha", 2usize);
        a.sample(0, &[]);

        let mut b = Recorder::to_buffer("d", 1, Format::Jsonl);
        b.config("alpha", 2usize);
        b.config("zebra", 1usize);
        b.sample(0, &[]);

        assert_eq!(lines(&a), lines(&b));
    }
}
