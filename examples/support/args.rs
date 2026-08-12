// Minimal command-line parsing for the demos.
//
// Deliberately dependency-free: the demos take a handful of numeric knobs and a
// couple of flags, which does not justify pulling `clap` into the build.
//
// Accepted forms are `--key value` and `--flag`. A `--key` followed by another
// `--`-prefixed token (or by nothing) is treated as a flag, so `--quiet --steps 10`
// parses the way you would expect.

use std::collections::{HashMap, HashSet};

pub struct Args {
    values: HashMap<String, String>,
    flags: HashSet<String>,
}

impl Args {
    /// Parse `std::env::args()`, skipping the binary name.
    pub fn parse() -> Self {
        Self::from_iter(std::env::args().skip(1))
    }

    pub fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let argv: Vec<String> = iter.into_iter().collect();
        let mut values = HashMap::new();
        let mut flags = HashSet::new();

        let mut i = 0;
        while i < argv.len() {
            let tok = &argv[i];
            if let Some(key) = tok.strip_prefix("--") {
                // `--key=value` is accepted too, since it costs nothing to support.
                if let Some((k, v)) = key.split_once('=') {
                    values.insert(k.to_string(), v.to_string());
                    i += 1;
                    continue;
                }
                match argv.get(i + 1) {
                    Some(next) if !next.starts_with("--") => {
                        values.insert(key.to_string(), next.clone());
                        i += 2;
                    }
                    _ => {
                        flags.insert(key.to_string());
                        i += 1;
                    }
                }
            } else {
                i += 1;
            }
        }

        Args { values, flags }
    }

    pub fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    /// Look up `key`, falling back to `default`. A value that fails to parse is a
    /// hard error rather than a silent fallback — a typo'd `--steps 10O` that
    /// quietly ran the default would be worse than a panic.
    pub fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T
    where
        T::Err: std::fmt::Display,
    {
        match self.values.get(key) {
            None => default,
            Some(raw) => match raw.parse::<T>() {
                Ok(v) => v,
                Err(e) => panic!("--{key}: cannot parse {raw:?}: {e}"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Args;

    fn args(s: &[&str]) -> Args {
        Args::from_iter(s.iter().map(|x| x.to_string()))
    }

    #[test]
    fn parses_values_flags_and_equals_form() {
        let a = args(&["--steps", "500", "--quiet", "--seed=7"]);
        assert_eq!(a.get::<usize>("steps", 1), 500);
        assert_eq!(a.get::<u64>("seed", 1), 7);
        assert!(a.flag("quiet"));
        assert!(!a.flag("steps"));
        assert_eq!(a.get::<usize>("missing", 42), 42);
    }

    #[test]
    fn flag_before_value_does_not_swallow_the_next_key() {
        let a = args(&["--quiet", "--steps", "10"]);
        assert!(a.flag("quiet"));
        assert_eq!(a.get::<usize>("steps", 1), 10);
    }
}
