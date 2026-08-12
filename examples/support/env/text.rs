// A word corpus for `vsa_char`.
//
// Upstream reads `resources/ts_snippet.txt`, which is **absent from the
// repository**. It is read as raw bytes against a 128-vector alphabet, so any ASCII
// works — but rather than commit someone else's prose, the default is generated: a
// small vocabulary walked by a sparse Markov chain, so next-word prediction is
// genuinely learnable and chance is exactly `1 / vocabulary`. `--text <path>` reads
// a real file instead.
//
// Generating it also makes the difficulty a dial rather than a property of whatever
// text happened to be lying around.

use std::path::Path;

use crate::support::rng::Rng;

/// The alphabet the character vectors are drawn from. Upstream uses 128 (ASCII).
pub const ALPHABET: usize = 128;

pub struct Corpus {
    pub words: Vec<String>,
    /// The sequence actually emitted, as indices into `words`.
    pub sequence: Vec<usize>,
}

impl Corpus {
    /// A vocabulary of random pronounceable-ish words walked by a sparse chain.
    ///
    /// `successors` controls how deterministic the sequence is: 1 makes it a fixed
    /// cycle, larger values make next-word prediction correspondingly harder. Chance
    /// accuracy is `1 / vocab` either way, so the demo's headline number stays
    /// comparable as this is varied.
    pub fn generated(
        vocab: usize,
        word_len: usize,
        successors: usize,
        length: usize,
        rng: &mut Rng,
    ) -> Self {
        assert!(vocab >= 2, "need at least two words");
        assert!(successors >= 1);

        const CONSONANTS: &[u8] = b"bcdfghklmnprstvwz";
        const VOWELS: &[u8] = b"aeiou";

        let mut words: Vec<String> = Vec::with_capacity(vocab);
        while words.len() < vocab {
            let w: String = (0..word_len)
                .map(|i| {
                    let set = if i % 2 == 0 { CONSONANTS } else { VOWELS };
                    set[rng.below(set.len())] as char
                })
                .collect();
            // Duplicates would make the target genuinely ambiguous rather than
            // merely hard, so reject them.
            if !words.contains(&w) {
                words.push(w);
            }
        }

        // Each word gets a small fixed set of possible successors.
        let table: Vec<Vec<usize>> = (0..vocab)
            .map(|_| (0..successors).map(|_| rng.below(vocab)).collect())
            .collect();

        let mut sequence = Vec::with_capacity(length);
        let mut current = rng.below(vocab);
        for _ in 0..length {
            sequence.push(current);
            let opts = &table[current];
            current = opts[rng.below(opts.len())];
        }

        Corpus { words, sequence }
    }

    /// Split a real file into whitespace-separated words.
    pub fn from_file(path: &Path, max_words: usize) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;

        let mut words: Vec<String> = Vec::new();
        let mut sequence: Vec<usize> = Vec::new();

        for token in text.split_whitespace().take(max_words) {
            // Keep only ASCII, since the alphabet is 128 vectors wide.
            let w: String = token.chars().filter(|c| c.is_ascii_graphic()).collect();
            if w.is_empty() {
                continue;
            }
            let idx = match words.iter().position(|x| *x == w) {
                Some(i) => i,
                None => {
                    words.push(w);
                    words.len() - 1
                }
            };
            sequence.push(idx);
        }

        if words.len() < 2 || sequence.len() < 2 {
            return Err(format!("{} yielded too little text", path.display()));
        }

        Ok(Corpus { words, sequence })
    }

    /// Build from `--text <path>` if given, else generate.
    pub fn from_args(args: &crate::support::args::Args, rng: &mut Rng) -> Self {
        match args.str("text") {
            Some(path) => Corpus::from_file(Path::new(path), args.get("max-words", 20_000))
                .unwrap_or_else(|e| panic!("--text {path}: {e}")),
            None => Corpus::generated(
                args.get("vocab", 12),
                args.get("word-len", 4),
                args.get("successors", 2),
                args.get("corpus-length", 4_000),
                rng,
            ),
        }
    }

    pub fn vocab(&self) -> usize {
        self.words.len()
    }

    /// Accuracy a uniform guesser would reach.
    pub fn chance(&self) -> f64 {
        1.0 / self.vocab() as f64
    }

    /// Longest word, which is how many positional vectors are needed.
    pub fn max_word_len(&self) -> usize {
        self.words.iter().map(|w| w.len()).max().unwrap_or(0)
    }

    /// The best next-word accuracy any model could reach on this sequence.
    ///
    /// Measured, not assumed: for each word, how often its single most common
    /// successor actually follows, averaged over the words that appear. A
    /// deterministic chain gives 1.0; two equally likely successors give about 0.5.
    ///
    /// This is what a demo should be judged against. "Above chance" is a weak claim
    /// when the ceiling is 0.5 — the interesting question is whether the model
    /// reached the ceiling, and without this the answer looks like a mediocre 50%.
    pub fn predictability(&self) -> f64 {
        use std::collections::HashMap;

        let mut counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); self.vocab()];
        for pair in self.sequence.windows(2) {
            *counts[pair[0]].entry(pair[1]).or_insert(0) += 1;
        }

        let mut total_seen = 0usize;
        let mut total_best = 0usize;
        for m in &counts {
            if m.is_empty() {
                continue;
            }
            total_seen += m.values().sum::<usize>();
            total_best += *m.values().max().unwrap();
        }

        if total_seen == 0 {
            f64::NAN
        } else {
            total_best as f64 / total_seen as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::Rng;

    #[test]
    fn a_generated_corpus_has_the_requested_shape() {
        let mut rng = Rng::new(1);
        let c = Corpus::generated(10, 4, 2, 500, &mut rng);
        assert_eq!(c.vocab(), 10);
        assert_eq!(c.sequence.len(), 500);
        assert!(c.words.iter().all(|w| w.len() == 4));
        assert!(c.sequence.iter().all(|&i| i < 10));
        assert!((c.chance() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn the_vocabulary_has_no_duplicates() {
        let mut rng = Rng::new(2);
        let c = Corpus::generated(20, 4, 2, 100, &mut rng);
        let mut sorted = c.words.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), c.words.len(), "duplicate words make the target ambiguous");
    }

    #[test]
    fn a_single_successor_makes_the_sequence_deterministic() {
        // With one successor each, the next word is a function of the current one,
        // so the task is fully learnable and the demo has a reachable ceiling.
        let mut rng = Rng::new(3);
        let c = Corpus::generated(8, 4, 1, 400, &mut rng);

        let mut next: Vec<Option<usize>> = vec![None; c.vocab()];
        for pair in c.sequence.windows(2) {
            match next[pair[0]] {
                None => next[pair[0]] = Some(pair[1]),
                Some(n) => assert_eq!(n, pair[1], "successor was not deterministic"),
            }
        }
    }

    #[test]
    fn more_successors_make_the_sequence_less_predictable() {
        let mut rng = Rng::new(4);
        let hard = Corpus::generated(8, 4, 4, 2000, &mut rng);

        // Count how often the most common successor of each word actually follows.
        let mut counts = vec![std::collections::HashMap::new(); hard.vocab()];
        for pair in hard.sequence.windows(2) {
            *counts[pair[0]].entry(pair[1]).or_insert(0usize) += 1;
        }
        let best_rate: f64 = counts
            .iter()
            .filter(|m| !m.is_empty())
            .map(|m| {
                let total: usize = m.values().sum();
                *m.values().max().unwrap() as f64 / total as f64
            })
            .sum::<f64>()
            / hard.vocab() as f64;

        assert!(best_rate < 0.95, "four successors should not be near-deterministic");
    }

    #[test]
    fn predictability_is_one_for_a_deterministic_chain_and_lower_otherwise() {
        let mut rng = Rng::new(5);
        let det = Corpus::generated(8, 4, 1, 2000, &mut rng);
        assert!(
            (det.predictability() - 1.0).abs() < 1e-9,
            "a one-successor chain should be perfectly predictable, got {}",
            det.predictability()
        );

        let two = Corpus::generated(8, 4, 2, 4000, &mut rng);
        let p = two.predictability();
        // Two successors drawn at random sometimes collide, so the ceiling sits at
        // or a little above 0.5 rather than exactly on it.
        assert!((0.45..0.85).contains(&p), "two-successor ceiling was {p}");
        assert!(p > two.chance(), "ceiling should exceed chance");
    }

    #[test]
    fn a_missing_text_file_is_an_error() {
        assert!(Corpus::from_file(Path::new("/nonexistent/corpus.txt"), 100).is_err());
    }
}
