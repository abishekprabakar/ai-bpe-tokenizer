"""Training: hand-computed merge sequences and naive == incremental."""

import random
import unittest

from bpe import train_incremental, train_naive, merge_word
from bpe.trainer import _base_vocab

# Sennrich et al. (2016) toy corpus, without the end-of-word marker.
SENNRICH = {b"low": 5, b"lower": 2, b"newest": 6, b"widest": 3}


def by_bytes(result):
    return [(result.vocab[a], result.vocab[b]) for a, b in result.merges]


class HandComputed(unittest.TestCase):
    """Expected merges worked out on paper; see comments for the pair counts."""

    def test_sennrich_sequence(self):
        # Step 1 pair counts: es 9, st 9, we 8, lo 7, ow 7, ne 6, ew 6, wi/id/de 3, er 2.
        #   tie es/st -> (e, s) wins on bytes.
        # 2: (es, t) 9.  3: lo 7 ties ow 7 -> (l, o).  4: (lo, w) 7.
        # 5: n-e, e-w, w-est all 6 -> (e, w).  6: n-ew 6 vs ew-est 6 -> (ew, est).
        # 7: (n, ewest) 6.  8: low-e 2, e-r 2, w-i 3, i-d 3, d-est 3 -> (d, est).
        expected = [(b"e", b"s"), (b"es", b"t"), (b"l", b"o"), (b"lo", b"w"),
                    (b"e", b"w"), (b"ew", b"est"), (b"n", b"ewest"), (b"d", b"est")]
        for trainer in (train_naive, train_incremental):
            with self.subTest(trainer=trainer.__name__):
                r = trainer(SENNRICH, 8)
                self.assertEqual(by_bytes(r), expected)
                self.assertEqual(r.counts, [9, 9, 7, 7, 6, 6, 6, 3])
                self.assertEqual(r.vocab[256 + 6], b"newest")

    def test_min_frequency_stops_training(self):
        # After 7 merges the best remaining pair has count 3 < 4.
        for trainer in (train_naive, train_incremental):
            self.assertEqual(len(trainer(SENNRICH, 100, min_frequency=4).merges), 7)

    def test_runs_and_overlap(self):
        # "aaaa" x10: (a,a) occurs 3 times per word -> 30; then (aa,aa) -> 10.
        for trainer in (train_naive, train_incremental):
            r = trainer({b"aaaa": 10}, 5)
            self.assertEqual(by_bytes(r), [(b"a", b"a"), (b"aa", b"aa")])
            self.assertEqual(r.counts, [30, 10])
        # "aaa" x3: (a,a) counted twice per word = 6; merged left to right -> [aa, a];
        # then (aa, a) has count 3.
        for trainer in (train_naive, train_incremental):
            r = trainer({b"aaa": 3}, 5)
            self.assertEqual(r.counts, [6, 3])
            self.assertEqual(r.vocab[257], b"aaa")

    def test_merge_word_left_to_right(self):
        self.assertEqual(merge_word([1, 1, 1], (1, 1), 9), [9, 1])
        self.assertEqual(merge_word([1, 1, 1, 1], (1, 1), 9), [9, 9])
        self.assertEqual(merge_word([1, 2, 1, 2, 3], (1, 2), 7), [7, 7, 3])
        self.assertEqual(merge_word([2, 1], (1, 2), 7), [2, 1])

    def test_new_token_is_concatenation(self):
        r = train_incremental(SENNRICH, 8)
        base = _base_vocab()
        for rank, (a, b) in enumerate(r.merges):
            self.assertLess(a, 256 + rank)
            self.assertLess(b, 256 + rank)
            self.assertEqual(r.vocab[256 + rank], r.vocab[a] + r.vocab[b])
        self.assertEqual({i: r.vocab[i] for i in range(256)}, base)


class NaiveEqualsIncremental(unittest.TestCase):
    def test_random_corpora(self):
        rng = random.Random(1234)
        for trial in range(40):
            alphabet = rng.choice([b"ab", b"abc", b"abcde", bytes(range(97, 123)), "aé🚀".encode()])
            counts = {}
            for _ in range(rng.randint(1, 60)):
                w = bytes(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
                counts[w] = counts.get(w, 0) + rng.randint(1, 20)
            n = rng.randint(1, 80)
            mf = rng.choice([1, 2, 3])
            with self.subTest(trial=trial):
                a = train_naive(counts, n, mf)
                b = train_incremental(counts, n, mf)
                self.assertEqual(a.merges, b.merges)
                self.assertEqual(a.counts, b.counts)

    def test_chosen_counts_respect_min_frequency(self):
        # Greedy choice: the chosen count is the max at that step, so it must be >= min_frequency.
        r = train_incremental({b"abracadabra": 7, b"cadabra": 3, b"abba": 11}, 50, min_frequency=2)
        self.assertTrue(all(c >= 2 for c in r.counts))

    def test_empty_inputs(self):
        for trainer in (train_naive, train_incremental):
            self.assertEqual(trainer({}, 10).merges, [])
            self.assertEqual(trainer({b"x": 100}, 10).merges, [])  # no pairs at all
            self.assertEqual(trainer({b"": 5, b"ab": 0}, 10).merges, [])


if __name__ == "__main__":
    unittest.main()
