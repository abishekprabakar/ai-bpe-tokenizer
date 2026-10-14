"""Encoding, decoding, pre-tokenization, persistence and special tokens."""

import random
import tempfile
import unittest
from pathlib import Path

from bpe import BPETokenizer, merge_word, word_counts, train_incremental
from bpe import pretokenize
from bpe.corpus import Language, train_and_heldout

SENNRICH = {b"low": 5, b"lower": 2, b"newest": 6, b"widest": 3}


def random_unicode(rng: random.Random, n: int) -> str:
    pools = [
        (0x20, 0x7E), (0x09, 0x0D), (0xC0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF),
        (0x4E00, 0x4FFF), (0x1F300, 0x1F64F), (0x2000, 0x206F), (0x5F, 0x5F), (0x27, 0x27),
    ]
    out = []
    for _ in range(n):
        lo, hi = rng.choice(pools)
        out.append(chr(rng.randint(lo, hi)))
    return "".join(out)


class Shared:
    tok: BPETokenizer
    held: str

    @classmethod
    def build(cls):
        if not hasattr(Shared, "tok"):
            train, held = train_and_heldout(800, 200)
            Shared.tok = BPETokenizer.train(train, 900, ["<|endoftext|>", "<|pad|>"])
            Shared.held = held
        return Shared.tok, Shared.held


class Pretokenize(unittest.TestCase):
    def test_lossless_on_random_unicode(self):
        rng = random.Random(7)
        for _ in range(500):
            s = random_unicode(rng, rng.randint(0, 40))
            self.assertEqual("".join(pretokenize.split(s)), s)

    def test_known_splits(self):
        self.assertEqual(pretokenize.split("Hello, world!"), ["Hello", ",", " world", "!"])
        self.assertEqual(pretokenize.split("it's 42x"), ["it", "'s", " 42", "x"])
        self.assertEqual(pretokenize.split("a   b"), ["a", "  ", " b"])
        self.assertEqual(pretokenize.split("snake_case"), ["snake", "_", "case"])
        self.assertEqual(pretokenize.split("x\n"), ["x", "\n"])


class HandEncoding(unittest.TestCase):
    def setUp(self):
        r = train_incremental(SENNRICH, 8)
        self.tok = BPETokenizer(r.merges)

    def test_lowest(self):
        # l o w e s t -> (e,s)#0 -> (es,t)#1 -> (l,o)#2 -> (lo,w)#3 -> [low, est]; no (low, est) merge.
        ids = self.tok.encode("lowest")
        self.assertEqual(ids, [256 + 3, 256 + 1])
        self.assertEqual([self.tok.vocab[i] for i in ids], [b"low", b"est"])

    def test_training_words_are_single_tokens_when_fully_merged(self):
        self.assertEqual(self.tok.encode("newest"), [256 + 6])
        self.assertEqual(self.tok.encode("low"), [256 + 3])
        # "lower": low e r, (low, e) never merged.
        self.assertEqual(self.tok.encode("lower"), [256 + 3, ord("e"), ord("r")])

    def test_untrained_bytes_pass_through(self):
        self.assertEqual(self.tok.encode("Q"), [ord("Q")])
        self.assertEqual(self.tok.encode("é"), list("é".encode()))


class Trained(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok, cls.held = Shared.build()

    def test_roundtrip_random_unicode(self):
        rng = random.Random(11)
        for _ in range(300):
            s = random_unicode(rng, rng.randint(0, 60))
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_roundtrip_heldout(self):
        self.assertEqual(self.tok.decode(self.tok.encode(self.held)), self.held)

    def test_greedy_equals_replay(self):
        rng = random.Random(3)
        texts = [self.held[:3000]] + [random_unicode(rng, 50) for _ in range(50)]
        lang = Language()
        texts += [lang.sentence(rng) for _ in range(100)]
        for t in texts:
            self.assertEqual(self.tok.encode(t), self.tok.encode_reference(t))

    def test_prefix_extension_and_monotone_length(self):
        # Encoding with merges[:k+1] == apply merge k to the encoding with merges[:k],
        # chunk by chunk; hence token count can never go up as merges are added.
        merges = self.tok.merges
        text = self.held[:1500]
        chunks = [c.encode() for c in pretokenize.split(text)]
        prev_len = sum(map(len, chunks))
        for k in range(0, len(merges), 37):
            sub = BPETokenizer(merges[:k + 1])
            cur_len = sum(len(sub._encode_chunk(c)) for c in chunks)
            self.assertLessEqual(cur_len, prev_len)
            prev_len = cur_len
        # exact one-step extension at a few ranks
        for k in (0, 5, 50, 200, len(merges) - 1):
            a, b = BPETokenizer(merges[:k]), BPETokenizer(merges[:k + 1])
            for c in chunks[:200]:
                self.assertEqual(list(b._encode_chunk(c)),
                                 merge_word(list(a._encode_chunk(c)), merges[k], 256 + k))

    def test_compression_beats_bytes_on_heldout(self):
        n_bytes = len(self.held.encode())
        n_tok = len(self.tok.encode(self.held))
        self.assertLess(n_tok, n_bytes / 2.5)  # ~3.9 bytes/token at this vocab size

    def test_vocab_bookkeeping(self):
        tok = self.tok
        self.assertEqual(tok.vocab_size, 900)
        self.assertEqual(len(tok.merges), 900 - 256 - 2)
        for r, (a, b) in enumerate(tok.merges):
            self.assertEqual(tok.vocab[256 + r], tok.vocab[a] + tok.vocab[b])
        self.assertEqual(len(set(tok.vocab.values())), len(tok.vocab))  # no duplicate strings


class Specials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok, _ = Shared.build()

    def test_atomic(self):
        eot = self.tok.special["<|endoftext|>"]
        pad = self.tok.special["<|pad|>"]
        self.assertEqual((eot, pad), (898, 899))
        ids = self.tok.encode("hi<|endoftext|><|pad|>")
        self.assertEqual(ids[-2:], [eot, pad])
        self.assertEqual(self.tok.decode(ids), "hi<|endoftext|><|pad|>")

    def test_disallowed_specials_are_plain_text(self):
        s = "a<|endoftext|>b"
        ids = self.tok.encode(s, allowed_special="none")
        self.assertNotIn(self.tok.special["<|endoftext|>"], ids)
        self.assertEqual(self.tok.decode(ids), s)
        ids2 = self.tok.encode("<|pad|><|endoftext|>", allowed_special={"<|pad|>"})
        self.assertEqual(ids2[0], self.tok.special["<|pad|>"])
        self.assertNotIn(self.tok.special["<|endoftext|>"], ids2)

    def test_specials_excluded_from_training_counts(self):
        wc = word_counts("x<|endoftext|>y<|endoftext|>", ["<|endoftext|>"])
        self.assertEqual(wc, {b"x": 1, b"y": 1})


class Persistence(unittest.TestCase):
    def test_save_load(self):
        tok, held = Shared.build()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            tok.save(p)
            t2 = BPETokenizer.load(p)
        self.assertEqual(t2.merges, tok.merges)
        self.assertEqual(t2.special, tok.special)
        self.assertEqual(t2.encode(held[:2000]), tok.encode(held[:2000]))

    def test_rejects_bad_merge_lists(self):
        with self.assertRaises(ValueError):
            BPETokenizer([(97, 300)])  # id 300 not defined yet
        with self.assertRaises(ValueError):
            BPETokenizer([(97, 98), (97, 98)])

    def test_unknown_id(self):
        with self.assertRaises(KeyError):
            BPETokenizer([]).decode([5000])


class CorpusDeterminism(unittest.TestCase):
    def test_seeded(self):
        a = train_and_heldout(50, 10)
        b = train_and_heldout(50, 10)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0][:200], a[1][:200])


if __name__ == "__main__":
    unittest.main()
