"""A seeded procedural corpus, so training and tests need no downloads.

The "language" has real morphology, which is what makes BPE interesting: words
are built as [prefix] + root + [suffix...] from fixed morpheme inventories, and
roots are drawn from a Zipf distribution. A good tokenizer should therefore learn
the morphemes ("un", "ation", " the") as tokens and generalise to held-out words
it never saw whole -- new combinations of known morphemes.

Sentences mix in numbers, punctuation, contractions and a small share of
non-ASCII text (accented Latin, Greek, CJK, emoji) so byte-level fallback is
exercised too.
"""

from __future__ import annotations

import random

ONSETS = ["b", "br", "c", "ch", "d", "dr", "f", "fl", "g", "gr", "h", "k", "l", "m",
          "n", "p", "pl", "qu", "r", "s", "sh", "sk", "st", "t", "th", "tr", "v", "w", "z"]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "ou", "oo", "ie"]
CODAS = ["", "", "n", "r", "l", "st", "nd", "ck", "m", "rt", "sh", "x"]
PREFIXES = ["un", "re", "pre", "dis", "over", "inter", "sub", "trans", "anti", "co"]
SUFFIXES = ["s", "ed", "ing", "er", "ation", "ness", "ly", "able", "ment", "ful", "ize", "ist"]
FUNCTION = ["the", "a", "of", "and", "to", "in", "is", "that", "for", "with", "as", "on",
            "it", "was", "by", "at", "from", "this", "be", "or"]
CONTRACT = ["it's", "don't", "we're", "they've", "I'm", "you'll", "she'd", "can't"]
FOREIGN = ["café", "naïve", "façade", "Zürich", "λόγος", "δεδομένα", "東京", "字节", "😀", "🚀", "→", "€"]


def _make_roots(rng: random.Random, n: int) -> list[str]:
    roots: set[str] = set()
    while len(roots) < n:
        syl = rng.choice([1, 1, 2, 2, 2, 3])
        roots.add("".join(rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS) for _ in range(syl)))
    return sorted(roots)


class Language:
    """Fixed vocabulary (by ``lang_seed``); text is then sampled with any seed."""

    def __init__(self, lang_seed: int = 7, n_roots: int = 600, zipf_s: float = 1.1):
        rng = random.Random(lang_seed)
        self.roots = _make_roots(rng, n_roots)
        rng.shuffle(self.roots)
        self.weights = [1.0 / (k + 1) ** zipf_s for k in range(n_roots)]

    def word(self, rng: random.Random) -> str:
        w = rng.choices(self.roots, self.weights)[0]
        if rng.random() < 0.25:
            w = rng.choice(PREFIXES) + w
        r = rng.random()
        if r < 0.45:
            w += rng.choice(SUFFIXES)
        if r < 0.10:
            w += rng.choice(SUFFIXES)
        return w

    def sentence(self, rng: random.Random) -> str:
        toks: list[str] = []
        for _ in range(rng.randint(5, 16)):
            u = rng.random()
            if u < 0.38:
                toks.append(rng.choice(FUNCTION))
            elif u < 0.42:
                toks.append(rng.choice(CONTRACT))
            elif u < 0.45:
                toks.append(str(rng.choice([rng.randint(0, 99), rng.randint(100, 9999), 2026])))
            elif u < 0.47:
                toks.append(rng.choice(FOREIGN))
            else:
                toks.append(self.word(rng))
        s = " ".join(toks)
        s = s[0].upper() + s[1:]
        if rng.random() < 0.15:
            k = s.find(" ", len(s) // 2)
            if k > 0:
                s = s[:k] + "," + s[k:]
        return s + rng.choice([".", ".", ".", "?", "!", "..."])

    def document(self, seed: int, n_sentences: int) -> str:
        rng = random.Random(seed)
        lines, para = [], []
        for _ in range(n_sentences):
            para.append(self.sentence(rng))
            if rng.random() < 0.2:
                lines.append(" ".join(para))
                para = []
        if para:
            lines.append(" ".join(para))
        return "\n\n".join(lines) + "\n"


def train_and_heldout(n_train: int = 6000, n_heldout: int = 1500, lang_seed: int = 7) -> tuple[str, str]:
    """Same language, disjoint random streams."""
    lang = Language(lang_seed)
    return lang.document(1000 + lang_seed, n_train), lang.document(2000 + lang_seed, n_heldout)
