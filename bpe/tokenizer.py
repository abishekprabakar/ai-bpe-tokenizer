"""Byte-level BPE tokenizer: encode, decode, special tokens, save/load.

Encoding a pre-token: start from its UTF-8 bytes, then repeatedly find the
adjacent pair with the *lowest merge rank* and merge all its occurrences, until
no adjacent pair has a rank. ``encode_reference`` does the same job a second,
independent way -- replaying every merge in training order -- and the tests
check the two agree. (They must: a pair of rank r is built only from ids below
256 + r, so once the lowest rank present is s > r, no rank-r pair can reappear.
The greedy loop therefore visits ranks in increasing order, exactly like the
replay.)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from . import pretokenize
from .trainer import Pair, TrainResult, merge_word, train_incremental


class BPETokenizer:
    def __init__(self, merges: list[Pair], special_tokens: Iterable[str] = ()):
        self.merges: list[Pair] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: dict[Pair, int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for r, (a, b) in enumerate(self.merges):
            if a not in self.vocab or b not in self.vocab:
                raise ValueError(f"merge {r} ({a}, {b}) refers to an id not yet defined")
            if (a, b) in self.ranks:
                raise ValueError(f"duplicate merge {(a, b)} at rank {r}")
            self.ranks[(a, b)] = r
            self.vocab[256 + r] = self.vocab[a] + self.vocab[b]
        self.special: dict[str, int] = {}
        for tok in special_tokens:
            if tok not in self.special:
                self.special[tok] = len(self.vocab) + len(self.special)
        self.special_inv = {i: t for t, i in self.special.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in sorted(self.special, key=len, reverse=True)) + ")")
            if self.special else None
        )
        self._cache: dict[bytes, tuple[int, ...]] = {}

    # ------------------------------------------------------------------ build
    @classmethod
    def train(cls, text: str | Iterable[str], vocab_size: int, special_tokens: Iterable[str] = (),
              min_frequency: int = 2) -> "BPETokenizer":
        special_tokens = list(special_tokens)
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError("vocab_size must be at least 256 + number of special tokens")
        counts = word_counts(text, special_tokens)
        result: TrainResult = train_incremental(counts, num_merges, min_frequency)
        return cls(result.merges, special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, chunk: bytes) -> tuple[int, ...]:
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit
        ids = list(chunk)
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for p in zip(ids, ids[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            ids = merge_word(ids, best_pair, 256 + best_rank)
        out = tuple(ids)
        if len(self._cache) < 200_000:
            self._cache[chunk] = out
        return out

    def _segments(self, text: str, allowed_special: str | set[str]):
        """Yield (is_special, piece). Specials not allowed are encoded as plain text."""
        if allowed_special == "all":
            allowed = set(self.special)
        elif allowed_special == "none":
            allowed = set()
        else:
            allowed = set(allowed_special) & set(self.special)
        if not allowed or self._special_re is None:
            yield False, text
            return
        pos = 0
        for m in self._special_re.finditer(text):
            if m.group() not in allowed:
                continue
            if m.start() > pos:
                yield False, text[pos:m.start()]
            yield True, m.group()
            pos = m.end()
        if pos < len(text):
            yield False, text[pos:]

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        ids: list[int] = []
        for is_special, piece in self._segments(text, allowed_special):
            if is_special:
                ids.append(self.special[piece])
                continue
            for chunk in pretokenize.split(piece):
                ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode_reference(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        """Slow, independent encoder: replay every merge in rank order."""
        ids: list[int] = []
        for is_special, piece in self._segments(text, allowed_special):
            if is_special:
                ids.append(self.special[piece])
                continue
            for chunk in pretokenize.split(piece):
                w = list(chunk.encode("utf-8"))
                for r, pair in enumerate(self.merges):
                    if len(w) < 2:
                        break
                    w = merge_word(w, pair, 256 + r)
                ids.extend(w)
        return ids

    # ----------------------------------------------------------------- decode
    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.special_inv:
                parts.append(self.special_inv[i].encode("utf-8"))
            else:
                raise KeyError(f"unknown token id {i}")
        return b"".join(parts)

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def token_str(self, i: int) -> str:
        if i in self.special_inv:
            return self.special_inv[i]
        return self.vocab[i].decode("utf-8", errors="backslashreplace")

    # ---------------------------------------------------------------- persist
    def to_dict(self) -> dict:
        return {"format": "byte-bpe/1", "merges": [list(m) for m in self.merges],
                "special_tokens": list(self.special)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=0) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "byte-bpe/1":
            raise ValueError("not a byte-bpe/1 file")
        return cls([tuple(m) for m in d["merges"]], d.get("special_tokens", []))


def word_counts(text: str | Iterable[str], special_tokens: Iterable[str] = ()) -> dict[bytes, int]:
    """Pre-tokenize a corpus and count chunk frequencies (special tokens excluded)."""
    docs = [text] if isinstance(text, str) else list(text)
    specials = list(special_tokens)
    split_re = re.compile("|".join(re.escape(t) for t in specials)) if specials else None
    c: Counter[bytes] = Counter()
    for doc in docs:
        pieces = split_re.split(doc) if split_re else [doc]
        for piece in pieces:
            c.update(ch.encode("utf-8") for ch in pretokenize.split(piece))
    return dict(c)
