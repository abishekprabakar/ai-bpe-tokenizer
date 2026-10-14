"""Pre-tokenization: split text into chunks that BPE merges never cross.

GPT-2 style splitting, written for the standard-library ``re`` module (which has
no ``\\p{L}``): ``[^\\W\\d_]`` is "a Unicode letter", ``\\d`` is a Unicode digit.

The one property that matters for a tokenizer is that the split is *lossless*:
concatenating the chunks must give back the input exactly, for every string.
``split`` asserts that instead of trusting the regex.
"""

from __future__ import annotations

import re

# Order matters: regex alternation is first-match, not longest-match.
#   1. English contractions as their own chunk ("'s", "'ll", ...)
#   2. an optional leading space + a run of letters
#   3. an optional leading space + a run of digits
#   4. an optional leading space + a run of anything that is neither space,
#      letter nor digit (punctuation, symbols, emoji, and "_")
#   5. a whitespace run, backing off one char if a word follows, so "a   b"
#      splits as "a", "  ", " b" and the word keeps its leading space
#   6. any remaining whitespace
PATTERN = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)"""
    r"""| ?[^\W\d_]+"""
    r"""| ?\d+"""
    r"""| ?(?:[^\s\w]|_)+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


def split(text: str) -> list[str]:
    """Split ``text`` into pre-tokens. Raises if any character would be lost."""
    chunks: list[str] = []
    pos = 0
    for m in PATTERN.finditer(text):
        if m.start() != pos:
            raise ValueError(f"pre-tokenizer skipped {text[pos:m.start()]!r} at {pos}")
        chunks.append(m.group())
        pos = m.end()
    if pos != len(text):
        raise ValueError(f"pre-tokenizer dropped trailing {text[pos:]!r}")
    return chunks
