"""Byte-level BPE training, two ways.

Both trainers learn an ordered merge list from a ``{word_bytes: count}`` table.
Every word starts as its raw bytes (ids 0..255). Each step:

  * count every adjacent pair across all words, weighted by word frequency
    (overlapping occurrences count: "aaa" contributes 2 to (a, a));
  * pick the most frequent pair; ties go to the pair whose byte strings are
    lexicographically smallest, so training is fully deterministic;
  * give it the next id (256, 257, ...) and rewrite every word, merging
    non-overlapping occurrences left to right.

``train_naive`` recounts everything each step: O(total_symbols) per merge. It is
short enough to check by eye and is the reference.

``train_incremental`` keeps the pair counts and an inverted index
pair -> {words containing it}. A merge only touches the words that contain the
chosen pair, so it subtracts those words' old pair counts, rewrites them and adds
the new pair counts back. The best pair comes off a max-heap with lazy deletion:
every time a pair's count changes a fresh entry is pushed, and popped entries
whose count no longer matches are discarded.

The two must produce *identical* merge lists; the test suite checks that on
random corpora.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field

Pair = tuple[int, int]


@dataclass
class TrainResult:
    merges: list[Pair]
    counts: list[int] = field(default_factory=list)  # frequency of each merge when chosen
    vocab: dict[int, bytes] = field(default_factory=dict)


def _base_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(256)}


def merge_word(word: list[int] | tuple[int, ...], pair: Pair, new_id: int) -> list[int]:
    """Replace non-overlapping occurrences of ``pair``, scanning left to right."""
    a, b = pair
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i + 1 < n and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _sort_key(pair: Pair, count: int, vocab: dict[int, bytes]):
    # Smaller key wins: highest count first, then lexicographically smallest bytes.
    return (-count, vocab[pair[0]], vocab[pair[1]])


def train_naive(word_counts: dict[bytes, int], num_merges: int, min_frequency: int = 2) -> TrainResult:
    vocab = _base_vocab()
    words = [(list(w), c) for w, c in word_counts.items() if c > 0 and w]
    merges: list[Pair] = []
    counts: list[int] = []
    for step in range(num_merges):
        pair_counts: dict[Pair, int] = defaultdict(int)
        for w, c in words:
            for p in zip(w, w[1:]):
                pair_counts[p] += c
        if not pair_counts:
            break
        best = min(pair_counts, key=lambda p: _sort_key(p, pair_counts[p], vocab))
        if pair_counts[best] < min_frequency:
            break
        new_id = 256 + step
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merges.append(best)
        counts.append(pair_counts[best])
        words = [(merge_word(w, best, new_id), c) for w, c in words]
    return TrainResult(merges, counts, vocab)


def train_incremental(word_counts: dict[bytes, int], num_merges: int, min_frequency: int = 2) -> TrainResult:
    vocab = _base_vocab()
    words: list[list[int]] = []
    freqs: list[int] = []
    for w, c in word_counts.items():
        if c > 0 and w:
            words.append(list(w))
            freqs.append(c)

    pair_counts: dict[Pair, int] = defaultdict(int)
    where: dict[Pair, set[int]] = defaultdict(set)
    for idx, (w, c) in enumerate(zip(words, freqs)):
        for p in zip(w, w[1:]):
            pair_counts[p] += c
            where[p].add(idx)

    heap: list = []

    def push(p: Pair) -> None:
        cnt = pair_counts.get(p, 0)
        if cnt > 0:
            heapq.heappush(heap, (-cnt, vocab[p[0]], vocab[p[1]], p))

    for p in pair_counts:
        push(p)

    merges: list[Pair] = []
    counts: list[int] = []
    step = 0
    while step < num_merges and heap:
        neg, _, _, best = heapq.heappop(heap)
        if pair_counts.get(best, 0) != -neg:
            continue  # stale entry; a current one was pushed when the count changed
        if -neg < min_frequency:
            break
        new_id = 256 + step
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merges.append(best)
        counts.append(-neg)

        touched: set[Pair] = set()
        for idx in list(where[best]):
            w, c = words[idx], freqs[idx]
            for p in zip(w, w[1:]):
                pair_counts[p] -= c
                touched.add(p)
            neww = merge_word(w, best, new_id)
            words[idx] = neww
            for p in zip(neww, neww[1:]):
                pair_counts[p] += c
                where[p].add(idx)
                touched.add(p)
            # Drop idx from the index of pairs it no longer contains.
            still = set(zip(neww, neww[1:]))
            for p in set(zip(w, w[1:])) - still:
                where[p].discard(idx)
        for p in touched:
            if pair_counts.get(p, 0) <= 0:
                pair_counts.pop(p, None)
                where.pop(p, None)
            else:
                push(p)
        step += 1
    return TrainResult(merges, counts, vocab)
