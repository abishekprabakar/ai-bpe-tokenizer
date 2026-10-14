# ai-bpe-tokenizer

Byte-level BPE tokenizer built from the merge algorithm up, in pure Python
(standard library only). It trains a merge list two independent ways, encodes two
independent ways, and the test suite checks each pair agrees, along with merge
sequences worked out by hand.

## What's in it

| File | What it does |
|---|---|
| `bpe/pretokenize.py` | GPT-2-style pre-tokenizer for stdlib `re` (`[^\W\d_]` stands in for `\p{L}`). It checks that no character is dropped. |
| `bpe/trainer.py` | `train_naive` recounts every pair on every step. `train_incremental` keeps pair counts, an inverted index pair → words and a lazy-deletion max-heap, so each merge only touches the words that contain the chosen pair. |
| `bpe/tokenizer.py` | `BPETokenizer`: greedy lowest-rank encoding with a cache, a merge-replay reference encoder, lossless byte decoding, atomic special tokens (allowed / disallowed per call) and JSON save/load. |
| `bpe/corpus.py` | A seeded procedural language with prefixes, Zipf-distributed roots, suffixes, numbers, contractions and some non-ASCII text, so nothing is downloaded. |
| `demo.py` | Trains a 2,048-token vocabulary, times the two trainers, sweeps compression against merge count, tests generalisation to unseen words, and runs the round-trip and encoder-equivalence checks. |
| `tests/` | 26 `unittest` tests. Nothing to install: `pytest` also picks them up. |

### The mechanism

* **Training.** Every pre-token starts as its UTF-8 bytes (ids 0–255). Each step
  counts adjacent pairs weighted by word frequency. Overlapping occurrences count,
  so `aaa` gives (a,a) a count of 2. The most frequent pair becomes id `256 + step`.
  Ties go to the pair whose byte strings sort first, which makes training fully
  deterministic. Training stops at the target vocabulary size or when the best
  count drops below `min_frequency`.
* **Encoding.** The tokenizer repeatedly merges the adjacent pair with the lowest
  rank. The replay encoder applies every merge in training order instead, and the
  two must match. A rank-*r* pair is made only from ids below `256 + r`, so once the
  lowest rank present is *s > r*, a rank-*r* pair can never reappear. The greedy loop
  therefore visits ranks in increasing order, exactly as replay does.
* **Prefix extension.** Encoding with `merges[:k+1]` gives the same result as
  applying merge *k* to the encoding with `merges[:k]`. A merge can only shorten a
  sequence, so on **any** text the token count never increases as merges are added.
  The tests check both facts directly.

### What the tests derive by hand

* Sennrich et al.'s toy corpus `{low:5, lower:2, newest:6, widest:3}` gives
  exactly `es, est, lo, low, ew, ewest, newest, dest` with counts
  `9, 9, 7, 7, 6, 6, 6, 3`. The comments in `tests/test_training.py` list every tie.
  With that merge list, `lowest` encodes to `[low, est]`.
* `aaaa`×10 gives counts 30 then 10. `aaa`×3 gives 6 then 3, because the left-to-right
  merge leaves `[aa, a]`.
* The naive and incremental trainers produce identical merges and counts on 40
  random corpora, including tiny alphabets that force ties and overlaps.
* `decode(encode(x)) == x` on 300 random mixed-script strings (Latin, Greek,
  Cyrillic, CJK, emoji, control whitespace, `_`, `'`).

Mutation check: each of 9 hand-made bugs makes the suite fail. The bugs were a
reversed tie-break, a skipped count decrement, overlapping merges, a max-rank
encoder, a lost `_` in pre-tokenization, an off-by-one special id, a missing
stale-heap check, an off-by-one `min_frequency`, and specials that ignore
`allowed_special`.

## Run it

```bash
python3 -m unittest discover -s tests -t .   # 26 tests, about 1.3 s
python3 demo.py                              # full report (about 17 s), writes output/
python3 demo.py --quick                      # smaller corpus, leaves output/ untouched
python3 demo.py --encode "The retoungrieshment of Zürich 🚀"
```

Requires Python 3.10+ and nothing else.

## Sample output

From an actual `python3 demo.py` run (Python 3.13, Linux):

```
corpus: train 507,259 bytes, held-out 125,894 bytes

[trainers] 11,294 distinct pre-tokens, 400 merges
  naive recount     10.85 s
  incremental        1.22 s   (8.9x faster)
  identical merge lists and counts: True

[train] vocab 2048 = 256 bytes + 1790 merges + 2 specials  (1.94 s)
  first merges: 'sh', 'ou', ' t', 'ie', 'gr', 'ea', 'ck', 'oun', 'grie', 'in', 'griesh', 'oungriesh'
  longest tokens: ' shouchisttharation', ' intergonraidrairt', ' interbrabiestgrem', ' brabiestgremation', ' δεδομένα', ' brabiestgremness'
  saved output/tokenizer.json; reload encodes held-out identically: True

[compression] bytes per token, prefixes of the same merge list
  merges   train  held-out  held tokens
       0   1.000     1.000      125,894
      64   1.830     1.832       68,705
     128   2.393     2.396       52,548
     256   3.067     3.067       41,042
     512   3.829     3.810       33,042
    1024   4.536     4.519       27,859
    1790   5.075     5.020       25,076
  held-out token count non-increasing in merges: True

[generalisation] held-out pre-tokens never seen in training: 1,912 of 4,064
  tokens per pre-token: seen 1.92, unseen 2.98, raw bytes for unseen 13.01
  novel compound 'Ungeshpeackationful rezoudooizement'
    -> U|n|geshpeack|ation|ful| rezoudoo|izement

[checks] decode(encode(x)) == x on mixed-script samples: True
  greedy min-rank encoder == merge-replay encoder: True
  special tokens: <|endoftext|>|ne|w| d|o|c|<|pad|>|<|pad|>
  specials disabled: 29 ordinary tokens
  emoji seen in training '🚀' -> [b'\xf0\x9f', b'\x9a\x80']
  emoji never seen '🧬' -> [b'\xf0\x9f', b'\xa7', b'\xac']
```

```
$ python3 demo.py --encode "The retoungrieshment of Zürich 🚀"
[1416, 570, 307, 344, 661, 692]
The| retoungriesh|ment| of| Zürich| 🚀
```

Some observations from that run:

* The trainers return identical merge lists. The incremental one is about 7–9× faster (timings vary between runs)
  at 400 merges, and the gap grows with the number of merges, because the naive
  trainer re-scans the whole corpus on every step.
* Held-out bytes per token closely track the training figures up to about 1,000
  merges. After that the gap widens: the last merges are whole training words,
  which the held-out text partly lacks.
* The tokenizer never saw `Ungeshpeackationful` whole, but splits it at morpheme
  boundaries (`ation`, `ful`, `izement`). Pre-tokens absent from training average
  3 tokens instead of 13 raw bytes.
* An emoji absent from training still encodes. It reuses the learned `F0 9F`
  prefix token (common to most emoji) and falls back to raw bytes for the rest.

## License

MIT
