"""End-to-end demo: train a byte-level BPE on the procedural corpus and evaluate it.

    python3 demo.py                 # full report (writes output/)
    python3 demo.py --quick         # smaller corpus, same checks
    python3 demo.py --encode "text" # tokenize a string with the saved tokenizer
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from bpe import BPETokenizer, train_incremental, train_naive, word_counts
from bpe.corpus import Language, train_and_heldout

OUT = Path(__file__).resolve().parent / "output"
SPECIALS = ["<|endoftext|>", "<|pad|>"]


def bytes_per_token(tok: BPETokenizer, text: str) -> float:
    return len(text.encode("utf-8")) / max(1, len(tok.encode(text)))


def show(tok: BPETokenizer, text: str) -> str:
    return "|".join(tok.token_str(i) for i in tok.encode(text))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--encode", metavar="TEXT")
    args = ap.parse_args(argv)

    if args.encode is not None:
        tok = BPETokenizer.load(OUT / "tokenizer.json")
        ids = tok.encode(args.encode)
        print(ids)
        print(show(tok, args.encode))
        assert tok.decode(ids) == args.encode
        return 0

    n_train, n_held = (1500, 400) if args.quick else (6000, 1500)
    train, held = train_and_heldout(n_train, n_held)
    print(f"corpus: train {len(train.encode()):,} bytes, held-out {len(held.encode()):,} bytes")

    # 1. naive vs incremental: identical merges, very different cost --------
    counts = word_counts(train, SPECIALS)
    k = 200 if args.quick else 400
    t0 = time.perf_counter(); naive = train_naive(counts, k); t_naive = time.perf_counter() - t0
    t0 = time.perf_counter(); inc = train_incremental(counts, k); t_inc = time.perf_counter() - t0
    same = naive.merges == inc.merges and naive.counts == inc.counts
    print(f"\n[trainers] {len(counts):,} distinct pre-tokens, {k} merges")
    print(f"  naive recount   {t_naive:7.2f} s")
    print(f"  incremental     {t_inc:7.2f} s   ({t_naive / t_inc:.1f}x faster)")
    print(f"  identical merge lists and counts: {same}")
    if not same:
        return 1

    # 2. train the real tokenizer -------------------------------------------
    t0 = time.perf_counter()
    tok = BPETokenizer.train(train, args.vocab, SPECIALS)
    print(f"\n[train] vocab {tok.vocab_size} = 256 bytes + {len(tok.merges)} merges + {len(tok.special)} specials"
          f"  ({time.perf_counter() - t0:.2f} s)")
    first = ", ".join(repr(tok.token_str(256 + i)) for i in range(12))
    print(f"  first merges: {first}")
    longest = sorted(range(256, 256 + len(tok.merges)), key=lambda i: -len(tok.vocab[i]))[:6]
    print(f"  longest tokens: {', '.join(repr(tok.token_str(i)) for i in longest)}")
    # --quick never overwrites the committed artifacts in output/.
    out = Path(tempfile.mkdtemp()) if args.quick else OUT
    out.mkdir(exist_ok=True)
    tok.save(out / "tokenizer.json")
    reloaded = BPETokenizer.load(out / "tokenizer.json")
    print(f"  saved {'(temp) ' if args.quick else 'output/'}tokenizer.json; reload encodes held-out identically: "
          f"{reloaded.encode(held) == tok.encode(held)}")

    # 3. compression vs number of merges (prefixes of one merge list) --------
    print("\n[compression] bytes per token, prefixes of the same merge list")
    print(f"  {'merges':>6}  {'train':>6}  {'held-out':>8}  {'held tokens':>11}")
    rows = []
    for m in [0, 64, 128, 256, 512, 1024, len(tok.merges)]:
        sub = BPETokenizer(tok.merges[:m])
        tr, he = bytes_per_token(sub, train), bytes_per_token(sub, held)
        n_he = len(sub.encode(held))
        rows.append((m, tr, he, n_he))
        print(f"  {m:>6}  {tr:6.3f}  {he:8.3f}  {n_he:>11,}")
    mono = all(a[3] >= b[3] for a, b in zip(rows, rows[1:]))
    print(f"  held-out token count non-increasing in merges: {mono}")
    (out / "compression.tsv").write_text(
        "merges\ttrain_bytes_per_token\theldout_bytes_per_token\theldout_tokens\n"
        + "".join(f"{m}\t{a:.4f}\t{b:.4f}\t{n}\n" for m, a, b, n in rows), encoding="utf-8")

    # 4. generalisation to unseen words --------------------------------------
    train_words = {w for w in counts}
    held_counts = word_counts(held, SPECIALS)
    unseen = [w for w in held_counts if w not in train_words]
    seen = [w for w in held_counts if w in train_words]
    def fert(ws): return sum(len(tok._encode_chunk(w)) for w in ws) / max(1, len(ws))
    print(f"\n[generalisation] held-out pre-tokens never seen in training: {len(unseen):,} of {len(held_counts):,}")
    print(f"  tokens per pre-token: seen {fert(seen):.2f}, unseen {fert(unseen):.2f}, "
          f"raw bytes for unseen {sum(map(len, unseen)) / max(1, len(unseen)):.2f}")
    lang = Language()
    novel = "Un" + lang.roots[3] + "ationful " + "re" + lang.roots[10] + "izement"
    print(f"  novel compound {novel!r}\n    -> {show(tok, novel)}")

    # 5. losslessness and the two encoders -----------------------------------
    samples = [held[:2000], "Zürich → 東京 🚀 naïve façade's €12.50!!",
               "tabs\tand  double  spaces\n\n\ttrail  ", "<|endoftext|>new doc<|pad|><|pad|>"]
    ok_rt = all(tok.decode(tok.encode(s)) == s for s in samples)
    ok_ref = all(tok.encode(s) == tok.encode_reference(s) for s in samples)
    print(f"\n[checks] decode(encode(x)) == x on mixed-script samples: {ok_rt}")
    print(f"  greedy min-rank encoder == merge-replay encoder: {ok_ref}")
    print(f"  special tokens: {show(tok, samples[3])}")
    print(f"  specials disabled: {len(tok.encode(samples[3], allowed_special='none'))} ordinary tokens")
    rare = "\U0001F9EC"  # never in the corpus
    print(f"  emoji seen in training '🚀' -> {[tok.vocab[i] for i in tok.encode('🚀')]}")
    print(f"  emoji never seen '{rare}' -> {[tok.vocab[i] for i in tok.encode(rare)]}")
    return 0 if (ok_rt and ok_ref and mono) else 1


if __name__ == "__main__":
    sys.exit(main())
