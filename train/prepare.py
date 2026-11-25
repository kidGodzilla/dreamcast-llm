#!/usr/bin/env python3
"""Merge generated data -> tokenizer -> token .bin files with loss masks.

Text format (one conversation):
  <u>hi there<dc>Hello! I'm the Dreamcast.<u>...<dc><look>Lion</look><book>The lion is...<dc>...<e>
Loss is only taken on DC's tokens (everything after <dc> up to the next role marker, plus <e>).

  ../.venv/bin/python prepare.py [--vocab 4096]
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "train" / "out"
ROLE = {"u": "<u>", "dc": "<dc>", "book": "<book>"}
SPECIALS = ["<u>", "<dc>", "<book>", "<e>", "<look>", "</look>"]


def load_conversations():
    convs = []
    judged = DATA / "judged"
    for d in ("pilot", "raw"):
        for f in sorted((DATA / d).glob("*.jsonl")):
            drop = set()
            j = judged / (f.stem + ".json")
            if j.exists():
                drop = set(json.loads(j.read_text()).get("drop", []))
            for i, l in enumerate(open(f)):
                if i not in drop:
                    convs.append(json.loads(l)["c"])
    for f in sorted((DATA / "fb" / "dialogs").glob("*.jsonl")) + sorted((DATA / "gfb" / "chats").glob("*.jsonl")):
        for l in open(f):
            convs.append(json.loads(l)["c"])
    # de-duplicate exact conversations
    seen, out = set(), []
    for c in convs:
        if not all(isinstance(t, list) and len(t) == 2 and t[0] in ROLE and isinstance(t[1], str) for t in c):
            continue
        k = json.dumps(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def render(c):
    return "".join(ROLE[r] + t.strip() for r, t in c) + "<e>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    convs = load_conversations()
    random.Random(0).shuffle(convs)
    texts = [render(c) for c in convs]
    print(f"{len(convs)} conversations, {sum(map(len, texts)) / 1e6:.1f}M chars")

    corpus = OUT / "corpus.txt"
    corpus.write_text("\n".join(texts))
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(OUT / f"tok{args.vocab}"), model_type="bpe",
        vocab_size=args.vocab, user_defined_symbols=SPECIALS, byte_fallback=True,
        character_coverage=1.0, split_digits=True, allow_whitespace_only_pieces=True,
        normalization_rule_name="identity", max_sentence_length=8192, num_threads=8,
        add_dummy_prefix=False,
        pad_id=-1, unk_id=0, bos_id=1, eos_id=2)
    sp = spm.SentencePieceProcessor(model_file=str(OUT / f"tok{args.vocab}.model"))
    ids = {s: sp.piece_to_id(s) for s in SPECIALS}

    toks, mask = [], []
    for t in texts:
        seq = [sp.bos_id()] + sp.encode(t)
        on = False
        for x in seq:
            if x in (ids["<u>"], ids["<book>"]):
                on = False
            m = on or x == ids["<e>"]
            if x == ids["<dc>"]:
                on = True
            toks.append(x)
            mask.append(1 if m else 0)
    toks = np.array(toks, dtype=np.uint16)
    mask = np.array(mask, dtype=np.uint8)
    n_val = len(toks) // 50
    for name, sl in (("train", slice(n_val, None)), ("val", slice(0, n_val))):
        toks[sl].tofile(OUT / f"{name}.tok")
        mask[sl].tofile(OUT / f"{name}.mask")
    print(f"{len(toks) / 1e6:.2f}M tokens ({mask.mean() * 100:.0f}% DC), "
          f"{len(toks) / len(convs):.0f} tokens/conversation")


if __name__ == "__main__":
    main()
