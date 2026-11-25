#!/usr/bin/env python3
"""Chat with a trained checkpoint on the Mac.  ../.venv/bin/python chat.py --name dc7m"""
import argparse, json, re
from pathlib import Path
import torch, sentencepiece as spm
from model import Config, Transformer

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="dc7m")
ap.add_argument("--temp", type=float, default=0.7)
args = ap.parse_args()

ck = torch.load(ROOT / "out" / f"{args.name}.pt", map_location="cpu")
model = Transformer(Config(**ck["cfg"]))
model.load_state_dict(ck["model"])
model.eval()
sp = spm.SentencePieceProcessor(model_file=str(ROOT / "out" / f"tok{model.c.vocab_size}.model"))
I = {s: sp.piece_to_id(s) for s in ("<u>", "<dc>", "<e>", "<book>", "</look>")}

book = {}
for d in ("fb/entries", "gfb/entries"):
    for f in (ROOT.parent / "data" / d).glob("*.jsonl"):
        for l in open(f):
            o = json.loads(l)
            book.setdefault(o["t"].lower(), o["e"])
print(f"{args.name} (val {ck['val']:.3f}), fact book {len(book)} entries. Ctrl-C to quit.\n")

ctx = [sp.bos_id()]
while True:
    u = input("> ")
    ctx += [I["<u>"]] + sp.encode(u) + [I["<dc>"]]
    for _ in range(2):  # allow one lookup
        x = torch.tensor([ctx[-model.c.max_seq_len + 80:]])
        y = model.generate(x, 80, temperature=args.temp, stop={I["<e>"], I["<u>"], I["</look>"]})
        new = y[0, x.shape[1]:].tolist()
        ctx += [t for t in new if t not in (I["<e>"], I["<u>"])]
        text = sp.decode(new)
        m = re.search(r"<look>(.*?)</look>", text)
        if not m:
            print("DC:", text.replace("<e>", "").replace("<u>", "").strip(), "\n")
            break
        entry = book.get(m.group(1).strip().lower(), "(no entry)")
        print(f"   [looks up {m.group(1)!r}: {entry[:70]}]")
        ctx += [I["<book>"]] + sp.encode(entry) + [I["<dc>"]]
