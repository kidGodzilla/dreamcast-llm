#!/usr/bin/env python3
"""Side-by-side: same prompts, same seeds, sampling like the disc (temp 0.7, top-p 0.9), with lookups.
  ../.venv/bin/python compare.py out-dc7m-v1/dc7m.pt:out-dc7m-v1/tok4096.model out/dc7m-full.pt:out/tok4096.model"""
import json, re, sys
from pathlib import Path
import torch, sentencepiece as spm
from model import Config, Transformer

ROOT = Path(__file__).resolve().parent
PROMPTS = ["hi", "who are you", "what's your favorite game", "how much ram do you have",
           "what is a lion", "who is ryo hazuki", "what is crazy taxi", "tell me about jet set radio",
           "what is the capital of france", "what is a smartphone", "you were discontinued lol",
           "its christmas 2000 and i just got you!!", "tell me a joke", "i'm sad today",
           "you're dumb", "what is blap"]

book = {}
for d in ("data/fb/entries", "data/gfb/entries"):
    for f in (ROOT.parent / d).glob("*.jsonl"):
        for l in open(f):
            o = json.loads(l)
            book.setdefault(o["t"].lower(), o["e"])


def load(spec):
    ck, tk = spec.split(":")
    c = torch.load(ROOT / ck, map_location="cpu")
    m = Transformer(Config(**c["cfg"]))
    m.load_state_dict(c["model"])
    m.eval()
    return m, spm.SentencePieceProcessor(model_file=str(ROOT / tk))


def reply(m, sp, prompt, seed):
    torch.manual_seed(seed)
    I = {s: sp.piece_to_id(s) for s in ("<u>", "<dc>", "<e>", "<book>", "</look>")}
    ctx = [sp.bos_id(), I["<u>"]] + sp.encode(prompt) + [I["<dc>"]]
    out = ""
    for _ in range(2):
        x = torch.tensor([ctx])
        y = m.generate(x, 80, temperature=0.7, top_p=0.9, stop={I["<e>"], I["<u>"], I["</look>"]})
        new = y[0, x.shape[1]:].tolist()
        ctx += new
        text = sp.decode(new)
        mt = re.search(r"<look>(.*?)</look>", text)
        if not mt:
            return out + text.replace("<e>", "").replace("<u>", "").strip()
        entry = book.get(mt.group(1).strip().lower(), "(no entry)")
        out += f"[look: {mt.group(1)}{'' if entry != '(no entry)' else ' -> no entry'}] "
        ctx += [I["<book>"]] + sp.encode(entry) + [I["<dc>"]]
    return out


models = [(s.split(":")[0], *load(s)) for s in sys.argv[1:]]
for p in PROMPTS:
    print(f"\n> {p}")
    for name, m, sp in models:
        for seed in (1, 2):
            print(f"  [{Path(name).stem:9s} s{seed}] {reply(m, sp, p, seed)}")
