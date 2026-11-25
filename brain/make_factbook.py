#!/usr/bin/env python3
"""Build the on-disc fact book: one "title<TAB>entry" line per topic, sorted by lowercase title.
  python3 make_factbook.py  -> disc/factbook.txt"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
book = {}
for d in ("data/gfb/entries", "data/fb/entries"):   # sourced entries win over generic ones
    for f in sorted((ROOT / d).glob("*.jsonl")):
        for l in open(f):
            o = json.loads(l)
            t, e = o["t"].strip(), " ".join(o["e"].split())
            if t and e and t.isascii() and e.isascii() and "\t" not in t:
                book.setdefault(t.lower(), (t, e))
out = ROOT / "brain" / "disc"
out.mkdir(exist_ok=True)
lines = [f"{t}\t{e}" for k, (t, e) in sorted(book.items())]
(out / "factbook.txt").write_text("\n".join(lines) + "\n")
print(f"{len(lines)} entries, {sum(map(len, lines)) / 1e6:.2f} MB")
