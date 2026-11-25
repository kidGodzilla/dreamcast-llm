#!/usr/bin/env python3
"""Grounded generation: every fact comes from a fetched source passage (Wikipedia / StrategyWiki, CC BY-SA).

  python3 grounded.py entries --jobs 8 --model opus    # fact-book entries per passage -> gfb/entries/
  python3 grounded.py chats   --jobs 8 --model sonnet  # DC chats about the passage   -> gfb/chats/

Resumable: one output file per passage.
"""
import argparse, hashlib, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from claude_util import claude, HERE

SRC = HERE.parent / "sources"
OUT = HERE / "gfb"
MAX_CHUNKS = {"wikipedia": 6, "strategywiki": 3}
CHUNK = 2200


def passages():
    out = []
    for site in ("wikipedia", "strategywiki"):
        seen = set()
        for f in sorted((SRC / site).glob("*.json")):
            d = json.loads(f.read_text())
            if d["title"] in seen or len(d["text"]) < 300:
                continue
            seen.add(d["title"])
            paras = [p.strip() for p in d["text"].split("\n") if len(p.strip()) > 40]
            chunks, cur = [], ""
            for p in paras:
                if len(cur) + len(p) > CHUNK and cur:
                    chunks.append(cur)
                    cur = ""
                cur += p + "\n"
            if cur:
                chunks.append(cur)
            for i, c in enumerate(chunks[:MAX_CHUNKS[site]]):
                key = hashlib.sha1(f"{site}:{d['title']}:{i}".encode()).hexdigest()[:12]
                out.append({"key": key, "site": site, "title": d["title"], "url": d["url"], "part": i, "text": c})
    return out


ENTRY_PROMPT = """Below is a passage from {site} about "{title}" (part {part}).
Write 3 to 6 fact-book entries using ONLY facts stated in this passage. Rules:
- Each entry has a short topic title a user might ask about (the game, a character, a place, a feature,
  a tip for a level or boss if it's a strategy guide), and a 2-3 sentence entry, at most 45 words.
- Simple words, plain ASCII. Write as if it is the year 2000-2001: do not mention later events, re-releases,
  sequels or ports after 2001, sales figures, critics, or reception.
- Never add a fact that isn't in the passage. Skip anything you can't state clearly from it.
Output one JSON object per line: {{"t": "<topic>", "e": "<entry>"}}

PASSAGE:
{text}"""

CHAT_PROMPT = """{persona}

## Grounded batch
Below is a passage from {site} about "{title}". Write 8 different conversations where users ask DC about
this subject (casual keyboard typing, varied users) and DC answers from its own memory in its usual voice.
Every fact DC states must come from this passage, simplified. Nothing after 2001. If a user asks
something the passage doesn't cover, DC says it doesn't remember that part. Vary length: 1-4 exchanges.
Output exactly 8 JSONL lines in the usual format.

PASSAGE:
{text}"""


def work(stage, p, model, persona):
    out = OUT / stage / f"{p['key']}.jsonl"
    if out.exists():
        return p["key"], "skip"
    site = "Wikipedia" if p["site"] == "wikipedia" else "StrategyWiki, a game strategy guide"
    if stage == "entries":
        lines = claude(ENTRY_PROMPT.format(**{**p, "site": site}), model=model)
        good = []
        for l in lines:
            try:
                o = json.loads(l)
            except Exception:
                continue
            if o.get("t") and o.get("e") and o["e"].isascii() and len(o["e"].split()) <= 60:
                good.append({"t": o["t"], "e": o["e"], "src": p["title"], "url": p["url"], "site": p["site"]})
    else:
        lines = claude(CHAT_PROMPT.format(persona=persona, **{**p, "site": site}), model=model)
        good = []
        for l in lines:
            try:
                o = json.loads(l)
                c = o["c"]
                if c[0][0] == "u" and c[-1][0] == "dc" and all(t.isascii() for _, t in c):
                    good.append({"c": c, "src": p["title"]})
            except Exception:
                continue
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(g) + "\n" for g in good))
    return p["key"], len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["entries", "chats"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    persona = (HERE / "persona.md").read_text()
    ps = passages()
    if args.limit:
        ps = ps[:args.limit]
    print(f"{len(ps)} passages", flush=True)
    total, consecutive = 0, 0
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(work, args.stage, p, args.model, persona): p for p in ps}
        for f in as_completed(futs):
            try:
                k, n = f.result()
                consecutive = 0
            except Exception as e:
                consecutive += 1
                print(f"{futs[f]['title']} FAIL {str(e)[:120]}", flush=True)
                if consecutive >= 10:
                    print("10 failures in a row, stopping.", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    sys.exit(2)
                continue
            if n != "skip":
                total += n
                print(f"{time.strftime('%H:%M:%S')} {futs[f]['title'][:40]} -> {n} total={total}", flush=True)


if __name__ == "__main__":
    main()
