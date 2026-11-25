#!/usr/bin/env python3
"""Review each raw/<id>.jsonl batch; write judged/<id>.json = list of indexes to DROP.

  python3 judge.py --jobs 8           # judges every raw batch not yet judged; re-run any time
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from claude_util import claude, HERE

RAW, OUT = HERE / "raw", HERE / "judged"
CRITERIA = """You are reviewing training conversations for a tiny chatbot that plays a Sega Dreamcast
console living around the year 2000 ("DC"). Flag a conversation if ANY of these is true:
1. DC states a factual error (wrong release date, wrong game detail, wrong character, wrong hardware spec,
   wrong real-world fact). Be strict: if you are confident it's wrong, flag it.
2. DC shows knowledge of things after 2001 as if it knows them (e.g. explains what a smartphone is).
3. DC breaks persona: says it is an AI, a language model, a program, or a human.
4. DC gives medical, legal, or financial advice, swears, or is unkind.
5. A DC reply is longer than about 50 words, or uses lists/markdown/emoji.
6. The conversation is incoherent or DC ignores what the user said.
Reply with ONE line of JSON: {"drop": [indexes], "why": {"index": "short reason"}}. Use [] if all are fine."""


def one(path, model):
    out = OUT / (path.stem + ".json")
    if out.exists():
        return path.stem, "skip"
    convs = [json.loads(l) for l in open(path)]
    body = "\n".join(f"{i}: {json.dumps(c['c'])}" for i, c in enumerate(convs))
    lines = claude(f"{CRITERIA}\n\nConversations:\n{body}", model=model, system="Output only JSON.")
    res = json.loads(lines[-1])
    out.write_text(json.dumps(res))
    return path.stem, len(res.get("drop", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    todo = [p for p in sorted(RAW.glob("*.jsonl")) if not (OUT / (p.stem + ".json")).exists()]
    dropped, done, consecutive = 0, 0, 0
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(one, p, args.model): p for p in todo}
        for f in as_completed(futs):
            try:
                k, n = f.result()
                consecutive = 0
            except Exception as e:
                consecutive += 1
                print(f"{futs[f].stem} FAIL {str(e)[:150]}", flush=True)
                if consecutive >= 10:
                    print("10 failures in a row, stopping.", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    sys.exit(2)
                continue
            if n == "skip":
                continue
            done += 1
            dropped += n
            print(f"{time.strftime('%H:%M:%S')} {k} drop={n} batches={done} dropped={dropped}", flush=True)


if __name__ == "__main__":
    main()
