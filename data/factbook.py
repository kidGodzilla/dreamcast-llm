#!/usr/bin/env python3
"""DC's fact book: short encyclopedia entries, written as if in the year 2000.

  python3 factbook.py topics  --jobs 8     # stage 1: topic lists per area
  python3 factbook.py entries --jobs 12    # stage 2: 2-3 sentence entries
  python3 factbook.py dialogs --jobs 8     # stage 3: chats where DC looks things up

Resumable: each call writes its own file under fb/.
"""
import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from claude_util import claude, HERE

FB = HERE / "fb"
AREAS = {
    "sega_games": "Sega and Dreamcast games, 1985-2001: titles, characters, places, items",
    "game_history": "video game history up to 2001: consoles, famous games, companies, arcades",
    "animals": "animals kids ask about: mammals, birds, fish, insects, dinosaurs",
    "space": "space: planets, moons, stars, astronauts, missions up to 2000",
    "earth": "geography: countries, capitals, oceans, mountains, rivers, big cities",
    "science": "simple science: weather, the body, electricity, light, sound, magnets, computers",
    "history": "famous history up to 2000: inventions, explorers, ancient civilizations",
    "culture_90s": "late-1990s culture: movies, toys, music styles, TV, technology (no real living people)",
    "food_sports": "foods from around the world, sports and their rules, hobbies",
    "words": "common words and sayings a kid might ask the meaning of",
}
ENTRY_RULES = """Rules for each entry:
- Written as if it is the year 2000. Nothing after 2001 exists.
- 2 or 3 short sentences, at most 45 words, simple common words, plain ASCII.
- Only well-established facts you are highly confident about. If unsure about a detail, leave it out.
- No living real people (except as part of an established company name). No dates more precise than needed."""


def stage_topics(args):
    def one(area, k):
        out = FB / "topics" / f"{area}_{k}.json"
        if out.exists():
            return area, "skip"
        lines = claude(
            f"List 150 distinct encyclopedia topic titles in this area: {AREAS[area]}.\n"
            f"Batch {k}: {'common, well-known topics' if k == 0 else 'less obvious but still well-known topics, different from the obvious ones'}.\n"
            "Output one JSON array of strings on a single line, nothing else.", model=args.model,
            system="Output only JSON.")
        topics = json.loads("".join(lines))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(topics))
        return area, len(topics)
    run_all([(a, k) for a in AREAS for k in range(2)], lambda t: one(*t), args.jobs)


def all_topics():
    seen, out = set(), []
    for f in sorted((FB / "topics").glob("*.json")):
        area = f.stem.rsplit("_", 1)[0]
        for t in json.loads(f.read_text()):
            key = t.lower().strip()
            if key not in seen:
                seen.add(key)
                out.append((area, t))
    return out


def stage_entries(args):
    topics = all_topics()
    batches = [topics[i:i + 25] for i in range(0, len(topics), 25)]

    def one(i):
        out = FB / "entries" / f"{i:05d}.jsonl"
        if out.exists():
            return i, "skip"
        b = batches[i]
        lines = claude(
            f"Write one fact-book entry for each topic below.\n{ENTRY_RULES}\n"
            'Output one JSON object per line: {"t": "<topic>", "e": "<entry>"}\n\nTopics:\n' +
            "\n".join(t for _, t in b), model=args.model)
        good = []
        for l in lines:
            try:
                o = json.loads(l)
                if o.get("t") and o.get("e") and o["e"].isascii() and len(o["e"].split()) <= 60:
                    good.append(o)
            except Exception:
                pass
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(g) + "\n" for g in good))
        return i, len(good)
    run_all(range(len(batches)), one, args.jobs)


def stage_dialogs(args):
    entries = [json.loads(l) for f in sorted((FB / "entries").glob("*.jsonl")) for l in open(f)]
    persona = (HERE / "persona.md").read_text()
    n_calls = args.calls

    def one(i):
        out = FB / "dialogs" / f"{i:05d}.jsonl"
        if out.exists():
            return i, "skip"
        rng = random.Random(i)
        picks = rng.sample(entries, 12)
        book = "\n".join(f"[{e['t']}] {e['e']}" for e in picks)
        lines = claude(
            f"{persona}\n\n## Fact book lookups\n"
            "DC has a fact book on its disc. When the user asks a factual question, DC can look it up by "
            "writing exactly <look>TOPIC</look> as its whole reply. The system then shows the entry as a "
            'turn with role "book". DC then answers in its own voice, in 1-2 sentences, using ONLY what the '
            "book says, and may add a short in-character comment. If the book entry doesn't answer the "
            "question, DC says it doesn't know.\n\n"
            f"Here are 12 fact-book entries:\n{book}\n\n"
            "Write 12 conversations. In 9 of them the user asks (casually, varied typing) something one of the "
            "entries answers, and DC looks it up with the exact topic title. In 2, the user asks something "
            "NOT in these entries; DC looks up a plausible title, and the book turn says exactly \"(no entry)\"; "
            "DC then admits it doesn't know. In 1, the user just chats and DC answers without looking anything up.\n"
            'Format, one per line: {"c":[["u","..."],["dc","<look>TOPIC</look>"],["book","<entry text>"],["dc","..."]]}\n'
            "Output only the 12 JSONL lines.", model=args.model)
        good = []
        for l in lines:
            try:
                o = json.loads(l)
                if isinstance(o.get("c"), list) and o["c"][0][0] == "u" and o["c"][-1][0] == "dc":
                    good.append(o)
            except Exception:
                pass
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(g) + "\n" for g in good))
        return i, len(good)
    run_all(range(n_calls), one, args.jobs)


def run_all(items, fn, jobs):
    total, consecutive = 0, 0
    with ThreadPoolExecutor(jobs) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            try:
                key, n = f.result()
            except Exception as e:
                consecutive += 1
                print(f"{time.strftime('%H:%M:%S')} {futs[f]} FAIL {e}", flush=True)
                if consecutive >= 10:
                    print("10 failures in a row, stopping.", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    sys.exit(2)
                continue
            consecutive = 0
            if n != "skip":
                total += n
                print(f"{time.strftime('%H:%M:%S')} {key} -> {n} total={total}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["topics", "entries", "dialogs"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--calls", type=int, default=1500)
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    {"topics": stage_topics, "entries": stage_entries, "dialogs": stage_dialogs}[args.stage](args)
