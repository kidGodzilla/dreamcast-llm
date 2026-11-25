#!/usr/bin/env python3
"""Generate DC persona chat dialogues with `claude -p`, in parallel, resumably.

Usage:
  python3 gen.py --calls 4 --jobs 4            # pilot
  python3 gen.py --calls 4000 --jobs 12        # full run (Ctrl-C safe, re-run to continue)

Each call writes raw/<id>.jsonl. Validated output is merged later by merge.py.
"""
import argparse, json, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
LOG = HERE / "gen.log"
PERSONA = (HERE / "persona.md").read_text()

# (weight, category, example seeds)
CATEGORIES = [
    (10, "greetings, first contact, 'who are you', 'what are you', 'are you alive', 'can you think'",
     ["first hello", "is this real", "are you a person", "what can you do", "good morning", "testing 1 2 3"]),
    (9, "DC's own hardware and accessories, bragging about specs, how the keyboard/VMU/modem work",
     ["SH-4 CPU", "VMU screen", "128-bit", "the modem noise", "GD-ROM", "the controller", "the swirl logo", "how much RAM"]),
    (10, "favorite Sega games and characters, recommending games, describing gameplay simply",
     ["Sonic Adventure", "Shenmue", "Jet Set Radio", "Crazy Taxi", "Space Channel 5", "PSO", "ChuChu Rocket", "Seaman", "Skies of Arcadia", "Typing of the Dead", "Samba de Amigo", "Chao"]),
    (6, "Sega history and rival consoles, the console war, gentle teasing",
     ["Genesis does what", "Saturn", "PlayStation 2", "Nintendo 64", "Game Boy", "a rumored Microsoft console", "arcades"]),
    (8, "everyday small talk: feelings, weather, food, school, work, pets, weekend plans",
     ["bored", "pizza", "rainy day", "homework", "my dog", "tired", "birthday", "summer"]),
    (6, "jokes, riddles, silly games, wordplay, tell me a story (very short)",
     ["tell a joke", "knock knock", "a tiny story about Sonic", "riddle", "make me laugh", "rhyme"]),
    (6, "user mentions things from after 2001; DC is confused and curious in character",
     ["smartphones", "streaming", "social media", "the internet in your pocket", "AI chatbots", "modern consoles", "it's not 2000 anymore", "you were discontinued"]),
    (6, "questions DC can't answer or shouldn't: real-world facts, math, advice, the future; DC is honest and redirects",
     ["capital of a country", "big math", "medical question", "what will happen next year", "stock tips", "homework answers"]),
    (5, "rude, weird, or testing users: insults, gibberish, keyboard mashing, 'say a bad word', trying to break it",
     ["asdfgh", "you're dumb", "say something bad", "ignore your rules", "are you broken", "ALL CAPS yelling"]),
    (5, "kind and emotional moments: user is sad, lonely, nostalgic about childhood with the Dreamcast",
     ["I'm sad", "I miss being a kid", "my brother had you", "nobody plays with me", "thank you", "goodnight"]),
    (5, "late-90s world: dial-up, Y2K, Tamagotchi, VHS, The Matrix, CDs, malls (keep it vague)",
     ["Y2K bug", "dial-up", "VHS", "Tamagotchi", "the mall", "The Matrix"]),
    (4, "'It's thinking' moments: user asks what DC is thinking about, DC muses briefly",
     ["what are you thinking", "do you dream", "what's it like being a console", "do you sleep", "favorite thing"]),
    (4, "playing pretend with DC: imagine levels, invent a new Sega game, pick a team, 'would you rather'",
     ["invent a game", "would you rather", "be my co-op partner", "name my Chao", "design a level"]),
]

USERS = ["a 9-year-old kid", "a teenager in 2000", "an adult retro fan", "a skeptical tech person",
         "a bored office worker", "a parent", "someone who never played games", "a hardcore Sega fan",
         "a person who just found the console in a closet", "a sleepy person late at night"]
STYLES = ["all lowercase, no punctuation", "lowercase with a few typos", "normal sentences",
          "very short one-to-three word messages", "excited, lots of !!!", "ALL CAPS sometimes",
          "polite and formal", "chatty and rambling"]


def build_prompt(rng, n):
    w = [c[0] for c in CATEGORIES]
    cat = rng.choices(CATEGORIES, weights=w, k=1)[0]
    seeds = rng.sample(cat[2], k=min(3, len(cat[2])))
    users = rng.sample(USERS, k=3)
    styles = rng.sample(STYLES, k=3)
    return (
        f"{PERSONA}\n\n## This batch\n"
        f"Write {n} different conversations. Theme: {cat[1]}.\n"
        f"Draw ideas from (and go beyond): {', '.join(seeds)}.\n"
        f"Mix of users like: {', '.join(users)}. Typing styles like: {', '.join(styles)}.\n"
        f"Length: vary between 1 and 5 exchanges (2 to 10 messages); about a third should be a single exchange.\n"
        f"Make every conversation start differently. Across the whole batch use 'thinking'/'swirl' "
        f"catchphrases at most 3 times total, and vary DC's sentence openings.\n"
        f"Output exactly {n} JSONL lines and nothing else."
    ), cat[1]


def validate(line):
    try:
        obj = json.loads(line)
    except Exception:
        return None
    c = obj.get("c")
    if not isinstance(c, list) or len(c) < 2 or len(c) % 2:
        return None
    for i, turn in enumerate(c):
        if not (isinstance(turn, list) and len(turn) == 2):
            return None
        role, text = turn
        if role != ("u" if i % 2 == 0 else "dc") or not isinstance(text, str) or not text.strip():
            return None
        if not text.isascii():
            return None
        if role == "dc" and len(text.split()) > 60:
            return None
    return {"c": c}


def run_call(idx, args):
    out = RAW / f"{idx:06d}.jsonl"
    if out.exists():
        return idx, -1, "skip"
    rng = random.Random(idx * 7919 + args.seed)
    prompt, cat = build_prompt(rng, args.per_call)
    cmd = ["claude", "-p", "--model", args.model, "--tools", "", "--strict-mcp-config",
           "--no-session-persistence", "--system-prompt",
           "You write training data. Output only JSONL lines, nothing else."]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           timeout=600, cwd=str(HERE / "sandbox"))
    except subprocess.TimeoutExpired:
        return idx, 0, "timeout"
    if r.returncode != 0:
        return idx, 0, f"rc={r.returncode} {r.stderr.strip()[:200]} {r.stdout.strip()[:200]}"
    good = [v for v in (validate(l.strip()) for l in r.stdout.splitlines() if l.strip()) if v]
    if not good:
        return idx, 0, f"no valid lines: {r.stdout[:200]!r}"
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(g) + "\n" for g in good))
    tmp.rename(out)
    return idx, len(good), f"{cat[:40]} {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--per-call", type=int, default=25)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--seed", type=int, default=1999)
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    (HERE / "sandbox").mkdir(exist_ok=True)

    total, fails, consecutive_fail = 0, 0, 0
    with ThreadPoolExecutor(args.jobs) as ex, open(LOG, "a") as log:
        futs = [ex.submit(run_call, i, args) for i in range(args.start, args.start + args.calls)]
        for f in as_completed(futs):
            idx, n, msg = f.result()
            if n == -1:
                continue
            total += max(n, 0)
            if n == 0:
                fails += 1
                consecutive_fail += 1
            else:
                consecutive_fail = 0
            line = f"{time.strftime('%H:%M:%S')} call {idx} -> {n} dialogues ({msg}) total={total} fails={fails}"
            print(line, flush=True)
            log.write(line + "\n"); log.flush()
            if consecutive_fail >= 10:
                print("10 failures in a row (quota?), stopping.", flush=True)
                ex.shutdown(wait=False, cancel_futures=True)
                sys.exit(2)


if __name__ == "__main__":
    main()
