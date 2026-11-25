#!/usr/bin/env python3
"""Fetch plain-text Wikipedia articles (CC BY-SA 4.0) for DC's grounded knowledge.

  python3 fetch_wikipedia.py      # resumable; writes wikipedia/<slug>.json {title, url, revid, text}
"""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "wikipedia"
API = "https://en.wikipedia.org/w/api.php"
UA = "dreamcast-thinking-homebrew/0.1 (non-commercial hobby project)"

CATEGORIES = ["Category:Dreamcast games", "Category:Dreamcast-only games",
              "Category:Sega video game consoles", "Category:Sega characters",
              "Category:Sonic the Hedgehog characters", "Category:Dreamcast accessories"]
EXTRA = ["Dreamcast", "Visual Memory Unit", "GD-ROM", "SuperH", "PowerVR", "SegaNet", "Sega",
         "History of Sega", "Sonic Team", "Sega AM2", "Smilebit", "Hitmaker", "United Game Artists",
         "Sega Genesis", "Sega Saturn", "Master System", "Game Gear", "Sega CD", "32X", "SG-1000",
         "Sega NAOMI", "Console war", "PlayStation 2", "Nintendo 64", "GameCube", "Xbox (console)",
         "Sega Dreamcast keyboard", "Dreamcast Broadband Adapter", "Seaman (video game)",
         "Sonic the Hedgehog (character)", "Tails (Sonic the Hedgehog)", "Knuckles the Echidna",
         "Dr. Eggman", "Shadow the Hedgehog", "Amy Rose", "Chao (Sonic)", "Ulala (Space Channel 5)",
         "Year 2000 problem", "Tamagotchi", "Dial-up Internet access", "Sixth generation of video game consoles",
         "Fifth generation of video game consoles", "Fourth generation of video game consoles"]


def api(**params):
    params.update(format="json", formatversion="2")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {attempt}: {e}", flush=True)
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"API failed: {params}")


def category_members(cat):
    titles, cont = [], {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=cat, cmlimit="500", cmnamespace="0", **cont)
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return titles
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def slug(t):
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")[:120]


def fetch(title):
    out = OUT / f"{slug(title)}.json"
    if out.exists():
        return "skip"
    d = api(action="query", prop="extracts|revisions|info", explaintext="1", exsectionformat="plain",
            rvprop="ids", inprop="url", titles=title, redirects="1")
    pages = d["query"]["pages"]
    if not pages or pages[0].get("missing"):
        return "missing"
    p = pages[0]
    text = p.get("extract", "")
    # drop trailing reference-ish sections
    text = re.split(r"\n(References|External links|See also|Notes|Further reading)\n", text)[0].strip()
    out.write_text(json.dumps({"title": p["title"], "url": p.get("fullurl"),
                               "revid": p["revisions"][0]["revid"], "text": text}))
    return len(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    titles = list(EXTRA)
    for c in CATEGORIES:
        m = category_members(c)
        print(f"{c}: {len(m)}", flush=True)
        titles += m
    titles = list(dict.fromkeys(t for t in titles if not t.startswith("List of")))
    print(f"{len(titles)} titles", flush=True)
    for i, t in enumerate(titles):
        try:
            r = fetch(t)
        except Exception as e:
            r = f"FAILED {e}"
            print(f"{t}: {r}", flush=True)
        if r != "skip":
            time.sleep(1.0)  # be polite
        if i % 50 == 0:
            print(f"{i}/{len(titles)} {t}: {r}", flush=True)
    print("done", flush=True)


def wikitext_to_plain(wt):
    import mwparserfromhell
    code = mwparserfromhell.parse(wt)
    for t in code.filter_templates(recursive=False):
        try:
            code.remove(t)
        except ValueError:
            pass
    for tag in code.filter_tags(matches=lambda n: n.tag in ("ref", "gallery", "table")):
        try:
            code.remove(tag)
        except ValueError:
            pass
    text = code.strip_code()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.split(r"\n(References|External links|See also|Notes|Further reading)\n", text)[0]
    return text.strip()


def main_batched():
    OUT.mkdir(parents=True, exist_ok=True)
    titles = list(EXTRA)
    for c in CATEGORIES:
        titles += category_members(c)
        time.sleep(2)
    titles = [t for t in dict.fromkeys(titles) if not t.startswith("List of")]
    todo = [t for t in titles if not (OUT / f"{slug(t)}.json").exists()]
    print(f"{len(titles)} titles, {len(todo)} to fetch", flush=True)
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        d = api(action="query", prop="revisions|info", rvprop="ids|content", rvslots="main",
                inprop="url", titles="|".join(chunk), redirects="1")
        redir = {r["to"]: r["from"] for r in d["query"].get("redirects", [])}
        norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
        n_ok = 0
        for p in d["query"]["pages"]:
            if p.get("missing") or "revisions" not in p:
                continue
            asked = redir.get(p["title"], p["title"])
            asked = norm.get(asked, asked)
            rev = p["revisions"][0]
            text = wikitext_to_plain(rev["slots"]["main"]["content"])
            for name in {asked, p["title"]}:
                (OUT / f"{slug(name)}.json").write_text(json.dumps(
                    {"title": p["title"], "url": p.get("fullurl"), "revid": rev["revid"], "text": text}))
            n_ok += 1
        print(f"batch {i // 50}: {n_ok}/{len(chunk)}", flush=True)
        time.sleep(5)
    print("done", flush=True)


if __name__ == "__main__":
    main_batched()
