#!/usr/bin/env python3
"""Fetch StrategyWiki (CC BY-SA) guides for Dreamcast games: main page + up to 12 subpages each.

  ../.venv/bin/python fetch_strategywiki.py     # resumable; writes strategywiki/<slug>.json
"""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path
from fetch_wikipedia import wikitext_to_plain, slug

HERE = Path(__file__).resolve().parent
OUT = HERE / "strategywiki"
API = "https://strategywiki.org/w/api.php"
UA = "dreamcast-thinking-homebrew/0.1 (non-commercial hobby project)"
MAX_SUB = 12


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


def members(cat):
    out, cont = [], {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=cat, cmlimit="500", cmnamespace="0", **cont)
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return out
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}
        time.sleep(2)


def subpages(game):
    d = api(action="query", list="allpages", apprefix=game + "/", apnamespace="0", aplimit="50")
    return [p["title"] for p in d["query"]["allpages"]][:MAX_SUB]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games = members("Category:Sega Dreamcast")
    print(f"{len(games)} Dreamcast games on StrategyWiki", flush=True)
    titles = []
    for g in games:
        titles.append(g)
        titles += subpages(g)
        time.sleep(1)
    todo = [t for t in titles if not (OUT / f"{slug(t)}.json").exists()]
    print(f"{len(titles)} pages, {len(todo)} to fetch", flush=True)
    for i in range(0, len(todo), 40):
        chunk = todo[i:i + 40]
        d = api(action="query", prop="revisions|info", rvprop="ids|content", rvslots="main",
                inprop="url", titles="|".join(chunk), redirects="1")
        n = 0
        for p in d["query"]["pages"]:
            if p.get("missing") or "revisions" not in p:
                continue
            rev = p["revisions"][0]
            text = wikitext_to_plain(rev["slots"]["main"]["content"])
            if len(text) < 200:
                continue
            (OUT / f"{slug(p['title'])}.json").write_text(json.dumps(
                {"title": p["title"], "url": p.get("fullurl"), "revid": rev["revid"], "text": text}))
            n += 1
        print(f"batch {i // 40}: {n}/{len(chunk)}", flush=True)
        time.sleep(5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
