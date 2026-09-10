# Read-only HF metadata (used 2026-09-10 for SOURCE_REPORT)/card inspection (paginated tree listing).
import json, re, sys, urllib.request
from collections import Counter
def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace"), r.headers.get("Link", "")
def tree(repo):
    url, out = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true", []
    while url:
        body, link = fetch(url); out += [x for x in json.loads(body) if x["type"] == "file"]
        url = next((p.split("<", 1)[1].split(">", 1)[0] for p in link.split(",") if 'rel="next"' in p), None)
    return out
for repo in sys.argv[1:]:
    print("=" * 100); print(repo)
    try:
        t = tree(repo)
    except Exception as e:
        print("tree ERR", e); continue
    tot = sum(x.get("size", 0) for x in t)
    top = Counter("/".join(x["path"].split("/")[:2]) for x in t)
    print(f"files {len(t)} total {tot/1e9:.2f} GB; top dirs {dict(top.most_common(8))}")
    for x in t[:5] + (t[-3:] if len(t) > 8 else []): print("   ", x["path"], x.get("size"))
    btc = [x for x in t if re.search(r"(?i)btc", x["path"])]
    if btc and len(btc) != len(t): print(f"   btc-files {len(btc)} {sum(x.get('size',0) for x in btc)/1e9:.2f} GB e.g. {btc[0]['path']} .. {btc[-1]['path']}")
    try:
        card, _ = fetch(f"https://huggingface.co/datasets/{repo}/raw/main/README.md")
    except Exception as e:
        card = ""
    lines = [l.strip() for l in card.splitlines() if re.search(r"(?i)binance|spot|futures|perp|level|depth|snapshot|diff|update|interval|\bms\b|second|coverage|period|20[12][0-9]-|rows|source|collected|websocket", l)]
    for l in lines[:18]: print("  |", l[:170])
