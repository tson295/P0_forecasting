# Generator of zenodo_20046390_scan.out/.csv (run inside data/orderbook/zenodo_20046390 with /home/ubuntu/venv-ob/bin/python).
# Read-only quality/continuity scan of Zenodo 20046390 JSON lines (no replay, no model).
import json, re, tarfile
import pandas as pd
TAIL = re.compile(rb'"timestamp":\s*(\d+),\s*"datetime":\s*[^,]*,\s*"nonce":\s*(\d+)')
NUM = re.compile(r'^-?\d+(\.\d+)?([eE]-?\d+)?, -?\d+(\.\d+)?([eE]-?\d+)?$')
def first_bad(line, side):
    m = re.search(r'"%s": \[\[(.*?)\]\]' % side, line)
    if not m:
        return 0
    for i, el in enumerate(m.group(1).split('], [')):
        if not NUM.match(el):
            return i
    return None
t = tarfile.open("btcusdt_lob_oct2023.tar.gz")
rows, prev = [], None
for mem in sorted((m for m in t.getmembers() if m.isfile() and m.name.startswith("./BTCUSDT_")), key=lambda m: m.name):
    ts, nonce, bad, bad_top10, levels = [], [], 0, 0, set()
    for raw in t.extractfile(mem):
        mt = TAIL.search(raw[-160:])
        if not mt:
            bad += 1; bad_top10 += 1; continue
        ts.append(int(mt.group(1))); nonce.append(int(mt.group(2)))
        try:
            o = json.loads(raw)
            levels.add((len(o["bids"]), len(o["asks"])))
        except ValueError:
            bad += 1
            line = raw.decode("utf-8", "replace")
            fb, fa = first_bad(line, "bids"), first_bad(line, "asks")
            if (fb is not None and fb < 10) or (fa is not None and fa < 10):
                bad_top10 += 1
    d = [b - a for a, b in zip(ts, ts[1:])]
    dn = [b - a for a, b in zip(nonce, nonce[1:])]
    rows.append(dict(file=mem.name[2:], n=len(ts), bad_json=bad, bad_top10=bad_top10, first=ts[0], last=ts[-1],
                     med_ms=pd.Series(d).median(), gt6s=sum(x > 6000 for x in d), gt10s=sum(x > 10000 for x in d),
                     max_gap_s=max(d) / 1000, ts_nonmono=sum(x <= 0 for x in d), nonce_nonincr=sum(x <= 0 for x in dn),
                     bridge_s=None if prev is None else (ts[0] - prev) / 1000, levels=sorted(levels)[:3]))
    prev = ts[-1]
df = pd.DataFrame(rows)
df["first_utc"] = pd.to_datetime(df["first"], unit="ms"); df["last_utc"] = pd.to_datetime(df["last"], unit="ms")
pd.set_option("display.width", 250)
print(df[["file", "n", "bad_json", "bad_top10", "first_utc", "last_utc", "med_ms", "gt6s", "gt10s", "max_gap_s",
          "ts_nonmono", "nonce_nonincr", "bridge_s", "levels"]].to_string())
print("TOTAL snapshots", int(df.n.sum()), "| bad_json", int(df.bad_json.sum()), "| bad in top10", int(df.bad_top10.sum()),
      "| gaps>10s", int(df.gt10s.sum()), "| max gap s", df.max_gap_s.max(), "| max bridge s", df.bridge_s.max())
df.to_csv("zenodo_scan.csv", index=False)
