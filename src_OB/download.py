"""Download a pinned public HF archive: BTCUSDT Spot depth and snapshots only."""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from .config import write_json

HOST = "https://huggingface.co"


def json_get(url):
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def download(cfg):
    root = Path(cfg["raw_dir"])
    root.mkdir(parents=True, exist_ok=True)
    journal = root / "download_manifest.json"
    repo = cfg["dataset_repo"]
    revision = json_get(f"{HOST}/api/datasets/{repo}/revision/{quote(cfg['dataset_revision'], safe='')}")["sha"]
    if journal.exists():
        manifest = json.loads(journal.read_text())
        if manifest["repo"] != repo or manifest["revision"] != revision:
            raise ValueError("Pinned archive khác manifest; dùng raw_dir mới, không trộn revision.")
    else:
        manifest = {"provider": "huggingface", "repo": repo, "revision": revision,
                    "exchange": cfg["exchange"], "asset": cfg["symbol"], "files": {},
                    "historical_fixed": True, "coverage": "read from reconstructed data, not filenames"}
    selected = []
    for kind in ("snapshots", "depth"):
        prefix = f"{kind}/{cfg['exchange']}/{cfg['symbol']}"
        url = f"{HOST}/api/datasets/{repo}/tree/{revision}/{prefix}?limit=1000"
        while url:
            with urlopen(url, timeout=60) as response:
                selected.extend(item for item in json.load(response)
                                if item["type"] == "file" and item["path"].endswith(".parquet"))
                link = response.headers.get("Link", "")
            url = next((part.split("<", 1)[1].split(">", 1)[0]
                        for part in link.split(",") if 'rel="next"' in part), None)
    if not any(f["path"].startswith("snapshots/") for f in selected) or not any(
            f["path"].startswith("depth/") for f in selected):
        raise ValueError("Archive phải có cả snapshot và depth diff.")
    manifest["selected_files"] = [{"path": f["path"], "bytes": f["size"]} for f in selected]
    manifest["status"] = "downloading"
    write_json(journal, manifest)
    print(f"HF revision={revision}, {len(selected)} files, {sum(f['size'] for f in selected):,} bytes", flush=True)
    for item in selected:
        name = item["path"]
        target = root / name
        if name in manifest["files"] and target.is_file() and target.stat().st_size == item["size"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".parquet.part")
        url = f"{HOST}/datasets/{repo}/resolve/{revision}/{quote(name, safe='/')}?download=true"
        for attempt in range(5):
            try:
                with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                if temporary.stat().st_size != item["size"]:
                    raise OSError("Incomplete archive transfer")
                temporary.replace(target)
                manifest["files"][name] = {"bytes": item["size"], "source_lfs_sha256": item.get("lfs", {}).get("oid")}
                write_json(journal, manifest)
                print(f"downloaded {name}: {item['size']:,} bytes", flush=True)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
    manifest["status"] = "complete"
    write_json(journal, manifest)
    print(f"Historical archive saved to {root}", flush=True)
