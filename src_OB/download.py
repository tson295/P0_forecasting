"""Download a pinned public archive: HF BTCUSDT Spot depth/snapshots, or one pinned Zenodo record file."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from .config import write_json

HOST = "https://huggingface.co"
ZENODO = "https://zenodo.org"


def json_get(url):
    for attempt in range(5):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 30))


def download(cfg):
    if cfg["provider"] == "zenodo":
        return download_zenodo(cfg)
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


def md5sum(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_zenodo(cfg):
    """One pinned Zenodo record file; its md5 on the record must equal the configured revision."""
    root = Path(cfg["raw_dir"])
    root.mkdir(parents=True, exist_ok=True)
    record, name = str(cfg["zenodo_record"]), cfg["zenodo_file"]
    entry = next((e for e in json_get(f"{ZENODO}/api/records/{record}/files")["entries"] if e["key"] == name), None)
    if entry is None:
        raise ValueError(f"Zenodo record {record} không có file {name}.")
    md5 = entry["checksum"].split(":", 1)[-1]
    if md5 != cfg["dataset_revision"]:
        raise ValueError("md5 trên Zenodo khác bản đã pin; dùng raw_dir mới, không trộn data.")
    target = root / name
    if target.is_file() and target.stat().st_size == entry["size"] and md5sum(target) == md5:
        print(f"verified existing {name}: {entry['size']:,} bytes, md5 {md5}", flush=True)
    else:
        temporary = target.with_name(target.name + ".part")
        url = f"{ZENODO}/records/{record}/files/{quote(name)}?download=1"
        for attempt in range(5):
            try:
                with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                if temporary.stat().st_size != entry["size"] or md5sum(temporary) != md5:
                    raise OSError("Incomplete or corrupted archive transfer")
                temporary.replace(target)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
        print(f"downloaded {name}: {entry['size']:,} bytes, md5 {md5}", flush=True)
    write_json(root / "download_manifest.json", {
        "provider": "zenodo", "repo": cfg["dataset_repo"], "revision": md5, "record": record,
        "exchange": cfg["exchange"], "asset": cfg["symbol"], "license": cfg.get("source_license"),
        "files": {name: {"bytes": entry["size"], "md5": md5}},
        "selected_files": [{"path": name, "bytes": entry["size"]}],
        "historical_fixed": True, "coverage": "read from reconstructed data, not filenames", "status": "complete"})
    print(f"Historical archive saved to {root}", flush=True)
