"""Download full daily L2 snapshots; no monthly sample substitution or paid purchase."""
from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import write_json, START, END


def download(cfg):
    start = date.fromisoformat(START)
    end = date.fromisoformat(END)
    key = os.environ.get("TARDIS_API_KEY")
    if not key and os.environ.get("TARDIS_API_KEY_FILE"):
        key = Path(os.environ["TARDIS_API_KEY_FILE"]).expanduser().read_text().strip()
    if not key:
        raise RuntimeError("Thiếu TARDIS_API_KEY hoặc TARDIS_API_KEY_FILE: lịch sử liên tục cần quyền truy cập. "
                           "Không thay bằng các ngày sample đầu tháng.")
    folder = Path(cfg["raw_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder / "download_manifest.json"
    import json
    manifest = json.loads(journal.read_text()) if journal.exists() else {
        "provider": "tardis", "exchange": cfg["exchange"], "symbol": cfg["symbol"],
        "type": "book_snapshot_25", "start_inclusive": START, "end_exclusive": END, "files": {}}
    day = start
    while day < end:
        name = f"{day.isoformat()}.csv.gz"
        target = folder / name
        if target.exists() and name in manifest["files"]:
            day += timedelta(days=1)
            continue
        url = (f"https://datasets.tardis.dev/v1/{quote(cfg['exchange'], safe='')}/"
               f"book_snapshot_25/{day:%Y/%m/%d}/{quote(cfg['symbol'], safe='')}.csv.gz")
        temporary = folder / (name + ".part")
        for attempt in range(5):
            try:
                req = Request(url, headers={"Authorization": f"Bearer {key}"})
                with urlopen(req, timeout=120) as response, temporary.open("wb") as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
                temporary.replace(target)
                manifest["files"][name] = {"url": url, "bytes": target.stat().st_size}
                write_json(journal, manifest)
                print(f"downloaded {day}", flush=True)
                break
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise RuntimeError("Tardis từ chối quyền truy cập lịch sử đã yêu cầu.") from None
                if exc.code == 404:
                    raise RuntimeError(f"Không có L2 ngày {day}; không tự bỏ ngày hoặc đổi dataset.") from None
                if attempt == 4:
                    raise RuntimeError(f"Download thất bại {day}: HTTP {exc.code}") from None
                time.sleep(min(2 ** attempt, 30))
            except (URLError, TimeoutError, ConnectionError):
                if attempt == 4:
                    raise RuntimeError(f"Download gián đoạn {day}; chạy lại để tiếp tục.") from None
                time.sleep(min(2 ** attempt, 30))
        day += timedelta(days=1)
