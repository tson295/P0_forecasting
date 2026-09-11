from __future__ import annotations

import argparse

from .config import ROOT, load


def main():
    parser = argparse.ArgumentParser(description="L2 10-level OF → Direct forecasting. No smoke/test commands.")
    parser.add_argument("command", choices=("download", "prepare", "data-report", "train", "summarize"))
    parser.add_argument("--config", default=str(ROOT / "configs/orderbook.json"))
    parser.add_argument("--models", help="Comma-separated families; one model per horizon")
    parser.add_argument("--folds", help="Comma-separated fold names, e.g. fold1,fold2")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells completed with this exact config/revision; move unfinished attempts to attempts/")
    args = parser.parse_args()
    cfg = load(args.config)
    if args.command == "download":
        from .download import download
        download(cfg)
    elif args.command == "prepare":
        from .prepare import prepare
        prepare(cfg)
    elif args.command == "data-report":
        from .report import data_report
        data_report(cfg)
    elif args.command == "summarize":
        from .results import refresh_summaries
        refresh_summaries(cfg["output_dir"])
    else:
        from .train import train
        train(cfg, args.models.split(",") if args.models else None,
              args.folds.split(",") if args.folds else None, resume=args.resume)


if __name__ == "__main__":
    main()
