"""Export best-checkpoint predictions for train/validation/test; never calls fit().

Test inference runs only here, after training and best-checkpoint selection finished.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.config import Config
from src.data.dataset import prepare_data
from src.models import build_model
from src.training.checkpoint import from_pretrained, read_metadata, verify_checkpoint, write_json
from src.training.trainer import seed_everything
from src.utils.predictions import export_split

SPLITS = ("train", "validation", "test")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Frozen run config, e.g. configs/lit_60s_base.json")
    p.add_argument("--csv", dest="csv_path")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--batch-size", type=int, default=256, help="Inference only; training batch stays 128")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = Config.load(args.config)
    if args.csv_path:
        config.data.csv_path = args.csv_path
    config.validate()
    seed_everything(config.training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    run_name = config.training.run_name
    run = Path(args.checkpoint_root)/config.model/run_name
    out = Path(args.artifact_root)/config.model/run_name
    out.mkdir(parents=True, exist_ok=True)

    # E0 is untrained by contract, so it has no checkpoint; every learned model must have one.
    saved = None
    if config.model == "e0":
        data = prepare_data(config)
        model = build_model("e0", data.history_rows, data.channels)
        verification = None
    else:
        best = run/"best"
        if not (best/"model.safetensors").exists():
            raise FileNotFoundError(f"Missing best checkpoint {best}")
        saved = read_metadata(best)
        data = prepare_data(config, saved)
        model = from_pretrained(best, device=device)
        expected = build_model(config.model, data.history_rows, data.channels, **config.model_kwargs)
        if model.model_config != expected.model_config:
            raise ValueError("Best checkpoint architecture differs from the frozen configuration")
        del expected
        # Both folders must reload to identical, finite predictions on a fixed batch.
        sample = torch.stack([data.datasets["validation"][i][0]
                              for i in range(min(8, len(data.datasets["validation"])))])
        verification = {}
        for name in ("best", "last"):
            reloaded = from_pretrained(run/name, device=device)
            verification[name] = verify_checkpoint(run/name, reloaded, sample, device)
            with torch.no_grad():
                verification[name]["prediction"] = reloaded(sample.to(device)).double().tolist()
            del reloaded
        verification["best_differs_from_last"] = (
            verification["best"]["prediction"] != verification["last"]["prediction"])

    counts = data.metadata["split_manifest"]["sample_counts"]
    started, metrics, timings = time.time(), {}, {}
    for split in SPLITS:
        split_started = time.time()
        _, split_metrics = export_split(model, data.datasets[split], data.raw, device,
                                        out/f"{split}_predictions.csv.gz",
                                        args.batch_size, args.num_workers)
        if split_metrics["samples"] != counts[split]:
            raise ValueError(f"{split}: exported {split_metrics['samples']} of {counts[split]} samples")
        write_json(out/f"{split}_metrics.json", split_metrics)
        metrics[split] = split_metrics
        timings[split] = time.time()-split_started
        print(f"{config.model} {split}: {split_metrics['samples']} rows "
              f"rmse={[round(v, 4) for v in split_metrics['rmse']]} price "
              f"gain={[round(v, 6) for v in split_metrics['rmse_gain_vs_e0']]}", flush=True)

    history = run/"training_history.jsonl"
    if history.exists():
        shutil.copy2(history, out/"training_history.jsonl")
    training_run = run/"training_run.json"
    training = json.loads(training_run.read_text()) if training_run.exists() else None

    summary = dict(
        model=config.model, run_name=run_name, experiment=config.to_dict(),
        parameters=int(sum(p.numel() for p in model.parameters())),
        architecture=model.model_config, training=training,
        best_epoch=(training or {}).get("best_epoch"),
        best_validation_mean_mse=(training or {}).get("best_validation_mean_mse"),
        checkpoint_verification=verification,
        metrics={split: metrics[split] for split in SPLITS},
        sample_counts=counts, history_rows=data.history_rows,
        stride_rows=data.metadata["preprocessing"]["stride_rows"],
        normalization=data.metadata["preprocessing"]["normalization"],
        split_manifest=data.metadata["split_manifest"],
        data_stats=data.raw.stats, target_config=data.metadata["target_config"],
        inference=dict(device=str(device), precision="fp32", batch_size=args.batch_size,
                       deterministic=True, seconds=timings, total_seconds=time.time()-started),
        checkpoint_paths=(None if config.model == "e0"
                          else {name: str(run/name) for name in ("best", "last")}),
        prediction_paths={split: str(out/f"{split}_predictions.csv.gz") for split in SPLITS})
    write_json(out/"run_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("model", "run_name", "parameters", "best_epoch")}), flush=True)
    return summary


if __name__ == "__main__":
    main()
