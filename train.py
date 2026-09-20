"""Prepare/smoke modes never enter fit() or auto batch probing."""
import argparse
import json
from pathlib import Path

import torch

from src.config import Config, MODELS
from src.data.dataset import prepare_data
from src.models import build_model
from src.training.checkpoint import from_pretrained, read_metadata, write_json
from src.training.trainer import (seed_everything, smoke_forward, fit,
                                  device_and_precision, make_loader, evaluate)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Experiment JSON; CLI options override fields")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--csv", dest="csv_path")
    p.add_argument("--history-seconds", type=int, choices=(60, 120, 180))
    p.add_argument("--history-rows", type=int, help="Explicit override (e.g. LiT 64)")
    p.add_argument("--stride-seconds", type=float)
    p.add_argument("--target-tolerance-seconds", type=float)
    p.add_argument("--train-end", help="Exclusive ISO timestamp, UTC if no timezone")
    p.add_argument("--validation-end", help="Exclusive ISO timestamp")
    p.add_argument("--of-representation", choices=("of", "ofi"))
    for name, kind in (("batch-size", int), ("epochs", int), ("num-workers", int),
                       ("prefetch-factor", int), ("gradient-accumulation", int),
                       ("learning-rate", float), ("seed", int), ("run-name", str),
                       ("device", str), ("checkpoint-root", str)):
        p.add_argument("--"+name, type=kind)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"))
    p.add_argument("--compile", action="store_true", default=None)
    p.add_argument("--auto-batch-size", action="store_true", default=None)
    p.add_argument("--model-kwargs", type=json.loads, help='JSON architecture overrides')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--evaluate", choices=("validation", "test"))
    p.add_argument("--backward", action="store_true", help="Smoke only: one backward, no optimizer")
    p.add_argument("--output", default="runs/inspection", help="Prepare/smoke/evaluation artifacts")
    checkpoint = p.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", help="Full epoch-boundary resume from best/last folder")
    checkpoint.add_argument("--init-from", help="Base checkpoint for fine-tuning or evaluation")
    p.add_argument("--head-only", action="store_true", help="Freeze backbone including dropout/BN")
    return p


def resolve_config(args):
    source = args.resume or args.init_from
    metadata = read_metadata(source) if source else None
    if args.config:
        config = Config.load(args.config)
    elif metadata and "experiment" in metadata:
        config = Config.from_dict(metadata["experiment"])
    else:
        config = Config()
    if args.model:
        config.model = args.model
    if args.model_kwargs is not None:
        config.model_kwargs = args.model_kwargs
    for group in (config.data, config.training):
        for field in vars(group):
            value = getattr(args, field, None)
            if value is not None:
                setattr(group, field, value)
    config.validate()
    if args.backward and not args.smoke_test:
        raise ValueError("--backward is only valid with --smoke-test")
    if args.head_only and not source:
        raise ValueError("--head-only requires --init-from or --resume")
    if args.evaluate and config.model != "e0" and not source:
        raise ValueError("Evaluation of a learned model requires a checkpoint")
    return config, metadata


def check_resume(config, data, saved):
    if not saved or "experiment" not in saved:
        raise ValueError("Resume requires full experiment/preprocessing metadata")
    previous = Config.from_dict(saved["experiment"])
    # A resumed scheduler, optimizer and batch sequence retain their original contract.
    ignored = {"num_workers", "prefetch_factor", "device", "compile", "checkpoint_root", "run_name"}
    for field, value in vars(config.training).items():
        if field not in ignored and value != getattr(previous.training, field):
            raise ValueError(f"Resume cannot change training.{field}; use --init-from for a new run")
    if config.model != previous.model or config.model_kwargs != previous.model_kwargs:
        raise ValueError("Resume model configuration differs")
    for key in ("preprocessing", "feature_schema", "target_config"):
        if data.metadata[key] != saved[key]:
            raise ValueError(f"Resume {key} differs")
    manifest, old = data.metadata["split_manifest"], saved["split_manifest"]
    for key in ("boundaries_ns", "sample_counts", "origin_index_sha256"):
        if manifest[key] != old[key]:
            raise ValueError(f"Resume split mismatch: {key}")
    if manifest["source"]["sha256"] != old["source"]["sha256"]:
        raise ValueError("Resume dataset fingerprint differs")


def main(argv=None):
    args = parser().parse_args(argv)
    config, saved = resolve_config(args)
    seed_everything(config.training.seed)
    if args.smoke_test:
        torch.set_num_threads(2)
    data = prepare_data(config, saved)
    print(json.dumps(dict(stats=data.raw.stats, history_rows=data.history_rows,
                          stride_rows=data.metadata["preprocessing"]["stride_rows"],
                          samples={k:len(v) for k,v in data.datasets.items()}), indent=2), flush=True)
    if args.resume:
        check_resume(config, data, saved)
    if args.prepare_only:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for name, value in (data.metadata | {"experiment": config.to_dict()}).items():
            write_json(output/f"{name}.json", value)
        return
    expected = build_model(config.model, data.history_rows, data.channels, **config.model_kwargs)
    if args.resume or args.init_from:
        model = from_pretrained(args.resume or args.init_from)
        if model.model_config != expected.model_config:
            raise ValueError("Checkpoint architecture/input shape differs from requested configuration")
        del expected
    else:
        model = expected
    if args.resume:
        state = torch.load(Path(args.resume)/"trainer_state.pt", map_location="cpu", weights_only=True)
        if state["head_only"]:
            model.freeze_backbone()
        elif args.head_only:
            raise ValueError("Cannot change a full-training resume to head-only; use --init-from")
    elif args.head_only:
        model.freeze_backbone()
    if args.smoke_test:
        result = smoke_forward(model, data.datasets["train"], args.backward)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/f"smoke_{config.model}.json", result)
        print(json.dumps(result, indent=2))
        return
    if args.evaluate or config.model == "e0":
        split = args.evaluate or "validation"
        device, dtype = device_and_precision(config.training)
        model.to(device)
        result = evaluate(model, make_loader(data.datasets[split], config.training, device), device, dtype)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/f"{config.model}_{split}_metrics.json", result)
        print(json.dumps(result, indent=2))
        return
    fit(model, data, config, args.resume)


if __name__ == "__main__":
    main()
