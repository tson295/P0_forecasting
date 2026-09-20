import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

METADATA_FILES = ("preprocessing", "feature_schema", "target_config", "split_manifest", "data_stats", "experiment")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def read_metadata(directory):
    directory = Path(directory)
    return {name: json.loads((directory/f"{name}.json").read_text())
            for name in METADATA_FILES if (directory/f"{name}.json").exists()}


def save_pretrained(model, directory, metadata=None):
    """Local Hub-uploadable folder; loading requires this package, not AutoModel."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
              str(directory/"model.safetensors"))
    write_json(directory/"config.json", model.model_config)
    for name, value in (metadata or {}).items():
        if name not in METADATA_FILES:
            raise ValueError(f"Unknown checkpoint metadata {name}")
        write_json(directory/f"{name}.json", value)
    write_json(directory/"environment.json", {
        package: importlib.metadata.version(package)
        for package in ("torch", "numpy", "pandas", "transformers", "safetensors", "pytorch-spiking")})


def from_pretrained(directory, device="cpu"):
    from src.models import build_model
    directory = Path(directory)
    config = json.loads((directory/"config.json").read_text())
    model = build_model(config.pop("name"), config.pop("history_rows"), config.pop("channels"), **config)
    model.load_state_dict(load_file(str(directory/"model.safetensors")), strict=True)
    model.to(device).eval()
    model.checkpoint_metadata = read_metadata(directory)
    return model


def capture_rng(generator=None):
    np_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=dict(kind=np_state[0], keys=np_state[1].tolist(), pos=np_state[2],
                           has_gauss=np_state[3], cached=np_state[4]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state() if generator is not None else None)


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    n = state["numpy"]
    np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if generator is not None and state["loader"] is not None:
        generator.set_state(state["loader"].cpu())


def save_training_checkpoint(model, directory, metadata, optimizer, scheduler, state):
    """Stage a complete snapshot before replacing the previous epoch's checkpoint."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    backup = directory.with_name(directory.name+".previous")
    try:
        save_pretrained(model, stage, metadata)
        torch.save(optimizer.state_dict(), stage/"optimizer.pt")
        torch.save(scheduler.state_dict(), stage/"scheduler.pt")
        torch.save(state, stage/"trainer_state.pt")
        if backup.exists():
            raise FileExistsError(f"Recover or remove interrupted snapshot {backup} before continuing")
        if directory.exists():
            directory.rename(backup)
        try:
            stage.rename(directory)
        except BaseException:
            if backup.exists():
                backup.rename(directory)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_training_state(directory, optimizer, scheduler, scaler, generator):
    directory = Path(directory)
    optimizer.load_state_dict(torch.load(directory/"optimizer.pt", map_location="cpu", weights_only=True))
    scheduler.load_state_dict(torch.load(directory/"scheduler.pt", map_location="cpu", weights_only=True))
    state = torch.load(directory/"trainer_state.pt", map_location="cpu", weights_only=True)
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"], generator)
    return state
