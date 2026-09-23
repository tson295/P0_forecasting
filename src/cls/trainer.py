"""Train, select, and export one classification run.

Recipe = the WF3 recipe (configs/*_wf3_f*.json): AdamW(lr 1e-4, weight decay 1e-4),
CosineAnnealingLR(T_max=epochs), 30 epochs, batch 128, gradient clip 1.0, bf16
autocast with TF32, seed 42. Only the loss changes:

    loss = mean over the run's horizons of CrossEntropy(logits_h, class_h)

(plain CrossEntropy: no class weights, focal, ordinal or auxiliary terms).
The best checkpoint is the epoch with the lowest validation mean CrossEntropy; the test
split is never read until export(), after training and selection are finished.
"""
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.training.trainer import seed_everything
from .data import ClassificationData, fingerprint
from .labels import LabelSet, displacement
from .metrics import DA_CONVENTION, official_metrics
from .models import build_classifier, load_classifier, parameter_count, save_weights

HORIZON_TAG = {60: "60s", 120: "120s", 180: "180s"}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=_default)+"\n")
    tmp.replace(path)


def _default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def environment():
    import importlib.metadata as md
    env = {p: md.version(p) for p in ("torch", "numpy", "pandas", "safetensors")}
    env.update(python=platform.python_version(), host=platform.node(),
               cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(gpu=props.name, gpu_total_bytes=int(props.total_memory))
    return env


def git_state(root):
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(commit=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(run("status", "--porcelain", "--untracked-files=no")))


def autocast(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


class Evaluator:
    """Streaming validation: CE per horizon and the four official metrics (argmax decode),
    plus the expected-value decode as a diagnostic, all accumulated on the device."""
    def __init__(self, label_set, horizons, device):
        self.reps = [torch.tensor(label_set[h].representatives, dtype=torch.float64, device=device)
                     for h in horizons]
        self.horizons = horizons
        self.device = device

    def reset(self):
        h = len(self.horizons)
        z = lambda: torch.zeros(h, dtype=torch.float64, device=self.device)
        self.n, self.ce = 0, z()
        self.stats = {d: dict(sse=z(), sae=z(), hits=z()) for d in ("argmax", "expected")}
        self.sse_e0, self.moved = z(), z()

    def update(self, logits, labels, delta):
        self.n += len(labels)
        for j, (lg, rep) in enumerate(zip(logits, self.reps)):
            lg = lg.float()
            self.ce[j] += F.cross_entropy(lg, labels[:, j], reduction="sum").double()
            true = delta[:, j]
            moved = true != 0
            self.sse_e0[j] += (true*true).sum()
            self.moved[j] += moved.sum()
            preds = dict(argmax=rep[lg.argmax(1)], expected=torch.softmax(lg, 1).double() @ rep)
            for name, pred in preds.items():
                err = pred-true
                s = self.stats[name]
                s["sse"][j] += (err*err).sum()
                s["sae"][j] += err.abs().sum()
                s["hits"][j] += (torch.sign(pred[moved]) == torch.sign(true[moved])).sum()

    def compute(self):
        n = self.n
        ce = (self.ce/n).tolist()
        out = dict(samples=n, ce=ce, mean_ce=float(np.mean(ce)))
        for name, s in self.stats.items():
            out[name] = dict(rmse=torch.sqrt(s["sse"]/n).tolist(), mae=(s["sae"]/n).tolist(),
                             r2_gain_vs_e0=(1-s["sse"]/self.sse_e0).tolist(),
                             da=(s["hits"]/self.moved).tolist())
        out["rmse_e0"] = torch.sqrt(self.sse_e0/n).tolist()
        if not math.isfinite(out["mean_ce"]):
            raise FloatingPointError("Nonfinite validation loss")
        return out


def evaluate(model, data, split, batch_size, device, dtype, evaluator):
    model.eval()
    evaluator.reset()
    with torch.no_grad():
        for x, y, delta, _ in data.batches(split, batch_size):
            with autocast(device, dtype):
                logits = model(x)
            evaluator.update(logits, y, delta)
    return evaluator.compute()


def _checkpoint(model, directory, metadata, extra=None):
    """Stage a complete folder, then swap it in (an interrupted save never leaves half a folder)."""
    directory = Path(directory)
    stage = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        save_weights(model, stage)
        for name, value in metadata.items():
            write_json(stage/f"{name}.json", value)
        for name, value in (extra or {}).items():
            torch.save(value, stage/name)
        old = directory.with_name(directory.name+".old")
        if old.exists():
            shutil.rmtree(old)
        if directory.exists():
            directory.rename(old)
        stage.rename(directory)
        if old.exists():
            shutil.rmtree(old)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


@torch.no_grad()
def _verify(directory, model, sample, device):
    """Contract V from WF3: a saved folder must reload to identical FP32 predictions."""
    restored = load_classifier(directory, device)
    was_training = model.training
    model.eval()
    before, after = model(sample), restored(sample)
    model.train(was_training)
    diff = max(float((a.double()-b.double()).abs().max()) for a, b in zip(before, after))
    if diff != 0.0:
        raise AssertionError(f"Reloaded {directory} deviates by {diff}")
    return dict(path=str(directory), samples=int(len(sample)), max_absolute_difference=diff, exact=True)


def multi_horizon_ce(logits, labels):
    """(CE_h1 + CE_h2 + CE_h3)/3 for multi-horizon runs; CE_h for single-horizon runs."""
    return torch.stack([F.cross_entropy(lg.float(), labels[:, j]) for j, lg in enumerate(logits)]).mean()


def training_functions(model, compile_mode, dtype, device):
    """The training loss function and the evaluation forward, compiled per architecture.

    compile_mode "none": eager (OFI-LSTM, as in WF3); "default": torch.compile of the
    model+loss (ModernTCN, compiled in WF3 too); "reduce-overhead": the same plus CUDA
    graphs (Transformer), which removes the per-kernel launch cost that otherwise makes
    a small model CPU-bound at batch 128. None of these changes the arithmetic.
    """
    def loss_fn(x, y):
        with autocast(device, dtype):
            return multi_horizon_ce(model(x), y)
    if compile_mode == "none":
        return loss_fn, model
    if compile_mode not in ("default", "reduce-overhead"):
        raise ValueError(f"Unknown compile_mode {compile_mode}")
    # dynamic=False: batch 128 keeps the static graph the benchmark measured; the epoch's
    # remainder batch compiles its own static entry instead of switching to dynamic shapes.
    eval_model = torch.compile(model, dynamic=False) if compile_mode == "default" else model
    return torch.compile(loss_fn, mode=compile_mode, dynamic=False), eval_model


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                generator=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["generator"])


def set_status(run_dir, **fields):
    path = Path(run_dir)/"status.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(fields, updated=pd.Timestamp.now(tz="UTC").isoformat())
    write_json(path, current)


def prepare(cfg, device):
    label_set = LabelSet.load(cfg["label_set_path"])
    data = ClassificationData(cfg["arch"], cfg["data"], cfg["csv_path"], label_set,
                              cfg["horizons"], device)
    return label_set, data


def fit(cfg, run_dir, device, dtype, label_set, data, repo_root):
    t = cfg["training"]
    run_dir = Path(run_dir)
    seed_everything(t["seed"])
    model = build_classifier(cfg["arch"], data.history_rows, data.channels, data.n_classes(),
                             cfg["horizons"], **cfg.get("model_kwargs", {})).to(device)
    loss_fn, eval_model = training_functions(model, t["compile_mode"], dtype, device)
    # fused=True: the same AdamW update in one kernel instead of a multi-tensor loop.
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
                                  fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    generator = torch.Generator().manual_seed(t["seed"])
    evaluator = Evaluator(label_set, cfg["horizons"], device)
    metadata = dict(run_config=cfg, label_set=label_set.to_dict() | {"definition_sha256": label_set.sha256()},
                    preprocessing=data.metadata["preprocessing"],
                    feature_schema=data.metadata["feature_schema"],
                    target_config=data.metadata["target_config"] | dict(
                        classification_target="delta_h = mid[target_h] - mid[origin] (USD)",
                        horizons_predicted=cfg["horizons"]),
                    split_manifest=data.metadata["split_manifest"], data_stats=data.metadata["data_stats"],
                    samples=data.summary())
    history_path = run_dir/"training_history.jsonl"
    start, best, best_epoch = 0, math.inf, None
    last = run_dir/"last"
    # A kill inside _checkpoint's rename window leaves only <name>.old: take it back.
    for folder in (last, run_dir/"best"):
        old = folder.with_name(folder.name+".old")
        if not folder.exists() and old.exists():
            old.rename(folder)
    if (last/"trainer_state.pt").exists():
        state = torch.load(last/"trainer_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(load_classifier(last).state_dict())
        optimizer.load_state_dict(torch.load(last/"optimizer.pt", map_location="cpu", weights_only=True))
        scheduler.load_state_dict(torch.load(last/"scheduler.pt", map_location="cpu", weights_only=True))
        restore_rng(state["rng"], generator)
        start, best, best_epoch = state["epoch"]+1, state["best"], state["best_epoch"]
        selection = run_dir/"best"/"selection.json"
        saved_best = json.loads(selection.read_text())["epoch"] if selection.exists() else None
        if saved_best != best_epoch:
            # last/ is committed before best/, so a kill between them leaves best/ one
            # improvement behind; the resumed weights ARE the best epoch's weights then.
            if best_epoch != state["epoch"]:
                raise RuntimeError(f"best/ holds epoch {saved_best}, state says {best_epoch}")
            _checkpoint(model, run_dir/"best", metadata | dict(selection=dict(
                criterion="validation mean CrossEntropy over the run's horizons", epoch=best_epoch,
                value=best, restored_from="last/ after an interrupted best/ save")))
        print(json.dumps(dict(resumed_from_epoch=state["epoch"], best=best, best_epoch=best_epoch)), flush=True)
    # History keeps exactly the epochs before `start` (WF3 truncate_jsonl semantics); a
    # fresh start drops everything, a torn final line is dropped rather than parsed.
    kept = []
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            try:
                if json.loads(line)["epoch"] < start:
                    kept.append(line)
            except (json.JSONDecodeError, KeyError):
                pass
    history_path.write_text("".join(l+"\n" for l in kept))
    n_train = len(data.splits["train"])
    sample = data.windows(data.device_split("validation")["origins"][:8]).float()
    started = time.time()
    for epoch in range(start, t["epochs"]):
        set_status(run_dir, state="running", epoch=epoch)
        epoch_started = time.time()
        model.train()
        order = torch.randperm(n_train, generator=generator)
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        steps = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for x, y, _, _ in data.batches("train", t["batch_size"], order):
            loss = loss_fn(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.detach().double()*len(y)
            steps += 1
        train_loss = float(loss_sum/n_train)
        if not math.isfinite(train_loss):
            raise FloatingPointError("Nonfinite training loss")
        train_seconds = time.time()-epoch_started
        validation = evaluate(eval_model, data, "validation", t["eval_batch_size"], device, dtype, evaluator)
        learning_rate = float(scheduler.get_last_lr()[0])
        scheduler.step()
        improved = validation["mean_ce"] < best
        if improved:
            best, best_epoch = validation["mean_ce"], epoch
        record = dict(epoch=epoch, learning_rate=learning_rate, train_mean_ce=train_loss,
                      validation=validation, is_best=bool(improved), best_epoch=best_epoch,
                      best_validation_mean_ce=best, steps=steps,
                      train_seconds=train_seconds, epoch_seconds=time.time()-epoch_started,
                      train_samples_per_second=n_train/train_seconds,
                      peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                      peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None)
        # Commit order: history, then last/ (resume point), then best/. A kill after the
        # history line re-runs the epoch (its line is dropped on resume); a kill between
        # last/ and best/ is repaired on resume from last/, which holds the best weights.
        with history_path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        state = dict(epoch=epoch, best=best, best_epoch=best_epoch, rng=rng_state(generator))
        _checkpoint(model, last, dict(run_config=cfg),
                    {"optimizer.pt": optimizer.state_dict(), "scheduler.pt": scheduler.state_dict(),
                     "trainer_state.pt": state})
        if improved:
            _checkpoint(model, run_dir/"best", metadata | dict(selection=dict(
                criterion="validation mean CrossEntropy over the run's horizons", epoch=epoch,
                value=best)))
            _verify(run_dir/"best", model, sample, device)
        print(json.dumps(dict(epoch=epoch, train_ce=round(train_loss, 5),
                              val_ce=[round(v, 5) for v in validation["ce"]],
                              val_r2_gain=[round(v, 5) for v in validation["argmax"]["r2_gain_vs_e0"]],
                              val_r2_gain_expected=[round(v, 5) for v in validation["expected"]["r2_gain_vs_e0"]],
                              best=improved, sec=round(record["epoch_seconds"], 1),
                              sps=round(record["train_samples_per_second"]))), flush=True)
    return dict(model=model, best_epoch=best_epoch, best_validation_mean_ce=best,
                train_seconds_this_process=time.time()-started)


@torch.no_grad()
def predict_split(model, data, split, batch_size):
    """FP32, eval mode, sequential order (as WF3 src/utils/predictions.predict)."""
    model.eval()
    chunks = [[] for _ in data.horizons]
    for x, _, _, _ in data.batches(split, batch_size):
        for j, lg in enumerate(model(x)):
            chunks[j].append(torch.softmax(lg.float(), 1).cpu())
    return [torch.cat(c).numpy().astype(np.float64) for c in chunks]


def export(cfg, run_dir, device, label_set, data, splits=("train", "validation", "test")):
    """Best checkpoint -> probabilities -> argmax class -> representative -> predicted mid.

    Writes <split>_predictions.csv.gz (validation, test) with every column needed to
    recompute the four official metrics without the model, <split>_probs.npz (float16
    class probabilities, validation and test), and <split>_metrics.json for all splits.
    """
    run_dir = Path(run_dir)
    model = load_classifier(run_dir/"best", device)
    batch = cfg["training"]["eval_batch_size"]
    results = {}
    # GPU phase: every split's probabilities; then the CPU-only writing phase, during which
    # the scheduler does not count this job against the GPU capacity.
    all_probs = {split: predict_split(model, data, split, batch) for split in splits}
    set_status(run_dir, state="writing")
    for split in splits:
        s = data.splits[split]
        probs = all_probs.pop(split)
        if any(len(p) != len(s) for p in probs):
            raise ValueError("Prediction count does not match the split")
        origin_mid = data.mid[s.origins].astype(np.float64)
        table = {"origin_index": s.origins.astype(np.int64),
                 "origin_timestamp_ns": data.timestamps[s.origins].astype(np.int64),
                 "origin_mid": origin_mid}
        metrics, diagnostics = {}, {}
        for j, h in enumerate(data.horizons):
            tag, spec, p = HORIZON_TAG[h], label_set[h], probs[j]
            target_index = s.targets[:, data.columns[j]]
            target_mid = data.mid[target_index].astype(np.float64)
            pred_class = p.argmax(1)
            pred_delta = spec.decode(pred_class)
            expected_delta = spec.expected(p)
            true_delta = displacement(target_mid, origin_mid)
            if not np.array_equal(spec.assign(true_delta), s.labels[:, j]):
                raise AssertionError("Exported true classes disagree with the training labels")
            table |= {f"target_index_{tag}": target_index.astype(np.int64),
                      f"target_timestamp_ns_{tag}": data.timestamps[target_index].astype(np.int64),
                      f"target_mid_{tag}": target_mid,
                      f"true_delta_{tag}": true_delta,
                      f"true_class_{tag}": s.labels[:, j],
                      f"pred_class_{tag}": pred_class.astype(np.int64),
                      f"pred_delta_{tag}": pred_delta,
                      f"pred_mid_{tag}": origin_mid+pred_delta,
                      f"expected_delta_{tag}": expected_delta,
                      f"expected_mid_{tag}": origin_mid+expected_delta,
                      f"p_max_{tag}": p.max(1)}
            official, counts = official_metrics(origin_mid, target_mid, origin_mid+pred_delta)
            expected, _ = official_metrics(origin_mid, target_mid, origin_mid+expected_delta)
            metrics[tag] = official
            ce = float(-np.log(np.clip(p[np.arange(len(p)), s.labels[:, j]], 1e-300, None)).mean())
            diagnostics[tag] = dict(counts=counts, expected_value_decoding=expected, cross_entropy=ce,
                                    predicted_class_histogram=np.bincount(pred_class, minlength=spec.n_classes).tolist(),
                                    true_class_histogram=np.bincount(s.labels[:, j], minlength=spec.n_classes).tolist())
        frame = pd.DataFrame(table)
        if not np.isfinite(frame.select_dtypes("number").to_numpy(dtype=np.float64)).all():
            raise FloatingPointError("Nonfinite prediction table")
        result = dict(split=split, samples=len(s), sample_fingerprint=fingerprint(s.origins, s.targets),
                      origin_index_sha256=s.origin_sha256, decoding="argmax class -> representative",
                      official_metrics=metrics, da_convention=DA_CONVENTION,
                      diagnostics_not_official=diagnostics)
        if split != "train":
            frame.to_csv(run_dir/f"{split}_predictions.csv.gz", index=False,
                         compression={"method": "gzip", "mtime": 0})
            np.savez_compressed(run_dir/f"{split}_probs.npz",
                                **{HORIZON_TAG[h]: probs[j].astype(np.float16) for j, h in enumerate(data.horizons)},
                                origin_index=s.origins)
        write_json(run_dir/f"{split}_metrics.json", result)
        results[split] = result
    return model, results


def run(cfg, run_dir, repo_root):
    """One job end to end: fit (resuming from last/ if present) -> export -> summary."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir/"run_config.json", cfg)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if device.type == "cuda" and cfg["training"]["precision"] == "bf16" else None
    set_status(run_dir, state="preparing", pid=os.getpid(), host=platform.node())
    label_set, data = prepare(cfg, device)
    write_json(run_dir/"label_set.json", label_set.to_dict() | {"definition_sha256": label_set.sha256()})
    write_json(run_dir/"environment.json", environment())
    write_json(run_dir/"git.json", git_state(repo_root))
    fitted = fit(cfg, run_dir, device, dtype, label_set, data, repo_root)
    set_status(run_dir, state="exporting")
    model, results = export(cfg, run_dir, device, label_set, data)
    history = [json.loads(l) for l in (run_dir/"training_history.jsonl").read_text().splitlines() if l.strip()]
    summary = dict(job_id=cfg["job_id"], stage=cfg["stage"], arch=cfg["arch"], method=cfg["method"],
                   label_set=cfg["label_set_path"], label_definition_sha256=label_set.sha256(),
                   fold=cfg["fold"], horizons=cfg["horizons"], variant=cfg.get("variant", "baseline"),
                   parameters=parameter_count(model), model_config=model.model_config,
                   best_epoch=fitted["best_epoch"], best_validation_mean_ce=fitted["best_validation_mean_ce"],
                   epochs_completed=len(history), sample_counts={k: len(v) for k, v in data.splits.items()},
                   history_rows=data.history_rows, stride_rows=data.metadata["preprocessing"]["stride_rows"],
                   feature_schema=data.metadata["feature_schema"]["names"],
                   normalization=data.metadata["preprocessing"]["normalization"],
                   metrics={split: r["official_metrics"] for split, r in results.items()},
                   test_sample_fingerprint=results["test"]["sample_fingerprint"],
                   validation_sample_fingerprint=results["validation"]["sample_fingerprint"],
                   train_seconds=sum(r["epoch_seconds"] for r in history),
                   mean_train_samples_per_second=float(np.mean([r["train_samples_per_second"] for r in history])),
                   peak_allocated_bytes=max((r["peak_allocated_bytes"] or 0) for r in history),
                   peak_reserved_bytes=max((r["peak_reserved_bytes"] or 0) for r in history),
                   wall_seconds_this_process=time.time()-started,
                   environment=environment(), git=git_state(repo_root), da_convention=DA_CONVENTION)
    write_json(run_dir/"run_summary.json", summary)
    set_status(run_dir, state="completed")
    return summary
