"""Pre-training gate: every frozen dataset, split, sampling, capacity and LoRA fact."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.config import MODELS, Config
from src.data.dataset import prepare_data
from src.data.preprocessing import TRAIN_GLOBAL_NORMALIZATION
from src.models import WINDOW_LOCAL_NORMALIZATION, build_model
from src.training.checkpoint import write_json
from src.training.trainer import seed_everything

ROOT = Path(__file__).resolve().parents[1]
CSV_SHA256 = "e42bb29e79bdff68f94540916983b1cba2bbbfd745b30933e755e04ec92a60ab"
DATA_STATS = dict(rows=678362, median_dt_seconds=1.236, p99_dt_seconds=1.683,
                  max_dt_seconds=6.221, gaps_gt_2_seconds=60, segment_transitions=1)
RANGES = dict(train=[0, 477086], validation=[477086, 578798], test=[578798, 678362])
SAMPLE_COUNTS = dict(train=58774, validation=12427, test=12145)
HISTORY_ROWS, STRIDE_SECONDS, STRIDE_ROWS = 49, 10.0, 8
HORIZONS, TOLERANCE_SECONDS, MAX_GAP_SECONDS, HISTORY_SECONDS = (60, 120, 180), 2.0, 2.0, 60
PARAMETERS = dict(e0=0, ofi_lstm=55491, hfformer=22026, patchtst=477059,
                  moderntcn=50568195, lit=736547)
CHANNELS = dict(e0=40, ofi_lstm=20, hfformer=38, patchtst=40, moderntcn=40, lit=40)
RUN_CONFIGS = dict(e0="e0_60s", ofi_lstm="ofi_lstm_60s_base", hfformer="hfformer_60s_base",
                   patchtst="patchtst_60s_base", moderntcn="moderntcn_60s_base",
                   lit="lit_60s_base")
# Section M: one retuned field silently changes what a 30-epoch run means.
FROZEN_TRAINING = dict(epochs=30, batch_size=128, learning_rate=1e-4, weight_decay=1e-4,
                       gradient_clip=1.0, gradient_accumulation=1, seed=42, loss="mse",
                       auto_batch_size=False)
# Section H: 49 and 8 must be resolved from the cadence, never pinned in the config.
FROZEN_DATA = dict(horizons_seconds=HORIZONS, target_tolerance_seconds=TOLERANCE_SECONDS,
                   max_gap_seconds=MAX_GAP_SECONDS, history_seconds=HISTORY_SECONDS,
                   history_rows_override=None, stride_seconds_override=None)
# Section W: future LoRA injection needs unmerged, individually named Linear modules.
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")
LORA_SUFFIXES = ATTENTION_PROJECTIONS+("fc1", "fc2")
# Sections K2 and K5 name fc1/fc2 for the two in-repo Transformers. Section K3 pins
# PatchTST to the stock transformers.PatchTSTModel at exactly 477,059 parameters, and
# that module keeps its feed-forward Linears inside an nn.Sequential named ff.0/ff.3;
# renaming them is the architecture change K3 forbids, so PatchTST is held to its four
# unmerged attention projections, which are the LoRA injection points that matter.
LORA_REQUIRED = dict(hfformer=LORA_SUFFIXES, lit=LORA_SUFFIXES, patchtst=ATTENTION_PROJECTIONS)
TOLERANCE = 1e-3
STANDARDIZED = 1e-4


class Gate:
    """Records every check instead of raising, so one run reports all violations."""
    def __init__(self):
        self.checks = []

    def check(self, name, passed, expected, actual):
        self.checks.append(dict(name=name, passed=bool(passed), expected=expected, actual=actual))
        return bool(passed)

    def equal(self, name, actual, expected):
        return self.check(name, actual == expected, expected, actual)

    def close(self, name, actual, expected, tolerance=TOLERANCE):
        return self.check(name, abs(actual-expected) <= tolerance, expected, actual)

    def failures(self):
        return [c for c in self.checks if not c["passed"]]


def fingerprint(datasets):
    """Origins AND target indices, so E0 shares the learned models' exact samples."""
    return {name: dict(origins=hashlib.sha256(ds.origins.tobytes()).hexdigest(),
                       targets=hashlib.sha256(ds.target_indices.tobytes()).hexdigest())
            for name, ds in datasets.items()}


def dataset_checks(gate, data):
    stats, manifest = data.raw.stats, data.metadata["split_manifest"]
    gate.equal("dataset.sha256", manifest["source"]["sha256"], CSV_SHA256)
    gate.equal("dataset.rows", stats["rows"], DATA_STATS["rows"])
    for key in ("median_dt_seconds", "p99_dt_seconds", "max_dt_seconds"):
        gate.close(f"dataset.{key}", stats[key], DATA_STATS[key])
    gate.equal("dataset.gaps_gt_2_seconds", stats["gaps_gt_2_seconds"], DATA_STATS["gaps_gt_2_seconds"])
    gate.equal("dataset.segment_transitions", stats["segment_transitions"], DATA_STATS["segment_transitions"])


def config_checks(gate, model, config):
    training, data = config.training, config.data
    gate.equal(f"{model}.config_model", config.model, model)
    gate.equal(f"{model}.run_name", training.run_name, RUN_CONFIGS[model])
    gate.equal(f"{model}.training_hyperparameters",
               {k: getattr(training, k) for k in FROZEN_TRAINING}, FROZEN_TRAINING)
    # Section N: BF16 with TF32; "auto" resolves to BF16 on this GPU, fp16/fp32 do not.
    gate.check(f"{model}.precision", training.precision in ("auto", "bf16"),
               "auto or bf16", training.precision)
    gate.equal(f"{model}.data_contract",
               dict(horizons_seconds=tuple(data.horizons_seconds),
                    target_tolerance_seconds=data.target_tolerance_seconds,
                    max_gap_seconds=data.max_gap_seconds, history_seconds=data.history_seconds,
                    history_rows_override=data.history_rows,
                    stride_seconds_override=data.stride_seconds), FROZEN_DATA)


def split_checks(gate, model, data):
    pre, manifest = data.metadata["preprocessing"], data.metadata["split_manifest"]
    gate.equal(f"{model}.split_ranges", {k: list(v) for k, v in manifest["ranges"].items()}, RANGES)
    gate.equal(f"{model}.sample_counts", manifest["sample_counts"], SAMPLE_COUNTS)
    gate.equal(f"{model}.history_rows", data.history_rows, HISTORY_ROWS)
    gate.equal(f"{model}.history_rows_metadata", pre["history_rows"], HISTORY_ROWS)
    gate.equal(f"{model}.stride_seconds", pre["stride_seconds"], STRIDE_SECONDS)
    gate.equal(f"{model}.stride_rows", pre["stride_rows"], STRIDE_ROWS)
    gate.equal(f"{model}.channels", data.channels, CHANNELS[model])


def sampling_checks(gate, model, data):
    """Sections E/F/G re-derived from raw timestamps: the stored indices cannot self-certify."""
    raw, ranges = data.raw, data.metadata["split_manifest"]["ranges"]
    timestamps, segments = raw.timestamps, raw.segments
    # dt == max_gap is still a good edge, so only a strict overshoot breaks continuity.
    bad = np.r_[0, np.cumsum((np.diff(timestamps)/1e9 > MAX_GAP_SECONDS)
                             | (segments[1:] != segments[:-1]), dtype=np.int64)]
    horizons = np.array(HORIZONS, dtype=np.int64)*10**9
    tolerance = round(TOLERANCE_SECONDS*1e9)
    for name, (lo, hi) in ranges.items():
        dataset = data.datasets[name]
        origins = np.arange(lo+HISTORY_ROWS-1, hi, STRIDE_ROWS, dtype=np.int64)
        wanted = timestamps[origins, None]+horizons
        target = np.searchsorted(timestamps, wanted, side="left")
        safe = np.minimum(target, len(timestamps)-1)
        overshoot = timestamps[safe]-wanted
        keep = ((target < hi).all(axis=1) & ((overshoot >= 0) & (overshoot <= tolerance)).all(axis=1)
                & (bad[safe[:, -1]]-bad[origins-HISTORY_ROWS+1] == 0))
        stored = timestamps[dataset.target_indices]-(timestamps[dataset.origins, None]+horizons)
        gate.equal(f"{model}.{name}_sampling", dict(
            count=len(dataset.origins),
            origins=np.array_equal(origins[keep], dataset.origins),
            targets=np.array_equal(target[keep], dataset.target_indices),
            inside_split=bool(dataset.origins.min()-HISTORY_ROWS+1 >= lo
                              and dataset.target_indices.max() < hi),
            within_tolerance=bool(((stored >= 0) & (stored <= tolerance)).all()),
            continuous=bool((bad[dataset.target_indices[:, -1]]
                             - bad[dataset.origins-HISTORY_ROWS+1] == 0).all()),
        ), dict(count=int(keep.sum()), origins=True, targets=True, inside_split=True,
                within_tolerance=True, continuous=True))


def normalization_checks(gate, model, data):
    pre = data.metadata["preprocessing"]
    scaler = pre["standardizer"]
    lo, hi = data.metadata["split_manifest"]["ranges"]["train"]
    x = data.datasets["train"].x
    offset = float(np.abs(x[lo:hi].mean(axis=0, dtype=np.float64)).max())
    spread = float(np.abs(x[lo:hi].std(axis=0, dtype=np.float64)-1).max())
    measured = dict(train_feature_mean_max=offset, train_feature_std_deviation_max=spread)
    if model == "hfformer":
        gate.equal("hfformer.standardizer", scaler, None)
        gate.equal("hfformer.normalization", pre["normalization"], WINDOW_LOCAL_NORMALIZATION)
        gate.equal("hfformer.feature_schema_normalization",
                   data.metadata["feature_schema"]["normalization"], WINDOW_LOCAL_NORMALIZATION)
        # Any corpus z-score would leave the train rows at mean 0; L10 prices sit far away.
        gate.check("hfformer.features_are_not_corpus_scaled", offset > 1,
                   "train-slice feature mean far from zero", offset)
        return dict(standardizer=None, normalization=pre["normalization"], **measured)
    fitted = scaler["fitted_on"] if isinstance(scaler, dict) else None
    gate.equal(f"{model}.standardizer_fitted_on", fitted, "train_only")
    gate.equal(f"{model}.normalization", pre["normalization"], TRAIN_GLOBAL_NORMALIZATION)
    # "train_only" is a hard-coded label, so measure it: only a fit over exactly the
    # train rows leaves exactly those rows at mean 0 / std 1.
    gate.check(f"{model}.standardized_on_train_rows_only", max(offset, spread) <= STANDARDIZED,
               f"train-slice mean 0 and std 1 within {STANDARDIZED}", measured)
    features = np.size(scaler["mean"]) if isinstance(scaler, dict) else 0
    return dict(standardizer=f"{fitted}, {features} features",
                normalization=pre["normalization"], **measured)


def model_checks(gate, model, data):
    net = build_model(model, data.history_rows, data.channels)
    parameters = sum(p.numel() for p in net.parameters())
    gate.equal(f"{model}.parameters", parameters, PARAMETERS[model])
    record = dict(parameters=parameters)
    required = LORA_REQUIRED.get(model)
    if required:
        # Exact Linear leaf names: a merged qkv or a renamed projection must not pass.
        linears = {name.rsplit(".", 1)[-1] for name, module in net.named_modules()
                   if isinstance(module, torch.nn.Linear)}
        record["lora_modules"] = [suffix for suffix in required if suffix in linears]
        # Always recorded, so a missing adapter point stays locatable from the report.
        record["linear_leaf_names"] = sorted(linears)
        gate.equal(f"{model}.lora_modules", record["lora_modules"], list(required))
    del net
    gc.collect()
    return record


def render(checks):
    width = max(len(c["name"]) for c in checks)
    for c in checks:
        detail = "" if c["passed"] else f"  expected {c['expected']!r:.200} got {c['actual']!r:.200}"
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']:<{width}}{detail}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=os.environ.get("LOB_CSV") or str(ROOT/"BTCUSDT_L10_oct2023.csv"))
    p.add_argument("--output", default=str(ROOT/"reports/contract_check.json"))
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(42)
    gate, summary, reference = Gate(), {}, None
    for model in MODELS:
        config = Config.load(ROOT/"configs"/f"{RUN_CONFIGS[model]}.json")
        config_checks(gate, model, config)
        config.data.csv_path = args.csv
        data = prepare_data(config)
        if reference is None:
            dataset_checks(gate, data)
        split_checks(gate, model, data)
        sampling_checks(gate, model, data)
        marks = fingerprint(data.datasets)
        gate.equal(f"{model}.origin_index_sha256",
                   {k: v["origins"] for k, v in marks.items()},
                   data.metadata["split_manifest"]["origin_index_sha256"])
        if reference is None:
            reference = marks  # E0 defines the frozen origins/targets every model must share.
        else:
            gate.equal(f"{model}.same_origins_and_targets_as_e0", marks, reference)
        summary[model] = dict(sample_counts=data.metadata["split_manifest"]["sample_counts"],
                              channels=data.channels, samples=marks,
                              **normalization_checks(gate, model, data),
                              **model_checks(gate, model, data))
        print(f"checked {model}", flush=True)
        del data
        gc.collect()
    failures = gate.failures()
    render(gate.checks)
    report = dict(status="fail" if failures else "pass", csv=str(Path(args.csv).resolve()),
                  checks=gate.checks, failed=[c["name"] for c in failures], models=summary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f"\n{len(gate.checks)-len(failures)}/{len(gate.checks)} contracts hold; report {output}")
    if failures:
        print("CONTRACT FAILURE, do not train: " + ", ".join(c["name"] for c in failures))
        sys.exit(1)
    print("All frozen contracts hold; training may start.")


if __name__ == "__main__":
    main()
