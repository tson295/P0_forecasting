"""Full CSV integrity audit and <=2-sample model smoke tests; never calls fit()."""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.config import Config, MODELS
from src.data.dataset import prepare_data, LOBDataset
from src.models import build_model
from src.training.checkpoint import from_pretrained, write_json
from src.training.trainer import smoke_forward, seed_everything


def audit_windows(data, config):
    raw = data.raw
    report = {}
    for name, ds in data.datasets.items():
        origins, targets = ds.origins, ds.target_indices
        starts = origins-ds.history_rows+1
        lo, hi = data.metadata["split_manifest"]["ranges"][name]
        assert (starts >= lo).all() and (targets < hi).all()
        assert (raw.bad_prefix[targets[:, -1]] == raw.bad_prefix[starts]).all()
        wanted = raw.timestamps[origins, None]+np.array([60, 120, 180])*10**9
        actual = raw.timestamps[targets]
        assert (actual >= wanted).all()
        assert (actual-wanted <= config.data.target_tolerance_seconds*1e9).all()
        assert (raw.timestamps[targets-1] < wanted).all()  # FIRST observation at/after.
        samples = []
        for idx in (0, len(ds)//2, len(ds)-1):
            _, y = ds[idx]
            expected = np.log(raw.mid[targets[idx]]/raw.mid[origins[idx]])
            np.testing.assert_allclose(y.numpy(), expected, rtol=1e-6, atol=1e-10)
            samples.append(dict(origin=int(origins[idx]), origin_ns=int(raw.timestamps[origins[idx]]),
                                target_ns=actual[idx].tolist(),
                                target_delay_seconds=((actual[idx]-raw.timestamps[origins[idx]])/1e9).tolist(),
                                returns=y.tolist()))
        report[name] = dict(samples=len(ds), all_windows_continuous=True,
                            max_overshoot_seconds=((actual-wanted).max(axis=0)/1e9).tolist(), examples=samples)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="reports/validation.json")
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(42)
    result = dict(training_epochs_run=0, optimizer_steps_run=0, models={})
    configs = Path(__file__).resolve().parents[1]/"configs"
    for name in MODELS:
        # Audit the frozen base-experiment runs, not the dataclass defaults.
        run_name = "e0_60s" if name == "e0" else f"{name}_60s_base"
        config = Config.load(configs/f"{run_name}.json")
        config.data.csv_path = args.csv
        data = prepare_data(config)
        if name == "e0":
            result.update(stats=data.raw.stats, metadata=data.metadata,
                          windows=audit_windows(data, config))
            result["histories"] = {}
            for seconds in (60, 120, 180):
                dt = data.metadata["preprocessing"]["median_train_dt_seconds"]
                h, stride = int(np.ceil(seconds/dt)), max(1, round(seconds/6/dt))
                counts = {}
                for split, bounds in data.metadata["split_manifest"]["ranges"].items():
                    ds = LOBDataset(data.datasets[split].x, data.raw, bounds, h, stride, config.data)
                    assert len(ds) and (data.raw.bad_prefix[ds.target_indices[:, -1]] ==
                                        data.raw.bad_prefix[ds.origins-h+1]).all()
                    counts[split] = len(ds)
                result["histories"][str(seconds)] = dict(history_rows=h, stride_rows=stride, samples=counts)
        model = build_model(name, data.history_rows, data.channels)
        check = smoke_forward(model, data.datasets["train"], args.backward and name != "e0")
        check["architecture"] = model.model_config
        check["features"] = data.metadata["feature_schema"]
        model.eval()
        x = data.datasets["train"][0][0].unsqueeze(0)
        with torch.no_grad():
            before = model(x)
        with tempfile.TemporaryDirectory(prefix="lob-checkpoint-") as tmp:
            model.save_pretrained(tmp, data.metadata | {"experiment": config.to_dict()})
            restored = from_pretrained(tmp)
            with torch.no_grad():
                torch.testing.assert_close(restored(x), before, rtol=0, atol=0)
            check["safetensors_roundtrip_exact"] = True
            del restored
        result["models"][name] = check
        print(name, check["input_shape"], "->", check["output_shape"], "PASS", flush=True)
        del model, data, x
        gc.collect()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
