"""Contracts of the classification branch (src/cls)."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.cls import HORIZONS
from src.cls.data import ClassificationData
from src.cls.fast_moderntcn import FastModernTCN
from src.cls.labels import LabelSet, fit_equal_width, fit_quantile
from src.cls.metrics import official_metrics
from src.cls.models import build_classifier, load_classifier, save_weights
from src.data.dataset import LOBDataset
from src.models import build_model
from src.utils.metrics import price_metrics
from tests.test_gate1y import START_US, STEP_US

ROWS = 2000


def write_walk_csv(path, rows=ROWS, seed=0):
    """A 10 s grid whose mid follows a random walk on a 0.05 USD grid (so delta varies)."""
    rng = np.random.default_rng(seed)
    mid = 100_000+np.cumsum(rng.integers(-40, 41, rows))*0.05
    frame = pd.DataFrame({"timestamp_us": START_US+np.arange(rows)*STEP_US})
    for side, sign in (("bid", -1), ("ask", 1)):
        for level in range(1, 11):
            frame[f"{side}_price_{level}"] = mid+sign*(0.05+0.1*(level-1))
            frame[f"{side}_qty_{level}"] = rng.uniform(0.1, 5, rows)
    frame.to_csv(path, index=False)


def data_fields(fold=1):
    return dict(timestamp_column="timestamp_us", timestamp_unit="us", segment_column="segment_id",
                history_seconds=490, history_rows=None, stride_seconds=20.0,
                target_tolerance_seconds=10.0, max_gap_seconds=10.0, of_representation="of",
                sort_by_timestamp=True, duplicate_timestamp_policy="keep_first",
                split_scheme="walk_forward", folds=3, fold=fold, test_fraction=0.15,
                embargo_seconds=180.0)


@pytest.fixture(scope="module")
def walk_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("cls")/"walk.csv"
    write_walk_csv(path)
    return str(path)


# ---------------------------------------------------------------- labels

def test_equal_width_edges_zero_boundary_and_overflow():
    rng = np.random.default_rng(1)
    delta = np.round(rng.standard_t(3, 50_000)*20, 2)
    spec = fit_equal_width(delta, 60, bins=32, percentile=99.0)
    q, w = spec.params["q"], spec.params["width"]
    assert q == np.percentile(np.abs(delta), 99.0)
    assert spec.n_classes == 34 and len(spec.edges) == 33
    assert spec.edges[0] == -q and spec.edges[-1] == q and spec.edges[16] == 0.0
    assert np.allclose(np.diff(spec.edges), w)
    # No finite bin crosses zero; zero opens the upper central bin [0, w).
    # (Displacements are snapped to 1e-6 USD, the smallest distinguishable move.)
    assert spec.assign([0.0])[0] == 17 and spec.assign([-1e-6])[0] == 16 and spec.assign([-1e-9])[0] == 17
    # Closed finite range [-q, +q]; overflow strictly outside it, never clipped.
    assert spec.assign([-q])[0] == 1 and spec.assign([q])[0] == 32
    assert spec.assign([q+1e-6])[0] == 33 and spec.assign([-q-1e-6])[0] == 0
    assert spec.assign([-10*q])[0] == 0 and spec.assign([10*q])[0] == 33
    # Official decoding: midpoints; overflow at -q - w/2 and +q + w/2.
    assert spec.representatives[0] == -q-w/2 and spec.representatives[-1] == q+w/2
    assert np.allclose(spec.representatives[1:-1], (np.array(spec.edges[:-1])+spec.edges[1:])/2)
    counts = np.bincount(spec.assign(delta), minlength=34)
    assert counts.sum() == len(delta) and spec.fit["counts"] == counts.tolist()


def test_interval_assignment_is_deterministic_not_nearest():
    # The prompt's example: [20,30) and [30,40) are neighbours; 36.7 belongs to [30,40).
    spec = fit_equal_width(np.r_[np.linspace(-100, 100, 1001)], 60, bins=20, percentile=100.0)
    assert spec.params["width"] == 10.0
    c = spec.assign([36.7, 30.0, 29.999, 39.999])
    lo = np.array(spec.edges)[c-1]
    assert lo.tolist() == [30.0, 30.0, 20.0, 30.0]
    # 36.7 is nearer the [30,40) midpoint 35 anyway; 38.9 is nearer 45 but still [30,40).
    assert np.array(spec.edges)[spec.assign([38.9])[0]-1] == 30.0


def test_quantile_bins_are_equal_frequency_and_merge_point_masses():
    rng = np.random.default_rng(2)
    delta = np.round(rng.normal(0, 10, 34_000), 6)  # already on the 1e-6 USD snap grid
    spec = fit_quantile(delta, 120, n_classes=34)
    counts = np.bincount(spec.assign(delta), minlength=spec.n_classes)
    assert spec.n_classes == 34 and abs(counts.max()-counts.min()) <= 2
    assert spec.representatives[0] == np.median(delta[delta < spec.edges[0]])
    assert spec.representatives[-1] == np.median(delta[delta >= spec.edges[-1]])
    massed = np.r_[np.zeros(4000), rng.normal(0, 10, 30_000)]  # 11.8% exactly zero
    spec = fit_quantile(massed, 60, n_classes=34)
    assert spec.params["merged_duplicate_edges"] >= 2 and spec.n_classes < 34
    assert len(set(spec.edges)) == len(spec.edges) and 0.0 in spec.edges


def test_label_set_roundtrip_and_fingerprint(tmp_path):
    d = np.random.default_rng(3).normal(0, 30, (5000, 3))
    ls = LabelSet("x", [fit_equal_width(d[:, j], h) for j, h in enumerate(HORIZONS)])
    path = ls.save(tmp_path/"x.json")
    back = LabelSet.load(path)
    assert back.sha256() == ls.sha256() and back[60].edges == ls[60].edges
    obj = json.loads(path.read_text())
    obj["horizons"]["60"]["edges"][3] += 1e-9
    with pytest.raises(ValueError, match="fingerprint"):
        LabelSet.from_dict(obj)


# ---------------------------------------------------------------- metrics

def test_official_metrics_match_wf3_price_metrics_and_da_convention():
    rng = np.random.default_rng(4)
    origin = 100_000+rng.normal(0, 100, (500, 1)).repeat(3, 1)
    target = origin+np.round(rng.normal(0, 30, (500, 3)), 2)
    target[:20, :] = origin[:20, :]  # exact zero moves are excluded from DA
    pred = origin+rng.normal(0, 10, (500, 3))
    wf3 = price_metrics(origin, target, pred)
    for j in range(3):
        m, c = official_metrics(origin[:, j], target[:, j], pred[:, j])
        assert np.isclose(m["rmse"], wf3["rmse"][j]) and np.isclose(m["mae"], wf3["mae"][j])
        assert np.isclose(m["r2_gain_vs_e0"], wf3["r2_gain_vs_e0"][j])
        moved = target[:, j] != origin[:, j]
        assert c["da_samples"] == moved.sum() == 480
        assert m["da"] == np.mean(np.sign(pred[moved, j]-origin[moved, j]) == np.sign(target[moved, j]-origin[moved, j]))
    e0, _ = official_metrics(origin[:, 0], target[:, 0], origin[:, 0])
    assert e0["r2_gain_vs_e0"] == 0.0 and e0["da"] == 0.0  # no call is a miss


# ---------------------------------------------------------------- models

def test_fast_moderntcn_is_the_wf3_module_numerically():
    torch.manual_seed(0)
    kwargs = dict(dims=[8, 8, 8, 8])
    ref = build_model("moderntcn", 49, 4, **kwargs).double()
    fast = FastModernTCN(copy.deepcopy(ref.model_config)).double()
    fast.load_state_dict(ref.state_dict())
    for m in (ref, fast):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
    x = torch.randn(5, 49, 4, dtype=torch.float64)
    for training in (True, False):
        ref.train(training)
        fast.train(training)
        a, b = ref(x), fast(x)
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-10)
        if training:
            a.square().sum().backward()
            b.square().sum().backward()
            for (name, p1), (_, p2) in zip(ref.named_parameters(), fast.named_parameters()):
                if name != "stem.0.bias":  # feeds a BatchNorm: its gradient is exactly zero
                    assert torch.allclose(p1.grad, p2.grad, atol=1e-10, rtol=1e-8), name
    for b1, b2 in zip(ref.buffers(), fast.buffers()):
        assert torch.allclose(b1.double(), b2.double())


@pytest.mark.parametrize("arch,channels,kw", [("ofi_lstm", 20, dict(hidden_size=16, layers=2)),
                                               ("moderntcn", 40, dict(dims=[8, 8, 8, 8])),
                                               ("transformer", 40, dict(d_model=32, heads=4, layers=2, ffn_dim=64))])
def test_classifiers_emit_one_logit_block_per_horizon_and_reload(tmp_path, arch, channels, kw):
    for horizons, n in (([60, 120, 180], [34, 33, 34]), ([180], [34])):
        model = build_classifier(arch, 49, channels, n, horizons, **kw).eval()
        x = torch.randn(3, 49, channels)
        out = model(x)
        assert [tuple(o.shape) for o in out] == [(3, k) for k in n]
        save_weights(model, tmp_path/arch)
        back = load_classifier(tmp_path/arch)
        assert all(torch.equal(a, b) for a, b in zip(out, back(x)))


def test_backbones_are_the_wf3_backbones():
    lstm = build_classifier("ofi_lstm", 49, 20, [34]*3, HORIZONS)
    wf3 = build_model("ofi_lstm", 49, 20, hidden_size=384, layers=3)
    assert {k: v.shape for k, v in lstm.state_dict().items() if not k.startswith("head")} == \
        {k: v.shape for k, v in wf3.state_dict().items() if not k.startswith("head")}
    tcn = build_classifier("moderntcn", 49, 40, [34]*3, HORIZONS)
    wf3 = build_model("moderntcn", 49, 40)
    assert {k: v.shape for k, v in tcn.state_dict().items() if not k.startswith("head")} == \
        {k: v.shape for k, v in wf3.state_dict().items() if not k.startswith("head")}


# ---------------------------------------------------------------- data

def test_gpu_batching_reproduces_lobdataset_windows_and_labels(walk_csv):
    d = np.random.default_rng(5).normal(0, 1, (1000, 3))
    ls = LabelSet("t", [fit_equal_width(d[:, j], h, bins=8) for j, h in enumerate(HORIZONS)])
    data = ClassificationData("moderntcn", data_fields(), walk_csv, ls, HORIZONS, torch.device("cpu"))
    for name, ds in data.prepared.datasets.items():
        assert isinstance(ds, LOBDataset)
        split = data.splits[name]
        assert np.array_equal(split.origins, ds.origins)
        seen = 0
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))
        for x, y, delta, index in data.batches(name, 7, order):
            for row in range(len(index)):
                i = int(index[row])
                ref_x, ref_y = ds[i]
                assert torch.equal(x[row], ref_x)
                # WF3 target is the log return of the same target row; ours is its USD delta.
                mid = data.mid
                o, t = ds.origins[i], ds.target_indices[i]
                assert np.allclose(np.log(mid[t]/mid[o]), ref_y.numpy(), atol=1e-7)
                assert np.array_equal(delta[row].numpy(), np.round(mid[t]-mid[o], 6))  # snapped USD delta
                assert [int(v) for v in y[row]] == [int(ls[h].assign([mid[t[j]]-mid[o]])[0]) for j, h in enumerate(HORIZONS)]
            seen += len(index)
        assert seen == len(ds)


def test_shuffle_order_is_the_wf3_random_sampler_permutation():
    from torch.utils.data import RandomSampler
    n = 1234
    ours = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    theirs = list(RandomSampler(range(n), generator=torch.Generator().manual_seed(42)))
    assert ours == theirs


# ---------------------------------------------------------------- snapping and end to end

def test_displacement_snapping_makes_classes_a_function_of_the_usd_value():
    from src.cls.labels import displacement, snap
    a, b = np.array([-0.10000000000582]), np.array([-0.09999999999127])
    assert snap(a)[0] == snap(b)[0] == -0.1
    massed = np.r_[np.full(500, -0.10000000000582), np.full(500, -0.09999999999127),
                   np.random.default_rng(6).normal(0, 1, 9000)]
    spec = fit_quantile(massed, 60, n_classes=34)
    assert len(set(spec.assign(massed[:1000]).tolist())) == 1
    assert displacement(np.array([100.15]), np.array([100.05]))[0] == 0.1


def test_end_to_end_run_resume_export_and_audit_comparisons(tmp_path, walk_csv):
    """A tiny CPU run: 1 epoch, then resume to 2; exported tables satisfy the audit's
    exact per-run comparisons after a round-trip CSV read."""
    from src.cls.labels import displacement
    from src.cls.metrics import official_metrics
    from src.cls.trainer import run
    fields = data_fields()
    probe = ClassificationData("transformer", fields, walk_csv,
                               LabelSet("p", [fit_equal_width(np.r_[-1., 1.], h, bins=2) for h in HORIZONS]),
                               HORIZONS, torch.device("cpu"))
    train_delta = displacement(probe.mid[probe.splits["train"].targets], probe.mid[probe.splits["train"].origins][:, None])
    for method in ("equal_width", "quantile"):
        specs = [fit_equal_width(train_delta[:, j], h, bins=8) if method == "equal_width"
                 else fit_quantile(train_delta[:, j], h, n_classes=10) for j, h in enumerate(HORIZONS)]
        path = LabelSet(method, specs, meta=dict(variant=dict(method=method))).save(tmp_path/f"{method}.json")
        for horizons in ([60, 120, 180], [120]):
            cfg = dict(job_id="t", stage="test", method=method, formulation="multi" if len(horizons) == 3 else "single",
                       arch="transformer", horizons=horizons, fold=1, label_set_path=str(path), data=fields,
                       csv_path=walk_csv, model_kwargs=dict(d_model=16, heads=2, layers=1, ffn_dim=32),
                       training=dict(epochs=1, batch_size=16, learning_rate=1e-3, weight_decay=1e-4, gradient_clip=1.0,
                                     precision="fp32", seed=42, eval_batch_size=64, compile_mode="none"))
            run_dir = tmp_path/f"{method}_{len(horizons)}"
            run(cfg, run_dir, ".")
            cfg["training"]["epochs"] = 2
            summary = run(cfg, run_dir, ".")          # resumes from last/
            history = [json.loads(l) for l in (run_dir/"training_history.jsonl").read_text().splitlines()]
            assert [r["epoch"] for r in history] == [0, 1]
            ces = [r["validation"]["mean_ce"] for r in history]
            selected = json.loads((run_dir/"best/selection.json").read_text())["epoch"]
            assert history[int(np.argmin(ces))]["epoch"] == summary["best_epoch"] == selected
            ls = LabelSet.load(path)
            for split in ("validation", "test"):
                frame = pd.read_csv(run_dir/f"{split}_predictions.csv.gz", float_precision="round_trip")
                origins = frame["origin_index"].to_numpy(np.int64)
                metrics = json.loads((run_dir/f"{split}_metrics.json").read_text())["official_metrics"]
                for h in horizons:
                    t = f"{h}s"
                    true = displacement(probe.mid[frame[f"target_index_{t}"].to_numpy(np.int64)], probe.mid[origins])
                    assert np.array_equal(frame[f"true_delta_{t}"].to_numpy(), true)
                    assert np.array_equal(frame[f"true_class_{t}"].to_numpy(np.int64), ls[h].assign(true))
                    assert np.array_equal(frame[f"pred_delta_{t}"].to_numpy(),
                                          ls[h].decode(frame[f"pred_class_{t}"].to_numpy(np.int64)))
                    again, _ = official_metrics(frame["origin_mid"], frame[f"target_mid_{t}"], frame[f"pred_mid_{t}"])
                    assert all(np.isclose(again[k], metrics[t][k], rtol=1e-12) for k in again)
