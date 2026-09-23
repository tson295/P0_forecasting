"""Aggregate every completed classification run and write the report.

  python scripts/cls_report.py [--stage multi]

Outputs
  reports/cls_results_long.csv          one row per run x split x horizon (official + diagnostics)
  reports/FINAL_REPORT_CLASSIFICATION.md  (or reports/STAGE1_MULTI_REPORT.md with --stage multi)
Interpretation text lives in reports/cls_interpretation.md (hand-written after reading
the numbers) and is spliced in verbatim; everything else is generated from artifacts.
Official tables carry exactly four metrics: RMSE, MAE, R2 gain vs E0, DA.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.cls.labels import LabelSet
from src.cls.metrics import DA_CONVENTION

TAGS = ["60s", "120s", "180s"]
HNAME = {"60s": "h1 (60 s)", "120s": "h2 (120 s)", "180s": "h3 (180 s)"}
ARCH_NAME = dict(ofi_lstm="OFI-LSTM", moderntcn="ModernTCN", transformer="Transformer")
METHOD_NAME = dict(equal_width="equal-width ±p99", quantile="quantile")


def fmt(v, k):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if k in ("rmse", "mae"):
        return f"{v:.4f}"
    if k == "da":
        return f"{100*v:.2f}%"
    return f"{v:+.6f}"


def collect():
    rows = []
    for summary_path in sorted((ROOT/"runs/cls").rglob("run_summary.json")):
        run = summary_path.parent
        if "benchmark_work" in run.parts:
            continue
        s = json.loads(summary_path.read_text())
        cfg = json.loads((run/"run_config.json").read_text())
        for split in ("train", "validation", "test"):
            path = run/f"{split}_metrics.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())
            for tag, off in m["official_metrics"].items():
                diag = m["diagnostics_not_official"][tag]
                ev = diag["expected_value_decoding"]
                rows.append(dict(job_id=s["job_id"], stage=s["stage"], experiment=cfg.get("experiment") or "",
                                 variant=s.get("variant", "baseline"), method=s["method"],
                                 label_set=Path(s["label_set"]).stem, formulation=cfg["formulation"],
                                 arch=s["arch"], fold=s["fold"], horizon=tag, split=split,
                                 rmse=off["rmse"], mae=off["mae"], r2_gain_vs_e0=off["r2_gain_vs_e0"], da=off["da"],
                                 rmse_e0=diag["counts"]["rmse_e0"], mae_e0=diag["counts"]["mae_e0"],
                                 ev_rmse=ev["rmse"], ev_mae=ev["mae"], ev_r2_gain_vs_e0=ev["r2_gain_vs_e0"], ev_da=ev["da"],
                                 cross_entropy=diag["cross_entropy"], best_epoch=s["best_epoch"],
                                 parameters=s["parameters"], run_dir=str(run.relative_to(ROOT))))
    return pd.DataFrame(rows)


def md_table(header, rows):
    out = ["| "+" | ".join(header)+" |", "|"+"|".join("---:" if i else "---" for i in range(len(header)))+"|"]
    out += ["| "+" | ".join(str(c) for c in r)+" |" for r in rows]
    return "\n".join(out)


def official_block(df, formulation, split="test", baseline_stage=True):
    """Per horizon: rows = method x arch x fold; columns = the four official metrics."""
    parts = []
    sub = df[(df.formulation == formulation) & (df.split == split) & (df.experiment == "")]
    for tag in TAGS:
        d = sub[sub.horizon == tag].sort_values(["method", "arch", "fold"])
        if d.empty:
            continue
        # E0 depends only on the samples: one row for the shared test split, one per fold otherwise.
        e0s = d.drop_duplicates("fold").sort_values("fold")
        if split == "test":
            if d.rmse_e0.round(9).nunique() != 1:
                raise AssertionError("Test E0 differs across runs: the test split is not shared")
            e0s = e0s.iloc[:1]
        rows = [["E0 (pred = origin mid)", "", "" if split == "test" else f"F{r.fold}", fmt(r.rmse_e0, "rmse"),
                 fmt(r.mae_e0, "mae"), fmt(0.0, "r2"), fmt(0.0, "da")] for _, r in e0s.iterrows()]
        for _, r in d.iterrows():
            rows.append([METHOD_NAME[r.method], ARCH_NAME[r.arch], f"F{r.fold}", fmt(r.rmse, "rmse"), fmt(r.mae, "mae"),
                         fmt(r.r2_gain_vs_e0, "r2"), fmt(r.da, "da")])
        parts.append(f"**{HNAME[tag]} — {split}** ({int(len(d))} runs)\n\n"
                     + md_table(["Binning", "Architecture", "Fold", "RMSE (USD)", "MAE (USD)", "R² gain vs E0", "DA"], rows))
    return "\n\n".join(parts)


def pivot(df, value, formulation, split="test", index=("method", "arch"), columns=("horizon", "fold")):
    sub = df[(df.formulation == formulation) & (df.split == split) & (df.experiment == "")]
    if sub.empty:
        return "(no runs)"
    p = sub.pivot_table(index=list(index), columns=list(columns), values=value, aggfunc="first")
    p = p[sorted(p.columns, key=lambda c: (TAGS.index(c[0]), c[1]))]
    header = [" / ".join(index)]+[f"{c[0]} F{c[1]}" if isinstance(c, tuple) else str(c) for c in p.columns]
    rows = []
    for idx, r in p.iterrows():
        name = " / ".join(METHOD_NAME.get(i, ARCH_NAME.get(i, str(i))) for i in (idx if isinstance(idx, tuple) else (idx,)))
        rows.append([name]+[fmt(v, "da" if value.endswith("da") else ("rmse" if "rmse" in value or "mae" in value else "r2")) for v in r.values])
    return md_table(header, rows)


def wf3_block():
    path = ROOT/"reports/wf3_regression_reference.json"
    if not path.exists():
        return "(reference not computed)"
    ref = json.loads(path.read_text())
    rows = []
    for run, r in ref["runs"].items():
        for tag in TAGS:
            m = r["test"][tag]
            rows.append([ARCH_NAME[r["model"]], f"F{r['fold']}", tag, fmt(m["rmse"], "rmse"), fmt(m["mae"], "mae"),
                         fmt(m["r2_gain_vs_e0"], "r2"), fmt(m["da"], "da")])
    return md_table(["WF3 regression model", "Fold", "Horizon", "RMSE (USD)", "MAE (USD)", "R² gain vs E0", "DA"], rows)


def label_sections():
    out = {}
    for name in ("equal_width_k32_p99", "quantile_c34"):
        ls = LabelSet.load(ROOT/f"labeling/{name}.json")
        lines = []
        for h in (60, 120, 180):
            s = ls[h]
            p = s.params
            if s.method == "equal_width":
                lines.append(md_table(["Field", "Value"], [
                    ["horizon", f"{h} s"], ["q_h = p99(|delta|), F1 train", f"{p['q']:.6f} USD"], ["K (finite bins)", p["bins"]],
                    ["bin width", f"{p['width']:.6f} USD"], ["finite min / max", f"{p['finite_min']:.6f} / {p['finite_max']:.6f}"],
                    ["classes", s.n_classes], ["lower overflow (delta < -q) F1 train", f"{s.fit['lower_overflow_percent']:.4f}%"],
                    ["upper overflow (delta > +q) F1 train", f"{s.fit['upper_overflow_percent']:.4f}%"],
                    ["overflow representatives", f"{s.representatives[0]:.6f} / {s.representatives[-1]:.6f} (∓(q + width/2))"]]))
            else:
                lines.append(md_table(["Field", "Value"], [
                    ["horizon", f"{h} s"], ["requested / realized classes", f"{p['requested_classes']} / {s.n_classes}"],
                    ["merged duplicate edges (point mass at 0)", p["merged_duplicate_edges"]],
                    ["edge range", f"{s.edges[0]:.4f} .. {s.edges[-1]:.4f} USD"],
                    ["outer representatives (F1-train class medians)", f"{s.representatives[0]:.4f} / {s.representatives[-1]:.4f}"],
                    ["outer class shares F1 train", f"{100*s.fit['lower_outer_fraction']:.3f}% / {100*s.fit['upper_outer_fraction']:.3f}%"]]))
            rows = []
            for k, (iv, rep, c) in enumerate(zip(s.class_intervals(), s.representatives, s.fit["counts"])):
                rows.append([k, iv, f"{rep:.4f}", c, f"{100*c/s.fit['samples']:.3f}%"])
            lines.append(f"<details><summary>{h} s: exact edges, representatives and F1-train counts</summary>\n\n"
                         + md_table(["Class", "Interval (USD)", "Representative", "F1-train count", "Share"], rows)
                         + "\n\n</details>")
        out[name] = "\n\n".join(lines)
    return out


def distribution_section():
    d = json.loads((ROOT/"labeling/label_distribution.json").read_text())
    rows = []
    for fold, splits in d["folds"].items():
        for split, entry in splits.items():
            if split == "test" and fold != "1":
                continue
            for h, e in entry["horizons"].items():
                ew = e["label_sets"]["equal_width_k32_p99"]
                q = e["label_sets"]["quantile_c34"]
                central = ew["fractions"][16]+ew["fractions"][17]
                rows.append([f"F{fold} {split}" if split != "test" else "test (all folds)", f"{h} s", entry["provenance"]["samples"],
                             f"{e['rmse_e0']:.3f}", f"{100*e['zero_fraction']:.2f}%",
                             f"{100*ew['lower_outer_fraction']:.3f}% / {100*ew['upper_outer_fraction']:.3f}%",
                             f"{100*central:.1f}%", f"{100*q['lower_outer_fraction']:.2f}% / {100*q['upper_outer_fraction']:.2f}%"])
    return md_table(["Split", "Horizon", "Samples", "RMSE_E0 (USD)", "delta = 0", "EW overflow low / high",
                     "EW two central bins", "Q outer low / high"], rows)


def benchmark_section():
    path = ROOT/"benchmarks/concurrency_benchmark.json"
    if not path.exists():
        return "(benchmark missing)"
    b = json.loads(path.read_text())
    rows = []
    for c in b["configurations"]:
        rows.append([c["config"], " + ".join(c["archs"]), f"{c['normalized_throughput']:.2f}",
                     " / ".join(f"{s:.0f}" for s in c["samples_per_second"]),
                     f"{c['mean_gpu_util']:.0f}%", f"{c['max_gpu_used_bytes']/2**30:.1f}", f"{c['mean_cpu_percent']:.0f}%",
                     f"{c['max_ram_used_bytes']/2**30:.1f}"])
    single = b["single_job"]
    srows = [[ARCH_NAME[a], v["parameters"], f"{v['samples_per_second']:.0f}", f"{1000*v['mean_step_seconds']:.2f}",
              f"{v['peak_allocated_bytes']/2**30:.2f}", f"{v['peak_reserved_bytes']/2**30:.2f}", f"{v['gpu_util']:.0f}%",
              f"{v['process_cpu_percent']:.0f}%", v["compile_mode"]] for a, v in single.items()]
    s = json.loads((ROOT/"benchmarks/training_schedule.json").read_text())
    return ("**Single job** (fold 1, batch 128, the exact training step)\n\n"
            + md_table(["Architecture", "Parameters", "samples/s", "step (ms)", "peak alloc (GiB)", "peak reserved (GiB)",
                        "GPU util", "process CPU", "compile"], srows)
            + "\n\n**Concurrency** (normalized throughput = Σ samples/s ÷ single-job samples/s of each job's architecture; serial = 1.00)\n\n"
            + md_table(["Config", "Jobs", "Normalized", "samples/s per job", "GPU util", "GPU mem (GiB)", "CPU", "RAM (GiB)"], rows)
            + "\n\n**Chosen schedule**\n\n```json\n"+json.dumps(s["policy"], indent=2)+"\n```\n\n"+s.get("rationale", ""))


def audit_section():
    path = ROOT/"audits/leakage_audit_classification.json"
    if not path.exists():
        path = ROOT/"audits/leakage_audit_preflight.json"
    a = json.loads(path.read_text())
    groups = {}
    for c in a["checks"]:
        name = c["name"]
        key = ("run-level" if name.startswith("run.") else "labels" if name.startswith("labels.")
               else "dataset" if name.startswith("dataset") else "split / window / embargo / normalization")
        g = groups.setdefault(key, [0, 0])
        g[0] += c["passed"]
        g[1] += 1
    rows = [[k, f"{v[0]}/{v[1]}"] for k, v in groups.items()]
    return (f"Report `{path.relative_to(ROOT)}`: status **{a['status']}**, {a['checks_passed']}/{a['checks_total']} checks, "
            f"{len(a.get('runs_audited', []))} runs audited, failed: {a['failed'] or 'none'}.\n\n"
            + md_table(["Check family", "Passed"], rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("multi", "all"), default="all")
    args = p.parse_args()
    df = collect()
    (ROOT/"reports").mkdir(exist_ok=True)
    df.to_csv(ROOT/"reports/cls_results_long.csv", index=False)
    registry = json.loads((ROOT/"runs/cls/registry.json").read_text()) if (ROOT/"runs/cls/registry.json").exists() else dict(jobs=[])
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    labels = label_sections()
    interp_path = ROOT/"reports/cls_interpretation.md"
    interp = interp_path.read_text() if interp_path.exists() else ""
    sections = {}
    for chunk in interp.split("\n<!-- section: ")[1:]:
        key, body = chunk.split(" -->\n", 1)
        sections[key.strip()] = body.strip()
    base = [j for j in registry["jobs"] if j["stage"] in ("multi", "single")]
    follow = [j for j in registry["jobs"] if j["stage"] == "followup"]
    count = lambda js, st: sum(j["status"] == st for j in js)
    title = "Stage 1 (multi-horizon) report" if args.stage == "multi" else "Final report"
    md = [f"# Discretized classification of BTC mid-price displacement on the WF3 contract — {title}",
          "",
          f"Branch `classification` of `tson295/P0_forecasting` at `{head}`; artifacts on the private Hugging Face repo "
          "`Tson29/LOB_Classification_WF3`; data from the private dataset `Tson29/btc-l10-gate-1y`.",
          "",
          f"Baseline jobs: {len(base)} planned, {count(base, 'completed')} completed, {count(base, 'failed')} failed, "
          f"{sum(bool(j.get('retried')) for j in base)} retried. Follow-up jobs: {len(follow)} "
          f"({count(follow, 'completed')} completed).",
          "",
          "## 1. Experiment contract", sections.get("contract", ""),
          "## 2. Dataset provenance", sections.get("dataset", ""),
          "## 3. Git/HF provenance", sections.get("provenance", ""),
          "## 4. Leakage audit", audit_section(), sections.get("audit", ""),
          "## 5. Label-distribution analysis", sections.get("distribution", ""), distribution_section(),
          "## 6. Equal-width p99 definition", sections.get("equal_width", ""), labels["equal_width_k32_p99"],
          "## 7. Quantile definition", sections.get("quantile", ""), labels["quantile_c34"],
          "## 8. Model architectures / parameter counts", sections.get("models", ""), params_table(df),
          "## 9. GPU benchmark / concurrency strategy", sections.get("benchmark", ""), benchmark_section(),
          "## 10. Multi-horizon results", sections.get("multi", ""),
          f"Directional accuracy convention: {DA_CONVENTION}. E0 has DA 0 under it (it never calls a direction); 50% is the coin-flip reference.",
          official_block(df, "multi"),
          "**Previous WF3 regression on the same test samples, scored with the same four metrics**\n\n"+wf3_block()]
    if args.stage == "all":
        md += ["## 11. Single-horizon results", sections.get("single", ""), official_block(df, "single"),
               "## 12. F1 → F2 → F3 scaling", sections.get("scaling", ""),
               "**R² gain vs E0 (test), multi-horizon**\n\n"+pivot(df, "r2_gain_vs_e0", "multi"),
               "**R² gain vs E0 (test), single-horizon**\n\n"+pivot(df, "r2_gain_vs_e0", "single"),
               "**DA (test), multi-horizon**\n\n"+pivot(df, "da", "multi"),
               "**DA (test), single-horizon**\n\n"+pivot(df, "da", "single"),
               "## 13. h1 vs h2 vs h3", sections.get("horizons", ""),
               "## 14. Equal-width vs quantile", sections.get("methods", ""),
               "## 15. Follow-up experiments", sections.get("followups", ""), decoding_block(),
               followup_block(df, registry),
               "## 16. Interpretation", sections.get("interpretation", ""),
               "## 17. Limitations", sections.get("limitations", ""),
               "## 18. Final conclusions", sections.get("conclusions", "")]
    else:
        md += ["## Stage-1 reading", sections.get("stage1", "")]
    md += ["## Appendix A. Secondary diagnostic (NOT an official metric): expected-value decoding",
           "Σ_k p_k · representative_k instead of the argmax representative, same test samples. Shown only to explain the "
           "official numbers; the official decoding is argmax.\n\n"
           + "**Expected-value decoding, R² gain vs E0 (test), multi-horizon**\n\n"
           + pivot(df, "ev_r2_gain_vs_e0", "multi", index=("method", "arch"))
           + ("\n\n**Expected-value decoding, R² gain vs E0 (test), single-horizon**\n\n"
              + pivot(df, "ev_r2_gain_vs_e0", "single", index=("method", "arch")) if args.stage == "all" else ""),
           "## Appendix B. Validation (selection split), multi-horizon, official metrics",
           official_block(df, "multi", split="validation")]
    name = "STAGE1_MULTI_REPORT.md" if args.stage == "multi" else "FINAL_REPORT_CLASSIFICATION.md"
    out = ROOT/"reports"/name
    out.write_text("\n\n".join(x for x in md if x is not None)+"\n")
    write_model_card(df, registry, head)
    print(f"wrote {out} ({len(df)} result rows from {df.job_id.nunique() if len(df) else 0} runs)")


def write_model_card(df, registry, head):
    """reports/HF_README.md: the Hugging Face model card (uploaded as README.md)."""
    base = [j for j in registry["jobs"] if j["stage"] in ("multi", "single")]
    follow = [j for j in registry["jobs"] if j["stage"] == "followup"]
    count = lambda js, st: sum(j["status"] == st for j in js)
    card = f"""---
license: mit
library_name: pytorch
tags:
- time-series-forecasting
- limit-order-book
- classification
- bitcoin
---

# LOB_Classification_WF3 — discretized classification of BTC mid-price displacement

The future USD mid-price displacement delta_h = mid(t+h) − mid(t), h ∈ {{60, 120, 180}} s, is mapped to
frozen class intervals (fitted on fold-1 train only), predicted with CrossEntropy by OFI-LSTM, ModernTCN and a
Transformer, and decoded back to a price (argmax class → interval representative). Everything else is the WF3
contract of `Tson29/Pretrain_Model_WF3`: 490 s history, 20 s stride, 3 expanding walk-forward folds, a shared
held-out test split (last 15 % of time, 235,848 samples), 180 s embargo, train-only normalization.

| Item | Value |
|---|---|
| Code | GitHub `tson295/P0_forecasting`, branch `classification`, commit `{head}` |
| Data | private dataset `Tson29/btc-l10-gate-1y`, `BTC_L10_gate_1y.csv`, SHA256 `6d8f82fb…e6df` (not copied here) |
| Baseline jobs | {len(base)} planned, {count(base, 'completed')} completed, {count(base, 'failed')} failed |
| Follow-up jobs | {len(follow)} ({count(follow, 'completed')} completed) |
| Official metrics | RMSE, MAE, R² gain vs E0, directional accuracy — on the decoded future mid price |

Full report: `reports/FINAL_REPORT_CLASSIFICATION.md`. Leakage audit: `audits/leakage_audit_classification.json`.
Frozen label definitions: `labeling/`. Throughput benchmark and schedule: `benchmarks/`. Run registry: `registry/registry.json`.

## Layout

```text
<equal_width|quantile>/<multi|single>/<ofi_lstm|moderntcn|transformer>/[h<seconds>/]f<fold>/
    best/                      weights (safetensors), config.json and every metadata JSON
    run_config.json label_set.json training_history.jsonl run_summary.json
    {{train,validation,test}}_metrics.json  {{validation,test}}_predictions.csv.gz  {{validation,test}}_probs.npz
followups/<experiment>/<variant>/<method>/<multi|single>/<arch>/[h<seconds>/]f<fold>/   same files
labeling/ reports/ audits/ benchmarks/ registry/
```

Prediction tables hold one row per origin with origin/target indices, timestamps and mids, the true displacement
and class, the predicted class, decoded displacement and mid, and the expected-value diagnostic, so all four
official metrics can be recomputed without the model.

## Loading a checkpoint

```python
from huggingface_hub import snapshot_download
from src.cls.models import load_classifier   # code from the GitHub branch above

path = snapshot_download("Tson29/LOB_Classification_WF3", allow_patterns="equal_width/multi/moderntcn/f3/best/*")
model = load_classifier(f"{{path}}/equal_width/multi/moderntcn/f3/best")   # returns one logits tensor per horizon
```
"""
    (ROOT/"reports/HF_README.md").write_text(card)


def params_table(df):
    """Exact counts differ with the class count (33 vs 34 at 60 s) and the number of heads."""
    d = df[df.experiment == ""]
    if d.empty:
        return ""
    d = d.assign(h=np.where(d.formulation == "multi", "60/120/180 s", d.horizon))
    d = d.drop_duplicates(["arch", "formulation", "method", "h"]).sort_values(["arch", "formulation", "method", "h"])
    return md_table(["Architecture", "Formulation", "Binning", "Horizon(s)", "Parameters (exact)"],
                    [[ARCH_NAME[r.arch], r.formulation, METHOD_NAME[r.method], r.h, f"{r.parameters:,}"]
                     for _, r in d.iterrows()])


def followup_block(df, registry):
    """Every follow-up variant against its baseline twin (same method, arch, formulation,
    fold, horizon), official metrics only, validation first (the selection split)."""
    fu = df[df.experiment != ""]
    if fu.empty:
        return "(no follow-up runs)"
    base = df[df.experiment == ""]
    hyp = {}
    for j in registry["jobs"]:
        if j["stage"] == "followup":
            hyp[(j["experiment"], j["variant"])] = (j.get("extra", {}).get("hypothesis", ""), j["label_set_path"],
                                                    j.get("model_kwargs", {}))
    keys = ["method", "formulation", "arch", "fold", "horizon", "split"]
    out = []
    for (exp, var), g in fu.groupby(["experiment", "variant"], sort=True):
        h, labels, kwargs = hyp.get((exp, var), ("", "", {}))
        m = g.merge(base, on=keys, suffixes=("_var", "_base"))
        lines = [f"### {exp} / {var}", f"*Hypothesis:* {h}" if h else "",
                 f"*Controlled change:* label set `{labels}`" + (f", model_kwargs `{json.dumps(kwargs)}`" if kwargs else "")
                 + "; everything else identical to the baseline twin."]
        if len(m) != len(g):
            lines.append(f"**{len(g)-len(m)} follow-up rows have no baseline twin.**")
        for split in ("validation", "test"):
            mm = m[m.split == split].sort_values(["method", "arch", "horizon", "fold"],
                                                 key=lambda c: c.map(TAGS.index) if c.name == "horizon" else c)
            rows = [[METHOD_NAME[r.method], ARCH_NAME[r.arch], r.formulation, r.horizon, f"F{r.fold}",
                     fmt(r.rmse_base, "rmse"), fmt(r.rmse_var, "rmse"), fmt(r.mae_base, "mae"), fmt(r.mae_var, "mae"),
                     fmt(r.r2_gain_vs_e0_base, "r2"), fmt(r.r2_gain_vs_e0_var, "r2"), fmt(r.da_base, "da"), fmt(r.da_var, "da")]
                    for _, r in mm.iterrows()]
            lines.append(f"**{split}**\n\n" + md_table(
                ["Binning", "Arch", "Form.", "Horizon", "Fold", "RMSE base", "RMSE var", "MAE base", "MAE var",
                 "R² gain base", "R² gain var", "DA base", "DA var"], rows))
        out.append("\n\n".join(x for x in lines if x))
    return "\n\n".join(out)


def decoding_block():
    path = ROOT/"reports/followups/decoding_long.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    d = d[d.experiment.fillna("") == ""]
    ce = (d[d.rule == "argmax"].groupby(["split", "method", "formulation", "arch", "horizon"])[["model_ce", "prior_ce"]]
          .mean().reset_index())
    ce_rows = []
    for _, r in ce.sort_values(["split", "method", "formulation", "arch", "horizon"],
                               key=lambda c: c.map(TAGS.index) if c.name == "horizon" else c).iterrows():
        ce_rows.append([r.split, METHOD_NAME[r.method], r.formulation, ARCH_NAME[r.arch], r.horizon,
                        f"{r.model_ce:.5f}", f"{r.prior_ce:.5f}", f"{r.prior_ce-r.model_ce:+.5f}"])
    ce_table = ("**Did the classifier learn beyond the class prior?** (diagnostic, not official) Cross-entropy of the "
                "model vs a predictor that always outputs the run's own train class frequencies; mean over folds.\n\n"
                + md_table(["Split", "Binning", "Form.", "Arch", "Horizon", "Model CE", "Prior CE", "Prior − model"], ce_rows))
    agg = (d.groupby(["split", "rule", "method", "formulation", "arch", "horizon"])[["rmse", "mae", "r2_gain_vs_e0", "da"]]
           .mean().reset_index())
    out = []
    for split in ("validation", "test"):
        a = agg[agg.split == split].sort_values(["method", "formulation", "arch", "horizon", "rule"],
                                                key=lambda c: c.map(TAGS.index) if c.name == "horizon" else c)
        rows = [[r.rule, METHOD_NAME[r.method], r.formulation, ARCH_NAME[r.arch], r.horizon, fmt(r.rmse, "rmse"),
                 fmt(r.mae, "mae"), fmt(r.r2_gain_vs_e0, "r2"), fmt(r.da, "da")] for _, r in a.iterrows()]
        out.append(f"**Decoding rules, {split}, mean over folds F1–F3** (same models, same classes; only the decoding changes)\n\n"
                   + md_table(["Decoding", "Binning", "Form.", "Arch", "Horizon", "RMSE", "MAE", "R² gain vs E0", "DA"], rows))
    return "\n\n".join([ce_table]+out)


if __name__ == "__main__":
    main()
