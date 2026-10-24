#!/usr/bin/env python3
"""Reproducible, read-only figures for the frozen P4-B/P5 evidence.

The script never calls MuJoCo or SB3.  It reads persisted CSV evidence and
writes derived tables and figures only under this directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TAB = OUT / "derived_tables"
MODES = ["pure_sac", "pd_residual", "mlp_residual", "gp_residual", "rbf_residual", "pd_gravity_residual"]
LABEL = {
    "pure_sac": "Pure SAC", "pd_residual": "PD + SAC",
    "mlp_residual": "MLP + PD + SAC", "gp_residual": "GP + PD + SAC",
    "rbf_residual": "RBF + PD + SAC", "pd_gravity_residual": "PD + Gravity + SAC",
}
COLORS = {"pure_sac":"#555555", "pd_residual":"#4C78A8", "mlp_residual":"#59A14F",
          "gp_residual":"#F28E2B", "rbf_residual":"#E15759", "pd_gravity_residual":"#B279A2"}

P4 = ROOT / "p4b_full_attempt02" / "independent_selected_validation_v1" / "episodes.csv"
P4G = ROOT / "p4b_pd_gravity_attempt02" / "independent_selected_validation_v1" / "episodes.csv"
P5 = ROOT / "p5_nominal_prearmed_attempt06"
P5G = ROOT / "p5_pd_gravity_exploratory_attempt01" / "episodes.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def savefig(name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        plt.savefig(FIG / f"{name}.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close()


def ci95(x: pd.Series) -> float:
    x = x.dropna().to_numpy(float)
    return 0.0 if len(x) < 2 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def p4_validation() -> pd.DataFrame:
    frames = [pd.read_csv(P4), pd.read_csv(P4G)]
    df = pd.concat(frames, ignore_index=True)
    df["completed"] = (~df["terminated"].astype(bool)).astype(int)
    # Training seed is the inferential repeat; rows/scenarios are nested.
    per_seed = (df.groupby(["mode", "policy_seed"], as_index=False)
                  .agg(completion_rate=("completed", "mean"), mean_steps=("steps", "mean"),
                       q_rmse_mean=("q_rmse_rad", "mean"), dq_rmse_mean=("dq_rmse_rad_s", "mean"),
                       mean_return=("return", "mean"), projection_rate=("projection_steps", lambda s: (s / 500).mean()),
                       episodes=("validation_seed", "count")))
    summary = (per_seed.groupby("mode", as_index=False)
               .agg(seed_n=("policy_seed", "nunique"), completion_rate_mean=("completion_rate", "mean"),
                    completion_rate_ci95=("completion_rate", ci95), q_rmse_mean=("q_rmse_mean", "mean"),
                    q_rmse_ci95=("q_rmse_mean", ci95), dq_rmse_mean=("dq_rmse_mean", "mean"),
                    return_mean=("mean_return", "mean"), projection_rate_mean=("projection_rate", "mean")))
    per_seed.to_csv(TAB / "p4_validation_per_training_seed.csv", index=False)
    summary.to_csv(TAB / "p4_validation_summary.csv", index=False)
    order = [m for m in MODES if m in summary["mode"].tolist()]
    s = summary.set_index("mode").loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(order))
    ax[0].bar(x, 100 * s.completion_rate_mean, yerr=100 * s.completion_rate_ci95,
              color=[COLORS[m] for m in order], capsize=4)
    ax[0].set_ylim(0, 105); ax[0].set_ylabel("Validation completion rate (%)")
    ax[1].bar(x, s.q_rmse_mean, yerr=s.q_rmse_ci95, color=[COLORS[m] for m in order], capsize=4)
    ax[1].set_ylabel("Mean episode q-RMSE (rad)")
    for a in ax:
        a.set_xticks(x, [LABEL[m] for m in order], rotation=30, ha="right")
        a.grid(axis="y", alpha=.25)
    fig.suptitle("P4-B frozen validation: mean ± 95% CI across five training seeds")
    savefig("p4b_validation_six_group_summary")
    return df


def p5_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = pd.concat([pd.read_csv(P5 / m / "episodes.csv") for m in MODES[:5]], ignore_index=True)
    strict["source_classification"] = "STRICT_NOMINAL_ONLY_FINAL"
    exploratory = pd.read_csv(P5G)
    exploratory["source_classification"] = "POSTHOC_EXPLORATORY_BASELINE"
    all_df = pd.concat([strict, exploratory], ignore_index=True)
    all_df["completed"] = (~all_df.terminated.astype(bool)).astype(int)
    # Aggregate episode rows to seed first: scenes are paired, not independent seed replicates.
    per_seed = (all_df.groupby(["source_classification", "mode", "condition", "policy_seed"], as_index=False)
                .agg(completion_rate=("completed", "mean"), mean_steps=("steps", "mean"),
                     q_rmse_prefix=("q_rmse_prefix_rad", "mean"), dq_rmse_prefix=("dq_rmse_prefix_rad_s", "mean"),
                     projection_steps=("projection_steps", "mean"), episodes=("scenario_seed", "count")))
    summary = (per_seed.groupby(["source_classification", "mode", "condition"], as_index=False)
               .agg(seed_n=("policy_seed", "nunique"), episodes=("episodes", "sum"),
                    completion_rate_mean=("completion_rate", "mean"), completion_rate_ci95=("completion_rate", ci95),
                    mean_steps=("mean_steps", "mean"), q_rmse_prefix=("q_rmse_prefix", "mean"),
                    dq_rmse_prefix=("dq_rmse_prefix", "mean"), projection_steps=("projection_steps", "mean")))
    per_seed.to_csv(TAB / "p5_per_training_seed_condition.csv", index=False)
    summary.to_csv(TAB / "p5_condition_summary.csv", index=False)
    all_df.to_csv(TAB / "p5_termination_records.csv", index=False)

    # Strict 5-group heatmap
    strict_s = summary[summary.source_classification == "STRICT_NOMINAL_ONLY_FINAL"]
    conditions = list(strict_s.condition.drop_duplicates())
    matrix = strict_s.pivot(index="mode", columns="condition", values="completion_rate_mean").reindex(MODES[:5])
    fig, ax = plt.subplots(figsize=(14, 5.2))
    im = ax.imshow(matrix.to_numpy() * 100, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    ax.set_xticks(range(len(conditions)), conditions, rotation=40, ha="right")
    ax.set_yticks(range(5), [LABEL[m] for m in MODES[:5]])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j, i, f"{matrix.iloc[i,j]*100:.0f}", ha="center", va="center", color="white" if matrix.iloc[i,j] < .5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="System completion rate (%)")
    ax.set_title("P5 strict nominal-only Final: condition-wise completion")
    savefig("p5_strict_completion_heatmap")

    # Overall completion bars, with sixth group deliberately separate/
    overall = (per_seed.groupby(["source_classification", "mode", "policy_seed"], as_index=False)
               .agg(completion_rate=("completion_rate", "mean")))
    o = (overall.groupby(["source_classification", "mode"], as_index=False)
         .agg(mean=("completion_rate", "mean"), ci95=("completion_rate", ci95)))
    o.to_csv(TAB / "p5_overall_completion_by_training_seed.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    oo = o.sort_values(["source_classification", "mode"])
    labels = [LABEL[m] + (" *" if c.startswith("POSTHOC") else "")
              for c, m in zip(oo["source_classification"], oo["mode"])]
    ax.bar(np.arange(len(oo)), oo["mean"]*100, yerr=oo["ci95"]*100, capsize=4, color=[COLORS[m] for m in oo["mode"]])
    ax.set_xticks(range(len(oo)), labels, rotation=27, ha="right"); ax.set_ylim(0, 105)
    ax.set_ylabel("System completion rate (%)")
    ax.set_title("P5 overall completion; * = posthoc exploratory, not confirmatory")
    ax.grid(axis="y", alpha=.25)
    savefig("p5_overall_completion_strict_vs_posthoc")

    # Terminations, strict and posthoc separated by source.
    fails = all_df[all_df.terminated.astype(bool)].copy()
    reason = (fails.groupby(["source_classification", "mode", "termination_reason"], as_index=False).size().rename(columns={"size":"episodes"}))
    reason.to_csv(TAB / "p5_termination_reason_counts.csv", index=False)
    top = reason.pivot_table(index=["source_classification", "mode"], columns="termination_reason", values="episodes", fill_value=0)
    top.plot(kind="bar", stacked=True, figsize=(14, 5.5), colormap="tab20", ax=plt.gca())
    plt.ylabel("Terminated episodes"); plt.xlabel("Evidence source / controller")
    plt.title("P5 safety termination reasons (strict and posthoc evidence remain separate)")
    plt.xticks(range(len(top.index)), [f"{'Strict' if c.startswith('STRICT') else 'Posthoc'}\n{LABEL[m]}" for c,m in top.index], rotation=30, ha="right")
    plt.legend(title="termination reason", bbox_to_anchor=(1.01, 1), loc="upper left")
    savefig("p5_termination_reasons")
    return strict, exploratory


def learning_curves() -> None:
    roots = {m: ROOT / "p4b_full_attempt02" / m for m in MODES[:5]}
    roots["pd_gravity_residual"] = ROOT / "p4b_pd_gravity_attempt02" / "pd_gravity_residual"
    rows = []
    for mode, root in roots.items():
        for path in sorted(root.glob("seed_*/validation.csv")):
            d = pd.read_csv(path); d["mode"] = mode; d["policy_seed"] = int(path.parent.name.split("_")[1]); rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["completed"] = (~df.terminated.astype(bool)).astype(int)
    agg = df.groupby(["mode", "checkpoint_timesteps"], as_index=False).agg(return_mean=("return","mean"), completion_rate=("completed","mean"))
    agg.to_csv(TAB / "p4b_learning_curves_validation.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for m in MODES:
        d = agg[agg["mode"] == m]
        ax[0].plot(d.checkpoint_timesteps / 1000, d.return_mean, marker="o", lw=1.8, label=LABEL[m], color=COLORS[m])
        ax[1].plot(d.checkpoint_timesteps / 1000, 100*d.completion_rate, marker="o", lw=1.8, label=LABEL[m], color=COLORS[m])
    ax[0].set_ylabel("Validation return (episode mean)"); ax[1].set_ylabel("Completion rate (%)")
    for a in ax: a.set_xlabel("Training transitions (thousands)"); a.grid(alpha=.25)
    ax[1].set_ylim(-5, 105); ax[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("P4-B validation learning curves (episode-level means; not independent scenario repeats)")
    savefig("p4b_validation_learning_curves")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    p4_validation(); p5_results(); learning_curves()
    sources = [P4, P4G, P5G] + [P5 / m / "episodes.csv" for m in MODES[:5]]
    manifest = {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sources}
    (OUT / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(list(FIG.glob('*')))} figure files and {len(list(TAB.glob('*.csv')))} derived CSV files")


if __name__ == "__main__":
    main()
