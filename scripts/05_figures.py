"""
05_figures.py — Generate figures for the thesis.

Generates:
    output/figures/fig_nonlinear_p90.png  — LOWESS / quadratic / linear fits for P90
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "output" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(ROOT / "output" / "cross_merged.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── (a) P90 vs log GDP ───────────────────────────────────────────────────
    ax = axes[0]
    mask = df[["log_agdp", "sd_w_p90"]].notna().all(axis=1)
    x, y = df.loc[mask, "log_agdp"].values, df.loc[mask, "sd_w_p90"].values

    ax.scatter(x, y, alpha=0.5, s=30, c="steelblue", edgecolors="white", linewidth=0.5)

    xline = np.linspace(x.min(), x.max(), 100)
    # Linear
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(xline, slope * xline + intercept, "r--",
            label=f"Linear (slope={slope:.1f})", linewidth=1.5)
    # LOWESS
    lw = lowess(y, x, frac=0.6)
    ax.plot(lw[:, 0], lw[:, 1], "darkgreen", linewidth=2.5, label="LOWESS")
    # Quadratic
    c2 = np.polyfit(x, y, 2)
    ax.plot(xline, np.polyval(c2, xline), "orange", linewidth=1.5,
            linestyle=":", label="Quadratic")

    ax.set_xlabel("Log GDP per Adult", fontsize=12)
    ax.set_ylabel("P90 of Hours Distribution (h/week)", fontsize=12)
    ax.set_title("(a) P90 vs. Log GDP per Adult", fontsize=13)
    ax.legend(fontsize=10)
    r_val = pearsonr(x, y)[0]
    ax.text(0.05, 0.95, f"r = {r_val:.2f}, N = {len(x)}",
            transform=ax.transAxes, fontsize=10, va="top")

    # ── (b) P90 vs tax rate ──────────────────────────────────────────────────
    ax = axes[1]
    mask = df[["taxr_lab", "sd_w_p90"]].notna().all(axis=1)
    x, y = df.loc[mask, "taxr_lab"].values, df.loc[mask, "sd_w_p90"].values

    ax.scatter(x, y, alpha=0.5, s=30, c="steelblue", edgecolors="white", linewidth=0.5)

    xline = np.linspace(x.min(), x.max(), 100)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(xline, slope * xline + intercept, "r--",
            label=f"Linear (slope={slope:.1f})", linewidth=1.5)
    lw = lowess(y, x, frac=0.6)
    ax.plot(lw[:, 0], lw[:, 1], "darkgreen", linewidth=2.5, label="LOWESS")
    c2 = np.polyfit(x, y, 2)
    ax.plot(xline, np.polyval(c2, xline), "orange", linewidth=1.5,
            linestyle=":", label="Quadratic")

    ax.set_xlabel("Labour Income Tax Rate", fontsize=12)
    ax.set_ylabel("P90 of Hours Distribution (h/week)", fontsize=12)
    ax.set_title("(b) P90 vs. Labour Tax Rate", fontsize=13)
    ax.legend(fontsize=10)
    r_val = pearsonr(x, y)[0]
    ax.text(0.05, 0.95, f"r = {r_val:.2f}, N = {len(x)}",
            transform=ax.transAxes, fontsize=10, va="top")

    plt.tight_layout()
    out = FIGS / "fig_nonlinear_p90.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
