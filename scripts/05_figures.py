"""
05_figures.py — Generate all thesis figures from merged data.

Generates:
    output/figures/fig3_tax_hours.png         — Figure 3: Tax rate vs weekly hours
    output/figures/fig15_pension_hours.png     — Figure 15: Pension coverage vs older-worker hours
    output/figures/fig_nonlinear_p90.png       — LOWESS / quadratic / linear fits for P90
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

REGION_COLORS = {
    "Western Europe and Anglosphere": "#2166AC",
    "Eastern Europe and ex-USSR": "#67A9CF",
    "East and Southeast Asia": "#D6604D",
    "Latin America": "#F4A582",
    "Middle East and North Africa": "#B2182B",
    "Sub-Saharan Africa": "#4DAF4A",
    "South Asia": "#FF7F00",
    "United States": "#2166AC",
}


def scatter_with_ols(ax, d, xvar, yvar, labels_dict, xlabel, ylabel):
    """Region-coloured bubble scatter with OLS line, correlation, and labels."""
    for region, color in REGION_COLORS.items():
        sub = d[d["region"] == region]
        if len(sub) == 0:
            continue
        sizes = np.clip(sub["npop_all"] / 1e6 * 3, 15, 400)
        ax.scatter(
            sub[xvar], sub[yvar], s=sizes, c=color, alpha=0.6,
            edgecolors="white", linewidth=0.5, label=region, zorder=3,
        )
    # OLS fit
    x, y = d[xvar].values, d[yvar].values
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, "k--", linewidth=1.5, alpha=0.7)
    r_val, _ = pearsonr(x, y)
    ax.text(
        0.03, 0.97, f"r = {r_val:.2f}, N = {len(d)}\nslope = {slope:.1f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    )
    # Country labels
    for _, row in d.iterrows():
        name = row["isoname"]
        if name in labels_dict:
            dx, dy = labels_dict[name]
            ax.annotate(
                name, (row[xvar], row[yvar]),
                xytext=(row[xvar] + dx, row[yvar] + dy),
                fontsize=7.5, alpha=0.8,
            )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.2)


# =====================================================================
#  Figure 3: Labour Income Tax Rate vs Weekly Hours per Worker
# =====================================================================
def fig3_tax_hours(df):
    fig, ax = plt.subplots(figsize=(11, 7))
    mask = df[["taxr_lab", "w"]].notna().all(axis=1)
    d = df[mask].copy()

    labels = {
        "Bhutan": (0.01, 1.5), "Djibouti": (0.01, -2),
        "Sudan": (-0.04, 1.5), "USA": (0.01, -1.5),
        "South Korea": (0.01, 1.2), "Netherlands": (0.01, -1.5),
        "Sweden": (0.01, 1.2), "Germany": (-0.02, -2),
        "India": (0.01, 1.2), "China": (0.01, -1.5),
        "Brazil": (0.01, 1.2), "Greece": (0.01, -1.5),
        "Moldova": (0.01, 1.2), "Bangladesh": (-0.04, -1.2),
        "Jordan": (-0.04, 0), "Norway": (-0.06, 0.5),
        "Azerbaijan": (0.01, 1),
    }
    scatter_with_ols(
        ax, d, "taxr_lab", "w", labels,
        "Labour Income Tax Rate", "Weekly Hours per Worker",
    )
    # Quartile gap annotation
    q1 = d[d["taxr_lab"] <= d["taxr_lab"].quantile(0.25)]["w"].mean()
    q4 = d[d["taxr_lab"] >= d["taxr_lab"].quantile(0.75)]["w"].mean()
    gap = (q1 - q4) * 52
    ax.text(
        0.03, 0.83,
        f"Q1 mean: {q1:.1f} h/wk  |  Q4 mean: {q4:.1f} h/wk\n"
        f"Annualised gap ≈ {gap:.0f} hours (~{gap/160:.0f} working days)",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9),
    )
    ax.set_title(
        "Figure 3. Labour Income Tax Rate and Weekly Working Hours per Worker:\n"
        f"{len(d)} Countries (Cross-Section, ~2019–2023)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, ncol=2)
    ax.set_xlim(-0.02, 0.56)
    ax.set_ylim(22, 58)
    plt.tight_layout()
    out = FIGS / "fig3_tax_hours.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# =====================================================================
#  Figure 15: Pension Coverage vs Older-Worker Hours
# =====================================================================
def fig15_pension_hours(df):
    fig, ax = plt.subplots(figsize=(11, 7))
    mask = df[["pens_covh", "w_old"]].notna().all(axis=1)
    d = df[mask].copy()

    labels = {
        "India": (1, 1.5), "Nigeria": (1, -2),
        "Bangladesh": (-12, 1.5), "Germany": (1, -2),
        "Sweden": (1, 1), "Norway": (1, -1.5),
        "Netherlands": (1, -1.5), "USA": (1, 1.2),
        "France": (-10, 1), "Japan": (1, -1.5),
        "South Korea": (1, 1), "Brazil": (1, -1.5),
        "China": (1, 1.2), "Poland": (1, -1.5),
        "Moldova": (-12, -1), "Bhutan": (1, 1),
        "Azerbaijan": (1, 1),
    }
    scatter_with_ols(
        ax, d, "pens_covh", "w_old", labels,
        "Pension Coverage (%)", "Weekly Hours per Worker, Ages 55+",
    )
    ax.set_title(
        "Figure 15. Pension Coverage and Working Hours Among Older Workers (Ages 55+):\n"
        f"{len(d)} Countries",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)
    plt.tight_layout()
    out = FIGS / "fig15_pension_hours.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# =====================================================================
#  LOWESS / Quadratic / Linear fits for P90
# =====================================================================
def fig_nonlinear_p90(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, xvar, xlabel, title_letter in [
        (axes[0], "log_agdp", "Log GDP per Adult", "a"),
        (axes[1], "taxr_lab", "Labour Income Tax Rate", "b"),
    ]:
        mask = df[[xvar, "sd_w_p90"]].notna().all(axis=1)
        x, y = df.loc[mask, xvar].values, df.loc[mask, "sd_w_p90"].values

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

        r_val = pearsonr(x, y)[0]
        ax.text(0.05, 0.95, f"r = {r_val:.2f}, N = {len(x)}",
                transform=ax.transAxes, fontsize=10, va="top")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("P90 of Hours Distribution (h/week)", fontsize=12)
        ax.set_title(f"({title_letter}) P90 vs. {xlabel}", fontsize=13)
        ax.legend(fontsize=10)

    plt.tight_layout()
    out = FIGS / "fig_nonlinear_p90.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    df = pd.read_csv(ROOT / "output" / "cross_merged.csv")
    fig3_tax_hours(df)
    fig15_pension_hours(df)
    fig_nonlinear_p90(df)


if __name__ == "__main__":
    main()
