"""
03_panel_twfe.py — Panel two-way fixed effects (Table 3.2).

Reads:  output/panel_merged.csv, output/cross_merged.csv (for region list)
Prints: TWFE results with clustered and unadjusted SEs.
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent


def sig(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def run_twfe(df, y, xvars, obs_count, min_obs=10, subset_isos=None, cov="clustered"):
    """Estimate TWFE with country + year FE."""
    d = df.copy()
    if subset_isos is not None:
        d = d[d["iso"].isin(subset_isos)]
    valid = obs_count[obs_count >= min_obs].index
    d = d[d["iso"].isin(valid)]
    mask = d[[y] + xvars + ["iso", "year"]].notna().all(axis=1)
    d = d[mask].set_index(["iso", "year"])
    formula = f'{y} ~ {" + ".join(xvars)} + EntityEffects + TimeEffects'
    model = PanelOLS.from_formula(formula, data=d)
    if cov == "clustered":
        return model.fit(cov_type="clustered", cluster_entity=True)
    return model.fit(cov_type="unadjusted")


def print_twfe(r, xvars, label):
    print(f"\n{label} (N={r.nobs}, C={int(r.entity_info.total)}, R²w={r.rsquared_within:.3f}):")
    for v in xvars:
        s = sig(r.pvalues[v])
        print(f"  {v:15s}  {r.params[v]:10.3f}  (SE={r.std_errors[v]:.3f}){s}")


def main():
    panel = pd.read_csv(ROOT / "output" / "panel_merged.csv")
    cross = pd.read_csv(ROOT / "output" / "cross_merged.csv")

    western_isos = list(
        cross[cross["region"].isin(["Western Europe and Anglosphere", "United States"])]["iso"]
    )
    obs_count = panel.groupby("iso")["w"].count()

    # ── TABLE 3.2: TWFE with entity-clustered SEs ────────────────────────────
    print("=" * 70)
    print("TABLE 3.2: PANEL TWFE — Entity-Clustered SEs")
    print("=" * 70)

    configs = [
        ("P1: All, taxr only", "w", ["taxr_lab"], None),
        ("P2: All, taxr + GDP", "w", ["taxr_lab", "log_agdp"], None),
        ("P3: Western, taxr only", "w", ["taxr_lab"], western_isos),
        ("P4: Western, taxr + GDP", "w", ["taxr_lab", "log_agdp"], western_isos),
        ("P5: All, log(w)", "log_w", ["taxr_lab", "log_agdp"], None),
        ("P6: Western, log(w)", "log_w", ["taxr_lab", "log_agdp"], western_isos),
    ]

    for label, yvar, xvars, subset in configs:
        r = run_twfe(panel, yvar, xvars, obs_count, subset_isos=subset)
        print_twfe(r, xvars, label)

    # ── Comparison: clustered vs unadjusted SEs ──────────────────────────────
    print("\n" + "=" * 70)
    print("SE COMPARISON (P4: Western, w ~ taxr + GDP)")
    print("=" * 70)
    for cov in ["clustered", "unadjusted"]:
        r = run_twfe(panel, "w", ["taxr_lab", "log_agdp"], obs_count,
                     subset_isos=western_isos, cov=cov)
        print(f"\n{cov} SEs:")
        for v in ["taxr_lab", "log_agdp"]:
            s = sig(r.pvalues[v])
            print(f"  {v}: {r.params[v]:.3f} (SE={r.std_errors[v]:.3f}){s}")

    # ── TWFE with SWIID market Gini ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PANEL TWFE WITH SWIID MARKET GINI")
    print("=" * 70)
    gini_configs = [
        ("All, w + gini", "w", ["taxr_lab", "log_agdp", "gini_mkt"], None),
        ("Western, w + gini", "w", ["taxr_lab", "log_agdp", "gini_mkt"], western_isos),
    ]
    for label, yvar, xvars, subset in gini_configs:
        try:
            r = run_twfe(panel, yvar, xvars, obs_count, subset_isos=subset)
            print_twfe(r, xvars, label)
        except Exception as e:
            print(f"\n{label}: FAILED — {e}")


if __name__ == "__main__":
    main()
