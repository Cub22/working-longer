"""
02_cross_section_ols.py — Cross-sectional OLS regressions (Table 3.1, robustness, H2 test).

Reads:  output/cross_merged.csv
Prints: All regression tables to stdout and saves to output/tables/
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def ols(df, y, xvars):
    """Unweighted OLS with HC1 standard errors."""
    mask = df[[y] + xvars].notna().all(axis=1)
    d = df[mask]
    return sm.OLS(d[y], sm.add_constant(d[xvars])).fit(cov_type="HC1")


def wls(df, y, xvars, wvar="emp_total"):
    """Population-weighted OLS with HC1 standard errors."""
    mask = df[[y] + xvars + [wvar]].notna().all(axis=1)
    d = df[mask]
    return sm.WLS(d[y], sm.add_constant(d[xvars]), weights=d[wvar]).fit(cov_type="HC1")


def sig(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def print_model(r, xvars, name=""):
    print(f"\n{name} (N={int(r.nobs)}, R²={r.rsquared:.3f}):")
    for v in xvars:
        s = sig(r.pvalues[v])
        print(f"  {v:25s}  {r.params[v]:10.3f}  ({r.bse[v]:.3f}){s}")


def main():
    df = pd.read_csv(ROOT / "output" / "cross_merged.csv")

    # ── Model specifications ─────────────────────────────────────────────────
    specs = {
        "M1": ["log_agdp"],
        "M2": ["log_agdp", "taxr_lab"],
        "M3": ["log_agdp", "taxr_lab", "gini_mkt"],
        "M4": ["log_agdp", "taxr_lab", "gova_soci", "ilo_emp_inf_tot"],
        "M5": ["log_agdp", "taxr_lab", "gini_mkt", "gova_soci", "ilo_emp_inf_tot", "rel_mush"],
        "M6": ["log_agdp", "taxr_lab", "gini_mkt", "gova_soci", "ilo_emp_inf_tot", "rel_mush", "regulations"],
    }

    # ── TABLE 3.1: Unweighted OLS, DV = w ────────────────────────────────────
    print("=" * 70)
    print("TABLE 3.1: CROSS-SECTIONAL OLS — UNWEIGHTED, DV = w")
    print("=" * 70)
    for name, xv in specs.items():
        r = ols(df, "w", xv)
        print_model(r, xv, name)

    # ── Same with DV = log(w) ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DV = log(w) — UNWEIGHTED")
    print("=" * 70)
    for name, xv in specs.items():
        r = ols(df, "log_w", xv)
        print_model(r, xv, name)

    # ── H2 TEST: standardised coefficients ────────────────────────────────────
    print("\n" + "=" * 70)
    print("H2 TEST: STANDARDISED COEFFICIENTS")
    print("=" * 70)
    for label, xv in [
        ("M3", ["log_agdp", "taxr_lab", "gini_mkt"]),
        ("M6", ["log_agdp", "taxr_lab", "gini_mkt", "gova_soci",
                "ilo_emp_inf_tot", "rel_mush", "regulations"]),
    ]:
        mask = df[["w"] + xv].notna().all(axis=1)
        d = df[mask].copy()
        for v in xv:
            d[v] = (d[v] - d[v].mean()) / d[v].std()
        d["w_z"] = (d["w"] - d["w"].mean()) / d["w"].std()
        r = sm.OLS(d["w_z"], sm.add_constant(d[xv])).fit(cov_type="HC1")
        print(f"\n{label} standardised (N={int(r.nobs)}):")
        for v in ["taxr_lab", "gini_mkt"]:
            print(f"  β*({v}) = {r.params[v]:.3f} (SE={r.bse[v]:.3f}){sig(r.pvalues[v])}")
        ratio = abs(r.params["taxr_lab"]) / max(abs(r.params["gini_mkt"]), 0.001)
        print(f"  |β*(taxr)| / |β*(gini)| = {ratio:.1f}")

    # ── ROBUSTNESS TABLE ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROBUSTNESS TABLE (M2 spec: DV ~ log_agdp + taxr_lab)")
    print("=" * 70)
    xv2 = ["log_agdp", "taxr_lab"]
    excl = df[~df["isoname"].isin(["China", "India"])]

    rows = [
        ("Unweighted, w", ols(df, "w", xv2)),
        ("Weighted, w", wls(df, "w", xv2)),
        ("Unweighted, log(w)", ols(df, "log_w", xv2)),
        ("Weighted, log(w)", wls(df, "log_w", xv2)),
        ("Excl. CN & IN", ols(excl, "w", xv2)),
        ("DV = h (per adult)", ols(df, "h", xv2)),
        ("DV = P90, unwtd", ols(df, "sd_w_p90", xv2)),
        ("DV = P90, wtd", wls(df, "sd_w_p90", xv2)),
    ]

    header = f"{'Specification':<25s} {'β(taxr)':>10s} {'SE':>8s} {'p':>8s}   {'N':>4s}  {'R²':>5s}"
    print(f"\n{header}")
    print("-" * 68)
    for name, r in rows:
        s = sig(r.pvalues["taxr_lab"])
        print(
            f"{name:<25s} {r.params['taxr_lab']:10.3f} {r.bse['taxr_lab']:8.3f}"
            f" {r.pvalues['taxr_lab']:8.4f} {s:3s} {int(r.nobs):4d}  {r.rsquared:.3f}"
        )


if __name__ == "__main__":
    main()
