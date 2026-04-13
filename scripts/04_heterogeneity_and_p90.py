"""
04_heterogeneity_and_p90.py — Subsample regressions and P90 supplementary analysis.

Tables: Income group heterogeneity (Table 3.5), Subsample regressions (Table 3.X),
        P90 as DV (Table 3.6).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent


def ols(df, y, xvars):
    mask = df[[y] + xvars].notna().all(axis=1)
    d = df[mask]
    return sm.OLS(d[y], sm.add_constant(d[xvars])).fit(cov_type="HC1")


def sig(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def main():
    df = pd.read_csv(ROOT / "output" / "cross_merged.csv")
    xv = ["log_agdp", "taxr_lab"]

    # ── HETEROGENEITY BY WORLD BANK INCOME GROUP ─────────────────────────────
    print("=" * 70)
    print("TABLE 3.5: HETEROGENEITY BY WORLD BANK INCOME GROUP")
    print("=" * 70)
    igroup_labels = {1: "Low income", 2: "Lower-middle", 3: "Upper-middle", 4: "High income"}

    print(f"\n{'Group':<18s} {'β(taxr)':>10s} {'SE':>8s} {'p':>8s}  {'N':>4s}  {'R²':>5s}")
    print("-" * 56)
    for g, label in igroup_labels.items():
        sub = df[df["igroup"] == g]
        r = ols(sub, "w", xv)
        s = sig(r.pvalues["taxr_lab"])
        print(
            f"{label:<18s} {r.params['taxr_lab']:10.1f} {r.bse['taxr_lab']:8.1f}"
            f" {r.pvalues['taxr_lab']:8.4f}{s:3s} {int(r.nobs):4d}  {r.rsquared:.3f}"
        )

    # Interaction model
    print("\nInteraction model: w ~ taxr_lab * income_group + log_agdp")
    d = df[["w", "log_agdp", "taxr_lab", "igroup"]].dropna().copy()
    for g in [2, 3, 4]:
        d[f"ig{g}"] = (d["igroup"] == g).astype(int)
        d[f"taxr_x_ig{g}"] = d["taxr_lab"] * d[f"ig{g}"]
    xv_int = ["log_agdp", "taxr_lab", "ig2", "ig3", "ig4",
              "taxr_x_ig2", "taxr_x_ig3", "taxr_x_ig4"]
    r = ols(d, "w", xv_int)
    print(f"N={int(r.nobs)}, R²={r.rsquared:.3f}")
    for v in xv_int:
        s = sig(r.pvalues[v])
        print(f"  {v:20s}: {r.params[v]:8.1f} ({r.bse[v]:.1f}){s}")

    # ── SUBSAMPLE REGRESSIONS: GENDER, AGE, EDUCATION ────────────────────────
    print("\n" + "=" * 70)
    print("TABLE SUBSAMPLES: GENDER, AGE, EDUCATION")
    print("=" * 70)

    groups = [
        ("Gender", [("All", "w"), ("Men", "w_men"), ("Women", "w_wom")]),
        ("Age", [("Youth 15-24", "w_you"), ("Prime 25-54", "w_pri"), ("Older 55+", "w_old")]),
        ("Education", [("No formal", "w_edu0"), ("Primary", "w_edu1"),
                       ("Secondary", "w_edu2"), ("Tertiary", "w_edu3")]),
    ]

    print(f"\n{'Dimension':<12s} {'Subgroup':<14s} {'β(taxr)':>10s} {'SE':>6s}  {'N':>4s}  {'R²':>5s}")
    print("-" * 58)
    for dim, subgroups in groups:
        for label, yvar in subgroups:
            try:
                r = ols(df, yvar, xv)
                s = sig(r.pvalues["taxr_lab"])
                print(
                    f"{dim:<12s} {label:<14s} {r.params['taxr_lab']:10.1f}"
                    f" {r.bse['taxr_lab']:6.1f}{s:3s} {int(r.nobs):4d}  {r.rsquared:.3f}"
                )
            except Exception as e:
                print(f"{dim:<12s} {label:<14s}  — failed: {e}")
            dim = ""  # don't repeat dimension label

    # ── P90 AS DEPENDENT VARIABLE ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLE 3.6: P90 AS DEPENDENT VARIABLE — UNWEIGHTED OLS")
    print("=" * 70)

    p90_specs = {
        "P1": ["log_agdp"],
        "P2": ["log_agdp", "taxr_lab"],
        "P3": ["log_agdp", "taxr_lab", "gini_mkt"],
        "P4": ["log_agdp", "taxr_lab", "gini_mkt", "gova_soci",
               "ilo_emp_inf_tot", "rel_mush", "regulations"],
    }

    for name, xv_spec in p90_specs.items():
        r = ols(df, "sd_w_p90", xv_spec)
        print(f"\n{name} (N={int(r.nobs)}, R²={r.rsquared:.3f}):")
        for v in xv_spec:
            s = sig(r.pvalues[v])
            print(f"  {v:25s}  {r.params[v]:10.3f}  ({r.bse[v]:.3f}){s}")

    # ── Formal nonlinearity test (quadratic GDP term) ────────────────────────
    print("\n— Quadratic GDP test for P90:")
    mask = df[["sd_w_p90", "log_agdp", "taxr_lab"]].notna().all(axis=1)
    d = df[mask].copy()
    d["log_agdp_sq"] = d["log_agdp"] ** 2
    r_lin = ols(d, "sd_w_p90", ["log_agdp", "taxr_lab"])
    r_quad = ols(d, "sd_w_p90", ["log_agdp", "log_agdp_sq", "taxr_lab"])
    print(f"  Linear R²={r_lin.rsquared:.3f}, Quadratic R²={r_quad.rsquared:.3f}")
    print(f"  log_agdp² coeff: {r_quad.params['log_agdp_sq']:.2f}"
          f" (p={r_quad.pvalues['log_agdp_sq']:.4f})")


if __name__ == "__main__":
    main()
