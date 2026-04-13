"""
01_merge_data.py — Load Gethin-Saez and SWIID, merge, export analysis-ready datasets.

Input:
    data/gethin-saez-cross-2026-03-02.dta
    data/gethin-saez-panel-2026-03-02.dta
    data/swiid9_91/swiid9_91_summary.csv

Output:
    output/cross_merged.csv
    output/panel_merged.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ── Country-name mapping GS → SWIID ─────────────────────────────────────────
NAME_MAP = {
    "Brunei Darussalam": "Brunei",
    "Cabo Verde": "Cape Verde",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "DR Congo": "Congo, the Democratic Republic of the",
    "Lao PDR": "Lao People's Democratic Republic",
    "Macedonia": "North Macedonia",
    "Palestine": "Palestine, State of",
    "Republic of the Congo": "Congo",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Swaziland": "Eswatini",
    "Syrian Arab Republic": "Syria",
    "UK": "United Kingdom",
    "USA": "United States",
}


def load_gethin_saez():
    """Load cross-section and panel from Gethin-Saez .dta files."""
    cross = pd.read_stata(DATA / "gethin-saez-cross-2026-03-02.dta")
    panel = pd.read_stata(DATA / "gethin-saez-panel-2026-03-02.dta")

    for df in [cross, panel]:
        df["log_agdp"] = np.log(df["agdp"])
        df["log_w"] = np.log(df["w"])
        df["swiid_name"] = df["isoname"].map(
            lambda x: NAME_MAP.get(x.strip(), x.strip())
        )

    cross["emp_total"] = cross["e"] * cross["npop_all"]
    return cross, panel


def load_swiid():
    """Load SWIID v9.91 summary CSV."""
    return pd.read_csv(DATA / "swiid9_91" / "swiid9_91_summary.csv")


def merge_cross(cross: pd.DataFrame, swiid: pd.DataFrame) -> pd.DataFrame:
    """Merge most-recent SWIID observation (≤ 2023) onto GS cross-section."""
    sw = (
        swiid[swiid["year"] <= 2023]
        .sort_values("year")
        .groupby("country", as_index=False)
        .last()[["country", "year", "gini_disp", "gini_mkt", "rel_red", "abs_red"]]
    )
    sw.rename(columns={"year": "swiid_year"}, inplace=True)
    merged = cross.merge(sw, left_on="swiid_name", right_on="country", how="left")
    merged["redist"] = merged["rel_red"]
    n = merged["gini_mkt"].notna().sum()
    print(f"Cross-section SWIID merge: {n}/{len(merged)} countries with market Gini")
    return merged


def merge_panel(panel: pd.DataFrame, swiid: pd.DataFrame) -> pd.DataFrame:
    """Merge SWIID by country-year onto GS panel."""
    sw = swiid[["country", "year", "gini_mkt", "gini_disp", "rel_red"]].copy()
    merged = panel.merge(
        sw, left_on=["swiid_name", "year"], right_on=["country", "year"], how="left"
    )
    merged["redist"] = merged["rel_red"]
    n = merged["gini_mkt"].notna().sum()
    print(f"Panel SWIID merge: {n}/{len(merged)} obs with market Gini")
    return merged


def main():
    cross, panel = load_gethin_saez()
    swiid = load_swiid()

    cross_m = merge_cross(cross, swiid)
    panel_m = merge_panel(panel, swiid)

    cross_m.to_csv(OUT / "cross_merged.csv", index=False)
    panel_m.to_csv(OUT / "panel_merged.csv", index=False)
    print(f"Saved cross_merged.csv ({len(cross_m)} rows) and panel_merged.csv ({len(panel_m)} rows)")


if __name__ == "__main__":
    main()
