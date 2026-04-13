# Working Longer in Unequal Societies

Replication code for the master's thesis *"Working Longer in Unequal Societies: Cross-Country Evidence from the Global Working Hours Database"* (University of Warsaw, Faculty of Economic Sciences, 2026).

## Data sources

| Dataset | Source | Filename |
|---------|--------|----------|
| Global Working Hours (cross-section) | [Gethin & Saez (2025)](https://amory-gethin.fr/data.html) | `gethin-saez-cross-2026-03-02.dta` |
| Global Working Hours (panel) | [Gethin & Saez (2025)](https://amory-gethin.fr/data.html) | `gethin-saez-panel-2026-03-02.dta` |
| SWIID v9.91 | [Solt (2020)](https://fsolt.org/swiid/) | `swiid9_91/swiid9_91_summary.csv` |

Place these files in the `data/` directory before running the scripts.

## Repository structure

```
working-longer/
├── data/                          # Raw data (not tracked by git)
│   ├── gethin-saez-cross-2026-03-02.dta
│   ├── gethin-saez-panel-2026-03-02.dta
│   └── swiid9_91/
│       └── swiid9_91_summary.csv
├── scripts/
│   ├── 01_merge_data.py           # Load & merge GS + SWIID
│   ├── 02_cross_section_ols.py    # Table 3.1, robustness, H2 test
│   ├── 03_panel_twfe.py           # Table 3.2, TWFE with clustered SEs
│   ├── 04_heterogeneity_and_p90.py # Tables 3.5, 3.6, subsample regressions
│   └── 05_figures.py              # LOWESS / nonlinear fit figures
├── output/                        # Generated outputs (not tracked)
│   ├── cross_merged.csv
│   ├── panel_merged.csv
│   ├── tables/
│   └── figures/
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Step 1: Merge datasets
python scripts/01_merge_data.py

# Step 2: Cross-sectional OLS (Table 3.1, robustness, H2 test)
python scripts/02_cross_section_ols.py

# Step 3: Panel TWFE (Table 3.2)
python scripts/03_panel_twfe.py

# Step 4: Heterogeneity & P90 (Tables 3.5, 3.6)
python scripts/04_heterogeneity_and_p90.py

# Step 5: Figures
python scripts/05_figures.py
```

## Key results

- **Cross-section (unweighted, M2):** β(taxr_lab) = −29.6 (SE = 3.3, p < 0.001)
- **Panel TWFE (Western, clustered SE):** β(taxr_lab) = −17.3 (SE = 6.9, p = 0.012)
- **P90 model (full, unweighted):** β(gini_mkt) = 0.291 (SE = 0.094, p < 0.01)
- **H2 test (standardised):** |β*(taxr)| / |β*(gini)| = 28:1 in M3

## License

Code: MIT. Data subject to original providers' terms of use.
