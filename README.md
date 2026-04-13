# Working Longer in Unequal Societies

**Cross-Country Evidence from the Global Working Hours Database**

Master's thesis · University of Warsaw, Faculty of Economic Sciences · 2026

---

## Overview

Do people in more unequal countries work longer hours? This thesis investigates the relationship between income redistribution and working time using the **Global Working Hours database** (Gethin & Saez, 2025) — the most comprehensive harmonised dataset on working hours available, covering **160 countries** across all world regions.

The analysis combines cross-sectional OLS regressions with panel two-way fixed-effects models and supplements the Gethin–Saez data with market income Gini coefficients from the SWIID (Solt, 2020).

### Key findings

| Result | Estimate | SE | p-value |
|--------|----------|-----|---------|
| Cross-section: 10pp ↑ tax rate → Δ hours/week | **−3.0 h** | 0.33 | < 0.001 |
| Panel TWFE (Western): 10pp ↑ tax rate → Δ hours/week | **−1.7 h** | 0.69 | 0.012 |
| P90 model: 1pt ↑ market Gini → Δ P90 hours | **+0.29 h** | 0.09 | < 0.01 |
| H2 test: \|β\*(tax rate)\| vs \|β\*(Gini)\| | **28 : 1** | — | — |

The annualised hours gap between the most and least redistributive quartiles amounts to approximately **900 hours per worker** — nearly five additional months of full-time work per year.

## Data

| Dataset | Source | Coverage |
|---------|--------|----------|
| Global Working Hours | [Gethin & Saez (2025)](https://amory-gethin.fr/data.html) | 160 countries, 1900–2023 |
| SWIID v9.91 | [Solt (2020)](https://fsolt.org/swiid/) | 199 countries, 1960–2025 |

Place `.dta` and SWIID files in the `data/` directory. They are excluded from version control via `.gitignore`.

## Replication

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/01_merge_data.py              # Merge Gethin-Saez + SWIID
python scripts/02_cross_section_ols.py       # Table 3.1, robustness, H2 test
python scripts/03_panel_twfe.py              # Table 3.2, clustered SEs
python scripts/04_heterogeneity_and_p90.py   # Income groups, gender/age/edu, P90
python scripts/05_figures.py                 # LOWESS nonlinear fit figures
```

## Repository structure

```
working-longer/
├── scripts/
│   ├── 01_merge_data.py              Load & merge GS + SWIID
│   ├── 02_cross_section_ols.py       Cross-sectional OLS (6 models + robustness)
│   ├── 03_panel_twfe.py              Panel TWFE with entity-clustered SEs
│   ├── 04_heterogeneity_and_p90.py   Subsample regressions & P90 analysis
│   └── 05_figures.py                 LOWESS / quadratic / linear fit figures
├── data/                             Raw data (not tracked)
├── output/                           Generated tables & figures
├── requirements.txt
└── README.md
```

## Methods

**Cross-section:** Unweighted OLS with HC1-robust standard errors across 6 nested specifications. Population-weighted regressions reported as robustness checks. The leverage of China and India (~60% of population weight) attenuates significance in weighted models — discussed explicitly in the thesis.

**Panel:** Two-way fixed effects (country + year) with entity-clustered standard errors. Estimated separately for all countries (80 countries, 2,001 observations) and Western economies (23 countries, 788 observations).

**Heterogeneity:** Subsample regressions by World Bank income group, gender, age group, and education level. The tax-rate effect is significant at 1% for every demographic subgroup.

## References

- Bowles, S. & Park, Y. (2005). Emulation, inequality, and work hours: Was Thorsten Veblen right? *Economic Journal*, 115(507), F397–F412.
- Behringer, J., Hein, E. & van Treeck, T. (2024). Income distribution and working hours in a comparative political economy perspective. *Oxford Economic Papers*, 76(3), 736–757.
- Gethin, A. & Saez, E. (2025). Global Working Hours. *NBER Working Paper* No. 34217.
- Liu, W., Sommet, N. & Du, H. (2025). Income inequality and working hours: A longitudinal cross-national analysis. *Social Psychological and Personality Science*.
- Solt, F. (2020). Measuring income inequality across countries and over time: The standardized world income inequality database. *Social Science Quarterly*, 101(3), 1183–1199.

## License

Code: MIT. Data subject to original providers' terms of use.
