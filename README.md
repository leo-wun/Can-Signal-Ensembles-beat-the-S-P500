# Can Signal Ensembles Beat the S&P 500?

**EPFL — Machine Learning for Finance, May 2026**  
Dario Frey (344524) · Leo Wunderli (329314)

An end-to-end empirical study of cross-sectional return predictability in the U.S. equity market. The project follows two stages: (1) a daily technical-signal analysis that documents the micro-cap bid–ask-bounce artifact; (2) a monthly ML pipeline combining CRSP returns with Compustat fundamentals, which produces a net Sharpe of **+1.12** on a small-cap universe (vs. S&P 500 **+0.77**) over the 2015–2024 test period.

## Links

- **GitHub repository:** `https://github.com/DarioFrey/ML_For_Finance_Project--DarioFrey-344524-LeoWunderli-329314` *(replace with the actual private repo URL; invite `DjoFE2021`)*
- **Raw data archive (zipped):** `https://drive.google.com/drive/folders/<RAW-DATA-ARCHIVE-PLACEHOLDER>` — a single zip of the raw inputs (Daily/Monthly CRSP, Compustat, Chen–Zimmermann, JKP) so the pipeline can be reproduced **without** re-downloading from WRDS/CRSP. Unzip into `data/raw/`.

> Compliance note: the headline monthly result uses only course-provided datasets (CRSP + Compustat, merged via the endorsed CCM table). VIX and CRSP market capitalisation are supplementary inputs used **only** for the robustness analyses (daily liquidity filter, low-vol, regime tables), as disclosed in §2 of the report.

---

## Repository Structure

```
.
├── config.py                        # All paths and constants
├── password.py                      # WRDS credentials (not committed — see Setup)
├── pyproject.toml / requirements.txt
│
├── data/
│   ├── raw/                         # Source files (CSV + WRDS pulls)
│   │   ├── daily_crsp.csv           # CRSP daily returns (provided by course)
│   │   ├── monthly_crsp.csv         # CRSP monthly returns (provided by course)
│   │   ├── Compustat_quarterly.csv  # Compustat fundamentals (provided by course)
│   │   ├── Chen_Zimmerman_monthly.csv
│   │   ├── [usa]_[all_factors]_[monthly]_[vw_cap].csv  # JKP factors
│   │   ├── ccm_linktable.parquet    # Pulled from WRDS (see Data section)
│   │   └── crsp_liquidity_raw.parquet  # Pulled from WRDS (optional)
│   └── processed/                   # Intermediate parquet files (auto-generated)
│
├── src/
│   ├── data_loading.py              # WRDS pull functions
│   ├── preprocessing.py             # CRSP / CZ cleaning
│   ├── feature_engineering.py       # Daily technical features
│   ├── monthly_features.py          # Monthly technical characteristics
│   ├── compustat_features.py        # Fundamental characteristics from Compustat
│   ├── backtest.py                  # Cost model, backtest engine, performance metrics
│   ├── utils.py                     # Shrink/dtypes, chronological split helpers
│   └── models/
│       ├── mlp_model.py             # Feed-forward neural network (PyTorch)
│       └── lstm_model.py            # LSTM recurrent model (PyTorch)
│
├── notebooks/                       # All steps are plain .py scripts (run with python)
│   ├── preprocess_eda.py         # Clean raw CSVs → parquet + 15 EDA figures
│   ├── features.py               # Rebuild features + CZ; optional WRDS VIX fetch
│   │
│   ├── ic_analysis.py               # Daily cross-sectional IC
│   ├── signal_correlation.py        # Feature correlation matrix
│   ├── regime_validation.py         # VIX / CZ regime analysis
│   ├── regime_comparison.py         # 2000–2018 vs 2019–2024
│   ├── pipeline_v1.py               # Daily XGBoost meta-model
│   ├── turnover_experiment.py       # Rebalancing-frequency sweep
│   ├── liquidity_filter.py          # Market-cap tradable-universe filter
│   ├── long_only_lowvol.py          # Long-only low-volatility portfolio
│   ├── nn_model.py                  # MLP on daily signals
│   ├── lstm_experiment.py           # LSTM on daily signals
│   │
│   ├── monthly_build_panel.py       # Monthly CRSP panel + technical features
│   ├── monthly_merge_compustat.py   # Merge with Compustat via CCM linktable
│   ├── monthly_model.py             # Lasso / XGBoost / MLP on full universe
│   ├── monthly_model_universe_sweep.py  # Universe-size sweep (expensive)
│   └── monthly_cost_robustness.py   # Trading-cost sensitivity sweep (expensive)
│
├── plots/                           # All generated figures
└── report/
    ├── main.tex
    ├── references.bib
    └── figures/
```

---

## Setup

### 1 · Python environment

Use pip:

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `polars`, `scipy`, `scikit-learn`, `xgboost`, `torch`, `wrds`, `pyarrow`, `matplotlib`, `seaborn`.

### 2 · WRDS credentials

Two scripts pull supplementary data directly from WRDS (Wharton Research Data Services): the CCM linking table and the CRSP liquidity file. Both require a WRDS account.

Create a file called **`password.py`** in the project root (this file is listed in `.gitignore` and is **never committed**):

```python
# password.py
WRDS_USERNAME = "your_wrds_username"
WRDS_PASSWORD = "your_wrds_password"
```

> **Note:** If you do not have WRDS access, the CCM linktable and CRSP liquidity parquets are included as pre-computed files in `data/raw/`, so the WRDS pull steps can be skipped.

---

## Data

### Provided by the course

Place the following files in `data/raw/` exactly as named:

| File | Description |
|---|---|
| `daily_crsp.csv` | CRSP daily stock returns, 2000–2024 (PERMNO, date, DlyRet, sprtrn) |
| `monthly_crsp.csv` | CRSP monthly stock returns (PERMNO, date, ret, sprtrn) |
| `Compustat_quarterly.csv` | Compustat quarterly fundamentals (YTD flow items) |
| `Chen_Zimmerman_monthly.csv` | Chen–Zimmermann (2022) monthly anomaly portfolio returns |
| `[usa]_[all_factors]_[monthly]_[vw_cap].csv` | JKP factor returns |

### Pulled from WRDS (one-time setup)

Run the following commands **once** to populate the remaining raw files. This requires the `password.py` file described above.

```python
# In a Python shell at the project root
import sys; sys.path.insert(0, '.')
from src.data_loading import fetch_ccm_linktable, fetch_crsp_liquidity, fetch_sp500, load_VIX

# CCM linking table — required for the monthly Compustat merge
fetch_ccm_linktable()           # saves to data/raw/ccm_linktable.parquet

# S&P 500 benchmark index — required by the regime/backtest scripts
fetch_sp500()                   # saves to data/raw/gspc.parquet

# CRSP daily price + market cap — required for the daily liquidity filter only
fetch_crsp_liquidity()          # saves to data/raw/crsp_liquidity_raw.parquet

# VIX — optional, used only in the daily regime EDA (not a provided dataset)
load_VIX()                      # saves to data/raw/vix_raw.parquet
```

> `fetch_ccm_linktable()` and `fetch_sp500()` are fast (~seconds). `fetch_crsp_liquidity()` pulls ~20 M rows and takes a few minutes. VIX and the CRSP price/market-cap pull are *not* among the course-provided datasets. They back the daily liquidity filter and regime EDA only; the main monthly result uses provided data exclusively.

---

## Reproducing the Results

All scripts are run from the **project root**. Processed parquet files are cached in `data/processed/`; if they already exist, the expensive steps do not need to be re-run.

### Step 1 — Preprocess raw data and generate EDA figures

Convert the raw CSVs to cleaned parquet files, build the daily technical
features, and produce the exploratory figures:

```bash
python notebooks/preprocess_eda.py
```

This script:
- bootstraps `data/raw/daily_crsp_raw.parquet` from `daily_crsp.csv` if it is missing;
- writes `data/processed/daily_crsp.parquet` (cleaned daily returns);
- writes `data/processed/Chen_Zimmerman_monthly.parquet` (cleaned CZ factors);
- writes `data/processed/features.parquet` (daily rank-normalised technical features);
- saves 15 EDA figures to `plots/` (return distribution, universe size, rank-IC,
  feature correlations, VIX dispersion, CZ factor plots, train/val/test split, …).

> VIX processing is skipped automatically unless `data/raw/vix_raw.parquet` exists
> (VIX is not a provided dataset — see the WRDS section above).
> The IC figures recompute per-date Spearman correlations across ~6,300 dates,
> so this script takes several minutes on the full universe.

Alternatively, `features.py` rebuilds only the feature and CZ parquets and can
fetch VIX from WRDS in the same run:

```bash
python notebooks/features.py --no-vix   # drop --no-vix to also fetch VIX from WRDS
```

### Step 2 — Daily signal analysis (Section 3–4 of the report)

Each script below is self-contained and reads from `data/processed/`.

```bash
# Cross-sectional IC and stability analysis
python notebooks/ic_analysis.py

# Feature correlation matrix
python notebooks/signal_correlation.py

# VIX and CZ factor regime analysis
python notebooks/regime_validation.py

# Daily XGBoost meta-model (pipeline v1) — gross vs net Sharpe
python notebooks/pipeline_v1.py

# Rebalancing-frequency experiment (daily / weekly / monthly)
python notebooks/turnover_experiment.py

# Liquidity filter: tradable universe with market-cap threshold
# Requires data/raw/crsp_liquidity_raw.parquet
python notebooks/liquidity_filter.py

# Long-only low-volatility portfolio
python notebooks/long_only_lowvol.py

# Sub-period regime comparison (2000–2018 vs 2019–2024)
python notebooks/regime_comparison.py
```

### Step 3 — Monthly cross-sectional ML pipeline (Section 5 of the report — main result)

Run the four scripts in order. Each step produces a parquet file consumed by the next.

```bash
# 3a. Build the monthly CRSP panel with technical features
#     Output: data/processed/monthly_panel.parquet
python notebooks/monthly_build_panel.py

# 3b. Merge with Compustat fundamentals via CCM linktable
#     Requires data/raw/ccm_linktable.parquet
#     Output: data/processed/monthly_dataset.parquet
python notebooks/monthly_merge_compustat.py

# 3c. Train Lasso / XGBoost / MLP on full CRSP universe
#     Output: data/processed/monthly_predictions.parquet
#             plots/monthly_model_equity.png
#     Runtime: ~5–10 min (XGBoost + MLP training)
python notebooks/monthly_model.py

# 3d. Universe-size sweep (full / top-N / bottom-N by revenue)
#     Output: plots/monthly_universe_sweep.png
#     Runtime: ~30–60 min (7 universes × 4 models)
python notebooks/monthly_model_universe_sweep.py

# 3e. Trading-cost robustness sweep (0–200 bps)
#     Output: plots/monthly_cost_robustness.png
#     Runtime: ~30–60 min (3 universes × 3 models × 6 cost levels)
python notebooks/monthly_cost_robustness.py
```

### Step 4 — Deep learning models on daily signals (Section 7)

```bash
# MLP on daily technical signals
python notebooks/nn_model.py

# LSTM recurrent model on raw return sequences
python notebooks/lstm_experiment.py
```

### Expected key outputs

| Script | Key metric |
|---|---|
| `monthly_model.py` | Full universe: XGBoost net Sharpe **+0.36**, MLP **+0.38** (S&P 500: **+0.77**) |
| `monthly_model_universe_sweep.py` | Bottom-500 XGBoost: Sharpe **+1.12**, IC **0.135** |
| `monthly_cost_robustness.py` | Bottom-500 XGBoost breakeven vs S&P 500: **~57 bps** |
| `liquidity_filter.py` | Weekly reversal on tradable top-1,000: net Sharpe **−0.12** |

## Notes on Runtime and Reproducibility

- **Randomness:** All models use `random_state=0`.
- **WRDS data versions:** The CRSP and Compustat files were pulled in May 2026.
- **Long-running scripts:** `monthly_model_universe_sweep.py` and `monthly_cost_robustness.py` each take ~30 minutes.
- **GPU:** The MLP and LSTM models automatically use a CUDA GPU if available.
