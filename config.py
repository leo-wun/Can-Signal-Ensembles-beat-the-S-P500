from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "plots"

CRSP_PATH = DATA_RAW / "daily_crsp.csv"
COMPUSTAT_PATH = DATA_RAW / "Compustat_quarterly.csv"
FUTURES_PATH = DATA_RAW / "daily_futures.csv"
JKP_PATH = DATA_RAW / "JKP_factors_US_monthly.csv"
CZ_PATH = DATA_RAW / "Chen_Zimmerman_monthly.csv"
CRSP_PATH_RAW = DATA_RAW / "daily_crsp_raw.parquet"
VIX_PATH_RAW = DATA_RAW / "vix_raw.parquet"
CZ_PATH_RAW = DATA_RAW/ "Chen_Zimmerman_monthly_raw.parquet"


CRSP_PATH_CLEAN = DATA_PROCESSED / "daily_crsp.parquet"
FEATURES_PATH_CLEAN = DATA_PROCESSED / "features.parquet"
CZ_PATH_CLEAN = DATA_PROCESSED / "Chen_Zimmerman_monthly.parquet"
VIX_PATH_CLEAN = DATA_PROCESSED / "vix.parquet"

GSPC_PATH_RAW = DATA_RAW / 'gspc.parquet'
GSPC_PATH_CLEAN = DATA_PROCESSED / 'gspc.parquet'





# Project date window (overlap of CRSP and futures with sufficient coverage)
START_DATE = "2000-01-01"
END_DATE = "2020-11-30"

# ── Feature windows ───────────────────────────────────────────────────────────

# Momentum: lookback windows in trading days
# Each window computes cumulative log return over [t-w-1, t-2]
# (skip most recent day per Jegadeesh-Titman to avoid microstructure noise)
MOMENTUM_WINDOWS = [5, 21, 63, 126, 252]   # 1w, 1m, 3m, 6m, 12m

# Short-term reversal: previous day return (kept separate from momentum)
REVERSAL_LAG = 1

# Volatility: windows for realized std of daily returns
VOLATILITY_WINDOWS = [21, 63, 252]          # 1m, 3m, 12m

# Futures: how many lags of futures log returns to include as macro features
FUTURES_LAGS = [1, 2, 3, 5]               # t-1 through t-5

# Moving average for VIX
MA_RANGE = [63, 126, 252]


# Minimum number of non-null observations for a futures instrument to be kept
FUTURES_MIN_HISTORY_YEARS = 5.0

# Train / validation / test split dates
TRAIN_END = "2015-12-31"
VAL_END = "2018-12-31"
# test: 2019-01-01 → END_DATE



# ── Global Variables ───────────────────────────────────────────────────────────

# Column Name containing daily return
VALUE_RETURN = 'ret'

# Number of stock within one random sampled batch
BATCH_SIZE = 500

# Total number of batch sampled
BATCH_NUMBER = 5

# Lookback period for normalization for the Chen-Zimmerman dataset (in month)
CZ_LOOKBACK = 12