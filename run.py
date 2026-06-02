"""
MLOps Signal Pipeline
---------------------

Generates trading signals using a rolling mean over time-series data.

Note: The first (window - 1) rows are excluded from signal_rate
calculation because a rolling mean requires a complete window
before producing valid values.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup – must happen before anything else so every module uses it
# ---------------------------------------------------------------------------

def setup_logging(log_path: str = "run.log") -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler – captures DEBUG and above
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler – INFO and above (stderr so it doesn't pollute stdout)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Config loading + validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"version", "seed", "window", "data_path"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    logger.info("Loading config from: %s", config_path)

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must be a YAML mapping/dict")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # Type / value validation
    if not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ValueError("'seed' must be a non-negative integer")

    if not isinstance(cfg["window"], int) or cfg["window"] < 2:
        raise ValueError("'window' must be an integer >= 2")

    if not isinstance(cfg["data_path"], str) or not cfg["data_path"]:
        raise ValueError("'data_path' must be a non-empty string")

    logger.info(
        "Config validated – version=%s | seed=%d | window=%d | data_path=%s",
        cfg["version"],
        cfg["seed"],
        cfg["window"],
        cfg["data_path"],
    )
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"close"}


def load_data(data_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading data from: %s", data_path)

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Data is missing required columns: {missing_cols}")

    null_close = df["close"].isna().sum()
    if null_close > 0:
        logger.warning("Found %d null values in 'close' column – they will be dropped", null_close)
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        logger.info("Rows after dropping nulls: %d", len(df))

    if len(df) == 0:
        raise ValueError("Data file is empty after null removal")

    return df


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, window: int, seed: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Deterministic signal pipeline:
      1. Set random seed (for reproducibility if any stochastic step is added later)
      2. Compute rolling mean of 'close' using the given window
      3. Generate binary signal: 1 if close > rolling_mean, else 0
         For the first (window-1) rows the rolling mean is NaN;
         those rows receive signal=0 and are excluded from signal_rate.
    """
    import numpy as np  # local import to keep top-level light

    np.random.seed(seed)
    logger.info("Computing rolling mean (window=%d) on %d rows", window, len(df))

    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()

    # Rows where rolling_mean is NaN (first window-1 rows)
    warmup_mask = df["rolling_mean"].isna()
    warmup_count = warmup_mask.sum()
    logger.debug("Warm-up rows (NaN rolling_mean, excluded from signal): %d", warmup_count)

    # Signal: 1 where close > rolling_mean; 0 otherwise; 0 during warm-up
    df["signal"] = 0
    valid_mask = ~warmup_mask
    df.loc[valid_mask, "signal"] = (
        df.loc[valid_mask, "close"] > df.loc[valid_mask, "rolling_mean"]
    ).astype(int)

    logger.info("Signal generation complete")
    logger.debug(
        "Signal distribution – 1s: %d | 0s: %d | warm-up (forced 0): %d",
        df.loc[valid_mask, "signal"].sum(),
        (df.loc[valid_mask, "signal"] == 0).sum(),
        warmup_count,
    )

    return df


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, version: str, seed: int, start_time: float, logger: logging.Logger) -> dict:
    window_size = int(df["rolling_mean"].isna().sum())  # warm-up rows == window-1

    # Exclude warm-up rows from signal_rate
    valid_signals = df.loc[~df["rolling_mean"].isna(), "signal"]
    signal_rate = round(float(valid_signals.mean()), 4)
    rows_processed = int(len(df))
    latency_ms = int(round((time.time() - start_time) * 1000))

    logger.info(
        "Metrics – rows_processed=%d | signal_rate=%.4f | latency_ms=%d",
        rows_processed,
        signal_rate,
        latency_ms,
    )

    return {
        "version": version,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": signal_rate,
        "latency_ms": latency_ms,
        "seed": seed,
        "status": "success",
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def write_metrics(metrics: dict, output_dir: str, logger: logging.Logger) -> None:
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("metrics.json written to: %s", metrics_path)


def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(description="MLOps Signal Pipeline")

    parser.add_argument("--input", help="Input CSV file")
    parser.add_argument("--config", help="Config YAML file")
    parser.add_argument("--output", help="Metrics output JSON file")
    parser.add_argument("--log-file", help="Log file path")

    args = parser.parse_args()
    log_path = args.log_file or os.environ.get("LOG_PATH", "run.log")
    config_path = args.config or os.environ.get("CONFIG_PATH", "config.yaml")
    output_dir = os.environ.get("OUTPUT_DIR", ".")

    logger = setup_logging(log_path)
    logger.info("=== MLOps Pipeline started ===")

    metrics: dict = {}

    try:
        # 1. Load + validate config
        cfg = load_config(config_path, logger)
        version = cfg["version"]
        seed = cfg["seed"]
        window = cfg["window"]
        data_path = args.input or cfg["data_path"]
        output_dir = cfg.get("output_dir", output_dir)

        # 2. Load data
        df = load_data(data_path, logger)

        # 3. Compute rolling mean + signals
        df = compute_signals(df, window=window, seed=seed, logger=logger)

        # 4. Compute metrics + timing
        metrics = compute_metrics(df, version=version, seed=seed, start_time=start_time, logger=logger)

        # 5. Write metrics.json
        write_metrics(metrics, output_dir=output_dir, logger=logger)

        logger.info("=== Pipeline finished successfully ===")

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        metrics = {
            "version": "v1",
            "status": "error",
            "error_message": str(exc),
        }
        try:
            write_metrics(metrics, output_dir=output_dir, logger=logger)
        except Exception as write_exc:
            logger.error("Could not write error metrics.json: %s", write_exc)
        print(json.dumps(metrics, indent=2))
        return 1

    # Print final metrics JSON to stdout (required by spec)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
