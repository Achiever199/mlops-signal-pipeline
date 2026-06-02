# MLOps Signal Pipeline

A lightweight, deterministic signal-generation pipeline that computes a rolling-mean crossover signal on OHLCV market data, outputs structured metrics, and ships as a self-contained Docker image.

---

## What it does

| Step | Description |
|------|-------------|
| 1 | Load `config.yaml` and validate all required fields |
| 2 | Load `data.csv` (OHLCV format) and validate schema |
| 3 | Compute rolling mean of `close` over the configured `window` |
| 4 | Generate binary signal: `1` if `close > rolling_mean`, else `0` |
| 5 | Compute `signal_rate`, `rows_processed`, `latency_ms` |
| 6 | Write `metrics.json` and `run.log`; print metrics to stdout |

---

## Local run instructions

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

The script reads `config.yaml` from the current directory by default.  
You can override paths with environment variables:

```bash
CONFIG_PATH=config.yaml LOG_PATH=run.log OUTPUT_DIR=. python run.py
```

---

## Docker build/run commands

```bash
# Build
docker build -t mlops-task .

# Run
docker run --rm mlops-task
```

To extract output files from the container:

```bash
docker run --rm -v $(pwd)/output:/app mlops-task
# metrics.json and run.log will appear in ./output/
```

---

## Configuration (`config.yaml`)

```yaml
version: "v1"    # Pipeline version string; appears in metrics.json
seed: 42         # Random seed for full reproducibility
window: 5      # Rolling mean window size (integer >= 2)
data_path: "data.csv"   # Path to input CSV (relative or absolute)
output_dir: "."  # Directory where metrics.json is written
```

---

## Input data format (`data.csv`)

CSV with at minimum a `close` column. Additional columns (`open`, `high`, `low`, `volume`, etc.) are accepted and ignored.

```
id,open,high,low,close,volume
1,100.0,101.2,99.5,100.8,123456
...
```

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4963,
  "latency_ms": 51,
  "seed": 42,
  "status": "success"
}
```

In error cases the file will contain:

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

> `metrics.json` is **always** written — even when the pipeline fails.

---

## Outputs

| File | Description |
|------|-------------|
| `metrics.json` | Structured metrics; always written |
| `run.log` | Timestamped log of every pipeline step |

---

## Design decisions

- **Warm-up rows excluded from `signal_rate`:** The first `window-1` rows have no rolling mean (NaN). They receive `signal=0` but are excluded from the `signal_rate` calculation to avoid skewing the metric.
- **Determinism:** `numpy.random.seed(seed)` is set at the start of signal computation; given the same `config.yaml` and `data.csv`, outputs are always identical.
- **Error handling:** All exceptions are caught, logged, and written to `metrics.json` as an error payload — the container exits with code `1`.
- **No hardcoded paths:** All paths come from `config.yaml` or environment variables.
