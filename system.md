# High-Frequency Statistical Arbitrage: System Design & Project Rundown

## 1. Project Overview

**Title:** High-Frequency Statistical Arbitrage: An End-to-End Pairs Trading Framework

**Purpose:** A production-oriented framework for identifying, backtesting, and evaluating mean-reverting price relationships (pairs trading) with rigorous statistics, event-driven simulation, and risk-aware performance attribution.

**Design Principles:**
- **No look-ahead bias** — All backtesting is strictly event-driven and time-ordered.
- **High throughput** — Columnar storage, async I/O, and C++ acceleration for hot paths.
- **Statistical rigor** — Cointegration, Kalman-filtered hedge ratios, and OU-process spread modeling.

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MODULE 1: DATA INFRASTRUCTURE                             │
│  [Raw Sources] → Ingestion (REST/WS) → Preprocessing → Columnar Storage (Parquet) │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MODULE 2: ALPHA RESEARCH & SIGNALS                           │
│  Cointegration (ADF/Johansen) → Kalman Filter (β) → OU Process → Entry/Exit       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MODULE 3: EVENT-DRIVEN BACKTESTING                             │
│  DataHandler ←→ Strategy ←→ Portfolio ←→ ExecutionHandler (C++ hot path)         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                 MODULE 4: RISK & PERFORMANCE ATTRIBUTION                           │
│  Sharpe/Sortino/Calmar → PCA (Alpha/Beta) → Kelly / Risk Parity sizing            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Dependency Graph

| Module | Depends On | Consumed By |
|--------|------------|-------------|
| **1. Data Infrastructure** | — | Module 2, 3 |
| **2. Alpha Research** | Module 1 | Module 3 |
| **3. Backtesting Engine** | Module 1, 2 | Module 4 |
| **4. Risk & Attribution** | Module 3 | — (output/reporting) |

---

## 3. Module Rundown

### Module 1: Data Infrastructure & Engineering

**Objective:** Robust, high-throughput pipeline for non-stationary financial time series.

| Component | Description | Deliverables |
|-----------|-------------|--------------|
| **Data Ingestion** | Async scrapers and API connectors (REST/WebSocket) for tick-level or bar-level data. | Connectors per venue/source; unified bar/tick schema. |
| **Preprocessing** | Corporate action adjustments (dividends/splits), missing-data handling (e.g. linear interpolation), outlier detection (Z-score or Hampel filter). | Adjusted OHLCV + metadata; outlier flags or cleaned series. |
| **Storage** | Migration from flat files to columnar formats: **Apache Parquet** or **HDF5** for I/O efficiency at scale. | Partitioned Parquet/HDF5 datasets; read/write APIs. |

**Key Interfaces:**
- `DataSource` (abstract): `fetch(symbol, start, end)` → raw bars/ticks.
- `Preprocessor`: `adjust(series)`, `interpolate(series)`, `detect_outliers(series)`.
- `StorageBackend`: `write(partition_key, dataframe)`, `read(symbols, start, end)`.

---

### Module 2: Alpha Research & Signal Generation

**Objective:** Identify mean-reverting price relationships using rigorous statistics.

| Component | Description | Deliverables |
|-----------|-------------|--------------|
| **Cointegration** | **Augmented Dickey-Fuller (ADF)** and **Johansen** tests to find pairs with long-term equilibrium. | Candidate pair universe; cointegration test results. |
| **Dynamic Hedge Ratio** | **Kalman Filter** to recursively estimate time-varying β; adapts to regime changes vs static OLS. | β(t) and spread series; filter state. |
| **Spread Modeling** | Model spread as **Ornstein-Uhlenbeck (OU)** process; estimate mean-reversion speed and optimal entry/exit thresholds. | OU parameters (θ, μ, σ); entry/exit bands. |

**Key Interfaces:**
- `CointegrationTest`: `adf(series)`, `johansen(price_matrix)` → test stats, p-values, cointegrating vectors.
- `KalmanHedgeRatio`: `update(price_a, price_b)` → β, spread; state.
- `OUModel`: `fit(spread_series)`, `entry_exit_thresholds()` → parameters and bands.

**Data Flow:** Clean prices (Module 1) → Cointegration screening → Kalman-filtered spread → OU fit → Signals (long/short spread, size).

---

### Module 3: Event-Driven Backtesting Engine

**Objective:** Realistic trading simulation with strict avoidance of look-ahead bias.

| Component | Description | Deliverables |
|-----------|-------------|--------------|
| **OOP Architecture** | Discrete components: `DataHandler`, `Strategy`, `Portfolio`, `ExecutionHandler`. Events flow in time order only. | Core engine classes; event types and queue. |
| **C++ Acceleration** | Offload Kalman recursions, order matching, and other hot loops via **pybind11**. | Python bindings; C++ lib for engine. |
| **Microstructure** | Bid-ask spread, configurable slippage, and transaction fee schedules. | Fill model; commission/slippage API. |

**Event Loop (conceptual):**
1. `DataHandler`: Emits `MarketEvent` (new bar/tick) in chronological order.
2. `Strategy`: Consumes events, requests signals using only past data.
3. `Portfolio`: Updates positions and equity; may emit `SignalEvent`.
4. `ExecutionHandler`: Simulates fills (spread, slippage, fees); emits `FillEvent`.
5. `Portfolio` updates on `FillEvent`; next bar continues.

**Key Interfaces:**
- `DataHandler`: `get_latest_bars(symbol, N)`, `update_bars()` (no future data).
- `Strategy`: `on_market_event(event)` → optional `SignalEvent`.
- `Portfolio`: `on_signal(signal)`, `on_fill(fill)`; `current_positions`, `equity_curve`.
- `ExecutionHandler`: `execute_order(signal)` → `FillEvent` with realistic costs.

---

### Module 4: Risk Management & Performance Attribution

**Objective:** Evaluate risk-adjusted profitability and factor exposure; keep strategy market-neutral.

| Component | Description | Deliverables |
|-----------|-------------|--------------|
| **Performance Metrics** | **Annualized Sharpe**, **Sortino**, **Calmar** to assess return quality. | Metrics report; time series of returns. |
| **Risk Attribution** | **PCA** to decompose returns into idiosyncratic Alpha vs systematic Beta; monitor market neutrality. | Alpha/Beta breakdown; factor loadings. |
| **Capital Allocation** | **Kelly Criterion** or **Risk Parity** for dynamic position sizing to limit risk of ruin. | Position sizing module; exposure limits. |

**Key Interfaces:**
- `PerformanceMetrics`: `sharpe(returns)`, `sortino(returns)`, `calmar(returns, drawdown)`.
- `RiskAttribution`: `pca_decompose(returns, factor_returns)` → alpha, beta, residuals.
- `PositionSizer`: `kelly(edge, odds)` or `risk_parity(volatilities)` → weights.

---

## 4. Recommended Directory Layout

```
BackTestEngine/
├── system.md                 # This document
├── README.md
├── requirements.txt
├── pyproject.toml            # Build & metadata (PEP 517); tool config (Ruff, pytest)
│
├── src/
│   ├── data/                 # Module 1
│   │   ├── ingestion/        # REST/WebSocket connectors, scrapers
│   │   ├── preprocessing/    # adjustments, interpolation, outliers
│   │   └── storage/          # Parquet/HDF5 backends
│   │
│   ├── alpha/                # Module 2
│   │   ├── cointegration/    # ADF, Johansen
│   │   ├── kalman/           # Kalman filter (Python + C++ bindings)
│   │   └── ou_process/      # OU fit, thresholds
│   │
│   ├── engine/               # Module 3
│   │   ├── events.py
│   │   ├── data_handler.py
│   │   ├── strategy.py
│   │   ├── portfolio.py
│   │   ├── execution.py
│   │   └── core/             # C++ extension via pybind11
│   │
│   └── risk/                 # Module 4
│       ├── metrics.py        # Sharpe, Sortino, Calmar
│       ├── attribution.py    # PCA, alpha/beta
│       └── sizing.py         # Kelly, risk parity
│
├── tests/
├── scripts/                  # Run pipelines, backtests, reports
├── config/                   # YAML/JSON configs per environment
└── data/                     # Local Parquet/HDF5 (gitignored or sample only)
```

---

## 5. Technology Stack (Suggested)

| Layer | Options |
|-------|--------|
| Language | Python 3.11+ (3.12 for new code); C++17 for hot path; type hints throughout |
| Data I/O | `polars` (Parquet, columnar DataFrames; lazy `scan_parquet` for scale), `tables` or `h5py` (HDF5) |
| Analytical SQL (optional) | `duckdb` — SQL on Parquet for ad-hoc queries and reporting |
| Numerics | `numpy`, `scipy` (ADF, linear algebra), `statsmodels` (Johansen, OLS) |
| Async / APIs | `httpx` (REST, preferred) or `aiohttp`, `websockets` for ingestion |
| C++ bindings | `pybind11` |
| Config | `pydantic` (v2) + YAML/JSON; `.env` for local secrets (e.g. `python-dotenv`) |
| Testing | `pytest`, `pytest-asyncio`; optional `pytest-cov` for coverage |
| Tooling | `ruff` (lint/format), `pyproject.toml` for single source of build and tool config |

---

## 6. Implementation Order

1. **Module 1 (Data):** Storage schema + one connector + preprocessing (adjustments, interpolation, outliers). Validate with a small dataset.
2. **Module 2 (Alpha):** Cointegration pipeline → Kalman filter (Python first) → OU fit and thresholds. Output: signal spec (when to enter/exit).
3. **Module 3 (Engine):** Event types → `DataHandler` (reads from Module 1) → `Strategy` (uses Module 2 logic) → `Portfolio` + `ExecutionHandler` (fees/slippage). No C++ until Python path is correct.
4. **C++ acceleration:** Replace Kalman/order-matching hot paths with pybind11 extensions; benchmark.
5. **Module 4 (Risk):** Metrics + PCA attribution + Kelly/Risk Parity sizing; wire into engine output.

---

## 7. Glossary

| Term | Meaning |
|------|--------|
| **ADF** | Augmented Dickey-Fuller test for unit root / stationarity. |
| **Johansen** | Multivariate cointegration test; identifies cointegrating vectors. |
| **Kalman Filter** | Recursive state estimator for time-varying β (hedge ratio). |
| **OU Process** | Ornstein-Uhlenbeck; mean-reverting diffusion for spread. |
| **Look-ahead bias** | Using future information in backtest; forbidden. |
| **Kelly / Risk Parity** | Position sizing methods for capital allocation. |

---

*This document is the single source of truth for the system design and project rundown. Update it as the architecture evolves.*
