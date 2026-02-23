# C++ Kalman filter (Step 4.1)

## Python extension (always built with `pip install -e .`)

- **Sources:** `kalman_hedge_ratio.cpp`, `kalman_filter.cpp` (pybind11 binding).
- **Header:** `kalman_hedge_ratio.hpp` — shared by extension and C++ test.
- From repo root: `pip install -e .` then `python -c "import kalman_core; print(kalman_core.KalmanFilter(1e-6, 1e-4).update(100, 80))"`.

## C++ unit tests (Step 4.1.6, optional)

Google Test is used; CMake fetches it via FetchContent (requires network and CMake).

**Requirements:** CMake 3.14+, C++17 compiler, Git (for FetchContent).

**Build and run** (from repo root):

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build
```

Or run the test executable directly:

```bash
cpp/build/kalman_filter_test   # Unix/macOS
cpp\build\Release\kalman_filter_test.exe   # Windows (e.g. MSVC)
```

**Tests:**

- `FixedSequenceMatchesReference` — fixed (price_a, price_b) sequence; beta and spread must match Python reference within 1e-10.
- `FirstObservationInitializes` — first update(100, 80) yields beta=1.25, spread=0.
- `PriceBZeroReturnsCurrentBetaAndPriceA` — when price_b=0, returns (current β, price_a).

Reference values in `kalman_filter_test.cpp` were generated from the Python `KalmanHedgeRatio` (Q=1e-6, R=1e-4, seed 47).
