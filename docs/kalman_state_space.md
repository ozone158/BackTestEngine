# Kalman Filter State-Space Formulation (Step 4.1.1)

Exact formulation used in Python `KalmanHedgeRatio` so that the C++ implementation can replicate it for bit-level comparable testing.

## State dimension

- **State:** \( x = \beta \) (scalar). State dimension = 1.
- **Interpretation:** \(\beta\) is the hedge ratio; spread = price_a - β × price_b.

## Transition (predict)

- **Model:** Random walk: \( \beta_t = \beta_{t-1} + w_t \), \( w_t \sim N(0, Q) \).
- **F** = 1 (scalar).
- **Process noise:** Q (scalar), configurable (e.g. `DEFAULT_PROCESS_NOISE = 1e-6`).
- **Predict equations:**
  - \( x_{pred} = x_{prev} \)  (i.e. \( \beta_{pred} = \beta_{prev} \))
  - \( P_{pred} = P_{prev} + Q \)

## Observation (measurement)

- **Observation:** \( z = \text{price\_a} \) (we observe price of leg A).
- **Observation equation:** \( z = H x + v \), with \( H = \text{price\_b} \) (time-varying).
- **Predicted observation:** \( z_{pred} = H \cdot x_{pred} = \text{price\_b} \times \beta_{pred} \).
- **Measurement noise:** R (scalar), configurable (e.g. `DEFAULT_MEASUREMENT_NOISE = 1e-4`).

## Update (correct)

- **Innovation:** \( y = z_{obs} - z_{pred} = \text{price\_a} - \beta_{pred} \times \text{price\_b} \).
- **Innovation covariance:** \( S = H P_{pred} H^T + R = \text{price\_b}^2 \times P_{pred} + R \).
- **Kalman gain:** \( K = P_{pred} H^T / S = (P_{pred} \times \text{price\_b}) / S \).
- **State update:** \( x_{new} = x_{pred} + K \cdot y \).
- **Covariance update:** \( P_{new} = (1 - K H) P_{pred} \). If \( P_{new} < 0 \), set \( P_{new} = 0 \).
- **Spread (after update):** \( \text{spread} = \text{price\_a} - \beta_{new} \times \text{price\_b} \).

## Initialization

- If no initial state provided: on first `update(price_a, price_b)`, set \( x = \text{price\_a} / \text{price\_b} \) (or 1.0 if price_b == 0), \( P = 1.0 \).
- If initial state provided: \( x = \text{initial\_beta} \), \( P = \text{initial\_P} \).

## Edge cases (Python behavior)

- If **price_b == 0**: do not update state; return (current β or 1.0, price_a) as spread.
- Use **double precision** throughout in C++ for numerical parity.

## Parameters (scalars)

| Symbol | Meaning | Python default |
|--------|---------|-----------------|
| Q | Process noise | 1e-6 |
| R | Measurement noise | 1e-4 |
| x0 | Initial state (β) | None (then from first obs or OLS) |
| P0 | Initial covariance | None (then 1.0 or from OLS) |
