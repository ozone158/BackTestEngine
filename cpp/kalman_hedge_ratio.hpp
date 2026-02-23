/**
 * Step 4.1 — C++ Kalman filter for dynamic hedge ratio (header).
 *
 * State-space: x = β (scalar). F=1, Q=process_noise, H=price_b, R=measurement_noise.
 * Formulation matches docs/kalman_state_space.md and Python KalmanHedgeRatio.
 */

#ifndef BACKTEST_CPP_KALMAN_HEDGE_RATIO_HPP_
#define BACKTEST_CPP_KALMAN_HEDGE_RATIO_HPP_

#include <optional>
#include <tuple>

namespace backtest {

class KalmanHedgeRatio {
 public:
  KalmanHedgeRatio(double process_noise, double measurement_noise,
                   std::optional<double> initial_beta = std::nullopt,
                   std::optional<double> initial_P = std::nullopt);

  /** One step: predict then update. Returns (beta, spread) after update. */
  std::tuple<double, double> update(double price_a, double price_b);

  double beta() const;
  double P() const;
  bool initialized() const;

 private:
  double Q_;
  double R_;
  std::optional<double> x_;
  std::optional<double> P_;
};

}  // namespace backtest

#endif  // BACKTEST_CPP_KALMAN_HEDGE_RATIO_HPP_
