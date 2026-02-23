/**
 * Step 4.1 — C++ Kalman filter implementation (no pybind11).
 */

#include "kalman_hedge_ratio.hpp"
#include <cmath>
#include <limits>
#include <tuple>

namespace backtest {

namespace {
std::tuple<double, double> sanitize(double beta, double spread) {
  auto bad = [](double v) {
    return std::isnan(v) || std::isinf(v);
  };
  if (bad(beta)) beta = 1.0;
  if (bad(spread)) spread = 0.0;
  return {beta, spread};
}
}  // namespace

KalmanHedgeRatio::KalmanHedgeRatio(double process_noise, double measurement_noise,
                                   std::optional<double> initial_beta,
                                   std::optional<double> initial_P)
    : Q_(process_noise),
      R_(measurement_noise),
      x_(initial_beta),
      P_(initial_P) {}

std::tuple<double, double> KalmanHedgeRatio::update(double price_a, double price_b) {
  if (price_b == 0.0) {
    double b = x_.value_or(1.0);
    return sanitize(b, price_a);
  }

  if (!x_.has_value() || !P_.has_value()) {
    x_ = (price_b != 0.0) ? (price_a / price_b) : 1.0;
    P_ = 1.0;
  }

  double x_pred = *x_;
  double P_pred = *P_ + Q_;
  double H = price_b;
  double z_obs = price_a;
  double z_pred = x_pred * H;
  double innovation = z_obs - z_pred;
  double S = H * P_pred * H + R_;
  double K = (S > 0.0) ? ((P_pred * H) / S) : 0.0;
  x_ = x_pred + K * innovation;
  double P_new = (1.0 - K * H) * P_pred;
  P_ = (P_new < 0.0) ? 0.0 : P_new;
  double spread = price_a - (*x_) * price_b;

  return sanitize(*x_, spread);
}

double KalmanHedgeRatio::beta() const { return x_.value_or(1.0); }
double KalmanHedgeRatio::P() const { return P_.value_or(1.0); }
bool KalmanHedgeRatio::initialized() const { return x_.has_value() && P_.has_value(); }

}  // namespace backtest
