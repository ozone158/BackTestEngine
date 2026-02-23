/**
 * Step 4.1.5 — pybind11 binding for C++ Kalman filter.
 */

#include "kalman_hedge_ratio.hpp"
#include <pybind11/pybind11.h>
#include <optional>

namespace py = pybind11;

PYBIND11_MODULE(kalman_core, m) {
  m.doc() = "C++ Kalman hedge ratio filter (Step 4.1); update(price_a, price_b) -> (beta, spread).";

  py::class_<backtest::KalmanHedgeRatio>(m, "KalmanFilter")
      .def(py::init([](double process_noise, double measurement_noise,
                       py::object initial_beta, py::object initial_P) {
        std::optional<double> ib = initial_beta.is_none()
            ? std::nullopt
            : std::optional<double>(initial_beta.cast<double>());
        std::optional<double> iP = initial_P.is_none()
            ? std::nullopt
            : std::optional<double>(initial_P.cast<double>());
        return new backtest::KalmanHedgeRatio(process_noise, measurement_noise, ib, iP);
      }),
           py::arg("process_noise"),
           py::arg("measurement_noise"),
           py::arg("initial_beta") = py::none(),
           py::arg("initial_P") = py::none())
      .def("update", &backtest::KalmanHedgeRatio::update,
           py::arg("price_a"),
           py::arg("price_b"),
           "One step: returns (beta, spread).")
      .def_property_readonly("beta", &backtest::KalmanHedgeRatio::beta)
      .def_property_readonly("P", &backtest::KalmanHedgeRatio::P)
      .def_property_readonly("initialized", &backtest::KalmanHedgeRatio::initialized);
}
