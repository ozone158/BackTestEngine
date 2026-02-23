/**
 * Step 4.4.2 — C++ fill simulator: slippage, commission, optional half-spread.
 * Input: (symbol, side, quantity, ref_price) per order; config (slippage_bps, commission).
 * Output: (fill_price, commission, slippage_bps) per order. Exposed via pybind11 as batch_fill.
 */

#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace backtest {

/** Apply half-spread then slippage; compute commission. Matches Python BacktestExecutionHandler. */
inline void fill_one(
    const std::string& side,
    double quantity,
    double ref_price,
    double slippage_bps,
    double commission_per_trade,
    double commission_per_share,
    double half_spread_bps,  // < 0 means not set
    double& out_fill_price,
    double& out_commission,
    double& out_slippage_bps)
{
    double effective_ref = ref_price;
    if (half_spread_bps >= 0.0) {
        if (side == "buy") {
            effective_ref = ref_price * (1.0 + half_spread_bps / 10000.0);
        } else {
            effective_ref = ref_price * (1.0 - half_spread_bps / 10000.0);
        }
    }
    if (side == "buy") {
        out_fill_price = effective_ref * (1.0 + slippage_bps / 10000.0);
    } else {
        out_fill_price = effective_ref * (1.0 - slippage_bps / 10000.0);
    }
    out_commission = commission_per_trade + quantity * commission_per_share;
    out_slippage_bps = slippage_bps;
}

/** Batch fill: same config for all orders. Returns (fill_prices, commissions, slippage_bps_list). */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> batch_fill(
    const std::vector<std::string>& symbols,
    const std::vector<std::string>& sides,
    const std::vector<double>& quantities,
    const std::vector<double>& ref_prices,
    double slippage_bps,
    double commission_per_trade,
    double commission_per_share,
    double half_spread_bps)
{
    const size_t n = symbols.size();
    if (sides.size() != n || quantities.size() != n || ref_prices.size() != n) {
        throw std::invalid_argument("batch_fill: symbols, sides, quantities, ref_prices must have same length");
    }
    std::vector<double> fill_prices(n), commissions(n), slippage_bps_list(n);
    for (size_t i = 0; i < n; ++i) {
        fill_one(
            sides[i],
            quantities[i],
            ref_prices[i],
            slippage_bps,
            commission_per_trade,
            commission_per_share,
            half_spread_bps,
            fill_prices[i],
            commissions[i],
            slippage_bps_list[i]);
    }
    return {fill_prices, commissions, slippage_bps_list};
}

}  // namespace backtest

PYBIND11_MODULE(execution_core, m) {
    m.doc() = "C++ fill simulator (Step 4.4); batch_fill(orders, config) -> fill prices and commissions.";

    m.def(
        "batch_fill",
        &backtest::batch_fill,
        py::arg("symbols"),
        py::arg("sides"),
        py::arg("quantities"),
        py::arg("ref_prices"),
        py::arg("slippage_bps"),
        py::arg("commission_per_trade"),
        py::arg("commission_per_share"),
        py::arg("half_spread_bps") = -1.0,
        "Compute fill price and commission for each order. half_spread_bps < 0 means not set.");
}
