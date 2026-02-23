/**
 * Step 4.1.6 — C++ unit test for Kalman hedge ratio.
 * Fixed (price_a, price_b) sequence; assert beta and spread match Python reference.
 */

#include "kalman_hedge_ratio.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr double kTol = 1e-10;

// Fixed sequence and reference (beta, spread) from Python KalmanHedgeRatio, Q=1e-6, R=1e-4, seed 47.
const double kQ = 1e-6;
const double kR = 1e-4;

struct Step {
  double price_a;
  double price_b;
  double ref_beta;
  double ref_spread;
};

const std::vector<Step> kSequence = {
    {129.91928952388008, 99.991519905242342, 1.2993030773709511, 0.0},
    {130.06116164305462, 100.00457896882183, 1.3005398207152614, 0.0012244402378769337},
    {129.97135034873423, 100.01382104848400, 1.2995437545592736, -0.00098616443196419823},
    {129.97913489460339, 100.02022516649635, 1.2995286664811760, -1.4937160557337847e-05},
    {130.03441144725215, 100.00967779666706, 1.3002115216740715, 0.0006760971140806759},
    {130.07569993846562, 100.02765540381831, 1.3003955480250180, 0.00018217195963643462},
    {130.05876245484487, 100.01752753631344, 1.3003600548595224, -3.5139189463961884e-05},
    {130.03787601325246, 100.02576376955686, 1.3000469189090647, -0.00030998688589534140},
    {129.96352502551747, 100.02281726602335, 1.2993457168063356, -0.00069417199276244901},
    {130.02455002734223, 100.01689256030824, 1.3000192257853849, 0.00066679563030902500},
};

TEST(KalmanHedgeRatioTest, FixedSequenceMatchesReference) {
  backtest::KalmanHedgeRatio kf(kQ, kR);
  for (size_t i = 0; i < kSequence.size(); ++i) {
    const auto& step = kSequence[i];
    auto [beta, spread] = kf.update(step.price_a, step.price_b);
    EXPECT_NEAR(beta, step.ref_beta, kTol) << "step " << i << " beta";
    EXPECT_NEAR(spread, step.ref_spread, kTol) << "step " << i << " spread";
  }
}

TEST(KalmanHedgeRatioTest, FirstObservationInitializes) {
  backtest::KalmanHedgeRatio kf(kQ, kR);
  EXPECT_FALSE(kf.initialized());
  kf.update(100.0, 80.0);
  EXPECT_TRUE(kf.initialized());
  EXPECT_DOUBLE_EQ(kf.beta(), 1.25);
  auto [beta, spread] = kf.update(100.0, 80.0);
  EXPECT_NEAR(spread, 0.0, 1e-9);
}

TEST(KalmanHedgeRatioTest, PriceBZeroReturnsCurrentBetaAndPriceA) {
  backtest::KalmanHedgeRatio kf(kQ, kR);
  kf.update(100.0, 80.0);
  auto [beta, spread] = kf.update(50.0, 0.0);
  EXPECT_DOUBLE_EQ(beta, 1.25);
  EXPECT_DOUBLE_EQ(spread, 50.0);
}

}  // namespace
