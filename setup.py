"""
Build C++ Kalman extension (Step 4.1.5).

Usage:
  pip install -e .   # build and install in development
  python -c "import kalman_core; print(kalman_core.KalmanFilter(1e-6, 1e-4).update(100.0, 80.0))"

Requires: pybind11, setuptools, and a C++17-capable compiler (MSVC, g++, clang).
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "kalman_core",
        ["cpp/kalman_hedge_ratio.cpp", "cpp/kalman_filter.cpp"],
        cxx_std=17,
        include_dirs=["cpp"],
    ),
    Pybind11Extension(
        "execution_core",
        ["cpp/fill_simulator.cpp"],
        cxx_std=17,
        include_dirs=["cpp"],
    ),
]

setup(
    name="backtest-engine",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
)
