# Step 4.1.6 — C++ unit tests for Kalman filter (optional build).
# From repo root: cmake -S cpp -B cpp/build && cmake --build cpp/build && ctest --test-dir cpp/build

cmake_minimum_required(VERSION 3.14)
project(kalman_cpp_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(kalman_filter_test
  kalman_hedge_ratio.cpp
  kalman_filter_test.cpp
)
target_include_directories(kalman_filter_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kalman_filter_test PRIVATE GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(kalman_filter_test)
