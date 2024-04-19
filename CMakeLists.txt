cmake_minimum_required(VERSION 3.1)
project(rle)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

enable_testing()

add_library(
    rle
    rle.cpp
)

target_link_libraries (
    rle 
    Eigen3::Eigen
)

add_executable(
    rle_test
    rle_test.cpp
)

target_link_libraries(
    rle_test
    rle
    GTest::gmock
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(rle_test)

add_executable(
    rle_benchmark
    rle_benchmark.cpp
)
target_link_libraries(
    rle_benchmark
    rle
    benchmark::benchmark
    benchmark::benchmark_main
)