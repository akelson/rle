cmake_minimum_required(VERSION 3.1)
project(rle)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark

FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)

FetchContent_MakeAvailable(
        googletest
        googlebenchmark
        Eigen)

enable_testing()

add_library(
    rle
    rle_v1.cpp
    rle_v2.cpp
)
target_link_libraries (
    rle 
    Eigen3::Eigen
)

add_executable(
    rle_test
    rle_v1_test.cpp
    rle_v2_test.cpp
    leb128_test.cpp
)
target_link_libraries(
    rle_test
    rle
    GTest::gmock
    GTest::gtest_main
)

include(GoogleTest)
include(CTest)
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