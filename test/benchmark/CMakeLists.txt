include(FetchContent)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark
FetchContent_MakeAvailable(googlebenchmark)

add_executable(
    run_benchmark
    rle_benchmark.cpp
    sparse_image_benchmark.cpp
)
target_include_directories(
    run_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(
    run_benchmark
    imgproc
    benchmark::benchmark
    benchmark::benchmark_main
)