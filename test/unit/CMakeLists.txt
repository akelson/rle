include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)
FetchContent_MakeAvailable(
        googletest
)

add_executable(
    run_unit_test
    rle_v1_test.cpp
    rle_v2_test.cpp
    leb128_test.cpp
    sparse_image_test.cpp
)
target_link_libraries(
    run_unit_test
    codec
    imgproc
    GTest::gmock
    GTest::gtest_main
)

include(GoogleTest)
include(CTest)

gtest_discover_tests(run_unit_test)
