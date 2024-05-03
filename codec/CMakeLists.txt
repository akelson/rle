add_library(
    codec
    rle_v1.cpp
    rle_v2.cpp
)
target_include_directories(
    codec PUBLIC
    ${CMAKE_SOURCE_DIR}
)