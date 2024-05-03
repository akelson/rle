cmake_minimum_required(VERSION 3.1)

project(rle)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(Eigen)

add_subdirectory(codec)
add_subdirectory(imgproc)
add_subdirectory(test)
