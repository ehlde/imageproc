cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_GENERATOR "Ninja-Multi-Config")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Link filesystem for Clang.
    link_libraries(c++fs)
endif()

project(imageproc)

add_subdirectory(runner)
add_subdirectory(imageprocessing)
add_subdirectory(base)