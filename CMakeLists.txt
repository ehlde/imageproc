cmake_minimum_required(VERSION 3.25)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -g")


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Link filesystem for Clang.
    link_libraries(c++fs)
endif()

project(imageproc)

add_subdirectory(runner)
add_subdirectory(imageprocessing)