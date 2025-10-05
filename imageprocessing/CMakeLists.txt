find_package(OpenCV REQUIRED)
find_package(IPP REQUIRED)
find_package(GTest REQUIRED)
include_directories(${IPP_ROOT}/linux/include)
link_directories(${IPP_ROOT}/lib)


set(SRC
  src/gemm_cv.cpp
  src/contrasting.cpp
  src/thresholding.cpp
  src/highlights.cpp
  src/reflection_removal.cpp
  src/integral_image.cpp
  src/connected_components.cpp
)

add_library(imageprocessing STATIC
  ${SRC}
)

target_include_directories(imageprocessing 
  PUBLIC
    include
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${IPP_INCLUDE_DIRS}
    base
)

target_link_libraries(imageprocessing PRIVATE
  ${OpenCV_LIBS}
  base
)

# Tests.
enable_testing()

add_executable(
  thresholding_tests
  src/tests/thresholding_tests.cpp
)

target_include_directories(
  thresholding_tests 
  PRIVATE 
  include
  ${GTEST_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  thresholding_tests
  GTest::gtest_main
  ${OpenCV_LIBS}
  imageprocessing
)

include(GoogleTest)
gtest_discover_tests(thresholding_tests)

# Add test executable for niblackIntegral
add_executable(niblack_integral_tests
    src/tests/niblack_integral_tests.cpp
)

target_link_libraries(niblack_integral_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        imageprocessing
        ${OpenCV_LIBS}
)

add_test(NAME NiblackIntegralTests COMMAND niblack_integral_tests)
