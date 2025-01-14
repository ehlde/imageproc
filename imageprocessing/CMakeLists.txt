find_package(OpenCV REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${IPP_LIB_DIR})
find_package(IPP REQUIRED)

set(SRC
  src/gemm_cv.cpp
  src/contrasting.cpp
  src/thresholding.cpp
  src/highlights.cpp
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
)

target_link_libraries(imageprocessing PRIVATE
  ${OpenCV_LIBS}
  IPP::ippcore
)