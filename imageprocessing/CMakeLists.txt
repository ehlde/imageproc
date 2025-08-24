find_package(OpenCV REQUIRED)
find_package(IPP REQUIRED)
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