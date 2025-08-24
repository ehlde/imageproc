find_package(OpenCV REQUIRED)


set(SRC
  src/types.cpp
)

add_library(base STATIC
  ${SRC}
)

target_include_directories(base
  PUBLIC
    include
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(base PRIVATE
  ${OpenCV_LIBS}
)