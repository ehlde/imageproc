find_package(OpenCV REQUIRED)
find_package(benchmark REQUIRED)

add_executable(runner src/main.cpp)
target_include_directories(runner PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries( runner PRIVATE ${OpenCV_LIBS} benchmark::benchmark imageprocessing base)
