cmake_minimum_required(VERSION 2.8)

project(Test_zone)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
)

