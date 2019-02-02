cmake_minimum_required(VERSION 2.8)

add_compile_options(-std=c++11)

project(Test_zone)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package( Tesseract REQUIRED )
include_directories(${Tesseract_INCLUDE_DIRS})

find_package( Leptonica REQUIRED )
include_directories(${Leptonica_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(train train.cpp)
target_link_libraries(train
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(test test.cpp)
target_link_libraries(test
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
  ${Tesseract_LIBRARIES}
  ${Leptonica_LIBRARIES}
)

