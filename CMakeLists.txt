cmake_minimum_required(VERSION 3.5)
project(fr)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(include/alize)
include_directories(include/seeta)
link_directories(libs)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIR})

set(SOURCE_FILES main.cpp)
add_executable(fr ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} -lseeta_facedet_lib)
target_link_libraries(${PROJECT_NAME} -lseeta_fa_lib)
target_link_libraries(${PROJECT_NAME} -lviplnet)
target_link_libraries(${PROJECT_NAME} -lalize_Linux_x86_64)