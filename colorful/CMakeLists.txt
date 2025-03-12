cmake_minimum_required(VERSION 3.10)
project(colorful LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)

# 查找依赖包
list(APPEND CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu/cmake/pcl)
set(CMAKE_CUDA_ARCHITECTURES "native")
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# 包含头文件路径
include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${nlohmann_json_INCLUDE_DIRS}
    # third_party/cnpy
)

# 设置源文件
set(SOURCE_FILES
    src/main.cc
    src/pose.cc
    src/projector.cc
    src/distribute_project.cu
    src/image_process.cu
    # third_party/cnpy/cnpy.cpp
    # 其他源文件
)

# 添加可执行文件
add_executable(projector ${SOURCE_FILES})
set_target_properties(projector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接库
target_link_libraries(projector
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${nlohmann_json_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
)

