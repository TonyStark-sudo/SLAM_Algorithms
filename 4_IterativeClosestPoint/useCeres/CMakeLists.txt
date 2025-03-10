cmake_minimum_required(VERSION 3.10)
project(ICP)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 指定头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 查找Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 设置源文件
set(SOURCES
    src/iterativeClosestPoint.cc
    src/main.cc
)

# 生成可执行文件
add_executable(ICP ${SOURCES})

# 链接 OpenCV 和 Eigen
target_link_libraries(ICP ${OpenCV_LIBS} ${CERES_LIBRARIES} Eigen3::Eigen)
