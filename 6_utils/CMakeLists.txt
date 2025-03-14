cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

# 查找 Eigen3
find_package(Eigen3 REQUIRED)

# 生成静态库
add_library(utils src/pose.cpp src/utility.cpp)
target_include_directories(utils PUBLIC include ${EIGEN3_INCLUDE_DIR})

# 生成可执行文件并链接静态库
add_executable(test test.cpp)
target_link_libraries(test utils)
