cmake_minimum_required(VERSION 3.10)

project(useCeres)

set(CMAKE_CXX_STANDARD 11)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(helloWorld helloWorld.cpp)
add_executable(overloadOperator overloadOperator.cpp)
add_executable(curveFitting curveFitting.cpp)
add_executable(robustCurveFitting robustCurveFitting.cpp)
add_executable(official official.cpp)
add_executable(simpleBundleAdjustment simple_bundle_adjuster.cc)

target_link_libraries(helloWorld ${CERES_LIBRARIES})
target_link_libraries(curveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(robustCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(official ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(simpleBundleAdjustment ${CERES_LIBRARIES})

set(CMAKE_BUILD_TYPE "Release")