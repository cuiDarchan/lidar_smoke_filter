cmake_minimum_required(VERSION 2.8.3)
project(lidar_smoke_filter)

add_definitions(-std=c++11)
set(CMAKE_CXX_STANDARD 11) #采用C11版本
#SET(CMAKE_BUILD_TYPE "DEBUG") #DUBUG 模式
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_BUILD_TYPE "RELEASE")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_conversions
  pcl_ros
  sensor_msgs
  visualization_msgs
  geometry_msgs
  sensor_msgs
)

include_directories(
  include 
  ${catkin_INCLUDE_DIRS}
)

find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
include_directories(
  ${EIGEN3_INCLUDE_DIR} 
  ${YAML_CPP_INCLUDE_DIR}
  )
add_definitions(${EIGEN_DEFINITIONS})

add_executable(main src/c_plus_plus/main.cpp 
  src/c_plus_plus/datasetProcess.cc
  src/c_plus_plus/topicPrediction.cc
  src/c_plus_plus/trainModel.cc
)

target_link_libraries(main
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  yaml-cpp
)

