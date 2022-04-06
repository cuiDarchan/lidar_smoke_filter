/*
    File name: utility.h
    Author: cuiDarchan
    Reference: https://github.com/leo-stan/particles_detection
    Date created: 2021/07/18
*/

#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <chrono> // 时间库
#include <cstdlib>
#include <ctime>

struct PointXYZIR {
  PCL_ADD_POINT4D
  uint8_t intensity;
  uint8_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIR,
    (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity,
                                            intensity)(uint8_t, ring, ring))

typedef PointXYZIR Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointCloud<Point>::Ptr PointCloudPtr;

class Tictoc {
 public:
  Tictoc() { tic(); }
  void tic() { start = std::chrono::system_clock::now(); }  // s
  double toc() {
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count() * 1000; //ms
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start, end;
};