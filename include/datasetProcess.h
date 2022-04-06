/*
    File name: dataProcess.h
    Author: cuiDarchan
    Reference: https://github.com/leo-stan/particles_detection
    Date created: 2021/07/18
*/

#pragma once

#include <pcl/common/centroid.h> //svd
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>

#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

// using eigen lib
#include <Eigen/Dense>

#include "utility.h"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::cerr;
using namespace cv;
using namespace cv::ml;
using Eigen::JacobiSVD;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class DatasetProcess {
 public:
  DatasetProcess();
  bool featureExtract(const vector<float>& cloud_data, Mat& frame_feature,
                      Mat& cloud_feature);
  bool featureExtract(const PointCloudPtr& point_cloud_ptr, Mat& frame_feature,
                      Mat& cloud_feature);
  bool labelExtract(int label, Mat& cloud_feature, vector<int>& frame_label); 
  bool computeFeature(const pcl::PointCloud<pcl::PointXYZI>& point_cloud,
                      Mat& frame_feature);

  inline void SetPointCloud(const PointCloudPtr& point_cloud) {
    cloud_ = point_cloud;
  }
  inline PointCloudPtr GetPointCloud() { return cloud_; }
  inline vector<int>& GetOriTrainingLabel() { return ori_training_label_; }
  inline vector<vector<float>>& GetOriTrainingData() {
    return ori_training_data_;
  }
  inline vector<int>& GetOriTestingLabel() { return ori_testing_label_; }
  inline vector<vector<float>>& GetOriTestingData() {
    return ori_testing_data_;
  }
  inline Mat& GetTrainingFeature() { return training_feature_; }
  inline Mat& GetTrainingLabel() { return training_label_; }
  inline Mat& GetTestingFeature() { return testing_feature_; }
  inline Mat& GetTestingLabel() { return testing_label_; }
  inline vector<int>& GetVoxelFlag() {return voxel_flag_;}
  inline void ClearVoxelFlag() {
    voxel_flag_.clear();
    voxel_flag_.resize(delt_voxel_[0] * delt_voxel_[1] * delt_voxel_[2], 0);
  }
  inline vector<pcl::PointCloud<pcl::PointXYZI>>& GetVoxelCloud() {
    return voxel_cloud_;
  }
  inline void ClearVoxelCloud() {
    int max_size = delt_voxel_[0] * delt_voxel_[1] * delt_voxel_[2];
    for (int i = 0; i < max_size; i++) {
      voxel_cloud_[i].points.clear();
    }
  }
  void PrintfFeatureAndLabel(const string& debug_file_path);

  bool ProcessFramesData(const vector<vector<float>>& cloud_frames,
                         const vector<int>& label, int mode);
  bool RFtreesClassifier();

 private:
  // 待处理数据
  vector<int> ori_training_label_;
  // vector<int> frame_training_label_; // 每一帧体素标签 
  vector<vector<float>> ori_training_data_;  // n个文件 × 1frame
  Mat training_feature_;
  Mat training_label_;
  vector<int> training_label_vec_;
  
  vector<int> ori_testing_label_;
  vector<int> frame_testing_label_; // 每一帧体素标签
  vector<vector<float>> ori_testing_data_;
  Mat testing_feature_;
  Mat testing_label_;
  vector<int> testing_label_vec_;

  PointCloudPtr cloud_;

  // 边界
  float voxel_size_ = 0.35;
  float bound_x_[2] = {-10.0, 10.0};
  float bound_y_[2] = {-10.0, 10.0};
  float bound_z_[2] = {-3.0, 2.0};
  int voxel_min_[3] = {0, 0, 0}; // 体素边界索引
  int voxel_max_[3];
  int delt_voxel_[3];
  vector<pcl::PointCloud<pcl::PointXYZI>> voxel_cloud_;
  vector<int> voxel_flag_;

  // 调试打印目录
  string debug_feature_label_data_ = "../data/";
};