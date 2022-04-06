/*
    File name: trainModel.h
    Author: cuiDarchan
    Reference: https://github.com/leo-stan/particles_detection
    Date created: 2021/07/18
*/

#pragma once

#include <dirent.h>
#include <stdio.h>
#include <fstream>  // std::ifstream
#include <iostream>
#include <string>
#include <vector>

#include <ros/ros.h>

#include "datasetProcess.h"

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;

class TrainModel {
 public:
  TrainModel(const ros::NodeHandle &nh);

  bool loadLidarData(const string &lidar_path,
                     int mode);  // mode代表训练集还是测试集
  bool loadLabelData(const string &label_path, int mode);
  vector<string> getFileName(const string &file_path);
  bool train();
  bool loadModel(const string &model_path);
  Ptr<RTrees> getRTreesModel() { return RT_model_; }

 private:
  std::shared_ptr<DatasetProcess> data_process_;

  string training_label_path_;
  string training_lidar_path_;
  string testing_label_path_;
  string testing_lidar_path_;

  Ptr<RTrees> RT_model_;
  bool is_train_ = false;
};