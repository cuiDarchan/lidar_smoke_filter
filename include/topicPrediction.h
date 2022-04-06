/*
    File name: dataProcess.h
    Author: cuiDarchan
    Reference: https://github.com/leo-stan/particles_detection
    Date created: 2021/07/18
*/

#pragma once

#include <ros/ros.h> 
#include <sensor_msgs/PointCloud2.h>

#include "datasetProcess.h"
#include "trainModel.h"

class TopicPrediction {
 public:
  TopicPrediction(const ros::NodeHandle& nh, const Ptr<RTrees>& model);
  void MsgCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg_ptr);

 private:
  ros::NodeHandle tp_nh_;
  ros::Subscriber tp_sub_;
  ros::Publisher tp_pub_;
  std::shared_ptr<DatasetProcess> data_process_;
  // std::shared_ptr<TrainModel> train_model_;
  Ptr<RTrees> tp_model_;
};