#include <ros/ros.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// pcl
#include <pcl/conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

// eigen库
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <yaml-cpp/yaml.h>

#include "topicPrediction.h"
#include "trainModel.h"

using std::string;

int main(int argc, char **argv) {
  ros::init(argc, argv, "lidar_smoke_filter");
  ros::NodeHandle nh;

  // 0. 加载参数
  YAML::Node node = YAML::LoadFile("../cfg/cpp_config.yaml");
  bool is_trained = node["is_needed_trained"].as<bool>();
  bool is_predicted = node["is_needed_prediction"].as<bool>();
  string training_lidar_path = node["training_lidar_path"].as<string>();
  string training_label_path = node["training_label_path"].as<string>();
  string testing_lidar_path = node["testing_lidar_path"].as<string>();
  string testing_label_path = node["testing_lidar_path"].as<string>();
  string model_path = node["model_path"].as<string>();

  // 2. 加载模型
  TrainModel model(nh);
  if (is_trained) {
    model.train();
  } else {
    model.loadModel(model_path);
  }

  // 3. 预测
  if (is_predicted) {
    TopicPrediction tp(nh, model.getRTreesModel());
    ros::Rate loop_rate(10);
    while (ros::ok()) {
      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  return 0;
}