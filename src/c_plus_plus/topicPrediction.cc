
#include "topicPrediction.h"

TopicPrediction::TopicPrediction(const ros::NodeHandle& nh,
                                 const Ptr<RTrees>& model)
    : tp_nh_(nh) {
  data_process_ = std::make_shared<DatasetProcess>();
  // train_model_ = std::make_shared<TrainModel>(tp_nh_);
  tp_model_ = model;
  tp_sub_ = tp_nh_.subscribe<sensor_msgs::PointCloud2>(
      "/livox_lidar_front/compensator/PointCloud2", 10,
      &TopicPrediction::MsgCallback, this);
  tp_pub_ = tp_nh_.advertise<sensor_msgs::PointCloud2>("/lidar_smoke", 10);
}

void TopicPrediction::MsgCallback(
    const sensor_msgs::PointCloud2::ConstPtr& cloud_msg_ptr) {
  Tictoc tp_time;
  pcl::PointCloud<pcl::PointXYZI> smoke_cloud;
  smoke_cloud.points.clear();
  smoke_cloud.header.frame_id = "novatel";

  // 1. 获取当前帧点云
  PointCloudPtr point_cloud_ptr(new PointCloud);
  pcl::fromROSMsg(*cloud_msg_ptr, *point_cloud_ptr);
  if (!point_cloud_ptr->points.size()) return;
  
  // 2. 特征提取
  Mat frame_feature;
  Mat cloud_feature;  // 所有帧之和
  data_process_->featureExtract(point_cloud_ptr, frame_feature, cloud_feature);
  // cout << "cloud_feature.rows: " << cloud_feature.rows << endl;
  
  // 3. 利用模型进行预测
  vector<int> flag = data_process_->GetVoxelFlag();
  int j = 0;
  for (int i = 0; i < frame_feature.rows && j < flag.size();) {
    if (!flag[j]) {
      j++;
    } else {
      Mat row = frame_feature.rowRange(i, i + 1).clone();
      int res = tp_model_->predict(row);
      if (res) {
        for (auto& p : data_process_->GetVoxelCloud()[j].points) {
          smoke_cloud.points.emplace_back(p);
        }
      }
      j++;
      i++;
    }
  }
  smoke_cloud.width = smoke_cloud.points.size();
  smoke_cloud.height = 1;

  // 4. 发出smoke_cloud数据
  sensor_msgs::PointCloud2 cloud_msg;
  cloud_msg.header.frame_id = "novatel";
  pcl::toROSMsg(smoke_cloud, cloud_msg);
  tp_pub_.publish(cloud_msg);

  cout << "Topic prediction time :" << tp_time.toc() << " ms" << endl;
}