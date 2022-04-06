
#include "datasetProcess.h"

DatasetProcess::DatasetProcess() {
  cloud_.reset(new PointCloud);
  voxel_max_[0] = std::floor((bound_x_[1] - bound_x_[0]) / voxel_size_);
  voxel_max_[1] = std::floor((bound_y_[1] - bound_y_[0]) / voxel_size_);
  voxel_max_[2] = std::floor((bound_z_[1] - bound_z_[0])/ voxel_size_);
  delt_voxel_[0] = voxel_max_[0] - voxel_min_[0];
  delt_voxel_[1] = voxel_max_[1] - voxel_min_[1];
  delt_voxel_[2] = voxel_max_[2] - voxel_min_[2];

  voxel_cloud_.resize(delt_voxel_[0] * delt_voxel_[1] * delt_voxel_[2]);
  ClearVoxelFlag();
}


bool DatasetProcess::featureExtract(const PointCloudPtr& point_cloud_ptr,
                                    Mat& frame_feature, Mat& cloud_feature) {
  ClearVoxelCloud();
  ClearVoxelFlag();
  Mat tmp;
  tmp.copyTo(frame_feature);

  bool pc_first_flag = false;
  size_t point_num = point_cloud_ptr->points.size();
  
  if (!point_num) return false;

  for (size_t i = 0; i < point_num; i++) {
    float x = point_cloud_ptr->points[i].x;
    float y = point_cloud_ptr->points[i].y;
    float z = point_cloud_ptr->points[i].z;
    float intensity = static_cast<float>(point_cloud_ptr->points[i].intensity);
    if (x < bound_x_[0] || x > bound_x_[1] || y < bound_y_[0] ||
        y > bound_y_[1] || z < bound_z_[0] || z > bound_z_[1])
      continue;
    int voxel_x = std::floor((x - bound_x_[0]) / voxel_size_);
    int voxel_y = std::floor((y - bound_y_[0]) / voxel_size_);
    int voxel_z = std::floor((z - bound_z_[0]) / voxel_size_);

    int flag_pos = voxel_z * delt_voxel_[0] * delt_voxel_[1] +
                   voxel_y * delt_voxel_[0] + voxel_x;
    pcl::PointXYZI p;
    p.x = x;
    p.y = y;
    p.z = z;
    p.intensity = intensity;
    // cout << "x:" << x << ", y: " << y << " ,z: " << z << endl;
    // voxel_flag 为0，新建点云
    if (!voxel_flag_[flag_pos]) {
      // pcl::PointCloud<pcl::PointXYZI> point_cloud;
      // point_cloud.points.emplace_back(p);
      voxel_cloud_[flag_pos].points.emplace_back(p);
      voxel_flag_[flag_pos] = 1;
    } else {
      voxel_cloud_[flag_pos].points.emplace_back(p);
    }
  }
  
  // 点云头文件长、宽赋值，计算特征向量
  int cnt = 0;
  for (uint i = 0; i < voxel_flag_.size(); i++) {
    if (voxel_flag_[i]) {
      if (voxel_cloud_[i].points.size() < 30) {
        voxel_flag_[i] = 0;
        continue;
      }
      cnt++;
      voxel_cloud_[i].height = 1;
      voxel_cloud_[i].width = voxel_cloud_[i].points.size();
      if (!computeFeature(voxel_cloud_[i], frame_feature)) {
        cerr << "Failed to computeFeature: " << endl;
        return false;
      }
    }
  }

  // 按列合并 vconcat（B,C，A）; // 等同于A=[B ;C]
  if (!pc_first_flag) {
    cloud_feature = frame_feature;
    pc_first_flag = true;
  } else {
    // frame_feature : 每一帧点云特征
    vconcat(cloud_feature, frame_feature, cloud_feature);
  }
  return true;
}

bool DatasetProcess::featureExtract(const vector<float>& cloud_data,
                                    Mat& frame_feature, Mat& cloud_feature) {
  ClearVoxelCloud();
  ClearVoxelFlag();
  Mat tmp;
  tmp.copyTo(frame_feature);

  static bool first_flag = false;
  // 第n个点 [4×point_num,4×point_num+1,4×point_num+2,4×point_num+3]
  size_t point_num = cloud_data.size() / 4;
  if (!point_num) return false;

  for (size_t i = 0; i < point_num; i++) {
    float x = cloud_data[4 * i];
    float y = cloud_data[4 * i + 1];
    float z = cloud_data[4 * i + 2];
    float intensity = cloud_data[4 * i + 3];
    if (x < bound_x_[0] || x > bound_x_[1] || y < bound_y_[0] ||
        y > bound_y_[1] || z < bound_z_[0] || z > bound_z_[1])
      continue;
    int voxel_x = std::floor((x - bound_x_[0]) / voxel_size_);
    int voxel_y = std::floor((y - bound_y_[0]) / voxel_size_);
    int voxel_z = std::floor((z - bound_z_[0]) / voxel_size_);

    int flag_pos = voxel_z * delt_voxel_[0] * delt_voxel_[1] +
                   voxel_y * delt_voxel_[0] + voxel_x;
    pcl::PointXYZI p;
    p.x = x;
    p.y = y;
    p.z = z;
    p.intensity = intensity;
    // voxel_flag 为0，新建点云
    if (!voxel_flag_[flag_pos]) {
      // pcl::PointCloud<pcl::PointXYZI> point_cloud;
      // point_cloud.points.emplace_back(p);
      voxel_cloud_[flag_pos].points.emplace_back(p);
      voxel_flag_[flag_pos] = 1;
    } else {
      voxel_cloud_[flag_pos].points.emplace_back(p);
    }
  }

  // 点云头文件长、宽赋值，计算特征向量
  for (uint i = 0; i < voxel_flag_.size(); i++) {
    if (voxel_flag_[i]) {
      if (voxel_cloud_[i].points.size() < 4) {
        voxel_flag_[i] = 0;
        continue;
      }
      voxel_cloud_[i].height = 1;
      voxel_cloud_[i].width = voxel_cloud_[i].points.size();
      
      if (!computeFeature(voxel_cloud_[i], frame_feature)) {
        cerr << "Failed to computeFeature: " << endl;
        return false;
      }
    }
  }

  // 按列合并 vconcat（B,C，A）; // 等同于A=[B ;C]
  if (!first_flag) {
    cloud_feature = frame_feature;
    first_flag = true;
  } else {
    vconcat(cloud_feature, frame_feature, cloud_feature);
  }
  return true;
}

bool DatasetProcess::labelExtract(int label, Mat& cloud_feature,
                                  vector<int>& frame_label) {
  int rows = cloud_feature.rows;
  if (rows) {
    frame_label.insert(frame_label.end(), rows, label);
  }
  return true;
}

bool DatasetProcess::computeFeature(
    const pcl::PointCloud<pcl::PointXYZI>& point_cloud, Mat& frame_feature) {
  Mat feature(1, 4, CV_32F);  // 四维特征
  vector<float> intensity_vec;
  
  if (point_cloud.points.size() >= 3) {
    for (size_t i = 0; i < point_cloud.points.size(); i++) {
      intensity_vec.emplace_back(point_cloud.points[i].intensity);
    }

    // 1. intensity_mean
    auto sum = std::accumulate(intensity_vec.begin(), intensity_vec.end(), 0.0);
    auto mean = sum / intensity_vec.size();
    feature.at<float>(0) = mean;
    // 2. intensity_std
    auto accum = 0.0;
    std::for_each(intensity_vec.begin(), intensity_vec.end(),
                  [&](const float d) { accum += (d - mean) * (d - mean); });
    auto std = sqrtf(accum / (intensity_vec.size() - 1));
    feature.at<float>(1) = std;
    // 3. roughness
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;
    pcl::computeMeanAndCovarianceMatrix(point_cloud, cov, pc_mean);
    // singular Value Decomposition: SVD
    JacobiSVD<MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal (3,1)
    MatrixXf single_value = (svd.matrixU().col(2));
    auto roughness = svd.singularValues()[2];//特征值
    feature.at<float>(2) = roughness;
    // 4. slope (# 0 xy 平面, pi/2 -- 1.71 沿z轴)
    auto slope_paramA =
        sqrtf(pow(single_value(0, 0), 2) + pow(single_value(1, 0), 2));
    auto slope_paramB =
        sqrtf(pow(single_value(0, 0), 2) + pow(single_value(1, 0), 2) +
              pow(single_value(2, 0), 2));
    auto slope = std::abs(std::asin(slope_paramA / slope_paramB));
    feature.at<float>(3) = slope;
    frame_feature.push_back(feature);
  }

  return true;
}

//随机森林分类
bool DatasetProcess::RFtreesClassifier() {
  // 输入数据：training_feature_ 和training_label_
  Ptr<TrainData> tdata =
      TrainData::create(training_feature_, ROW_SAMPLE, training_label_);
  // 随机森林分类器
  Ptr<RTrees> model = RTrees::create();
  model->setMaxDepth(10);
  model->setMinSampleCount(10);
  model->setRegressionAccuracy(0);
  model->setUseSurrogates(false);
  model->setMaxCategories(15);
  model->setPriors(Mat());
  model->setCalculateVarImportance(true);
  model->setActiveVarCount(4);
  // model->setTermCriteria(TC(100, 0.01f));
  model->train(tdata);
  model->save("../model/new_model.xml");
  cout << "Number of trees: " << model->getRoots().size() << endl;

  // Print variable importance
  // Mat var_importance = model->getVarImportance();
  // if (!var_importance.empty()) {
  //   double rt_imp_sum = sum(var_importance)[0];
  //   printf("var#\timportance (in %%):\n");
  //   int i, n = (int)var_importance.total();
  //   for (i = 0; i < n; i++)
  //     printf("%-2d\t%-4.1f\n", i,
  //            100.f * var_importance.at<float>(i) / rt_imp_sum);
  // }
  return true;
}

bool DatasetProcess::ProcessFramesData(
    const vector<vector<float>>& cloud_frames, const vector<int>& label,
    int mode) {
  int frame_size = cloud_frames.size();
  int label_size = label.size();
  if (!frame_size || !label_size || frame_size != label_size) {
    cerr << "Cloud frames data size is: " << frame_size
         << ", label_size:" << label_size << endl;
    return false;
  }

  if (mode == 0) {
    int training_cnt = 0;
    for (auto& cloud_frame : cloud_frames) {
      Mat frame_feature;
      if (!featureExtract(cloud_frame, frame_feature, training_feature_)) {
        cerr << "Extracting training_feature failed! " << endl;
        return false;
      }
      if (!labelExtract(label[training_cnt], frame_feature,
                        training_label_vec_)) {
        cerr << "Extracting training_label failed! " << endl;
        return false;
      }
      training_cnt++;
    }
    Mat(training_label_vec_).copyTo(training_label_);
  } else if (mode == 1) {
    int testing_cnt = 0;
    for (auto& cloud_frame : cloud_frames) {
      Mat frame_feature;
      if (!featureExtract(cloud_frame, frame_feature,testing_feature_)) {
        cerr << "Extracting training_feature failed! " << endl;
        return false;
      }
      if (!labelExtract(label[testing_cnt], testing_feature_,
                        testing_label_vec_)) {
        cerr << "Extracting training_label failed! " << endl;
        return false;
      }
      testing_cnt++;
    }
    Mat(testing_label_vec_).copyTo(testing_label_);
  } else {
    cerr << " ProcessFramesData : There is no mode: " << mode << endl;
    return false;
  }
  // 打印label feature 特征文件
  PrintfFeatureAndLabel(debug_feature_label_data_);
  return true;
}

void DatasetProcess::PrintfFeatureAndLabel(const string& debug_file_path) {
  // 1. 打印训练集  -- 标签，4个特征
  std::ofstream ofs;
  string file_name = debug_file_path + "debug_data.txt";
  if (access(file_name.c_str(), 0) == 0) {
    remove(file_name.c_str());
  }
  ofs.open(file_name, std::ios::app);
  for (int i = 0; i < training_feature_.rows; i++) {
    ofs << training_label_.at<int>(i) << " "
        << training_feature_.at<float>(i, 0) << " "
        << training_feature_.at<float>(i, 1) << " "
        << " " << training_feature_.at<float>(i, 2) << " "
        << " " << training_feature_.at<float>(i, 3) << endl;
  }
  ofs.close();
}
