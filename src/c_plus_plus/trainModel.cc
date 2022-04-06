
#include "trainModel.h"

TrainModel::TrainModel(const ros::NodeHandle &nh) {
  nh.param<std::string>("training_lidar_path", training_lidar_path_,
                        "../data/training/lidar/");
  nh.param<std::string>("training_label_path", training_label_path_,
                        "../data/training/label/");
  nh.param<std::string>("testing_label_path", testing_label_path_,
                        "../data/testing/label/");
  nh.param<std::string>("testing_lidar_path", testing_lidar_path_,
                        "../data/testing/lidar/");
  // nh.param<std::string>("is_train", is_train_,"false");
  data_process_ = std::make_shared<DatasetProcess>();

}

vector<string> TrainModel::getFileName(const string &file_path) {
  vector<string> names = {};

  // 1. opendir无序读取
  // struct dirent *ptr;
  // DIR *dir;
  // dir = opendir(file_path.c_str());
  // while (ptr = readdir(dir)) {
  //   if (ptr->d_name[0] == '.')  // 去掉. 和..
  //     continue;
  //   names.push_back(file_path + string(ptr->d_name));
  //   cout << "In file_path , file_name is " << file_path + string(ptr->d_name) << endl;
  // }
  // closedir(dir);
  
  // 2. scandir有序读取
  struct dirent **namelist;  // struct dirent * namelist[];
  int n = scandir(file_path.c_str(), &namelist, NULL, alphasort);
  if (n < 0) {
    return names;
  } else {
    while (n--) {
      string file_name = namelist[n]->d_name;
      if (file_name == "." || file_name == "..") continue;
      names.push_back(file_path + file_name);
      cout << "In file_path , file_name is " << file_path + file_name << endl;
      free(namelist[n]);
    }
    free(namelist);
  }

  return names;
}

bool TrainModel::loadLidarData(const string &lidar_path, int mode) {
  vector<string> lidar_names = getFileName(lidar_path);
  for (auto &lidar_name : lidar_names) {
    ifstream ifs;
    ifs.open(lidar_name,
             std::ifstream::in | std::ifstream::binary);  // 只读、二进制
    if (!ifs.is_open()) {
      cerr << lidar_name << " open error." << endl;
      return false;
    }

    ifs.seekg(0, std::ios::end);  // 参数1：偏移；参数2：位置
    size_t num_elements = 0;
    // 注意区分double还是float读取
    if (mode == 0){ // ******0
      num_elements = ifs.tellg() / sizeof(double);  // 元素个数
    } else {
      num_elements = ifs.tellg() / sizeof(float);  // 元素个数
    }
    ifs.seekg(0, std::ios::beg);

    if (mode == 0) {  // train
      vector<double> tmp_data_buffer(num_elements);    
      ifs.read(reinterpret_cast<char *>(&tmp_data_buffer[0]),
             num_elements * sizeof(double));
      vector<float> lidar_data_buffer(tmp_data_buffer.begin(),tmp_data_buffer.end());
      data_process_->GetOriTrainingData().emplace_back(lidar_data_buffer);
    } else if (mode == 1) {  // test
      vector<float> lidar_data_buffer(num_elements);
      ifs.read(reinterpret_cast<char *>(&lidar_data_buffer[0]),
             num_elements * sizeof(float));
      data_process_->GetOriTestingData().emplace_back(lidar_data_buffer);
    } else {
      cout << "The mode :" << mode << " is not supported!" << endl;
    }
    ifs.close();
  }
  return true;
}

bool TrainModel::loadLabelData(const string &label_path, int mode) {
  vector<string> label_names = getFileName(label_path);

  for (auto &label_name : label_names) {
    FILE *file = fopen(label_name.c_str(), "r");  // 只读方式
    if (file) {
      int label;
      constexpr int ksize = 1;
      while (fscanf(file, "%d", &label) == ksize) {
        if (mode == 0) {  // train
          data_process_->GetOriTrainingLabel().push_back(label);
        } else if (mode == 1) {  // test
          data_process_->GetOriTestingLabel().push_back(label);
        } else {
          cerr << "The mode :" << mode << " is not supported!" << endl;
          return false;
        }
      }
    } else {
      cerr << "Label_name :" << label_name << " can't open!" << endl;
      return false;
    }
    fclose(file);
  }
  return true;
}

bool TrainModel::train() {
  // 1.加载原始训练集数据
  if (!loadLidarData(training_lidar_path_, 0)) { 
    cerr << "Can't load training lidar data!" << endl;
  }
  if (!loadLabelData(training_label_path_, 0)) { 
    cerr << "Can't load training label data!" << endl;
  }
  // 2.加载原始测试集数据
  if (!loadLidarData(testing_lidar_path_, 1)) {
    cerr << "Can't load testing lidar data!" << endl;
  }
  if (!loadLabelData(testing_label_path_, 1)) {
    cerr << "Can't load testing label data!" << endl;
  }

  // 3.转化为特征数据集合，用于训练
  data_process_->ProcessFramesData(data_process_->GetOriTrainingData(),
                                   data_process_->GetOriTrainingLabel(), 0);

  // 4.训练RF模型
  data_process_->RFtreesClassifier();
  return true;
}

bool TrainModel::loadModel(const string &model_path) {
  RT_model_ = RTrees::load(model_path);
  return true;
}