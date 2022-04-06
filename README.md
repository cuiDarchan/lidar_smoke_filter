# 矿区沙尘过滤
## 依赖
ROS、OpenCV、Eigen、Yaml  

## 编译
```
mkdir build && cd build
cmake.. && make -j4
```
## 运行
### C++版本
```
cd build 
./main
rosbag play xx.bag
```
### python版本
```
cd src/python
python topic_prediction.py
```

# 矿区扬尘数据集
## 文件名
1 -- person
2 -- car
3 -- ground
4 -- smoke 
## 类别
0 -- no smoke
1 -- smoke

## 参考
见paper中论文