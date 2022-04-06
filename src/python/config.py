#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from easydict import EasyDict as edict #作用：可以使得以属性的方式去访问字典的值！

__C = edict()
# Consumers can get config by:
#    import config as cfg
cfg = __C  #相当于引用

# for dataset dir
#__C.DATA_DIR = '/media/storage/leo-ws/voxelnet/sorted_data'
#__C.CALIB_DIR = '/media/storage/leo-ws/voxelnet/data/training/calib'
__C.ROOT_DIR = '/home/cui-dell/catkin_ws/src/lidar_smoke_filter'
__C.INPUT_MODEL = None             # 输入模型名称，None 或者 'official_model'
__C.OUTPUT_MODEL = 'official_model'             # 输出模型名称
__C.TRAINING_LIDAR = 'training/lidar/'                         # 训练集地址
__C.TRAINING_LABEL = 'training/label/'                         # 训练集标签                 
__C.TESTING_LIDAR = 'testing/lidar/'                          # 测试集地址
__C.TESTING_LABEL = 'testing/label/'                          # 测试集标签

__C.Z_MIN = -3
__C.Z_MAX = 2
__C.Y_MIN = -10
__C.Y_MAX = 10
__C.X_MIN = -10
__C.X_MAX = 10
__C.VOXEL_X_SIZE = 0.2
__C.VOXEL_Y_SIZE = 0.2
__C.VOXEL_Z_SIZE = 0.2
__C.VOXEL_POINT_COUNT = 50
__C.MAP_TO_FOOTPRINT_X = 0
__C.MAP_TO_FOOTPRINT_Y = 0
__C.MAP_TO_FOOTPRINT_Z = 0
__C.VFE1_OUT = 32
__C.VFE2_OUT = 128
__C.VFE3_OUT = 256
__C.TIMING_COUNT_MAX = 50
