#!/usr/bin/env python
# coding=UTF-8<code>

"""
    File name: lidar_dataset_process.py
    Author: cuiDarchan
    Reference: https://github.com/leo-stan/particles_detection
    Date created: 2021/07/13
    Python Version: 2.7
"""

import numpy as np
import os, os.path
import glob # glob 文件管理相关
from config import cfg
import sys
from sklearn.decomposition import PCA
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from config import cfg

class LidarDatasetHC():
    """Lidar dataset for Hand Crafted features and classifier."""
    # 构造函数__init__
    def __init__(self, features, root_dir=None, train=True, test=False, val=False, smoke=True, dust=True,
                 shuffle=False, mu=None, sigma=None):
        """
        Args:
            root_dir (string): Directory with training data:
            |---root_dir
                |--- label
                |--- lidar
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.array([]) #初始化为空
        self.labels = np.array([])
        if len(features) == 0:
            sys.exit('LidarDatasetHC: nb_features = 0, need at least one feature')
        self.features = np.asarray(features)
        self.root_dir = root_dir
        self.smoke = smoke
        self.dust = dust
        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(self.shuffle)
        
        # Scaling parameters
        self.mu = mu
        self.sigma = sigma
        if (self.mu is not None or self.sigma is not None) and len(mu) != len(sigma):
            sys.exit('Mu and Sigma need to be of same dimension')

        # Prepare Dataset
        self.smoke_data = []
        self.non_smoke_data = []
        self.smoke_labels = []
        self.non_smoke_labels = []

        # If root directory provided, build dataset out of that
        if root_dir is not None:
            self.f_lidar = []
            self.f_label = []

            if train: # train 为bool类型
                self.f_lidar = self.f_lidar + glob.glob(os.path.join(self.root_dir, cfg.TRAINING_LIDAR + '*.bin'))
                self.f_label = self.f_label + glob.glob(os.path.join(self.root_dir, cfg.TRAINING_LABEL + '*.txt'))
            if test:
                self.f_lidar = self.f_lidar + glob.glob(os.path.join(self.root_dir, cfg.TESTING_LIDAR + '*.bin'))
                self.f_label = self.f_label + glob.glob(os.path.join(self.root_dir, cfg.TESTING_LABEL + '*.txt'))
            #if val:
            #    self.f_lidar = self.f_lidar + glob.glob(os.path.join(self.root_dir, 'validation/lidar/*.bin'))
            #    self.f_label = self.f_label + glob.glob(os.path.join(self.root_dir, 'validation/label/*.txt'))

            self.f_lidar.sort()
            self.f_label.sort()

            for i in range(len(self.f_lidar)):
                if train:
                    raw_lidar_pcl = np.fromfile(self.f_lidar[i], dtype=np.float64).reshape((-1, 4)).astype(np.float32) #reshape中-1 代表不知道多少行
                    #raw_lidar_pcl = np.fromfile(self.f_lidar[i], dtype=np.float32).reshape((-1, 4))  # reshape中-1 代表不知道多少行
                    sys.stdout.write("\r Training Loading %s " % str(self.f_lidar[i]))
                    sys.stdout.flush()
                if test:
                    raw_lidar_pcl = np.fromfile(self.f_lidar[i], dtype=np.float32).reshape((-1, 4)) #reshape中-1 代表不知道多少行
                    sys.stdout.write("\r Testing Loading %s " % str(self.f_lidar[i]))
                    sys.stdout.flush()

                if raw_lidar_pcl.size:
                    label = [int(line.strip('\n')) for line in open(self.f_label[i], 'r').readlines()][0]
                    #if (label == 1 and not smoke) or (label == 2 and not dust):
                    #    break  # If unrequested label, skip
                    feature_buffer, _, _, _ = self.extract_scan(raw_lidar_pcl)
                    if label == 0:
                        self.non_smoke_data.append(feature_buffer)
                        self.non_smoke_labels.append(label * np.ones(shape=(feature_buffer.shape[0], 1), dtype=np.int64))
                    else:
                        label = 1
                        self.smoke_data.append(feature_buffer)
                        self.smoke_labels.append(label * np.ones(shape=(feature_buffer.shape[0], 1), dtype=np.int64))
            self.non_smoke_data = np.concatenate(self.non_smoke_data) #concatenate衔接
            self.smoke_data = np.concatenate(self.smoke_data)
            self.non_smoke_labels = np.concatenate(self.non_smoke_labels)
            self.smoke_labels = np.concatenate(self.smoke_labels)

            self.non_smoke_data = self.non_smoke_data.reshape((-1, 1, self.features.size))
            self.smoke_data = self.smoke_data.reshape((-1, 1, self.features.size))

            self.post_process_scan()

            self.data = self.data.reshape((-1, self.features.size))


    # 切分写入体素格网
    def extract_scan(self, raw_lidar_pcl):
        if self.shuffle:
            np.random.shuffle(raw_lidar_pcl)

        # Lidar points in map coordinate
        shifted_coord = raw_lidar_pcl[:, :3]

        voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE], dtype=np.float32)
        grid_size = np.array([(cfg.X_MAX - cfg.X_MIN) / cfg.VOXEL_X_SIZE, (cfg.Y_MAX - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE,
                              (cfg.Z_MAX - cfg.Z_MIN) / cfg.VOXEL_Z_SIZE], dtype=np.int64)

        voxel_index = np.floor(shifted_coord / voxel_size).astype(np.int)

        # Bound border x_min,x_max according to center(0.0,0)
        bound_min = np.array([cfg.X_MIN / cfg.VOXEL_X_SIZE, cfg.Y_MIN / cfg.VOXEL_Y_SIZE,
                              cfg.Z_MIN / cfg.VOXEL_Z_SIZE], dtype=np.int64)
        bound_max = np.array([cfg.X_MAX / cfg.VOXEL_X_SIZE, cfg.Y_MAX / cfg.VOXEL_Y_SIZE,
                              cfg.Z_MAX / cfg.VOXEL_Z_SIZE], dtype=np.int64)

        bound_x = np.logical_and(
            voxel_index[:, 0] >= bound_min[0], voxel_index[:, 0] < bound_max[0]) #np.logical_and 逻辑与
        bound_y = np.logical_and(
            voxel_index[:, 1] >= bound_min[1], voxel_index[:, 1] < bound_max[1])
        bound_z = np.logical_and(
            voxel_index[:, 2] >= bound_min[2], voxel_index[:, 2] < bound_max[2])
        bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

        # Raw scan within bounds
        raw_lidar_pcl = raw_lidar_pcl[bound_box]
        voxel_index = voxel_index[bound_box]

        # [K, 3] coordinate buffer as described in the paper
        if voxel_index.shape[0] > 0:
            coordinate_buffer = np.unique(voxel_index, axis=0) #去除数组中的重复数字，并进行排序之后输出

            # Number of voxels in scan
            K = len(coordinate_buffer)
            # Max number of lidar points in each voxel
            T = cfg.VOXEL_POINT_COUNT

            # [K, 1] store number of points in each voxel grid ，即得分值
            number_buffer = np.zeros(shape=K, dtype=np.int64)

            # [K, T, 8] feature buffer as described in the paper
            feature_buffer = np.zeros(shape=(K, T, 8), dtype=np.float32)
            final_buffer = np.zeros(shape=(K, self.features.size), dtype=np.float32)
            # build a reverse index for coordinate buffer
            index_buffer = {}
            for i in range(K):
                index_buffer[tuple(coordinate_buffer[i])] = i  #dic

            for voxel, point in zip(voxel_index, raw_lidar_pcl):
                index = index_buffer[tuple(voxel)]
                number = number_buffer[index]
                if number < T:
                    feature_buffer[index, number, :4] = point
                    number_buffer[index] += 1

            # Compute features
            for i in range(K):
                if not ((('roughness' in self.features) or ('slope' in self.features)) and number_buffer[i] < 4):  # Need at least 3 points to compute pca
                    final_buffer[i, :] = self.compute_features(feature_buffer[i, :, :])

            # If pca was necessary, remove rows that had too few points to compute pca (empty rows)
            if 'roughness' in self.features or 'slope' in self.features:
                final_ids = ~np.all(final_buffer == 0, axis=1)
                final_buffer = final_buffer[final_ids]
                coordinate_buffer = coordinate_buffer[final_ids]

            return final_buffer, coordinate_buffer, feature_buffer, index_buffer
        else:
            return [], [], [], []


    ## 计算特征 int_mean; int_var; echo ;roughness; slope
    def compute_features(self, voxel):

        voxel_features = []
        voxel = voxel[~np.all(voxel == 0, axis=1)]
        pca = None
        if 'int_mean' in self.features:
            voxel_features.append(np.mean(voxel[:, 3])) #强度均值
        if 'int_var' in self.features:
            voxel_features.append(np.std(voxel[:, 3]))  #强度标准差
        if 'echo' in self.features:
            unique, cnts = np.unique(voxel[:, 4], return_counts=True)
            voxel_features.append(unique[np.argmax(cnts)])
        if 'roughness' in self.features:
            pca = PCA()
            pca.fit(voxel[:, :3])
            voxel_features.append(pca.singular_values_[2])
        if 'slope' in self.features:
            if not pca:
                pca = PCA()
                pca.fit(voxel[:, :3])
            # 0 is any vector on the xy plane, pi/2 is a vector along the z axis
            voxel_features.append(abs(math.asin(pca.components_[2, 2])))
        return voxel_features

    def post_process_scan(self):
        # Shuffle prior to concatenation to avoid taking the same samples everytime,洗牌，避免出现相同样本
        if self.shuffle:
            p = np.random.permutation(len(self.non_smoke_data))
            self.non_smoke_data = self.non_smoke_data[p]
            self.non_smoke_labels = self.non_smoke_labels[p]
            p = np.random.permutation(len(self.smoke_data))
            self.smoke_data = self.smoke_data[p]
            self.smoke_labels = self.smoke_labels[p]

        # Adjust number of each label to be equal
        if self.non_smoke_data.shape[0] > self.smoke_data.shape[0]:
            self.non_smoke_data = self.non_smoke_data[:self.smoke_data.shape[0], :, :]
            self.non_smoke_labels = self.non_smoke_labels[:self.smoke_labels.shape[0]]
        else:
            self.smoke_data = self.smoke_data[:self.non_smoke_data.shape[0], :, :]
            self.smoke_labels = self.smoke_labels[:self.non_smoke_labels.shape[0]]

        self.data = np.concatenate((self.smoke_data, self.non_smoke_data), axis=0)
        self.labels = np.concatenate((self.smoke_labels, self.non_smoke_labels), axis=0)
        self.labels = self.labels.reshape((self.labels.shape[0]))
        # Need to re shuffle post concatenation
        if self.shuffle:
            p = np.random.permutation(len(self.data))
            self.data = self.data[p]
            self.labels = self.labels[p]

        self.normalise_data()

    def get_pred_scan(self, raw_lidar_pcl):
        self.data, coord_buffer, feature_buffer, index_buffer = self.extract_scan(raw_lidar_pcl)
        self.normalise_data()
        return self.data, coord_buffer, feature_buffer, index_buffer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        data, labels = self.data[idx], self.labels[idx]

        if self.transform:
            data = self.transform(data)

        sample = {'inputs': data, 'labels': labels}

        return sample

    def get_scale(self):
        data_tmp = np.reshape(self.data, (-1, self.data.shape[2]))
        return np.mean(data_tmp, axis=0), np.std(data_tmp, axis=0)

    def update_transform(self, transform):
        self.transform = transform

    def normalise_data(self):
        # Data normalisation
        if self.mu is None or self.sigma is None:
            print('No scaling values provided, calulating scaling values for normalisation...')
            data_tmp = np.reshape(self.data, (-1, self.data.shape[2]))
            data_tmp = data_tmp[~np.all(data_tmp == 0, axis=1)]

            self.mu = np.mean(data_tmp, axis=0)
            self.sigma = np.std(data_tmp, axis=0)

        if self.features.size != len(self.mu) or self.features.size != len(self.sigma):
            sys.exit('Number of features different than mu or sigma')

        # idx = np.arange(0, cfg.VOXEL_POINT_COUNT)
        # for i in range(0, self.data.shape[0]):
        #     self.data[i, idx[~np.all(self.data[i, :, :] == 0, axis=1)], :] = (self.data[i, idx[~np.all(
        #         self.data[i, :, :] == 0, axis=1)], :] - self.mu) / self.sigma

    # TODO: pass arguments for individual methods to be not hard coded
    def select_features(self, method):
        if method == 'SelectPercentile':
            select = SelectPercentile(percentile=50)
        elif method == 'SelectKBest':
            select = SelectKBest(chi2, k=2)
        elif method == 'VarianceThreshold':
            select = VarianceThreshold(threshold=(.8 * (1 - .8)))
        elif method == 'TreeBased':
            select = SelectFromModel(RandomForestClassifier(), threshold='median')
        elif method == 'L1Based':
            select = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False), threshold='median')
        else:
            sys.exit('Method name not valid')

        # Fit the selector
        select.fit(self.data, self.labels)
        # Apply to features
        self.data = select.transform(self.data)
        print('Feature selection using %s method:' % method)
        mask = select.get_support()
        print(mask)
        self.mu = self.mu[mask]
        self.sigma = self.sigma[mask]
        self.features = self.features[mask]
