from lib.processing_utils import load_mat
from models.resnet_ensemble import HSI_Lidar_Couple_DAD

import os
import numpy as np
import torch


# def test_numpy_reshape():
#     lidar_data = load_mat("/home/shicaiwei/data/remote_sensing/houston2013/ms/ms_X_train.mat")
#     sample = lidar_data[0, :]
#     sample_reshape = np.reshape(sample, (11, 11, -1), order='F')
#     sample_reshape_one = sample_reshape[:, :, 0]
#     max = np.max(lidar_data)
#     print(1)

def test_numpy_max():
    a = np.ones((3, 4, 4))
    a = torch.from_numpy(a)
    c = torch.max(a, 0, keepdim=True)[0]
    d = a - c
    print(c)


# def test_label():
#     a_label=load_mat("/home/shicaiwei/data/remote_sensing/honston_2013_7x7_liadr/TeLabel.mat")
#     b_label=load_mat("/home/shicaiwei/data/remote_sensing/houston2013/lidar/lidar_Y_test.mat")
#     c=a_label-b_label
#     print(1)

#
# def test_model():
#     # teacher_model = HSI_Lidar_Couple_DAD(args, pretrained=True)
#     model = torch.load(
#         '/home/shicaiwei/project/MI_LandCover/output/models/hsi_lidar_hallucination_ensemble_multi_version_0.pth')
#     print(1)


# def test_flatten():
#     a = torch.tensor([[[[1]],[[2]],[[3]]],[[[4]],[[5]],[[6]]]])
#     b = a.view(a.shape[0], -1)
#     c = torch.flatten(a, 1)
#     print(b.shape)
#     print(c.shape)
#     print(1)
def data_fix(label, data1, data2):
    from collections import Counter

    a = label
    a = list(a)
    read_begin = []
    begin_index = 0
    print(type(a))
    counter_result = Counter(a)
    for i in set(a):
        read_begin.append(begin_index)
        begin_index += counter_result[i]
    b = data1
    b = list(b)
    e = data2
    e = list(e)
    c = zip(a, b, e)
    c_sorted = sorted(c, key=lambda x: (x[0]))
    d = {}
    for i in range(len(a)):
        d[i] = c_sorted[i]

    label_sorted = []
    read_begin_diff = np.diff(read_begin)
    read_begin_diff = list(read_begin_diff)
    read_begin_diff.append(begin_index - read_begin[-1])

    for i in range(len(a)):
        index = np.mod(i, len(read_begin))
        step = i // len(read_begin)
        while step >= read_begin_diff[index]:
            index = np.mod(index + 1, len(read_begin))
        c_sorted_index = read_begin[index] + step
        label_sorted.append(c_sorted[c_sorted_index])

    result = zip(*label_sorted)
    label, data1, data2 = [list(x) for x in result]

    return np.array(label), np.array(data1), np.array(data2)


def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components


# def test_mat():
#     import h5py
#     import mat73
#     data = load_mat('/home/shicaiwei/data/sar/sar_Y_test.mat')
#     data_trans = np.transpose(data, [1, 0])
#     data_trans = list(data_trans)
#     data_trans = data_trans[0]
#     data, a, b = data_fix(data_trans, data, data)
#     data = [data]
#     data = np.transpose(data, [1, 0])
#     print(data[1:100])
#
#     print(1)


# def test_batch():
#     from collections import Counter
#
#     a = np.array([0, 0, 1, 2, 3, 1, 2, 3, 1, 1, ])
#     read_begin = []
#     begin_index = 0
#     counter_result = Counter(a)
#     for i in range(len(set(a))):
#         read_begin.append(begin_index)
#         begin_index += counter_result[i]
#     b = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
#     e = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
#     c = zip(a, b, e)
#     c_sorted = sorted(c, key=lambda x: (x[0]))
#     d = {}
#     for i in range(len(a)):
#         d[i] = c_sorted[i]
#
#     label_sorted = []
#     for i in range(len(a)):
#         index = np.mod(i, len(read_begin))
#         step = i // len(read_begin)
#         c_sorted_index = read_begin[index] + step
#         label_sorted.append(c_sorted[c_sorted_index])
#
#     result = zip(*label_sorted)
#     label, data1, data2 = [list(x) for x in result]
#     print(1)


def ma_test():
    import numpy as np
    from scipy.spatial.distance import cdist

    x = np.array([[1, 2, 3],
                  [9, 0, 1]])

    y = np.array([[8, 7, 6],
                  [0, 1, 2]])

    results = cdist(x, y, 'mahalanobis')

    results = np.diag(results)
    print(results)


def mainflod_test():
    from sklearn.manifold import Isomap
    import datetime
    a = np.random.randint(1, 10, (64, 512))
    embedding = Isomap(n_components=3, n_neighbors=5)
    begin = datetime.datetime.now()
    X_transformed = embedding.fit_transform(a)
    dist = embedding.dist_matrix_
    end = datetime.datetime.now()
    print((end - begin).total_seconds())
    print(dist.shape)


import torch.functional as tf
import torch.nn as nn


class PA_Measure(nn.Module):
    def __init__(self):
        super(PA_Measure, self).__init__()

    def forward(self, x, y):
        y = y.t()
        t = torch.mm(x, y)
        u, s, v = torch.svd(t)
        s_sum = torch.sum(s)
        return s_sum


def read_imglist(imglist_fp):
    ll = []
    with open(imglist_fp, 'r') as fd:
        for line in fd:
            ll.append(line.strip())
    return ll


def add_a(a):
    a['c']=3
    a=4
    print(id(a))
    print(1)


if __name__ == '__main__':
    a=[[1,2,3],[4,5,6]]
    b=a.copy()
    print(id(a),id(b))
    a[0]=[1]
    print(a)
    print(b)
