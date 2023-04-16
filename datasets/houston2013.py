import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import os
from lib.processing_utils import read_txt, get_file_list, load_mat
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as tt
import scipy.io as scio


class Huston2013_multi(Dataset):
    def __init__(self, modality_path_1, modality_path_2, label_path, args, data_transform=None, isdict=True):
        self.modality_1 = load_mat(modality_path_1)
        self.modality_2 = load_mat(modality_path_2)
        self.label_data = load_mat(label_path)
        self.data_transform = data_transform
        self.isdict = isdict
        self.args = args

    def __len__(self):
        dataset_len = self.modality_1.shape[0]
        return dataset_len

    def __getitem__(self, idx):

        modality_data_1 = self.modality_1[idx, :]
        modality_1 = np.reshape(modality_data_1, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_data_2 = self.modality_2[idx, :]
        modality_2 = np.reshape(modality_data_2, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_label = self.label_data[idx] - 1
        modality_label = int(modality_label)

        sample = {"m_1": modality_1, "m_2": modality_2, "label": modality_label}
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        if self.isdict:
            return sample
        else:
            return sample["m_1"], sample["m_2"], sample["label"]


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


class Huston2013_multi_fix(Dataset):
    def __init__(self, modality_path_1, modality_path_2, label_path, args, data_transform=None, isdict=True):
        self.modality_1 = load_mat(modality_path_1)
        self.modality_2 = load_mat(modality_path_2)
        self.label_data = load_mat(label_path)
        self.data_transform = data_transform
        self.isdict = isdict
        self.args = args
        print("fix data")

        self.label_data = np.transpose(self.label_data, [1, 0])
        self.label_data = self.label_data[0]
        self.label_data, self.modality_1, self.modality_2 = data_fix(self.label_data, self.modality_1, self.modality_2)
        self.label_data = [self.label_data]
        self.label_data = np.transpose(self.label_data, [1, 0])
        # print(self.label_data[1:100])

    def __len__(self):
        dataset_len = self.modality_1.shape[0]
        return dataset_len

    def __getitem__(self, idx):

        modality_data_1 = self.modality_1[idx, :]
        modality_1 = np.reshape(modality_data_1, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_data_2 = self.modality_2[idx, :]
        modality_2 = np.reshape(modality_data_2, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_label = self.label_data[idx] - 1
        modality_label = int(modality_label)

        sample = {"m_1": modality_1, "m_2": modality_2, "label": modality_label}
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        if self.isdict:
            return sample
        else:
            return sample["m_1"], sample["m_2"], sample["label"]


class Huston2013_single(Dataset):
    def __init__(self, modality_path_1, label_path, args, data_transform=None, isdict=True):
        self.modality_1 = load_mat(modality_path_1)
        self.label_data = load_mat(label_path)
        self.data_transform = data_transform
        self.isdict = isdict
        self.args = args

    def __len__(self):
        return self.modality_1.shape[0]

    def __getitem__(self, idx):
        modality_data_1 = self.modality_1[idx, :]
        modality_1 = np.reshape(modality_data_1, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_label = int(self.label_data[idx] - 1)

        if self.isdict:
            sample = {"m_1": modality_1, "label": modality_label}
            if self.data_transform is not None:
                sample = self.data_transform(sample)
            return sample
        else:

            if self.data_transform is not None:
                modality_1 = self.data_transform(modality_1)
            return modality_1, modality_label
