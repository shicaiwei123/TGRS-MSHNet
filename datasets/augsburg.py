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


class Augsburg_multi(Dataset):
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


class Augsburg_single(Dataset):
    def __init__(self, modality_path_1, label_path, args, data_transform=None, isdict=True):
        self.modality_1 = load_mat(modality_path_1)
        self.label_data = load_mat(label_path)
        self.data_transform = data_transform
        self.isdict = isdict
        self.args=args

    def __len__(self):
        return self.modality_1.shape[0]

    def __getitem__(self, idx):
        modality_data_1 = self.modality_1[idx, :]
        modality_1 = np.reshape(modality_data_1, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_label = int(self.label_data[idx]-1)

        if self.isdict:
            sample = {"m_1": modality_1, "label": modality_label}
            if self.data_transform is not None:
                sample = self.data_transform(sample)
            return sample
        else:

            if self.data_transform is not None:
                modality_1 = self.data_transform(modality_1)
            return modality_1, modality_label


class Augsburg_multi_tri(Dataset):
    def __init__(self, modality_path_1, modality_path_2, modality_path_3, label_path, args, data_transform=None,
                 isdict=True, miss=None):
        self.modality_1 = load_mat(modality_path_1)
        self.modality_2 = load_mat(modality_path_2)
        self.modality_3 = load_mat(modality_path_3)

        self.label_data = load_mat(label_path)
        self.data_transform = data_transform
        self.isdict = isdict
        self.args = args
        self.miss = miss

    def __len__(self):
        dataset_len = self.modality_1.shape[0]
        # print(dataset_len)
        return dataset_len

    def __getitem__(self, idx):
        modality_data_1 = self.modality_1[idx, :]
        modality_1 = np.reshape(modality_data_1, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_data_2 = self.modality_2[idx, :]
        modality_2 = np.reshape(modality_data_2, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_data_3 = self.modality_3[idx, :]
        modality_3 = np.reshape(modality_data_3, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_label = self.label_data[idx] - 1
        modality_label = int(modality_label)

        if self.miss is None:
            pass
        else:
            if "m1" in self.miss:
                modality_1 = np.zeros_like(modality_1)
            if "m2" in self.miss:
                modality_2 = np.zeros_like(modality_2)
            if "m3" in self.miss:
                modality_3 = np.zeros_like(modality_3)

        sample = {"m_1": modality_1, "m_2": modality_2, "m_3": modality_3, "label": modality_label}
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        if self.isdict:
            return sample
        else:
            return sample["m_1"], sample["m_2"], sample["label"]
