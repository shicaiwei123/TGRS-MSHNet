import os
import numpy as np
import torch
import csv
import cv2
import random
import scipy.io as scio
import h5py
import mat73


def get_file_list(read_path):
    '''
    获取文件夹下图片的地址
    :param read_path:
    :return:
    '''
    path = read_path
    dirs = os.listdir(path)
    floder_len = len(dirs)
    file_name_list = []
    for i in range(floder_len):

        # 设置路径
        floder = dirs[i]
        floder_path = path + "/" + floder

        # 如果路径下是文件，那么就再次读取
        if os.path.isdir(floder_path):
            file_one = os.listdir(floder_path)
            file_len_one = len(file_one)
            for j in range(file_len_one):
                # 读取视频
                floder_path_one = floder_path + "/" + file_one[j]
                if os.path.isdir(floder_path_one):
                    file_two = os.listdir(floder_path_one)
                    file_len_two = len(file_two)
                    for k in range(file_len_two):
                        floder_path_two = floder_path_one + "/" + file_two[k]
                        if os.path.isdir(floder_path_two):
                            file_three = os.listdir(floder_path_two)
                            file_len_three = len(file_three)
                            for m in range(file_len_three):
                                floder_path_three = floder_path_two + "/" + file_three[m]
                                file_name_list.append(floder_path_three)
                        else:
                            file_name_list.append(floder_path_two)

                else:
                    file_name_list.append(floder_path_one)

        # 如果路径下，没有文件夹，直接是文件，就加入进来
        else:
            file_name_list.append(floder_path)

    return file_name_list


def get_mean_std(dataset, ratio=1):
    """Get mean and std by sample ratio
    """
    '求数据集的均值方差'
    '本质是读取一个epoch的数据进行测试,只不过把一个epoch的大小设置成了所有数据'
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]  # 一个batch的数据
    train = train['image_rgb']
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


def seed_torch(seed=0):
    '''在使用模型的时候用于设置随机数'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_dataself_hist(arr):
    '''
    统计一个arr 不同数字出现的频率
    可以看做以本身为底 的直方图统计
    :param arr:
    :return:
    '''
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v

    return result


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_csv(csv_path, data):
    '''
    以csv格式,增量保存数据,常用域log的保存

    :param csv_path: csv 文件地址
    :param data: 要保存数据,list和arr 都可以,但是只能是一维的
    :return:
    '''
    with open(csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)
    f.close()


def save_args(args, save_path):
    if os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()


def read_csv(csv_path):
    '''
    读取csv文件的内容,并且返回
    '''
    data_list = []
    csvFile = open(csv_path, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        '''item 是一个list,一个item 就是一行'''
        data_list.append(item)

    return data_list


def read_txt(txt_path):
    '''
    读取txt 文件
    :param txt_path:
    :return: txt中每行的数据,结尾用'\n'
    '''

    f = open(txt_path)
    data = f.readlines()
    for index in range(len(data)):
        data[index] = data[index][:-1]
    return data


def load_mat(mat_path):
    try:
        data = scio.loadmat(mat_path)
        data = data[list(data.keys())[-1]]
    except Exception as e:
        # data = h5py.File(mat_path)
        # data_key = data.keys()
        # print(data_key)
        # data = data[list(data_key)[0]]
        # print(data.shape)
        # print(dir(data))
        # data = data.value
        data = mat73.loadmat(mat_path)
        data = data[list(data.keys())[0]]
    return data

