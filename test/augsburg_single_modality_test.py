'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_Share, HSI_Lidar_Couple_DAD
from models.single_modality_model import Single_Modality
from models.single_modality_model import Single_Modality_DAD

from src.huston2013_dataloader import huston2013_single_dataloader
from src.augsburg_dataloader import augsburg_single_dataloader
from configuration.augsburg_single_modality_config import args
import torch
import torch.nn as nn
from lib.model_develop_utils import train_base, calc_accuracy, calc_accuracy_full
from lib.processing_utils import get_file_list
import torch.optim as optim

import os
import numpy as np
import random


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deeppix_main(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/augsburg"
    args.modal = 'hsi'
    args.gpu = 0

    # print(args)
    test_loader = augsburg_single_dataloader(train=False, args=args)
    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    model = Single_Modality_DAD(modality_to_channel[args.modal], args, pretrained=True)

    for i in range(4):
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'augsburg_single_fc_class_7_modal_hsi_version_' + str(i) + '.pth')))
        model.eval()

        args.retrain = False
        result = calc_accuracy(model=model, args=args,loader=test_loader, hter=False, verbose=True)
        print(result)


def deeppix_main_full(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/augsburg"
    args.modal = 'hsi'
    args.gpu = 0
    args.class_num=7

    # print(args)
    test_loader = augsburg_single_dataloader(train=False, args=args)
    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    model = Single_Modality_DAD(modality_to_channel[args.modal], args, pretrained=True)

    for i in range(4):
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'tri_cross_patch_kd_jda_avg_fc_hsi+sar+dsm_hsi_lr_0.001_version_' + str(1) + '.pth')))
        model.eval()

        args.retrain = False
        result = calc_accuracy_full(model=model, loader=test_loader, args=args, hter=False, verbose=True)
        print(result)


if __name__ == '__main__':
    deeppix_main_full(args=args)
