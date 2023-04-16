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
from configuration.huston2013_single_modality_config import args
import torch
import torch.nn as nn
from lib.model_develop_utils import train_base, calc_accuracy
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
    args.data_root = "../data/huston2013"
    args.modal = 'hsi'

    # print(args)
    test_loader = huston2013_single_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    model = Single_Modality_DAD(modality_to_channel[args.modal], args, pretrained=True)
    model.load_state_dict(
        torch.load(os.path.join(args.model_root, 'jda_hsi+lidar_hsi_lr_0.001_version_10.pth')))
    model.eval()

    args.retrain = False
    result = calc_accuracy(model=model, args=args,loader=test_loader, hter=False, verbose=True)
    print(result)


if __name__ == '__main__':
    deeppix_main(args=args)
