'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_Share, HSI_Lidar_Couple_DAD,HSI_Lidar_Couple_Cross_DAD
from src.augsburg_dataloader import augsburg_multi_dataloader
from configuration.augsburg_multi_config import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi, calc_accuracy_multi
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
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'sar']

    # print(args)

    test_loader = augsburg_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 180, 'sar':4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = HSI_Lidar_Couple_Cross_DAD(args, modality_1_channel, modality_2_channel)
    model.load_state_dict(
        torch.load(os.path.join(args.model_root, 'augsburg_hsi_sar_couple_cross_fc_version_1.pth')))
    model.eval()

    args.retrain = False
    result = calc_accuracy_multi(model=model,args=args, loader=test_loader, hter=False, verbose=True)
    print(result)


if __name__ == '__main__':
    deeppix_main(args=args)
