'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Couple_Cross_DAD
from src.huston2013_dataloader import huston2013_multi_dataloader_fix, huston2013_multi_dataloader
from models.single_modality_model import Single_Modality_DAD
from configuration.huston2013_kd_jda_config import args
import torch
from loss.kd import *
from lib.model_develop import train_knowledge_distill_jda
from itertools import chain
import torch.nn as nn
import os
import numpy as np
import datetime
import random




def deeppix_main(args):
    train_loader = huston2013_multi_dataloader(train=True, args=args)
    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    teacher_model = HSI_Lidar_Couple_Cross_DAD(args, modality_1_channel, modality_2_channel)
    student_model = Single_Modality_DAD(modality_to_channel[args.student_data], args, pretrained=True)
    args.preserve_modality = args.student_data

    # initialize teacher

    if 'lidar' in args.pair_modalities:
        teacher_model.load_state_dict(
            torch.load(os.path.join(args.model_root, 'fusion_11__hsi_lidar_version_0.pth')))
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    elif 'ms' in args.pair_modalities:
        teacher_model.load_state_dict(
            torch.load(os.path.join(args.model_root, 'fusion_hsi_ms.pth')))
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    else:
        raise Exception('INVALID modality')

    if torch.cuda.is_available():
        teacher_model.cuda()  #
        student_model.cuda()
        print("GPU is using")

    # define loss functions

    criterionKD = JDA(args.alpha, args.beta)

    if torch.cuda.is_available():
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')

        optimizer = torch.optim.SGD(student_model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')

        optimizer = torch.optim.Adam(student_model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     )
    else:
        print('optim error')
        optimizer = None

    # warp nets and criterions for train and test
    nets = {'snet': student_model, 'tnet': teacher_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    train_knowledge_distill_jda(net_dict=nets, cost_dict=criterions, optimizer=optimizer,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
