'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Couple_Cross_TRI_DAD, HSI_Lidar_Couple_Cross_DAD
from src.augsburg_dataloader import augsburg_multi_dataloader_tri
from models.single_modality_model import Single_Modality_DAD
from configuration.augsburg_kd_jda_config import args
import torch
from loss.kd import *
from lib.model_develop import train_knowledge_distill_jda_tri
from itertools import chain
import torch.nn as nn
import os


def deeppix_main(args):
    train_loader = augsburg_multi_dataloader_tri(train=True, args=args)
    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)



    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    modality_3_channel = modality_to_channel[args.pair_modalities[2]]
    teacher_model = HSI_Lidar_Couple_Cross_TRI_DAD(args, modality_1_channel, modality_2_channel, modality_3_channel)

    args.student_pair_modalities=args.student_data.split('+')
    modality_1_channel = modality_to_channel[args.student_pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.student_pair_modalities[1]]
    student_model = HSI_Lidar_Couple_Cross_DAD(args, modality_1_channel, modality_2_channel)
    args.preserve_modality = args.student_data


    # initialize teacher
    teacher_model.load_state_dict(
        torch.load(os.path.join(args.model_root, 'augsburg_hsi_sar_dsm_couple_cross_fc_version_2.pth')))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False


    if torch.cuda.is_available():
        teacher_model.cuda()
        student_model.cuda()
        print("GPU is using")

    # define loss functions
    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    elif args.kd_mode == 'at':
        criterionKD = AT(args.p)
    elif args.kd_mode == 'fitnet':
        criterionKD = Hint()
    elif args.kd_mode == 'multi_st':
        criterionKD = MultiSoftTarget(args.T)

    else:
        raise Exception('Invalid kd mode...')
    if torch.cuda.is_available():
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.SGD(chain(student_model.parameters(),
                                              *[c.parameters() for c in criterionKD[1:]]),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
        else:
            optimizer = torch.optim.SGD(student_model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.Adam(chain(student_model.parameters(),
                                               *[c.parameters() for c in criterionKD[1:]]),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
        else:
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

    train_knowledge_distill_jda_tri(net_dict=nets, cost_dict=criterions, optimizer=optimizer,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
