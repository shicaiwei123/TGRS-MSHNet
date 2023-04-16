'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import Hallucination_ensemble
from src.huston2013_dataloader import huston2013_single_dataloader
from configuration.huston2013_s2f_config import args
import torch
import torch.nn as nn
from lib.model_develop import train_hall_infer, calc_accuracy_hall_infer
from lib.processing_utils import get_file_list
import torch.optim as optim

import os
import numpy as np
import datetime
import random




def deeppix_main(args):
    args.modal = args.pair_modalities[1]
    args.pair_modalities[0] = args.pair_modalities[1]
    train_loader = huston2013_single_dataloader(train=True, args=args)
    test_loader = huston2013_single_dataloader(train=False, args=args)



    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_pretrain_dir = os.path.join(args.model_root, args.modality_1_pretrain_dir)
    modality_1_dict = torch.load(modality_1_pretrain_dir)

    modality_2_pretrain_dir = os.path.join(args.model_root, args.modality_2_pretrain_dir)
    modality_2_dict = torch.load(modality_2_pretrain_dir)
    model = Hallucination_ensemble(args, channel_dict=modality_to_channel,
                                   modality_1_dict=modality_1_dict,
                                   modality_2_dict=modality_2_dict)

    for p in model.parameters():
        if p.requires_grad == True:
            print(p)
    result = calc_accuracy_hall_infer(model=model, args=args,loader=test_loader, verbose=True)
    print(result)


    if torch.cuda.is_available():
        model.cuda()
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


    args.retrain = False
    train_hall_infer(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                     test_loader=test_loader,
                     args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
