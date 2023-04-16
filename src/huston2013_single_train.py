

import sys
import os
sys.path.append('..')
from models.single_modality_model import Single_Modality
from src.huston2013_dataloader import huston2013_single_dataloader
from configuration.huston2013_single_modality_config import args
import torch
import torch.nn as nn
from lib.model_develop_utils import train_base
import torch.optim as optim



def deeppix_main(args):
    train_loader = huston2013_single_dataloader(train=True, args=args)
    test_loader = huston2013_single_dataloader(train=False, args=args)


    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    model = Single_Modality(modality_to_channel[args.modal], args, pretrained=True)

    if torch.cuda.is_available():
        model.cuda()
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


    args.retrain = False
    train_base(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
               test_loader=test_loader,
               args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
