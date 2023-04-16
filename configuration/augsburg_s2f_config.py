import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_warmup', type=bool, default=True)
parser.add_argument('--total_epoch', type=int, default=5)
parser.add_argument('--lr_decrease', type=str, default='cos', help='the methods of learning rate decay  ')
parser.add_argument('--mixup', type=bool, default=False, help='using mixup or not')
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=7)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
# parser.add_argument('modal', type=str, default='multi')

parser.add_argument('--model_name', type=str, default='mnist_cnn_best')
parser.add_argument('--log_name', type=str, default='.csv')
parser.add_argument('--backbone', type=str, default='augsburg_multihall_ensemble_infer')
parser.add_argument('--patch_size', type=int, default=7)

parser.add_argument('data_root', type=str,
                    default='/home/bigspace/shicaiwei/remote_sensing/houston2013')
parser.add_argument('pair_modalities', type=str, default='hsi+lidar')
parser.add_argument('modality_1_pretrain_dir', type=str, default='')
parser.add_argument('modality_2_pretrain_dir', type=str, default='')
parser.add_argument('gpu', type=int, default=0)
parser.add_argument('version', type=int, default=0)

args = parser.parse_args()
args.pair_modalities = args.pair_modalities.split('+')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.name = args.pair_modalities[0] + "_" + args.pair_modalities[1] + "_" + "to" + "_" + args.pair_modalities[1] + "_"+args.backbone + "_version_" + str(args.version)

args.log_name = args.name + '.csv'
args.model_name = args.name
