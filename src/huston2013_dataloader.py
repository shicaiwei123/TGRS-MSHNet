import sys

sys.path.append('..')

import torchvision.transforms as tt
import torch
import os

from datasets.houston2013 import Huston2013_multi, Huston2013_single, Huston2013_multi_fix
from lib.processing_utils import get_mean_std
from datasets.dataset_proceess_utils import ToTensor_multi, RandomHorizontalFlip_multi, Normaliztion_multi, \
    ColorAdjust_multi, RandomVerticalFlip_multi

huston2013_multi_transforms_train = tt.Compose(
    [
        RandomHorizontalFlip_multi(),
        RandomVerticalFlip_multi(),
        # ColorAdjust_multi(brightness=0.3),
        ToTensor_multi(),
        # Normaliztion_multi(),
    ]
)

huston2013_multi_transforms_test = tt.Compose(
    [
        ToTensor_multi(),
        # Normaliztion_multi(),
    ]
)

huston2013_single_transforms_train = tt.Compose(
    [
        RandomHorizontalFlip_multi(),
        RandomVerticalFlip_multi(),
        # ColorAdjust_multi(brightness=0.3),
        ToTensor_multi(),
        # Normaliztion_multi(),
    ]
)

huston2013_single_transforms_test = tt.Compose(
    [
        ToTensor_multi(),
        # Normaliztion_multi(),
    ]
)


def huston2013_multi_dataloader(train, args):
    # dataset and data loader
    if train:
        # print(args)
        modality_path_1 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_X_train.mat'))
        modality_path_2 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[1], args.pair_modalities[1] + '_X_train.mat'))
        label_train_path = os.path.join(args.data_root,
                                        os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_Y_train.mat'))
        huston2013_multi_dataset = Huston2013_multi(modality_path_1=modality_path_1, modality_path_2=modality_path_2,
                                                    label_path=label_train_path,
                                                    data_transform=huston2013_multi_transforms_train, args=args)
    else:
        # print(args)
        modality_path_1 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_X_test.mat'))
        modality_path_2 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[1], args.pair_modalities[1] + '_X_test.mat'))
        label_train_path = os.path.join(args.data_root,
                                        os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_Y_test.mat'))
        huston2013_multi_dataset = Huston2013_multi(modality_path_1=modality_path_1, modality_path_2=modality_path_2,
                                                    label_path=label_train_path,
                                                    data_transform=huston2013_multi_transforms_test, args=args)

    huston2013_data_loader = torch.utils.data.DataLoader(
        dataset=huston2013_multi_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)

    return huston2013_data_loader


def huston2013_multi_dataloader_fix(train, args):
    # dataset and data loader
    if train:
        # print(args)
        modality_path_1 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_X_train.mat'))
        modality_path_2 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[1], args.pair_modalities[1] + '_X_train.mat'))
        label_train_path = os.path.join(args.data_root,
                                        os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_Y_train.mat'))
        huston2013_multi_dataset = Huston2013_multi_fix(modality_path_1=modality_path_1,
                                                        modality_path_2=modality_path_2,
                                                        label_path=label_train_path,
                                                        data_transform=huston2013_multi_transforms_train, args=args)
    else:
        # print(args)
        modality_path_1 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_X_test.mat'))
        modality_path_2 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[1], args.pair_modalities[1] + '_X_test.mat'))
        label_train_path = os.path.join(args.data_root,
                                        os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_Y_test.mat'))
        huston2013_multi_dataset = Huston2013_multi(modality_path_1=modality_path_1,
                                                    modality_path_2=modality_path_2,
                                                    label_path=label_train_path,
                                                    data_transform=huston2013_multi_transforms_test, args=args)

    huston2013_data_loader = torch.utils.data.DataLoader(
        dataset=huston2013_multi_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)

    return huston2013_data_loader


def huston2013_single_dataloader(train, args):
    # dataset and data loader
    if train:

        single_train_path = os.path.join(args.data_root, os.path.join(args.modal, args.modal + '_X_train.mat'))
        label_train_path = os.path.join(args.data_root, os.path.join(args.modal, args.modal + '_Y_train.mat'))
        huston2013_single_dataset = Huston2013_single(modality_path_1=single_train_path,
                                                      label_path=label_train_path,
                                                      data_transform=huston2013_single_transforms_train, args=args)
        huston2013_data_loader = torch.utils.data.DataLoader(
            dataset=huston2013_single_dataset,
            batch_size=args.batch_size,
            shuffle=True)
    else:
        single_train_path = os.path.join(args.data_root, os.path.join(args.modal, args.modal + '_X_test.mat'))
        label_train_path = os.path.join(args.data_root, os.path.join(args.modal, args.modal + '_Y_test.mat'))
        # print(single_train_path, label_train_path)
        huston2013_single_dataset = Huston2013_single(modality_path_1=single_train_path,
                                                      label_path=label_train_path,
                                                      data_transform=huston2013_single_transforms_test, args=args)

        huston2013_data_loader = torch.utils.data.DataLoader(
            dataset=huston2013_single_dataset,
            batch_size=args.batch_size,
            shuffle=False)

    return huston2013_data_loader
