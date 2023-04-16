import sys

sys.path.append('..')

import torchvision.transforms as tt
import torch
import os

from datasets.augsburg import Augsburg_multi, Augsburg_single,Augsburg_multi_tri
from lib.processing_utils import get_mean_std
from datasets.dataset_proceess_utils import ToTensor_multi, RandomHorizontalFlip_multi, Normaliztion_multi, \
    ColorAdjust_multi, RandomVerticalFlip_multi

augsburg_multi_transforms_train = tt.Compose(
    [
        RandomHorizontalFlip_multi(),
        RandomVerticalFlip_multi(),
        # ColorAdjust_multi(brightness=0.3),
        ToTensor_multi(),
        # Normaliztion_multi(),
    ]
)

augsburg_multi_transforms_test = tt.Compose(
    [
        ToTensor_multi(),
        # Normaliztion_multi(),
    ]
)

augsburg_single_transforms_train = tt.Compose(
    [
        RandomHorizontalFlip_multi(),
        RandomVerticalFlip_multi(),
        # ColorAdjust_multi(brightness=0.3),
        ToTensor_multi(),
        # Normaliztion_multi(),
    ]
)

augsburg_single_transforms_test = tt.Compose(
    [
        ToTensor_multi(),
        # Normaliztion_multi(),
    ]
)


def augsburg_multi_dataloader(train, args):
    # dataset and data loader
    if train:
        print(args)
        modality_path_1 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_X_train.mat'))
        modality_path_2 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[1], args.pair_modalities[1] + '_X_train.mat'))
        label_train_path = os.path.join(args.data_root,
                                        os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_Y_train.mat'))
        augsburg_multi_dataset = Augsburg_multi(modality_path_1=modality_path_1, modality_path_2=modality_path_2,
                                                label_path=label_train_path,
                                                data_transform=augsburg_multi_transforms_train, args=args)
        augsburg_data_loader = torch.utils.data.DataLoader(
            dataset=augsburg_multi_dataset,
            batch_size=args.batch_size,
            shuffle=True)
    else:
        # print(args)
        modality_path_1 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_X_test.mat'))
        modality_path_2 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[1], args.pair_modalities[1] + '_X_test.mat'))
        label_train_path = os.path.join(args.data_root,
                                        os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_Y_test.mat'))
        augsburg_multi_dataset = Augsburg_multi(modality_path_1=modality_path_1, modality_path_2=modality_path_2,
                                                label_path=label_train_path,
                                                data_transform=augsburg_multi_transforms_test, args=args)

        augsburg_data_loader = torch.utils.data.DataLoader(
            dataset=augsburg_multi_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4)

    return augsburg_data_loader


def augsburg_single_dataloader(train, args):
    # dataset and data loader
    if train:

        single_train_path = os.path.join(args.data_root, os.path.join(args.modal, args.modal + '_X_train.mat'))
        label_train_path = os.path.join(args.data_root, os.path.join(args.modal, args.modal + '_Y_train.mat'))
        augsburg_single_dataset = Augsburg_single(modality_path_1=single_train_path,
                                                  label_path=label_train_path,
                                                  data_transform=augsburg_single_transforms_train, args=args)
    else:
        single_train_path = os.path.join(args.data_root, os.path.join(args.modal, args.modal + '_X_test.mat'))
        label_train_path = os.path.join(args.data_root, os.path.join(args.modal, args.modal + '_Y_test.mat'))
        augsburg_single_dataset = Augsburg_single(modality_path_1=single_train_path,
                                                  label_path=label_train_path,
                                                  data_transform=augsburg_single_transforms_test, args=args)

    augsburg_data_loader = torch.utils.data.DataLoader(
        dataset=augsburg_single_dataset,
        batch_size=args.batch_size,
        shuffle=True)

    return augsburg_data_loader

def augsburg_multi_dataloader_tri(train, args):
    # dataset and data loader
    if train:
        modality_path_1 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_X_train.mat'))
        modality_path_2 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[1], args.pair_modalities[1] + '_X_train.mat'))
        modality_path_3 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[2], args.pair_modalities[2] + '_X_train.mat'))
        label_train_path = os.path.join(args.data_root,
                                        os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_Y_train.mat'))
        augsburg_multi_dataset = Augsburg_multi_tri(modality_path_1=modality_path_1, modality_path_2=modality_path_2,
                                                    modality_path_3=modality_path_3,
                                                    label_path=label_train_path,
                                                    data_transform=augsburg_multi_transforms_train, args=args,
                                                    miss=args.miss)
        augsburg_data_loader = torch.utils.data.DataLoader(
            dataset=augsburg_multi_dataset,
            batch_size=args.batch_size,
            shuffle=True)
    else:
        # print(args)
        modality_path_1 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_X_test.mat'))
        modality_path_2 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[1], args.pair_modalities[1] + '_X_test.mat'))
        modality_path_3 = os.path.join(args.data_root,
                                       os.path.join(args.pair_modalities[2], args.pair_modalities[2] + '_X_test.mat'))
        label_train_path = os.path.join(args.data_root,
                                        os.path.join(args.pair_modalities[0], args.pair_modalities[0] + '_Y_test.mat'))
        augsburg_multi_dataset = Augsburg_multi_tri(modality_path_1=modality_path_1, modality_path_2=modality_path_2,
                                                    modality_path_3=modality_path_3,
                                                    label_path=label_train_path,
                                                    data_transform=augsburg_multi_transforms_test, args=args,
                                                    miss=args.miss)

        augsburg_data_loader = torch.utils.data.DataLoader(
            dataset=augsburg_multi_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4)

    return augsburg_data_loader

