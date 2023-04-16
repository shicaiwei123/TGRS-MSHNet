import torch
import torch.nn as nn
from models.base_model import MDMB_extract, MDMB_fusion, Couple_CNN, CCR, MDMB_fusion_late, MDMB_fusion_share
from lib.model_arch_utils import Flatten, MMTM, SPP

seed =2


class Single_Modality(nn.Module):
    def __init__(self, input_channel, args, pretrained):
        super().__init__()
        self.feature = Couple_CNN(input_channel=input_channel)
        self.block_5_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))
                                       )
        self.block_6 = nn.Sequential(nn.Conv2d(64, args.class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True)
                                     )

        self.fc = nn.Linear(64, args.class_num, bias=True)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = self.block_5_1(x)
        x_feature = x
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x_dropout = x
        return x_dropout, x

class Single_Modality_transfer(nn.Module):
    '''
    用作单模态幻觉的第三步
    '''
    def __init__(self, input_channel, args, pretrained):
        super().__init__()
        self.feature = Couple_CNN(input_channel=input_channel)
        self.block_5_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))
                                       )
        self.block_6 = nn.Sequential(nn.Conv2d(64, args.class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True)
                                     )

        self.fc = nn.Linear(64, args.class_num, bias=True)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = self.block_5_1(x)
        x_feature = x
        # x = self.block_6(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x_dropout = x
        return x_dropout, x, x_feature


class Single_Modality_Baseline(nn.Module):
    def __init__(self, input_channel, args, pretrained):
        super().__init__()
        self.feature = Couple_CNN(input_channel=input_channel)
        self.block_5_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))
                                       )
        self.block_6 = nn.Sequential(nn.Conv2d(64, args.class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True)
                                     )

        self.fc = nn.Linear(64, args.class_num, bias=True)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = self.block_5_1(x)
        x_feature = x
        # x = self.block_6(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x_dropout = x
        return x_dropout, x


class Single_Modality_SPP(nn.Module):
    def __init__(self, input_channel, args, pretrained):
        super().__init__()
        self.feature = Couple_CNN(input_channel=input_channel)
        self.block_5_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),

                                       )
        self.block_6 = nn.Sequential(nn.Conv2d(64, args.class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True)
                                     )

        self.fc = nn.Linear(64, args.class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.spp = SPP(merge='max')
        self.args = args
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x_feature = self.block_5_1(x)
        x = self.avgpooling(x_feature)
        # x = self.block_6(x)
        x_whole = torch.flatten(x, 1)
        x_whole = self.fc(x_whole)
        # x_whole = self.dropout(x_whole)

        x_spp = self.spp(x_feature)
        feature_num = x_spp.shape[-1]
        patch_score = torch.zeros(x_spp.shape[0], self.args.class_num, feature_num)
        patch_strength = torch.zeros(x_spp.shape[0], feature_num)

        for i in range(feature_num):
            patch_feature = x_spp[:, :, i]
            patch_strength[:, i] = torch.mean(patch_feature, dim=1)
            # patch_feature = torch.unsqueeze(patch_feature, 2)
            # patch_feature = torch.unsqueeze(patch_feature, 3)
            # patch_logits = self.block_6(patch_feature)
            # patch_logits = torch.flatten(patch_logits)
            patch_logits = self.fc(patch_feature)
            patch_score[:, :, i] = patch_logits

        return x_whole, patch_score, patch_strength


class Single_Modality_DAD(nn.Module):
    def __init__(self, input_channel, args, pretrained):
        super().__init__()
        self.feature = Couple_CNN(input_channel=input_channel)
        self.block_5_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       )
        self.block_6 = nn.Sequential(nn.Conv2d(64, args.class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True)
                                     )
        self.spp = SPP()
        self.fc = nn.Linear(64, args.class_num, bias=True)
        self.args = args
        self.dropout = nn.Dropout()
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x_feature = self.block_5_1(x)
        x = self.avgpooling(x_feature)
        # x_whole = self.block_6(x)
        # x_whole = torch.flatten(x_whole, 1)
        x = x.view(x.shape[0], -1)
        x_whole = self.fc(x)
        # x_whole = self.dropout(x_whole)

        return x_whole, x_feature
