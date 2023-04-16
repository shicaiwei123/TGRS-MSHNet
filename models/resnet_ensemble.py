import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from models.base_model import MDMB_extract, MDMB_fusion, Couple_CNN, CCR, MDMB_fusion_late, MDMB_fusion_share, \
    MDMB_fusion_spp, MDMB_fusion_baseline, MDMB_fusion_dad
from models.single_modality_model import Single_Modality



class HSI_Lidar_MDMB(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()

        self.special_bone_hsi = MDMB_extract(input_channel=144)
        self.special_bone_lidar = MDMB_extract(input_channel=1)
        self.share_bone = MDMB_fusion(256, 15)

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        x = torch.cat((x_hsi, x_lidar), dim=1)
        x = self.share_bone(x)
        return x


class HSI_Lidar_Couple(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_modality_1 = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_modality_2 = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion(256, args.class_num)

    def forward(self, modality_1, modality_2):
        x_modality_1 = self.special_bone_modality_1(modality_1)
        x_modality_2 = self.special_bone_modality_2(modality_2)

        x = torch.cat((x_modality_1, x_modality_2), dim=1)
        x_dropout, x = self.share_bone(x)
        return x_dropout, x


class HSI_Lidar_Couple_Baseline(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_modality_1 = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_modality_2 = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_baseline(256, args.class_num)

    def forward(self, modality_1, modality_2):
        x_modality_1 = self.special_bone_modality_1(modality_1)
        x_modality_2 = self.special_bone_modality_2(modality_2)

        x = torch.cat((x_modality_1, x_modality_2), dim=1)
        x = self.share_bone(x)
        return x



class HSI_Lidar_Couple_Late(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=144)
        self.special_bone_lidar = Couple_CNN(input_channel=1)
        self.share_bone = MDMB_fusion_late(128, 15)

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        x = self.share_bone(x_hsi, x_lidar)
        return x


class HSI_Lidar_Couple_Share(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=144)
        self.special_bone_lidar = Couple_CNN(input_channel=1)
        self.share_bone = MDMB_fusion_share(128, 15)

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        x = self.share_bone(x_hsi, x_lidar)
        return x


class HSI_Lidar_Couple_Cross(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()
        self.hsi_block_1 = nn.Sequential(

                                         nn.Conv2d(modality_1_channel, 32, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),

                                         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         )

        self.lidar_block_1 = nn.Sequential(

                                           nn.Conv2d(modality_2_channel, 32, kernel_size=3, stride=1, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(32),
                                           nn.ReLU(),
                                           nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(),
                                           )

        self.hsi_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.lidar_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(),
                                           )

        self.share_bone = MDMB_fusion(256, args.class_num)

    def forward(self, hsi, lidar):
        hsi = self.hsi_block_1(hsi)
        lidar = self.lidar_block_1(lidar)
        x_hsi = self.hsi_block_2(hsi)
        x_lidar = self.lidar_block_2(lidar)
        x_hsi_lidar = self.lidar_block_2(hsi)
        x_lidar_hsi = self.hsi_block_2(lidar)

        joint_1 = torch.cat(((x_hsi + x_lidar_hsi) / 2, (x_lidar + x_hsi_lidar) / 2), dim=1)
        joint_2 = torch.cat((x_hsi, x_hsi_lidar), dim=1)
        joint_3 = torch.cat((x_lidar_hsi, x_lidar), dim=1)


        x1 = self.share_bone(joint_1)
        x2 = self.share_bone(joint_2)
        x3 = self.share_bone(joint_3)
        return x1, x2, x3


class HSI_Lidar_Couple_Cross_DAD(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()
        self.hsi_block_1 = nn.Sequential(

                                         nn.Conv2d(modality_1_channel, 32, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),

                                         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         )

        self.lidar_block_1 = nn.Sequential(

                                           nn.Conv2d(modality_2_channel, 32, kernel_size=3, stride=1, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(32),
                                           nn.ReLU(),
                                           nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(),
                                           )

        self.hsi_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.lidar_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(),
                                           )

        self.share_bone = MDMB_fusion_dad(256, args.class_num)

    def forward(self, hsi, lidar):
        hsi = self.hsi_block_1(hsi)
        lidar = self.lidar_block_1(lidar)
        x_hsi = self.hsi_block_2(hsi)
        x_lidar = self.lidar_block_2(lidar)
        x_hsi_lidar = self.lidar_block_2(hsi)
        x_lidar_hsi = self.hsi_block_2(lidar)

        joint_1 = torch.cat(((x_hsi + x_lidar_hsi) / 2, (x_lidar + x_hsi_lidar) / 2), dim=1)
        joint_2 = torch.cat((x_hsi, x_hsi_lidar), dim=1)
        joint_3 = torch.cat((x_lidar_hsi, x_lidar), dim=1)


        x1, x_feature = self.share_bone(joint_1)
        return x1, x_feature


class HSI_Lidar_CCR(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=144)
        self.special_bone_lidar = Couple_CNN(input_channel=1)
        self.share_bone = CCR(256, 15)

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        x = torch.cat((x_hsi, x_lidar), dim=1)
        x_origin = torch.cat((x_lidar, x_hsi), dim=1)
        # x_origin = x
        x, x_rec = self.share_bone(x)
        return x, x_origin, x_rec


class Hallucination_ensemble(nn.Module):
    '''
   fusion shared and specific information
    '''

    def __init__(self, args, channel_dict, modality_1_dict=None, modality_2_dict=None):
        super(Hallucination_ensemble, self).__init__()
        modality_1_channel = channel_dict[args.pair_modalities[0]]
        modality_2_channel = channel_dict[args.pair_modalities[1]]
        self.modality_1_model = Single_Modality(input_channel=modality_1_channel, args=args, pretrained=True)
        self.modality_2_model = Single_Modality(input_channel=modality_2_channel, args=args, pretrained=True)
        if modality_1_dict is not None:
            self.modality_1_model.load_state_dict(modality_1_dict)
        if modality_2_dict is not None:
            self.modality_2_model.load_state_dict(modality_2_dict)
        self.dropout = nn.Dropout()
        for p in self.modality_1_model.parameters():
            p.requires_grad = False
        for p in self.modality_2_model.parameters():
            p.requires_grad = False


        for p in self.modality_1_model.fc.parameters():
            p.requires_grad = True
        for p in self.modality_2_model.fc.parameters():
            p.requires_grad = True


        self.fuse_weight_1 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.fuse_weight_1.data.fill_(1)
        self.fuse_weight_2.data.fill_(0)
        self.count = 0

    def forward(self, modality_1_batch, modality_2_batch):
        dropout_out_1, normal_out_1 = self.modality_1_model(modality_1_batch)
        dropout_out_2, normal_out_2 = self.modality_2_model(modality_2_batch)


        pred = normal_out_1 * self.fuse_weight_1 + normal_out_2 * self.fuse_weight_2
        self.count += 1
        if self.count == 640:
            print(self.fuse_weight_1.cpu().detach().numpy(), self.fuse_weight_2.cpu().detach().numpy())
            self.count = 0
        return pred


class HSI_Lidar_Couple_Cross_TRI(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()
        self.hsi_block_1 = nn.Sequential(
                                         nn.Conv2d(modality_1_channel, 32, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         # nn.MaxPool2d(kernel_size=2),

                                         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         )

        self.lidar_block_1 = nn.Sequential(
                                           nn.Conv2d(modality_2_channel, 32, kernel_size=3, stride=1, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(32),
                                           nn.ReLU(),
                                           # nn.MaxPool2d(kernel_size=2),
                                           nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(),
                                           )

        self.dsm_block_1 = nn.Sequential(
                                         nn.Conv2d(modality_3_channel, 32, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         # nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         )

        self.hsi_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.lidar_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(),
                                           )

        self.dsm_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.share_bone = MDMB_fusion(384, args.class_num)

    def forward(self, hsi, lidar, dsm):
        hsi = self.hsi_block_1(hsi)
        lidar = self.lidar_block_1(lidar)
        dsm = self.dsm_block_1(dsm)

        x_hsi = self.hsi_block_2(hsi)
        x_lidar = self.lidar_block_2(lidar)
        x_dsm = self.dsm_block_2(dsm)

        x_hsi_lidar = self.lidar_block_2(hsi)
        x_hsi_dsm = self.dsm_block_2(hsi)

        x_lidar_hsi = self.hsi_block_2(lidar)
        x_lidar_dsm = self.dsm_block_2(lidar)

        x_dsm_hsi = self.hsi_block_2(dsm)
        x_dsm_lidar = self.lidar_block_2(dsm)

        joint_1 = torch.cat(((x_hsi + x_lidar_hsi + x_dsm_hsi) / 3, (x_lidar + x_hsi_lidar + x_dsm_lidar) / 3,
                             (x_dsm + x_hsi_dsm + x_lidar_dsm) / 3),
                            dim=1)
        joint_2 = torch.cat((x_hsi, x_hsi_lidar, x_hsi_dsm), dim=1)
        joint_3 = torch.cat((x_lidar_hsi, x_lidar, x_dsm), dim=1)
        #
        # joint_1 = torch.cat((x_hsi, x_lidar_hsi), dim=1)
        # joint_2 = torch.cat((x_hsi_lidar, x_lidar), dim=1)
        # joint_3 = torch.cat((x_hsi + x_lidar_hsi, x_lidar + x_hsi_lidar), dim=1)

        x1 = self.share_bone(joint_1)
        x2 = self.share_bone(joint_2)
        x3 = self.share_bone(joint_3)
        # x2 = joint_2
        # x3 = joint_3
        # x = (x1 + x2 + x3) / 3
        return x1


class HSI_Lidar_Couple_Cross_TRI_DAD(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()
        self.hsi_block_1 = nn.Sequential(
                                         nn.Conv2d(modality_1_channel, 32, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),

                                         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         )

        self.lidar_block_1 = nn.Sequential(
                                           nn.Conv2d(modality_2_channel, 32, kernel_size=3, stride=1, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(32),
                                           nn.ReLU(),
                                           nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(),
                                           )

        self.dsm_block_1 = nn.Sequential(
                                         nn.Conv2d(modality_3_channel, 32, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         )

        self.hsi_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.lidar_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(),
                                           )

        self.dsm_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.share_bone = MDMB_fusion_dad(384, args.class_num)

    def forward(self, hsi, lidar, dsm):
        hsi = self.hsi_block_1(hsi)
        lidar = self.lidar_block_1(lidar)
        dsm = self.dsm_block_1(dsm)

        x_hsi = self.hsi_block_2(hsi)
        x_lidar = self.lidar_block_2(lidar)
        x_dsm = self.dsm_block_2(dsm)

        x_hsi_lidar = self.lidar_block_2(hsi)
        x_hsi_dsm = self.dsm_block_2(hsi)

        x_lidar_hsi = self.hsi_block_2(lidar)
        x_lidar_dsm = self.dsm_block_2(lidar)

        x_dsm_hsi = self.hsi_block_2(dsm)
        x_dsm_lidar = self.lidar_block_2(dsm)

        joint_1 = torch.cat(((x_hsi + x_lidar_hsi + x_dsm_hsi) / 3, (x_lidar + x_hsi_lidar + x_dsm_lidar) / 3,
                             (x_dsm + x_hsi_dsm + x_lidar_dsm) / 3),
                            dim=1)

        x1, x_feature = self.share_bone(joint_1)
        return x1, x_feature

