import torch
import torch.nn as nn


mdmb_seed = 7
couple_seed = 7


class WCRN(nn.Module):
    def __init__(self, num_classes=9):
        super(WCRN, self).__init__()

        self.conv1a = nn.Conv2d(103, 64, kernel_size=3, stride=1, padding=0)
        self.conv1b = nn.Conv2d(103, 64, kernel_size=1, stride=1, padding=0)
        self.maxp1 = nn.MaxPool2d(kernel_size=3)
        self.maxp2 = nn.MaxPool2d(kernel_size=5)

        self.bn1 = nn.BatchNorm2d(128)
        self.conv2a = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1a(x)
        out1 = self.conv1b(x)
        out = self.maxp1(out)
        out1 = self.maxp2(out1)

        out = torch.cat((out, out1), 1)

        out1 = self.bn1(out)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2a(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2b(out1)

        out = torch.add(out, out1)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out


class MDMB_extract(nn.Module):
    '''
    More Diverse Means Better: Multimodal Deep Learning Meets Remote Sensing Imagery Classificatio
    '''

    def __init__(self, input_channel):
        super(MDMB_extract, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.block4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )

        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class MDMB_fusion(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion, self).__init__()

        self.block_5 = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )

        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x = self.block_6(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x_dropout = x
        return x_dropout


class MDMB_fusion_baseline(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_baseline, self).__init__()

        self.block_5 = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )

        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x = self.block_6(x)
        # x = self.block_7(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # x = self.dropout(x)
        return x




class MDMB_fusion_late(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_late, self).__init__()

        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))
                                       )

        self.block_6 = nn.Sequential(nn.Conv2d(128, class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True),
                                     )
        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_hsi, x_lidar):
        x_hsi = self.block_5_1(x_hsi)
        x_lidar = self.block_5_2(x_lidar)
        x = torch.cat((x_hsi, x_lidar), dim=1)
        x = self.block_6(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        # x = self.dropout(x)
        return x


class MDMB_fusion_share(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_share, self).__init__()

        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))
                                       )

        self.block_6 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True),
                                     )
        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_hsi, x_lidar):
        x_hsi = self.block_5_1(x_hsi)
        x_lidar = self.block_5_1(x_lidar)

        x1 = self.block_6(x_hsi)
        x2 = self.block_6(x_lidar)
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x = (x1 + x2) / 2
        # x = self.fc(x)
        # x = self.dropout(x)
        return x


class CCR(nn.Module):
    '''
    Convolutional Neural Networks for Multimodal Remote Sensing Data Classification
    '''

    def __init__(self, input_channel, class_num):
        super(CCR, self).__init__()
        self.block_5 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(128),
                                     )

        self.block_8 = nn.Sequential(nn.Conv2d(128, input_channel, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(input_channel),
                                     )

        self.fc = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0,
                                          bias=True), )
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x_feature = self.block_6(x)
        x = self.fc(x_feature)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)

        x_rec = self.block_7(x_feature)
        x_rec = self.block_8(x_rec)
        return x, x_rec


class En_De(nn.Module):
    def __init__(self, input_channel, class_num):
        super(En_De, self).__init__()
        self.block_5 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     )
        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )
        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True),
                                     )


class Cross_Fusion(nn.Module):
    def __init__(self, input_channel, class_num):
        super(Cross_Fusion, self).__init__()

        self.block_5 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )


class Couple_CNN(nn.Module):
    def __init__(self, input_channel):
        super(Couple_CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1, padding_mode='replicate',
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate',
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='replicate',
                                              bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    # nn.MaxPool2d(kernel_size=2)
                                    )

        for m in self.modules():
            torch.manual_seed(couple_seed)
            torch.cuda.manual_seed(couple_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
