import torch
import torch.nn as nn


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
