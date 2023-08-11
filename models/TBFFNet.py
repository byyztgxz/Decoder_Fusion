import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from utils.misc import initialize_weights


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        for n, m in self.resnet.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.resnet.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.conv = nn.Sequential(nn.Conv2d(512+256+128, 128, 1), nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.conv)

    def forward(self, x):
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        xm = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(xm)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x4 = torch.cat([x2, x3, x4], 1)
        x4 = self.conv(x4)
        return x0, x1, x4


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = self._make_layers_(in_channels, out_channels)
        self.cb = ConvBlock(out_channels)

    def _make_layers_(self, in_channels, out_channels, blocks=2, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels))
        layers = [ResBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = self.cb(x)
        return x


class LCMDecoder(nn.Module):
    def __init__(self):
        super(LCMDecoder, self).__init__()
        self.db4_1 = DecoderBlock(128+64, 128)
        self.db1_0 = DecoderBlock(128+64, 128)

    def decode(self, x1, x2, db):
        x1 = F.upsample(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x1, x2], 1)
        x = db(x)
        return x

    def forward(self, x0, x1, x4):
        x1 = self.decode(x4, x1, self.db4_1)
        x0 = self.decode(x1, x0, self.db1_0)
        return x0, x1


class CDBranch(nn.Module):
    def __init__(self):
        super(CDBranch, self).__init__()
        self.db4 = DecoderBlock(256+128, 128)
        self.db1 = DecoderBlock(256, 128)
        self.db0 = DecoderBlock(256, 128)

    def decode(self, x1, x2, db):
        x1 = db(x1)
        x1 = F.upsample(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x1, x2], 1)
        return x

    def forward(self, x0, x1, x1_4, x2_4):
        x4 = torch.cat([x1_4, x2_4, torch.abs(x1_4 - x2_4)], 1)
        x1 = self.decode(x4, x1, self.db4)
        x0 = self.decode(x1, x0, self.db1)
        x0 = self.db0(x0)
        return x0


class TBFFNet(nn.Module):
    def __init__(self, channels=3, num_classes=7):
        super(TBFFNet, self).__init__()
        self.encoder = Encoder()
        self.lcm_decoder = LCMDecoder()
        self.cd_branch = CDBranch()
        self.lcm_classifier1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                             nn.Conv2d(64, num_classes, kernel_size=1))
        self.lcm_classifier2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                             nn.Conv2d(64, num_classes, kernel_size=1))
        self.cd_classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                           nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights(self.lcm_decoder, self.cd_branch, self.lcm_classifier1, self.lcm_classifier2, \
                           self.cd_classifier)

    def forward(self, x1, x2):
        x_size = x1.size()

        x1_0, x1_1, x1_4 = self.encoder(x1)
        x2_0, x2_1, x2_4 = self.encoder(x2)

        x1_0, x1_1 = self.lcm_decoder(x1_0, x1_1, x1_4)
        x2_0, x2_1 = self.lcm_decoder(x2_0, x2_1, x2_4)

        cd_map = self.cd_branch(torch.abs(x1_0 - x2_0), torch.abs(x1_1 - x2_1), x1_4, x2_4)
        change = self.cd_classifier(cd_map)

        out1 = self.lcm_classifier1(x1_0)
        out2 = self.lcm_classifier2(x2_0)

        return F.upsample(change, x_size[2:], mode='bilinear'), \
               F.upsample(out1, x_size[2:], mode='bilinear'), \
               F.upsample(out2, x_size[2:], mode='bilinear')