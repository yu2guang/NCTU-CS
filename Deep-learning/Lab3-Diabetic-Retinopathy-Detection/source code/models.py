import torch.nn as nn
import torchvision.models

def lr_decay(optim):
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr']*0.7

class Basic(nn.Module):
    def __init__(self, in_c, filter, stride=1, downsample=None):
        super(Basic, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, filter, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(filter),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(filter, filter, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter),
        )

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self, in_c, filter, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, filter, kernel_size=1, bias=False),
            nn.BatchNorm2d(filter)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(filter, filter, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(filter)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(filter, filter*4, kernel_size=1, bias=False),
            nn.BatchNorm2d(filter*4)
        )

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, layer, pretrained):
        super(ResNet, self).__init__()

        if(layer==18):
            self.classify = nn.Linear(512, 5)
        else:
            self.classify = nn.Linear(2048, 5)

        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layer)](pretrained=pretrained)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x


class ResNet_no(nn.Module):
    def __init__(self, layerN):
        super(ResNet_no, self).__init__()

        if (layerN == 18):
            self.classify = nn.Linear(512, 5)
            self.bas_bott_out = 1
            bas_bott = Basic
            blayers = [2, 2, 2, 2]
        else:
            self.classify = nn.Linear(2048, 5)
            self.bas_bott_out = 4
            bas_bott = Bottleneck
            blayers = [3, 4, 6, 3]

        self.in_c = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # to reduce memory
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._bBlock(bas_bott, blayers[0], 64)
        self.layer2 = self._bBlock(bas_bott, blayers[1], 128, stride=2)
        self.layer3 = self._bBlock(bas_bott, blayers[2], 256, stride=2)
        self.layer4 = self._bBlock(bas_bott, blayers[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _bBlock(self, BB, BB_n, planes, stride=1):

        downsample = None
        if (stride != 1) or (self.in_c != planes * self.bas_bott_out):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, planes * self.bas_bott_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.bas_bott_out),
            )

        layers = []
        layers.append(BB(self.in_c, planes, stride, downsample))
        self.in_c = planes * self.bas_bott_out
        for i in range(1, BB_n):
            layers.append(BB(self.in_c, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x


