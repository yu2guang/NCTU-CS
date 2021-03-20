import torch.nn as nn
import torchvision.models

def lr_decay(optim):
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr']*0.99

class ResNet_pre(nn.Module):
    def __init__(self, layer, pretrained=True):
        super(ResNet_pre, self).__init__()

        if(layer==18):
            self.classify = nn.Linear(512, 15)
        else:
            self.classify = nn.Linear(2048, 15)

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
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x



