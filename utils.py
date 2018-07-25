import torch.nn as nn
from torchvision.models import resnet as resnet_modules
from pretrainedmodels.models import senet as senet_modules

class Net(nn.Module):
    def __init__(self, features, classifer):
        super(Net, self).__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifer

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out).view(x.size(0), -1)
        return self.classifier(out)


def convert_resnet_family(model, se=False):
    """
    This function wraps any (se)resnet model from torchvision or from pretrained models
    :param model: nn.Sequential
    :return: nn.Sequential
    """

    features = list()
    if not se:
        layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        features.append(layer0)
    else:
        features.append(model.layer0)

    for ind in range(1, 5):
        modules_layer = model._modules[f'layer{ind}']._modules
        new_modules = []
        for block_name in modules_layer:
            b = modules_layer[block_name]
            if isinstance(b, resnet_modules.BasicBlock):
                b = BasicResnetBlock(b)

            if isinstance(b, resnet_modules.Bottleneck) or \
                    isinstance(b, senet_modules.SEBottleneck) or \
                    isinstance(b, senet_modules.SEResNetBottleneck) or \
                    isinstance(b, senet_modules.SEResNeXtBottleneck):

                b = BottleneckResnetBlock(b, se)
            new_modules.append(b)
        features.append(nn.Sequential(*new_modules))

    features = nn.Sequential(*features)
    if not se:
        classifier = model.fc
    else:
        classifier = model.last_linear

    return Net(features, classifier)


class BasicResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, source_block):
        super(BasicResnetBlock, self).__init__()
        self.block1 = nn.Sequential(
            source_block.conv1,
            source_block.bn1
        )

        self.block2 = nn.Sequential(
            source_block.conv2,
            source_block.bn2
        )

        self.downsample = source_block.downsample
        self.stride = source_block.stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.block1(x))
        out = self.block2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckResnetBlock(nn.Module):
    expansion = 4

    def __init__(self, source_block, se=False):
        super(BottleneckResnetBlock, self).__init__()
        self.block1 = nn.Sequential(
            source_block.conv1,
            source_block.bn1,
        )

        self.block2 = nn.Sequential(
            source_block.conv2,
            source_block.bn2
        )

        self.block3 = nn.Sequential(
            source_block.conv3,
            source_block.bn3
        )
        self.relu = nn.ReLU(inplace=True)

        self.downsample = source_block.downsample
        self.stride = source_block.stride
        if se:
            self.se_module = source_block.se_module
        else:
            self.se_module = None

    def forward(self, x):
        residual = x

        out = self.relu(self.block1(x))
        out = self.relu(self.block2(out))
        out = self.block3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se_module is not None:
            out += self.se_module(out) + residual
        out = self.relu(out)

        return out
