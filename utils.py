import torch.nn as nn
from torchvision.models import resnet as resnet_modules


def convert_resnet(model):
    """
    This function wraps
    :param model: nn.Sequential
    :return: nn.Sequential
    """

    layer0 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool
    )


    features = [layer0, ]
    for ind in range(1, 5):
        modules_layer = model._modules[f'layer{ind}']._modules
        new_modules = []
        for block_name in modules_layer:
            b = modules_layer[block_name]
            if isinstance(b, resnet_modules.BasicBlock):
                b = BasicBlock(b)
            if isinstance(b, resnet_modules.Bottleneck):
                b = Bottleneck(b)
            new_modules.append(b)
        features.append(nn.Sequential(*new_modules))

    features = nn.Sequential(*features)
    classifier = nn.Sequential(model.fc)

    return Net(features, classifier)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, source_block):
        super(BasicBlock, self).__init__()
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, source_block):
        super(Bottleneck, self).__init__()
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

    def forward(self, x):
        residual = x

        out = self.relu(self.block1(x))
        out = self.relu(self.block2(out))
        out = self.block3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
