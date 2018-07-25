import torch
from torch.nn import functional as F
import torchvision.models as models
from torchvision import transforms

import argparse
from PIL import Image
import numpy as np
from bn_fusion import fuse_bn_recursively
from utils import convert_resnet_family
import pretrainedmodels
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vgg16_bn')
    args = parser.parse_args()

    trf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           ])

    img = trf(Image.open('dog.jpg')).unsqueeze(0)

    try:
        net = getattr(models, args.model)(pretrained=True)
    except:
        net = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')



    if 'resnet' in args.model:
        se = True if 'se' in args.model else False
        net = convert_resnet_family(net, se)

    # Benchmarking
    # First, we run the network the way it is
    net.eval()
    with torch.no_grad():
        F.softmax(net(img), 1)
    # Measuring non-optimized model performance
    times = []
    for i in range(50):
        start = time.time()
        with torch.no_grad():
            res_0 = F.softmax(net(img), 1)
        times.append(time.time() - start)

    print('Non fused takes', np.mean(times), 'seconds')

    net = fuse_bn_recursively(net)
    net.eval()
    times = []
    for i in range(50):
        start = time.time()
        with torch.no_grad():
            res_1 = F.softmax(net(img), 1)
        times.append(time.time() - start)

    print('Fused takes', np.mean(times), 'seconds')

    diff = res_0 - res_1
    print('L2 Norm of the element-wise difference:', diff.norm().item())
