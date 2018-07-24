import torch
from torch.nn import functional as F
import torchvision.models as models
from torchvision import transforms

import argparse
from PIL import Image

from bn_fusion import fuse_bn_sequential
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

    net = getattr(models, args.model)(pretrained=True)
    net.eval()

    start = time.time()
    with torch.no_grad():
        res_0 = F.softmax(net(img), 1)

    print('Non fused takes', time.time()-start, 'seconds')

    net.features = fuse_bn_sequential(net.features)
    start = time.time()
    with torch.no_grad():
        res_1 = F.softmax(net(img), 1)
    print('Fused takes', time.time()-start, 'seconds')

    diff = res_0 - res_1
    print('L2 Norm of the element-wise difference:', diff.norm().item())


