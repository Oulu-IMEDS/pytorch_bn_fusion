# Batch Norm Fusion for Pytorch

## About 

In this repository, we present a simplistic implementation of batchnorm fusion for the most popular CNN architectures in PyTorch. 
This package is aimed to speed up the inference at the test time: **expected boost is 30%!** In the future

## How it works

We know that both - convolution and batchnorm are the linear operations to the data point x, and they can be written in terms of matrix multiplications:
 ![T_{bn}*S{bn}*Conv_W*(x)](https://latex.codecogs.com/gif.latex?T_{bn}*S_{bn}*W_{conv}*x), 
 where we first apply convolution to the data, scale it and eventually shift it using the batchnorm-trained parameters.

## Supported architectures

We support any architecture, where Conv and BN are combined in a Sequential module. 
If you want to optimize your own networks with this tool, just follow this design. 
For the conveniece, we wrapped VGG, ResNet and SeNet families to demonstrate how your models can be converted into such format.

- [x] VGG from torchvision.
- [x] ResNet Family from `torchvision`.
- [x] SeNet family from `pretrainedmodels`

## How to use

```python
import torchvision.models as models
from bn_fusion import fuse_bn_recursively

net = getattr(models,'vgg16_bn')(pretrained=True)
net = fuse_bn_recursively(net)
net.eval()
# Make inference with the converted model
```
## TODO

- [ ] Tests.
- [ ] Performance benchmarks.

## Acknowledgements

Thanks to [@ZFTurbo](https://github.com/ZFTurbo) for the idea, discussions and his [implementation for Keras](https://github.com/ZFTurbo/Keras-inference-time-optimizer).
