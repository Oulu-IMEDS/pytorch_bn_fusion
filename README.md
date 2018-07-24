# Batch Norm Fusion for Pytorch

## About 

In this repository, we present a simplistic implementation of batchnorm fusion for the most popular CNN archiectures. This is aimed to speed up the inference at the test time.

## How it works

We know that both - convolution and batchnorm are the linear operations, thus can be written in terms of matrix multiplications: ![T_{bn}*S{bn}*Conv_W*(x)](https://latex.codecogs.com/gif.latex?T_{bn}*S_{bn}*W_{conv}*x), where we first apply convolution to the data, scale it and eventually shift it using the batchnorm-trained parameters.

## Supported architectures

- [ ] VGG
- [ ] ResNet Family
- [ ] Dynamic UNet

## Acknowledgements

Thanks to [@ZFTurbo](https://github.com/ZFTurbo) for the idea and his [implementation for Keras](https://github.com/ZFTurbo/Keras-inference-time-optimizer).
