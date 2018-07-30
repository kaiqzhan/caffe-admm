# Installation

Same as the original version of Caffe.

See http://caffe.berkeleyvision.org/installation.html for the latest
installation instructions.

## How to train

Two steps are required to prune a neural network. Suppose we have a pretrained model "bvlc_alexnet.caffemodel".

Step 1: admm (suppress target weights)
```
./build/tools/caffe train --solver=models/bvlc_alexnet/solver_admm.prototxt --weights=/path/to/bvlc_alexnet.caffemodel
```

Step 2: retrain (set target weights to 0 and retrain)
```
./build/tools/caffe train --solver=models/bvlc_alexnet/solver_retrain.prototxt --weights=models/bvlc_alexnet/caffe_alexnet_train_admm_iter_2400000.caffemodel
```
Suppose `caffe_alexnet_train_admm_iter_2400000.caffemodel` is the trained caffemodel from step 1.

## Explanation of parameters
### solver prototxt

- **lr_policy**: "admm" for admm step; your prefered policy for retrain step
- **gamma**: drop the learning rate by a factor of this value
- **stepvalue**: drop the learning rate by gamma when meet this stepvalue in each admm iteration
- **admm_iter**: Caffe training iterations for each admm iteration
- **max_iter**: Total training iterations (total admm iterations = max_iter / admm_iter)
- **pruning_phase**: "admm" for admm step and "retrain" for retrain step

### model prototxt

- **prune_ratio**: the percent of weight parameters to prune for each layer
- **rho**: the rho parameter for admm pruning (detailed in our paper)

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
