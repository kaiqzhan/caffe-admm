#!/usr/bin/env sh
set -e

# ADMM. Use pretrained model
./build/tools/caffe train --solver=examples/mnist/lenet_solver_admm.prototxt --weights=examples/mnist/lenet.caffemodel $@

# retrain
./build/tools/caffe train --solver=examples/mnist/lenet_solver_retrain.prototxt --weights=examples/mnist/lenet_admm_lr01.caffemodel $@
