#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

echo "Installing pearl requirements..."
pip install --no-input --upgrade setuptools --no-user
conda install --yes swig
pip install --no-input  gymnasium[mujoco,atari]==1.0.0a2 moviepy opencv-python-headless matplotlib mujoco torch torchvision torchaudio --no-user
pip install git+https://github.com/AmiiThinks/AlphaEx.git --no-user
