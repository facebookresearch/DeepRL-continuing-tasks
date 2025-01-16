#!/usr/bin/env bash

# A script that installs miniconda3 in a 'local' directory under user's $HOME and installs the necessary packages for running Pearl.

echo "Activating conda environment pearl"
source /etc/bashrc
conda activate pearl || exit

active_env=$(conda info --envs | grep -E '\*' | awk '{print $1}')
echo "Active conda environment: $active_env"

echo "Installing pearl requirements..."
pip install --no-input --target ~/.conda/envs/pearl/lib/python3.10/site-packages/ --upgrade setuptools --no-user
conda install --yes swig
pip install --no-input  --target ~/.conda/envs/pearl/lib/python3.10/site-packages/ gymnasium[mujoco,atari,box2d]==0.28.1 moviepy opencv-python-headless matplotlib mujoco torch torchvision torchaudio --no-user
pip install --target ~/.conda/envs/pearl/lib/python3.10/site-packages/ git+https://github.com/AmiiThinks/AlphaEx.git --no-user

# We need a special version of fbgemm when there is no GPU
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "CUDA detected"
else
    echo "CUDA not detected -- installing CPU version of fbgemm"
    with-proxy pip uninstall fbgemm-gpu -y
    with-proxy pip install --no-input fbgemm-gpu-cpu
fi
