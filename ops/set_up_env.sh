#!/bin/bash
set -e

# Install specific versions of torch and torchvision compatible with CUDA 13.0
uv pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130

# Install the required Python packages
uv pip install -r requirements.txt
