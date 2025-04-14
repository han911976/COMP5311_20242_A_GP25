#!/usr/bin/env bash

pip3 install --user -r .devcontainer/requirements.txt
pip3 install --user -r .devcontainer/requirements-dev.txt
pip3 install torch==1.10.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip3 install torch==1.10.2 torchvision==1.10.2 torchaudio==1.10.2 --index-url https://download.pytorch.org/whl/cpu
