---
last_update: 2025-04-14
subject: COMP5311
---

# D-PACK Exp

This is a tiny README for our project's codebase.

> https://github.com/VincentLee077/D-PACK
> https://github.com/davidyslu/USTC-TFC2016

### Prerequisites

#### Data

`https://github.com/VincentLee077/D-PACK/tree/main/D-PACK/dataset`  
`https://drive.google.com/drive/u/0/folders/1LU9nm7ClK0IMOOTyJ_E4i3jVnOVy5bDM`

#### Environment

> https://stackoverflow.com/questions/78347799/why-my-anaconda-keeps-showing-error-while-loading-conda-entry-point

```
python -m pip uninstall anaconda-cloud-auth
conda update conda
conda install "anaconda-cloud-auth>0.5.0"
conda install “anaconda-cloud-auth>0.5.0”

conda install conda=24.11.1
```

Please check how to cuda, a non cuda version of code has also included.  
The following command is ran on COMP UDS machine for your reference:

```cmd
C:\Users\24g\AppData\Local\anaconda3\Scripts\activate

cd <your-code-dir>
conda create --name pytorch matplotlib pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate pytorch
pip install torchsummary  scikit-learn keras tensorflow seaborn
```

At local, devcontainer with python 3.6

requirement.txt
```
cycler==0.11.0
joblib==1.1.1
kiwisolver==1.3.1
matplotlib==3.3.4
numpy==1.19.5
pandas==1.1.5
Pillow==8.4.0
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==0.24.2
scipy==1.5.4
six==1.17.0
threadpoolctl==3.1.0
torch==1.4.0+cpu
torchaudio==0.4.0
torchsummary==1.5.1
torchvision==0.1.6
```

devcontainer.json
```
{
  "name": "Python 3",
  "image": "mcr.microsoft.com/devcontainers/python:3.6-bullseye",
  "remoteUser": "vscode",
  "postCreateCommand": "bash .devcontainer/post-create.sh",
  // "runArgs": ["--userns=keep-id"],
  "containerEnv": {
    "HOME": "/home/vscode"
    // "DISPLAY": "unix:0"
  },
  "customizations": {
    "vscode": {
      "settings": {},
      "extensions": [
        "ms-python.python",
        "ms-python.pylint",
        "ms-python.vscode-pylance"
      ]
    }
  }
}
```

#### CUDA

If you want to check the supportiveness of cuda in your PC, you may try to run `test_cuda.py`.

### Try the model

To try the D-PACK, please ensure you have download the preprocessed data from the github and put in relative folder before excuting the code.

```
python pycnn2.py
```

### Plot

The code of some of our plot has also included.

### Early cutoff

An early cutoff logic has added for testing the impact to the model.

### Appendix

#### model description
```py
CNN_AUTO(
  (conv1): Sequential(
    (0): Conv1d(1, 32, kernel_size=(6,), stride=(1,), padding=(5,))
    (1): ReLU()
    (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv1d(32, 64, kernel_size=(6,), stride=(1,), padding=(5,))
    (1): ReLU()
    (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense1): Linear(in_features=2752, out_features=1024, bias=True)
  (dense2): Linear(in_features=1024, out_features=25, bias=True)
  (cnn_out): Linear(in_features=25, out_features=10, bias=True)
  (encoder_1): Linear(in_features=1024, out_features=512, bias=True)
  (encoder_2): Linear(in_features=512, out_features=256, bias=True)
  (decoder_1): Linear(in_features=256, out_features=512, bias=True)
  (decoder_2): Linear(in_features=512, out_features=1024, bias=True)
)
```

#### Spec

CPU: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz  
RAM: 64GB  
GPU: NVIDIA GeForce RTX 3070  
python 3.12
