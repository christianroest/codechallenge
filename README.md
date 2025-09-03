# Machine Vision Challenge
## Overview
This repository contains the files used for training a segmentatic segmentation model for the SAR-RARP50 challenge.

## Description of methods
The data is modelled using a U-Net model, containing skip connections between layers at equal levels in the contracting and expanding paths.
Data are split at video-level, into training (90%) and validation datasets (10%).
The training data are used to optimize the model, while the validation split is held out, and used to track validation performance.
The model with the highest validation performance is retained and used as the final model.

The implementation is built using TensorFlow 2 in Python 3.

## Usage
### Requirements
The below requirements specify the exact versions of libraries used during development.
Other versions may be compatible, but have not been explored.

```
tensorflow==2.11.0
matpotlib==3.5.3
numpy==1.22.3
six==1.17.0
opencv-python==4.10.0.84
```

A GPU with sufficient VRAM is required to run the model with the parameters and batch size used during development.
The code was tested on 32 GB Tesla V100 and 40GB NVidia L40S.

It is recommended to install the packages in a Python virtual environment:

```
python3 -m venv .env
source .env/bin/activate
```