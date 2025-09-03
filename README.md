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
`tf.data.AUTOTUNE` is used to optimize performance at runtime depending on the hardware.

It is recommended to install the packages in a Python virtual environment:

```
python3 -m venv .env
source .env/bin/activate
```

## Preparing the dataset
This description applies to the SAR-RARP50 dataset, but the same instructions apply to other datasets following the same data structure.
Download the training dataset to a location on your filesystem. 
Extract each zip file, such that the file structure look as follows:
```
[training_data_root]
|- video_01
 |- segmentations
  |- 000000000.png
  |- 000000060.png
  |- 000000120.png
  |- [...]
 |- video_left.avi
 |- [...]
```

Next, run the following command to start sampling PNG files from the video files at 1HZ, which will serve as the input images. 
```
python scripts/sample_video.py [training_data_root] -f 1 -r
```

This will create a subfolder `rgb` in each of the video directories, containing images matching the segmentations.
After this process completes, your data folder should look as follows:
```
[training_data_root]
|- video_01
 |- segmentations
  |- 000000000.png
  |- 000000060.png
  |- 000000120.png
  |- [...]
 |- rgb
  |- 000000000.png
  |- 000000060.png
  |- 000000120.png
  |- [...]
 |- video_left.avi
 |- [...]
```

## Training a model
Train a model by loading the environment, and calling `main.py` as follows:
```
python main.py -r [train_data_root] -w [work_dir]
```
Where, `work_dir` is a directory where the output of the run can be stored (trained models and sample PNGs)

## Inference
Predict on new images by loading the environment, and calling `predict.py` as follows:
```
python predict.py -r [test_data_root] -m [path_to_model_1] [path_to_model_2] ...
```
This looks for subdirectories in `test_data_root` containing the `rgb/` folder.
Predictions (segmentation masks) are exported as PNGs to a new `predictions/` folder, adjacent to the existing `rgb/` folder.