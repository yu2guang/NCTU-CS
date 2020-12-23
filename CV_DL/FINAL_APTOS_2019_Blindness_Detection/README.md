# FINAL: APTOS 2019 Blindness Detection

## Hardware

The following specs were used to create the original solution.

- Ubuntu 18.04.5 LTS
- Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz
- GeForce GTX 1080 Ti

## Installation

All requirements are detailed in requirements.txt. 

```bash=
$ pip install -r requirements.txt
```

## Dataset Preparation

The data directory is structured as:

```
data
  +- train_images
  | | train images
  +- test_images
  | | test images
  +- cv2_img
  | | preprocessed train images
  | train.csv
  | test.csv
```

### 1. Download Official Image

If the Kaggle API is installed, run following command.

```bash=
$ kaggle competitions download -c aptos2019-blindness-detection
```

### 2. Data Preprocess

To preprocess the train images, run following command: 

```bash=
$ python data_process.py --src_path ./data/train_images/ 
                         --src_csv ./data/train.csv 
                         --saved_path ./data/cv2_img/ 
                         --img_size 256 --n_cpu 12 --batch_size 64
```

In order to speed up training, the preprocessed images will be saved at `./data/cv2_img/` and will be used directly while training.


## Train

### ResNet/EfficientNet

To train a ResNet/EfficientNet regression/classification model, change `model_type` in  `train.cfg` as one of the followings: 

```bash=
model_type=[ResNet, reg]
model_type=[ResNet, class]
model_type=[EfficientNet, reg]
model_type=[EfficientNet, class]
```

And then run the command:

```bash=
$ python train.py --cfg train.cfg
```

The terminal will show the training process as below:

```bash
# config file path
Namespace(cfg='train.cfg')  
# hyperparameters
{'model_type': ['EfficientNet', 'reg'], 'n_cpu': 12, 'epoch': 50, 'batch': 16, 'img_size': 256, 'lr': 0.001, 'w_decay': 1e-05, 'random_seed': 1234, 'valid_proportion': 0.1, 'src_path': 'final/data/', 'pretrain_path': 'None', 'saved_path': 'saved/EfficientNet_reg'}
# training process for every epoch
Epoch 1/50 	 loss=1.5267 	 train_qk=0.3342 	val_loss=2.8545 	 qk=0.0000 	 time=90.77s
Epoch 2/50 	 loss=1.0797 	 train_qk=0.5380 	val_loss=2.4479 	 qk=0.0000 	 time=91.74s
Epoch 3/50 	 loss=0.8517 	 train_qk=0.6610 	val_loss=0.9399 	 qk=0.6304 	 time=92.09s
# highest/every 5 epoch weight saved path info
Save weight: saved/EfficientNet_reg/weight_3_6304.pt
```

The weights and training process (`history.csv`) will be automatically saved at `saved/{ResNet/EfficientNet}_{reg/class}` and the structure is the following:

```
saved
  +- ResNet_reg
  |  weights
  |  history.csv
  +- ResNet_class
  |  weights
  |  history.csv
  +- EfficientNet_reg
  |  weights
  |  history.csv
  +- EfficientNet_class
  |  weights
  |  history.csv
```

The training data distribution and process could be ploted by:

```bash=
$ python plot.py --src_path ./saved/
                 --data_info_path ./data/train.csv
                 --plot_all_files True
                 --history_list ['ResNet_reg']
```

The image will be automatically saved at `src_path`.

### 11th Place Solution

`Reference github`:https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection
The qk score of 11th place soultion of this competition is about 0.93.
To reproduce 11th method, clone the reference github, and follow the README.md and start training first-level models.
After getting the weights of 1st-level-model weight, run the following notebook ,`11th-solution-seconde-stage-train.ipynb`, `11th-solution-submission.ipynb`,to do 2nd-level model training and inference.
The structure of the data directory should be modified as:
```
data
  +- train_images
  | | train images
  +- test_images
  | | test images
  +- cv2_img
  | | preprocessed train images
  +- images_288_scaled
  | | preprocessed train images from 11th place solution
  | | automatically save when training 2nd-level-model
  +- images_test_288_scaled
  | | preprocessed test images from 11th place solution
  | | automatically save when training 2nd-level-model
  | train.csv
  | test.csv
```
Before training 2nd-level-model, save the 1st-level-model weights at `saved/{se_resnext101_32x4d_pre2015/se_resnext101_32x4d_pre2015/se_resnext50_32x4d_pre2015}` 
When 2nd-level-model training, the weights will be automatically saved at `saved/{se_resnext101_32x4d/se_resnext101_32x4d/se_resnext50_32x4d}` 

```
saved
  +- 
  |  weights
  |  history.csv
  +- se_resnext50_32x4d_pre2015
  |  1st-level-model: model_{fold_num}.pth 
  +- se_resnext101_32x4d_pre2015
  |  1st-level-model: model_{fold_num}.pth 
  +- senet154_pre2015
  |  1st-level-model: model_{fold_num}.pth 
  |
  +- se_resnext50_32x4d
  |  2nd-level-model: model_{fold_num}.pth 
  +- se_resnext101_32x4d
  |  2nd-level-model: model_{fold_num}.pth 
  
```
#### 2nd-level-model
To train 2nd-level-model, run this notebook `11th-solution-seconde-stage-train.ipynb`

## Submission

### ResNet/EfficientNet

To submit to the kaggle competition, create a new notebook on competition website and upload `aptos-final.ipynb.ipynb`.
On the kaggle notebook, it is necessary to add `EfficientNet-pytorch` library into `output/` directory.

### 11th Place Solution

To submit to the kaggle competition, create a new notebook on competition website and upload `11th-solution-submission.ipynb` and two weight folder `se_resnext50_32x4d/`, `se_resnext101_32x4d/`.
On the kaggle notebook, it is necessary to add `pretrainedmodels` library into `input/` directory.
