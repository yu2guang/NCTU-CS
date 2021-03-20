# HW4: Image Super-Resolution README

## Reference

- yjn870 Github: https://github.com/yjn870/SRCNN-pytorch

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
  +- training_hr_images
  | | training images
  +- testing_lr_images
  | | testing images
  | train.h5
  | train_eval.h5
```

You can use `prepare.py` to create custom dataset.

##### Train data
```bash=
$ python prepare.py --images-dir ./data/training_hr_images 
                    --output-path ./data/train.h5 
                    --patch-size 33 --stride 14 --scale 2
```

##### Evaluation data
```bash=
$ python prepare.py --images-dir ./data/training_hr_images 
                    --output-path ./data/train_eval.h5 
                    --patch-size 33 --stride 14 --scale 3 
                    --eval
```

## Train

To train models, run following commands:

```bash=
$ mkdir ./saved/SRCNN/
$ python -u train.py --train-file ./data/train.h5 
                     --eval-file ./data/train_eval.h5
                     --outputs-dir ./saved/SRCNN
                     --pretrain-file None --scale 2 --lr 1e-4
                     --batch-size 1024 --num-epochs 100
                     --num-workers 12 --seed 54760
                     | tee ./saved/SRCNN/process.txt
```

The weights and training process (i.e. loss) will be automatically saved at `saved/SRCNN/` and the structure is the following:

```
saved
  +- SRCNN
  |  +- pkls
  |  | weights
  |  process.txt
```

The training loss could be ploted by:

```bash=
$ python plot_train_loss.py --src_path ./saved/SRCNN/process.txt
```

The image will be automatically saved at `src_path`.

## Test

To test models, run following commands:

```bash=
$ python test.py --weights-file ./saved/SRCNN/epoch_x_xx.xx.pth
                 --images-dir ./data/testing_lr_images/
                 --outputs-dir ./saved/test_out/
                 --scale 3
```

The detected images file will be automatically saved at `images_dir`.