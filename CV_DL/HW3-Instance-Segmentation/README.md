# HW3: Instance Segmentation

## Reference

- TA provided Github: https://github.com/NCTU-VRDL/CS_IOC5008/tree/master/HW4
- Torchvision Models: https://pytorch.org/docs/stable/torchvision/models.html

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
    | training images
  +- test_images
    | testing images
  | train.json
  | test.json
```

## Train

To train models, run following commands:

```bash=
$ python train.py --cfg train.cfg
```

The weights will be automatically saved at `saved/mRCNN/pkls/` and the structure is the following:

```
saved
  +- mRCNN
  |  +- pkls
  |  | weights
```

## Test

To test models, run following commands:

```bash=
$ python test.py --cfg test.cfg
```

The detected images & `maskRCNN_{$epoch}.json` file will be automatically saved at `saved/mRCNN/results/` (same saved dir as `Train`) and the structure is the following:

```
saved
  +- mRCNN
  |  +- pkls
  |  | weights
  |  +- results
  |  |  +- imgs
  |  |  | detected images
  |  | json file       
```

## Evaluate

To evaluate models, run following commands:

```bash=
$ python evaluate.py --truth dataset/train.json
                     --submit saved/mRCNN_0/results/maskRCNN_100_train.json
```

And it will take some time to show the APs.