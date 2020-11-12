# HW1: Car Brand Classification

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

## Prepare Images

The data directory is structured as:

```
data
  +- training_data
  | images
  | training_labels.csv
  +- testing_data
  | images
  | testing_labels.csv
```

### Download Official Image

Download and extract train.zip and test.zip. If the Kaggle API is installed, run following command.

```bash=
$ kaggle competitions download -c cs-t0828-2020-hw1
$ unzip cs-t0828-2020-hw1.zip
$ mkdir data
$ cp -r training_data/training_data/ data/
$ cp training_labels.csv data/training_data/
$ cp -r testing_data/testing_data/ data/
$ rm -rf cs-t0828-2020-hw1.zip training_data/ training_labels.csv testing_data/
```

## Training

To train models, run following commands.

```bash=
$ train main.py
```

The results from every epoch will be automatically saved at `./saved/` and the structure is the following:

```
saved
  +- ResNet18
  |  +- pkls
     | checkpoints
  |  +- preds
     | submissions
  | train_acc.txt
  | time.txt

```