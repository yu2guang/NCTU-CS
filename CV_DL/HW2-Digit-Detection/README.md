# HW2: Digits detection
## Reference

- YOLOv5 Github: https://github.com/ultralytics/yolov5

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
  +- images
  | +- train
    | training images
  | +- valid
    | testing images
  +- labels
    +- train
    | annotation files 
  | training_info.csv
  | digitStruct.mat
  | see_bboxes.m
```

### Image Folder

Move the images of training dataset to `data/images/train/`.
Move the images of testing dataset to `data/images/valid/`.

### Construct Dataset

To construct dataset, run following command:

```python=
$ python data_process.py
```

And you will get parsed annotations (`data/training_info.csv`), and processed annotation files in `data/labels/train/`.

#### Annotation Folder

The dataloader expects that the annotation file corresponding to the image `data/images/train/train.jpg` has the path `data/labels/train/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`.

#### Create data.yaml

```yaml=
# train and val data
train: ./data/images/train/
val: ./data/images/train/
test: ./data/images/valid/

# number of classes
nc: 10

# class names
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
```

## Train

To train on the custom dataset run:

```bash=
$ python train.py --model_def config/yolov3-custom.cfg \
                  --train_config config/train_params.cfg
```

The weights and training results will be automatically saved at `saved/` and the structure is the following:

```
runs
  +- train
  |  +- exp*
     | +- weights
       | best.pt
       | last.pt
     | results.txt
```

The training results could be ploted by:

```bash=
$ python plot_train_result.py --save_path runs/train/exp3
```

## Detect

To detect the digit of the image:

```bash=
$ python detect.py --source data/images/valid/ \
    --weights runs/train/exp3/weights/best.pt --conf 0.3
```

The detected images & `sub.json` file will be automatically saved at `runs/detect/exp*` (same dir as Train) and the structure is the following:

```
runs
  +- detect
  |  +- exp*
        | detected images
        | sub.json
```