import torch.nn as nn

BATCH_SIZE = 64
LR_DIS = 1e-4
LR_GEN_Q = 1e-3
Z_SIZE = 54
C_SIZE = 10
BCE = nn.BCELoss()
CE = nn.CrossEntropyLoss()
EPOCH = 80
DATA_ROOT = './mnist/'
IMAGE_SIZE = 64
RESULT_FILE = './result/6'
PRINT_EVERY = 100

DOWNLOAD_MNIST = False