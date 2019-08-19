import torch.nn as nn

# parameters
PRE_YET = True
EPOCH = 100
COND = ['sp', 'tp', 'pg', 'p']
COND_EMBEDDING_SIZE = 8
VOCAB_DICT_SIZE = 28
HIDDEN_SIZE = 256
LATENT_SIZE = 32
LR_DECAY = 0.95
NUM = 2
CRITERATION = nn.CrossEntropyLoss().cuda()
TEACHING_FORCE_RATIO = 1.0  # -=0.025, [0,0.5]
LR = 0.001

KL_W = 0.0  # +=0.0005, [0,0.5]



