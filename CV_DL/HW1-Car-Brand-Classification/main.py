import os, time, torch, sys, warnings
from itertools import count
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data

from dataloader import data_evaluation, dataLoader
from models import ResNet, lr_decay

def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # --- hyper parameters --- #
    BATCH_SIZE = 256
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    N_layer = 18
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- data process --- #
    # info
    src_path = './data/'
    target_path = './saved/ResNet18/'
    model_path = target_path + 'pkls/'
    pred_path = target_path + 'preds/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    # evaluation: num of classify labels & image size
    # output testing id csv
    label2num_dict, num2label_dict = data_evaluation(src_path)

    # load
    train_data = dataLoader(src_path, 'train', label2num_dict)
    train_len = len(train_data)
    test_data = dataLoader(src_path, 'test')

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
    )

    # --- model training --- #
    # fp: for storing data
    fp_train_acc = open(target_path + 'train_acc.txt', 'w')
    fp_time = open(target_path + 'time.txt', 'w')

    # train
    highest_acc, train_acc_seq = 0, []
    loss_funct = nn.CrossEntropyLoss()
    net = ResNet(N_layer).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    print(net)

    for epoch_i in count(1):
        right_count = 0

        # print('\nTraining epoch {}...'.format(epoch_i))
        # for batch_x, batch_y in tqdm(train_loader):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # clear gradient
            optimizer.zero_grad()

            # forward & backward
            output = net.forward(batch_x.float())
            highest_out = torch.max(output, 1)[1]
            right_count += sum(batch_y == highest_out).item()

            loss = loss_funct(output, batch_y)
            loss.backward()

            # update parameters
            optimizer.step()

        # calculate accuracy
        train_acc = right_count / train_len
        train_acc_seq.append(train_acc * 100)

        if train_acc > highest_acc:
            highest_acc = train_acc

        # save model
        torch.save(net.state_dict(),
                   '{}{}_{}_{}.pkl'.format(model_path, target_path.split('/')[2], round(train_acc * 1000), epoch_i))

        # write data
        fp_train_acc.write(str(train_acc * 100) + '\n')
        fp_time.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '\n')
        print('\n{} Epoch {}, Training accuracy: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_i, train_acc))

        # test
        net.eval()
        test_df = pd.read_csv(src_path + 'testing_data/testing_labels.csv')
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(test_loader):
                batch_x = batch_x.to(device)
                output = net.forward(batch_x.float())
                highest_out = torch.max(output, 1)[1].cpu()
                labels = [num2label_dict[out_j.item()] for out_j in highest_out]
                test_df['label'].iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = labels
        test_df.to_csv('{}{}_{}_{}.csv'.format(pred_path, target_path.split('/')[2], round(train_acc * 1000), epoch_i), index=False)
        net.train()

        lr_decay(optimizer)

    fp_train_acc.close()
    fp_time.close()

if __name__ == '__main__':
    main()
