import torch
import torch.nn as nn
import torch.utils.data as Data
from itertools import count
import time, os

from dataloader import *
from models import *
from plot import *


if __name__ == '__main__':

    # Hyper Parameter
    BATCH_SIZE = 64
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    N_layer = 50
    loss_funct = nn.CrossEntropyLoss()

    x1, x2, xd, y1, y2, yd = 0, 10, 2, 72, 92, 2
    title = 'ResNet'+str(N_layer)

    # load data
    src_path = './hw5_data/'
    target_path = './'+title+'_2/'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    train_data = dataLoader(src_path,'train')
    train_len = len(train_data)
    test_data = dataLoader(src_path, 'test')
    test_len = len(test_data)

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

    highest_accuracy = 0
    show_result_start(title, x1, x2, xd, y1, y2, yd)
    net = ResNet_pre(N_layer).cuda()
    optimizer = t.optim.Adam(
        net.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    fp_train = open(target_path + 'train_acc_pre.txt', 'w')
    fp_test = open(target_path + 'test_acc_pre.txt', 'w')

    # train
    train_accuracy = []
    test_accuracy = []
    print('START:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    epoch_i = 0
    for epoch_i in count(1):
        right = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            # clear gradient
            optimizer.zero_grad()

            # forward + backward
            output = net.forward(batch_x.float())
            y_o = torch.max(output, 1)[1]
            y_hat = sum(batch_y == y_o).item()
            right += y_hat

            loss = loss_funct(output, batch_y)
            loss.backward()

            # update parameters
            optimizer.step()

        accuracy = right/train_len
        train_accuracy.append(accuracy * 100)
        fp_train.write(str(accuracy * 100) + '\n')
        print('\nEpoch {}, Train accuracy: {}'.format(epoch_i, accuracy))

        # test
        right = 0
        net.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
            #for step, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                test_out = net.forward(batch_x.float())
                test_out = torch.max(test_out, 1)[1]
                test_hat = sum(batch_y==test_out).item()
                right += test_hat
        net.train()

        accuracy = right / test_len
        if(accuracy > highest_accuracy):
            torch.save(net.state_dict(), target_path+title+'_{}.pkl'.format(epoch_i))
            highest_accuracy = accuracy
            if highest_accuracy == 1:
                break
        test_accuracy.append(accuracy * 100)
        fp_test.write(str(accuracy * 100) + '\n')
        print('Test accuracy:', accuracy)

        # decay lr
        # lr_decay(optimizer)

    print('END:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'\n')

    plot_line(epoch_i, train_accuracy, 'train')
    plot_line(epoch_i, test_accuracy, 'test')

    show_result_end(target_path, title)

    fp_train.close()
    fp_test.close()

