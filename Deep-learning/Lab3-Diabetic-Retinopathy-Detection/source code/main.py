import torch as t
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm
import time

from dataloader import RetinopathyLoader as dload
import models
import plot


if __name__ == '__main__':

    # Hyper Parameter
    BATCH_SIZE = 18
    LR = 1e-3
    EPOCH= [10, 10]
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    loss_funct = nn.CrossEntropyLoss()

    x1, x2, xd, y1, y2, yd = 0, 10, 2, 72, 92, 2
    title = 'Result Comparison(ResNet'

    # load data
    train_data = dload('./data/','train')
    train_len = dload.__len__(train_data)
    test_data = dload('./data/', 'test')
    test_len = dload.__len__(test_data)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
    )

    resnet = [50, 18]
    pretrain = [True, False]
    hi_name = ['', '', '', '']
    highest_accur = [[0, 0], [0, 0], [0, 0], [0, 0]]
    j = 0
    for layer in resnet:
        plot.show_result_start(title+str(layer)+')', x1, x2, xd, y1, y2, yd)
        for p_i in pretrain:
            print('~~~Resnet%d:'%layer, p_i, '~~~')
            if(layer==18):
                TOTAL_EPOCH = EPOCH[0]
            else:
                TOTAL_EPOCH = EPOCH[1]

            net = models.ResNet(layer, p_i).cuda() if (p_i == True) else models.ResNet_no(layer).cuda()

            optimizer = t.optim.SGD(
                net.parameters(),
                lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
            )

            # train
            train_accuracy = []
            test_accuracy = []
            for epoch_i in range(1, TOTAL_EPOCH+1):
                print('epoch', epoch_i)
                print('START:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                right = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()

                    # clear gradient
                    optimizer.zero_grad()

                    # forward + backward
                    output = net.forward(batch_x.float())
                    y_o = t.max(output, 1)[1]
                    y_hat = sum(batch_y == y_o).item()
                    right += y_hat

                    loss = loss_funct(output, batch_y)
                    loss.backward()

                    # update parameters
                    optimizer.step()

                accuracy = right/train_len
                train_accuracy.append(accuracy * 100)
                print('epoch ', epoch_i, ', loss:', loss.item(), ', accuracy:', accuracy)

                # test
                right = 0
                net.eval()
                with t.no_grad():
                    for batch_x, batch_y in test_loader:
                    #for step, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()
                        test_out = net.forward(batch_x.float())
                        test_out = t.max(test_out, 1)[1]
                        test_hat = sum(batch_y==test_out).item()
                        right += test_hat
                net.train()

                accuracy = right / test_len
                if(accuracy > highest_accur[j][1]):
                    hi_name[j] = "resNet{layer}_{pre}_{e}".format(layer=layer, pre=p_i, e=epoch_i)
                    highest_accur[j][0] = epoch_i
                    highest_accur[j][1] = accuracy
                    if(accuracy>=0.8):
                        t.save(net.state_dict(), "resNet{layer}_{pre}_{e}.pkl".format(layer=layer, pre=p_i, e=epoch_i))
                test_accuracy.append(accuracy * 100)
                print('Test accuracy:', accuracy)

                # decay lr
                models.lr_decay(optimizer)

                print('END:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'\n')

            print("{name}: {acc}".format(name=hi_name[j], acc=highest_accur[j][1]))

            plot.plot_line(TOTAL_EPOCH, train_accuracy, 'train', p_i)
            plot.plot_line(TOTAL_EPOCH, test_accuracy, 'test', p_i)

            j += 1

        plot.show_result_end(layer)


