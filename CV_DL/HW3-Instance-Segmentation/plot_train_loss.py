import matplotlib.pyplot as plt
import numpy as np
import os, argparse


def plt_process_loss(src_path, target_path):
    # read file
    fp = open(src_path, 'r')
    lines = fp.readlines()
    fp.close()

    start_i = 0
    while True:
        if lines[start_i].startswith('# --- '):
            break
        start_i += 1

    x_label_i, y, batch_num = [], [], 0
    tmp_loss = []
    for i in range(start_i, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        elif line.startswith('# --- '):
            x_label_i.append(len(y))
            if int(line.split(' ')[5].strip()) != 1:
                y.append(np.mean(tmp_loss))
                tmp_loss = []
        else:
            tmp_loss.append(float(line))
    y.append(np.mean(tmp_loss))

    x = [int(i) for i in range(len(y))]

    # plot
    fig = plt.figure()

    plt.plot(x, y, color='#005ab5')

    plt.title('Training Loss', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yticks(np.arange(int(round(y[0])), int(round(y[-1])), 10))

    fig.savefig(target_path)
    print(target_path)


if __name__ == '__main__':
    import sys, warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_path", type=str, default="saved/mRCNN_0_3/process.txt", help="path to the training process file")
    parser.add_argument("--img_path", type=str, default="saved/mRCNN_0_3/train_loss.jpg", help="path to the saved training loss image")
    opt = parser.parse_args()
    print(opt)
    
    plt_process_loss(opt.loss_path, opt.img_path)
