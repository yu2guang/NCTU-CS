import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

# plot
def plot_line(n, y, mode, pre=True):
    x = [i for i in range(1,n+1)]

    if(mode=='test'):
        if(pre==False):
            plt.plot(x, y, color='blue', linewidth=1.0, marker='o', label='Test(w/o pretraining)')
        else:
            plt.plot(x, y, color='orange', linewidth=1.0, label='Test')
    else:
        if(pre==False):
            plt.plot(x, y, color='green', linewidth=1.0, marker='o', label='Train(w/o pretraining)')
        else:
            plt.plot(x, y, color='blue', linewidth=1.0, label='Train')


def show_result_start(title, x1, x2, xd, y1, y2, yd):

    plt.figure()
    plt.title(title, fontsize=18)
    plt.xlim(x1-xd/4, x2+xd/4)
    plt.xticks(np.arange(x1, x2+xd/4, xd))
    plt.ylim(y1-yd/4, y2+yd/4)
    plt.yticks(np.arange(y1, y2+yd/4, yd))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

def show_result_end(target_path, title):
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    plt.savefig(target_path+title+'.png')

if __name__ == '__main__':
    N_layer = 50
    title = 'ResNet' + str(N_layer)
    target_path = './' + title + '_2/'

    train_accuracy = np.loadtxt(target_path + 'train_acc_pre.txt')
    test_accuracy = np.loadtxt(target_path + 'test_acc_pre.txt')

    x1, x2, xd, y1, y2, yd = 1, 100, 10, 65, 100, 5

    show_result_start(title, x1, x2, xd, y1, y2, yd)
    plot_line(x2, train_accuracy[:x2], 'train')
    plot_line(x2, test_accuracy[:x2], 'test')
    show_result_end(target_path, title)






