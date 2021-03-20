import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

# plot
def plot_line(n, y, mode, pre):
    x = [i for i in range(1,n+1)]

    if(mode=='test'):
        if(pre==False):
            plt.plot(x, y, color='blue', linewidth=1.0, marker='o', label='Test(w/o pretraining)')
        else:
            plt.plot(x, y, color='orange', linewidth=1.0, label='Test(with pretraining)')
    else:
        if(pre==False):
            plt.plot(x, y, color='green', linewidth=1.0, marker='o', label='Train(w/o pretraining)')
        else:
            plt.plot(x, y, color='red', linewidth=1.0, label='Train(with pretraining)')


def show_result_start(title, x1, x2, xd, y1, y2, yd):

    plt.figure()
    plt.title(title, fontsize=18)
    plt.xlim(x1-xd/4, x2+xd/4)
    plt.xticks(np.arange(x1, x2+xd/4, xd))
    plt.ylim(y1-yd/4, y2+yd/4)
    plt.yticks(np.arange(y1, y2+yd/4, yd))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

def show_result_end(layers):
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    plt.savefig('ResNet%d.png'%layers)

# confusion matrix
def plot_confusion_matrix(y_true, y_pred, name):

    plt.figure()

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, None]
    df_cm = pd.DataFrame(cm, range(5), range(5))
    sn.heatmap(df_cm, annot=True, cmap='GnBu')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(name)
    plt.show()
    plt.savefig(name)






