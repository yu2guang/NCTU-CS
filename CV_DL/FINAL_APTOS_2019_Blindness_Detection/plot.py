import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, argparse


def plt_train_process(src_path, plot_all_files, history_list=None):

    def plt_fig(dirs_name, df_col_name, title, x_label, y_label, save_path, label_loc='upper right'):
        fig = plt.figure()
        for path in dirs_name:
            df = pd.read_csv(f'{path}history.csv')
            x = df['epoch'].values
            plt.plot(x, df[df_col_name], label=f'{path.split("/")[-2]}')

        plt.title(title, fontsize=18)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if df_col_name == 'lr':
            plt.yticks([1e-3, 1e-4, 1e-5, 1e-6])
        plt.legend(loc=label_loc)
        fig.savefig(save_path)

    if plot_all_files:
        dirs_name = glob.glob(f'{src_path}*/')
    else:
        dirs_name = [src_path + name + '/' for name in history_list]

    plt_fig(dirs_name, 'train_loss', 'Train Loss', 'Epoch', 'Loss', f'{src_path}train_loss.jpg')
    plt_fig(dirs_name, 'val_loss', 'Valid Loss', 'Epoch', 'Loss', f'{src_path}valid_loss.jpg')
    plt_fig(dirs_name, 'train_qk', 'Train Quadratic Kappa', 'Epoch', 'QK', f'{src_path}train_qk.jpg', 'lower right')
    plt_fig(dirs_name, 'qk', 'Valid Quadratic Kappa', 'Epoch', 'QK', f'{src_path}valid_qk.jpg', 'lower right')
    plt_fig(dirs_name, 'lr', 'Learning Rate', 'Epoch', 'LR', f'{src_path}lr.jpg')


def plt_data_dist(src_path, save_path):
    labels = np.array([0, 1, 2, 3, 4])
    df = pd.read_csv(src_path)
    df = df.sort_values('diagnosis')
    height = df.groupby(['diagnosis'])['diagnosis'].count().values

    # plot
    fig = plt.figure()
    plt.bar(labels, height)
    plt.title('Data Distrbution', fontsize=18)
    plt.xlabel('Label')
    plt.ylabel('Amount')
    fig.savefig(f'{save_path}data_dist.jpg')


if __name__ == '__main__':
    import sys, warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="./saved/", help="path to the source dirs")
    parser.add_argument("--data_info_path", type=str, default="./data/train.csv", help="path to the source data info csv")
    parser.add_argument("--plot_all_files", type=bool, default=True, help="to determine whether all dirs in src path should be plotted")
    parser.add_argument("--history_list", type=list, default=['ResNet_reg'], help="source dirs name should be plotted")
    opt = parser.parse_args()
    print(opt)

    # plt_data_dist(opt.data_info_path, opt.src_path)
    plt_train_process(opt.src_path, opt.plot_all_files, opt.history_list)
