import matplotlib.pyplot as plt
import os


def plt_train_acc(src_path, epoch_i, target_path='./train_acc_plot/'):

    # make dir
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # read file
    fp = open(src_path + 'train_acc.txt', 'r')
    lines = fp.readlines()
    fp.close()

    y = [float(acc.strip()) for acc in lines]
    x = [int(i) for i in range(len(y))]

    # plot
    fig = plt.figure()

    plt.plot(x, y, color='#005ab5')
    plt.axvline(epoch_i, color='#F00078')

    plt.title('Training Accuracy', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    fig.savefig('{}{}_train_acc.jpg'.format(target_path, src_path.split('/')[-2]))


if __name__ == '__main__':
    plt_train_acc('./saved/ResNet50_2/', 89)
    plt_train_acc('./saved/ResNet18_1/', 55)
    plt_train_acc('./saved/ResNet18_pers/', 646)
    plt_train_acc('./saved/ResNet18_wd54_rot/', 268)
    plt_train_acc('./saved/ResNet18_wd4_cj_2/', 867)
    