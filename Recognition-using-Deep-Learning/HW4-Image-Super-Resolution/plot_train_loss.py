import re, argparse
import matplotlib.pyplot as plt


def plot_train_loss(src_path):
    # file process
    fp = open(src_path, 'r')
    train_lines = fp.readlines()
    fp.close()

    # parse loss & psnr from each epoch
    loss_seq = []
    for i, line in enumerate(train_lines):
        if re.match(r'(.*)Namespace(.*)', line):
            num_epochs = int(line.split('num_epochs')[1].split(', ')[0].lstrip('='))
            loss_seq = [1 for _ in range(num_epochs)]
        if re.match(r'(.*)100%(.*)', line):
            loss_seq[int(line.split('/')[0].split(' ')[1])] = float(train_lines[i-1].split('loss')[1].strip(' \n=]'))
    x = [i for i in range(1, len(loss_seq) + 1)]

    # plot
    fig = plt.figure()
    plt.plot(x, loss_seq)
    plt.title('Train Loss', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(f'{src_path[:-11]}train_loss.jpg')
    print(f'Saved: {src_path[:-11]}train_loss.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./saved/SRCNN/process.txt')
    args = parser.parse_args()
    print(args)

    plot_train_loss(args.src_path)