import argparse
from utils.plots import plot_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='runs/train/exp3', help='training results path')
    opt = parser.parse_args()

    plot_results(save_dir=opt.save_path)