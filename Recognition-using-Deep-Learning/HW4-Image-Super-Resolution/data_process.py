import os, cv2, glob, tqdm, torch, random, argparse
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


def get_avg_resol(src_path):
    img_paths = glob.glob(src_path + '*.png')
    w_total, h_total = 0, 0
    for path in img_paths:
        img = cv2.imread(path)
        w_total += img.shape[0]
        h_total += img.shape[1]

    w_avg = w_total // len(img_paths)
    h_avg = h_total // len(img_paths)
    print(f'{src_path} --> average w: {w_avg}, h: {h_avg}')


def hr2lr(src_path, lr_path, hr_path):
    img_paths = glob.glob(src_path + '*.png')
    os.makedirs(hr_path, exist_ok=True)
    os.makedirs(lr_path, exist_ok=True)

    for path in img_paths:
        img = cv2.imread(path)
        # lr
        img_lr = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
        cv2.imwrite(lr_path+path.split('/')[-1], img_lr)
        # hr
        if img.shape[0] % 3 != 0 or img.shape[1] % 3 != 0:
            img_hr = cv2.resize(img, (img.shape[1] // 3 * 3, img.shape[0] // 3 * 3))
            cv2.imwrite(hr_path + path.split('/')[-1], img_hr)
        else:
            cv2.imwrite(hr_path + path.split('/')[-1], img)


def hr_resize(lr_path, hr_path, target_path):
    lr_img_paths = glob.glob(lr_path + '*.png')
    hr_img_paths = glob.glob(hr_path + '*.png')
    os.makedirs(target_path, exist_ok=True)

    for lr_p, hr_p in zip(lr_img_paths, hr_img_paths):
        lr_img = cv2.imread(lr_p)
        hr_img = cv2.imread(hr_p)
        # lr
        new_hr_img = cv2.resize(hr_img, (lr_img.shape[1] * 3, lr_img.shape[0] * 3))
        cv2.imwrite(target_path + lr_p.split('/')[-1], new_hr_img)


def cutblur(im1, im2, prob=1.0, alpha=0.6):
    if im1.shape != im2.shape:
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.shape[0], im2.shape[1]
    ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)
    print(0, np.abs(h - ch + 1))
    cy = np.random.randint(0, np.abs(h - ch + 1))
    print(0, np.abs(w - cw + 1))
    cx = np.random.randint(0, np.abs(w - cw + 1))

    # apply CutBlur to inside or outside
    # if np.random.random() > 0.5:
    #     im2[..., cy:cy + ch, cx:cx + cw] = im1[..., cy:cy + ch, cx:cx + cw]
    # else:
    #     im2_aug = im1.clone()
    #     im2_aug[..., cy:cy + ch, cx:cx + cw] = im2[..., cy:cy + ch, cx:cx + cw]
    #     im2 = im2_aug
    im2_aug = im1.copy()
    im2_aug[..., cy:cy + ch, cx:cx + cw] = im2[..., cy:cy + ch, cx:cx + cw]
    im2 = im2_aug

    return im1, im2


def save_aug_imgs(src_path, lr_path, hr_path, mode=['hflip', 'vflip']):
    img_paths = glob.glob(src_path + '*.png')
    os.makedirs(hr_path, exist_ok=True)
    os.makedirs(lr_path, exist_ok=True)

    for path in tqdm.tqdm(img_paths):
        img = Image.open(path, 'r').convert('RGB')
        img_name = path.split('/')[-1]

        # lr & hr origin img
        img_lr = transforms.Resize((img.size[1] // 3, img.size[0] // 3), interpolation=Image.BICUBIC)(img)
        img_lr.save(lr_path + img_name)
        if img.size[0] % 3 != 0 or img.size[1] % 3 != 0:
            img_hr = transforms.Resize((img.size[1] // 3 * 3, img.size[0] // 3 * 3))(img)
        else:
            img_hr = img.copy()
        img_hr.save(hr_path + img_name)

        if 'hflip' in mode:
            cur_name = img_name.split('.')
            cur_name[0] += '_hflip'
            cur_name = '.'.join(cur_name)
            img_lr_hflip = transforms.RandomHorizontalFlip(p=1)(img_lr)
            img_hr_hflip = transforms.RandomHorizontalFlip(p=1)(img_hr)
            img_lr_hflip.save(lr_path + cur_name)
            img_hr_hflip.save(hr_path + cur_name)

            if 'cj' in mode:
                cur_name = img_name.split('.')
                cur_name[0] += '_hflip_cj'
                cur_name = '.'.join(cur_name)
                img_hr_hflip_cj = transforms.ColorJitter(brightness=(1, 10), contrast=(1, 10),
                                                   saturation=(1, 10), hue=(0.2, 0.4))(img_hr_hflip)
                img_lr_hflip_cj = transforms.Resize((img_hr_hflip_cj.size[1] // 3, img_hr_hflip_cj.size[0] // 3))(img_hr_hflip_cj)
                img_lr_hflip_cj.save(lr_path + cur_name)
                img_hr_hflip_cj.save(hr_path + cur_name)

        if 'vflip' in mode:
            cur_name = img_name.split('.')
            cur_name[0] += '_vflip'
            cur_name = '.'.join(cur_name)
            img_lr_vflip = transforms.RandomVerticalFlip(p=1)(img_lr)
            img_hr_vflip = transforms.RandomVerticalFlip(p=1)(img_hr)
            img_lr_vflip.save(lr_path + cur_name)
            img_hr_vflip.save(hr_path + cur_name)

            if 'cj' in mode:
                cur_name = img_name.split('.')
                cur_name[0] += '_vflip_cj'
                cur_name = '.'.join(cur_name)
                img_hr_vflip_cj = transforms.ColorJitter(brightness=(1, 10), contrast=(1, 10),
                                                   saturation=(1, 10), hue=(0.2, 0.4))(img_hr_vflip)
                img_lr_vflip_cj = transforms.Resize((img_hr_vflip_cj.size[1] // 3, img_hr_vflip_cj.size[0] // 3))(img_hr_vflip_cj)
                img_lr_vflip_cj.save(lr_path + cur_name)
                img_hr_vflip_cj.save(hr_path + cur_name)

        if 'cj' in mode:
            cur_name = img_name.split('.')
            cur_name[0] += '_cj'
            cur_name = '.'.join(cur_name)
            img_hr_cj = transforms.ColorJitter(brightness=(1, 10), contrast=(1, 10),
                                               saturation=(1, 10), hue=(0.2, 0.4))(img_hr)
            img_lr_cj = transforms.Resize((img_hr_cj.size[1] // 3, img_hr_cj.size[0] // 3))(img_hr_cj)
            img_lr_cj.save(lr_path + cur_name)
            img_hr_cj.save(hr_path + cur_name)


def parse_info_cfg(src_path):
    """
        return the parsed config file info
    """
    fp = open(src_path, 'r')
    data_info = {}

    for line in fp.readlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        else:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            if key in ['n_cpu', 'epoch', 'batch', 'img_size', 'n_class', 'random_seed']:
                value = int(value)
            elif key in ['lr', 'w_decay', 'valid_proportion']:
                value = float(value)
            elif key in ['model_type']:
                value = value.strip('[]').split(', ')
                value = [v.strip() for v in value]

            data_info[key] = value

    fp.close()
    return data_info


if __name__ == '__main__':
    import sys, warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--src_path", type=str, default="./data/train_images/", help="path to the train images dir")
    # parser.add_argument("--src_csv", type=str, default="./data/train.csv", help="path to the train info csv")
    # parser.add_argument("--saved_path", type=str, default="./data/cv2_img/", help="path to the saved images dir")
    # parser.add_argument("--img_size", type=int, default=256, help="preprocessed image size")
    # parser.add_argument("--n_cpu", type=int, default=12, help="number of workers")
    # parser.add_argument("--batch_size", type=int, default=64, help="size of batch")
    # opt = parser.parse_args()
    # print(opt)
    #
    # train_df = pd.read_csv(opt.src_csv)
    # os.makedirs(opt.saved_path, exist_ok=True)
    #
    # trainset = AptosDataset(opt.src_path, train_df, 'preprocess', opt.img_size, opt.saved_path)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    #
    # for _, _ in tqdm.tqdm(train_loader):
    #     pass

    # get_avg_resol('./data/training_hr_images/')
    # get_avg_resol('./data/testing_lr_images/')

    # hr2lr('./data/training_hr_images0/', './data/training_lr_images/', './data/training_hr_images/')
    save_aug_imgs('./data/training_hr_images0/', './data/train_lr_aug2/', './data/train_hr_aug2/', mode=['hflip', 'vflip', 'cj'])
    # save_aug_imgs('./data/training_hr_images0/', './data/train_lr_aug/', './data/train_hr_aug2/', mode=['hflip', 'vflip'])

    # img1 = Image.open('./data/training_hr_images/25098.png', 'r').convert('RGB')
    # img2 = Image.open('./data/training_lr_images/25098.png', 'r').convert('RGB')
    # img2 = transforms.Resize((img2.size[1] * 3, img2.size[0] * 3))(img2)
    #
    # img1.save('./data/25098_hr.png')
    # img2.save('./data/25098_lr.png')

    # img1 = cv2.imread('./data/training_hr_images/25098.png')
    # img2 = cv2.imread('./data/training_lr_images/25098.png')
    # img2 = cv2.resize(img2, (img2.shape[1] * 3, img2.shape[0] * 3))
    #
    # cv2.imwrite('./data/25098_hr.png', img1)
    # cv2.imwrite('./data/25098_lr.png', img2)
    #
    #
    # img1, img2 = cutblur(img1, img2, prob=1.0, alpha=1.0)
    # cv2.imwrite('./data/25098_hr_cut.png', img1)
    # cv2.imwrite('./data/25098_lr_cut.png', img2)

    # hr_resize('./data/testing_lr_images/', './data/testing_hr_images0/', './data/testing_hr_images/')