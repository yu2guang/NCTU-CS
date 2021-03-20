import os, cv2, tqdm, torch, argparse
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class AptosDataset(Dataset):
    def __init__(self, src_path: str, df: pd.DataFrame, mode: str, img_size: int, saved_path: str=None):
        self.src_path = src_path
        self.df = df
        self.mode = mode
        self.img_size = img_size
        self.saved_path = saved_path

        if self.mode == 'train':
            self.trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation((-120, 120)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.trans = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)
        path = self.src_path + self.df.id_code.values[idx] + '.png'

        if self.mode == 'preprocess':
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = crop_image_from_gray(img)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 30), -4, 128)
            cv2.imwrite(self.saved_path + self.df.id_code.values[idx] + '.png', img)
        else:
            img = Image.open(path)
            img = self.trans(img)

        return img, label


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)

        return img


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="./data/train_images/", help="path to the train images dir")
    parser.add_argument("--src_csv", type=str, default="./data/train.csv", help="path to the train info csv")
    parser.add_argument("--saved_path", type=str, default="./data/cv2_img/", help="path to the saved images dir")
    parser.add_argument("--img_size", type=int, default=256, help="preprocessed image size")
    parser.add_argument("--n_cpu", type=int, default=12, help="number of workers")
    parser.add_argument("--batch_size", type=int, default=64, help="size of batch")
    opt = parser.parse_args()
    print(opt)

    train_df = pd.read_csv(opt.src_csv)
    os.makedirs(opt.saved_path, exist_ok=True)

    trainset = AptosDataset(opt.src_path, train_df, 'preprocess', opt.img_size, opt.saved_path)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    for _, _ in tqdm.tqdm(train_loader):
        pass
