import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import transforms


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode

        if(mode == 'train'):
            self.flip = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.ToTensor()
                                            ])
        else:
            self.flip = transforms.ToTensor()

        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.root + self.img_name[index] + '.jpeg'

        img_origin = Image.open(path, 'r')

        img = self.flip(img_origin)

        """pixels = img_origin.load()

        img = np.zeros(shape=(3,512,512))
        for i in range(512):
            for j in range(512):
                for k in range(3):
                    img[k][i][j] = pixels[i,j][k]/255.0"""

        label = self.label[index]

        return img, label

    """def create_npz(self, isCreate):
        if(isCreate == False):
            len = self.__len__()
            for i in range(len):
                data, label = self.__getitem__(i)
                if(i==0):
                    save_info = np.array(data).reshape(1,3,512,512)
                else:
                    save_info = np.concatenate((save_info, np.array(data).reshape(1,3,512,512)))

            if(self.mode=='train'):
                np.savez("train", data=save_info)
            else:
                np.savez("test", data=save_info)"""