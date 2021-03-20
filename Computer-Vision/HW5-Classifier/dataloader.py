import pandas as pd
import torch as t
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import transforms

class dataLoader(data.Dataset):
    def __init__(self, src_path, mode):
        """
        Args:
            src_path (string): source path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.src_path = src_path
        self.mode = mode
        self.map_label = {'Bedroom': 0, 'Coast': 1, 'Forest': 2, 'Highway': 3, 'Industrial': 4, 'InsideCity': 5, 'Kitchen': 6, 'LivingRoom': 7, 'Mountain': 8, 'Office': 9, 'OpenCountry': 10, 'Store': 11, 'Street': 12, 'Suburb': 13, 'TallBuilding': 14}
        self.img_name, self.label = self.getData()

        self.trans = transforms.Compose([transforms.Resize((250, 250), interpolation=Image.NEAREST),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])


        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        label_name = self.label[index]
        label = self.map_label[self.label[index]]
        path = self.src_path + self.mode + '/' + label_name + '/' + self.img_name[index]

        img_origin = Image.open(path, 'r').convert('RGB')
        img = self.trans(img_origin)

        return img, label

    def getData(self):
        img = pd.read_csv(self.src_path + self.mode + '/img.csv')
        label = pd.read_csv(self.src_path + self.mode + '/label.csv')

        return np.squeeze(img.values), np.squeeze(label.values)

