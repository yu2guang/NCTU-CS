import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import transforms
import glob, cv2, random


def data_evaluation(src_path):
    """
        Evaluate the training data & output testing csv
    """
    train_path = src_path + 'training_data/'
    test_path = src_path + 'testing_data/'

    # get the num of classify labels & mapping
    data = pd.read_csv(train_path + 'training_labels.csv')
    labels_name = data['label'].unique()
    label2num_dict = {name: i for i, name in enumerate(labels_name)}
    num2label_dict = {i: name for i, name in enumerate(labels_name)}
    print('> The num of classify labels is: {}'.format(len(labels_name)))

    # output testing id csv
    img_names = glob.glob(test_path + '*.jpg')
    img_names = pd.Series(img_names).str.split('/', expand=True)[3].str.split('.', expand=True)[0]
    test_df = pd.DataFrame({
        'id': img_names,
        'label': img_names
    })
    test_df.to_csv(test_path + 'testing_labels.csv', index=False)
    print('> Testing csv saved\n')

    return label2num_dict, num2label_dict


class dataLoader(data.Dataset):
    def __init__(self, src_path: str, mode: str, labels_dict={}):
        """
        Args:
            src_path (string): Source path of the dataset
            mode (string): Indicate procedure status (train/test)
            labels_dict (dict): Label mapping dictionary

            self.img_name (string list): Store all image names
            self.label (string list): Store all label values; testing dataset doesn't have
        """

        self.src_path = src_path + mode + 'ing_data/'
        self.mode = mode
        self.labels_dict = labels_dict
        self.img_name, self.label = self.get_data()

        if self.mode == 'train':

            self.trans = transforms.Compose([transforms.RandomPerspective(),
                                             # transforms.RandomRotation(random.randrange(0, 60)),
                                             transforms.ColorJitter(),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.Resize((250, 300), interpolation=Image.NEAREST),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        else:
            self.trans = transforms.Compose([transforms.Resize((250, 300), interpolation=Image.NEAREST),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        print("> {}: Found {} images".format(self.mode, len(self.img_name)))

    def __len__(self):
        """return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """
            return processed image and label
        """
        label = self.label[index]
        if self.mode == 'train':
            label = self.labels_dict[self.label[index]]

        img_name = self.img_name[index]
        img_origin = Image.open(self.src_path + img_name, 'r').convert('RGB')
        img = self.trans(img_origin)

        return img, label

    def get_data(self):
        """
            return the img_name & label list
        """
        data = pd.read_csv(self.src_path + self.mode + 'ing_labels.csv')
        data['id'] = data['id'].astype(str).str.zfill(6)
        data['id'] += '.jpg'

        return np.squeeze(data['id'].values), np.squeeze(data['label'].values)


