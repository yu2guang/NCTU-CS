import os, glob
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import csv



class data_preprocessing():
    def __init__(self, src_path, target_path,
                 img_csv_name='img.csv', label_csv_name='label.csv'):

        self.src_path = src_path
        self.target_path = target_path
        self.img_csv_name = img_csv_name
        self.label_csv_name = label_csv_name

        # training data
        print('Preprocessing the training data...')
        self.create_des_csv_npz('train')

        # testing data
        print('Preprocessing the testing data...')
        self.create_des_csv_npz('test')

    def create_des_csv_npz(self, mode):
        # src path & target path
        cur_src_path = self.src_path + mode + '/'
        cur_target_path = self.target_path + mode + '/'
        if not os.path.exists(cur_target_path):
            os.makedirs(cur_target_path)

        print('Saving the ' + cur_target_path + self.img_csv_name)
        print('Saving the ' + cur_target_path + self.label_csv_name)

        # get all the dirs of labels
        dirs = glob.glob('{}*/'.format(cur_src_path))
        dirs.sort()

        # fps of csv
        fp_img = open(cur_target_path + self.img_csv_name, 'w')
        fp_label = open(cur_target_path + self.label_csv_name, 'w')

        # create csv & find all descriptor
        first = True

        for dir in dirs:
            label = dir[len(cur_src_path):-1]
            imgs = glob.glob('{}*.jpg'.format(dir + '/'))
            for img in imgs:
                fp_img.write(img[len(cur_src_path+label)+1:] + '\n')
                fp_label.write(label + '\n')

        fp_img.close()
        fp_label.close()


def tiny_img(img, size=(16, 16)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compute_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

def knn(train_img, train_label, test_img, test_label, k):
    correct = 0
    for img, label in zip(test_img, test_label):
        distances = pow(train_img - img, 2).sum(1) ** 0.5
        index = distances.argsort()
        neighbors = [train_label[idx] for idx in index[:k]]
        distances = [distances[idx] for idx in index[:k]]
        pred_label = get_label(neighbors, distances)
        if pred_label == label:
            correct += 1
    return correct * 100 / test_label.shape[0]

def get_label(neighbors, distances):
    labels = {}
    for label, distance in zip(neighbors, distances):
        if label in labels:
            labels[label] += 1. / distance
        else:
            labels[label] = 1. / distance
    return max(labels.keys(), key=(lambda key: labels[key]))

if __name__ == '__main__':
    # Prepare Data
    src_path = './hw5_data/'
    target_path = './script/'
    pre_data = data_preprocessing(src_path, target_path)

    with open(os.path.join(target_path, 'train/img.csv'), newline='') as csvfile:
        train_img = list(csv.reader(csvfile))
    with open(os.path.join(target_path, 'train/label.csv'), newline='') as csvfile:
        train_label = list(csv.reader(csvfile))
    with open(os.path.join(target_path, 'test/img.csv'), newline='') as csvfile:
        test_img = list(csv.reader(csvfile))
    with open(os.path.join(target_path, 'test/label.csv'), newline='') as csvfile:
        test_label = list(csv.reader(csvfile))

    n_train_data = len(train_img)

    imgs = []
    for path, label in zip(train_img, train_label):
        img = cv2.imread(os.path.join(src_path, 'train', label[0], path[0]), 0)
        img = tiny_img(img)
        imgs.append(img)

    x_train = np.array(imgs).reshape((n_train_data, -1))
    y_train = np.array(train_label).reshape((n_train_data, ))

    n_test_data = len(test_img)

    imgs = []
    for path, label in zip(test_img, test_label):
        img = cv2.imread(os.path.join(src_path, 'test', label[0], path[0]), 0)
        img = tiny_img(img)
        imgs.append(img)

    x_test = np.array(imgs).reshape((n_test_data, -1))
    y_test = np.array(test_label).reshape((n_test_data, ))

    print("{:^3} | {:^10}".format("K", "Test Acc"))
    print('-' * 15)
    for k in range(1, 30+1):
        test_acc = knn(x_train, y_train, x_test, y_test, k)
        print("{:^3} | {:^10}".format(k, "%.4f"%test_acc))

