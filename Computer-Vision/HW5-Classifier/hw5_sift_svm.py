from cyvlfeat.kmeans import kmeans
from cyvlfeat.sift.dsift import dsift
from scipy.spatial import distance
import cv2
import numpy as np
import os, glob, time
import sys
sys.path.append("C:\libsvm-weights-3.24\python")
import pickle
from svmutil import *
from commonutil import *

def load_img_from_folder(folder_path):
    img_dic = {}
    catagory = glob.glob('{}*'.format(folder_path))
    catagory = np.array(catagory)
    for i in range(len(catagory)):
        folder_name = catagory[i].split('\\')[-1]
        imgs = glob.glob('{}*'.format(folder_path + "/" + folder_name + "/"))
        img_in_cat = []
        for j in range(len(imgs)):
            img_name = imgs[j].split('\\')[-1]
            fullimgname = folder_path + "/" + folder_name + "/" + img_name
            img = cv2.imread(fullimgname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img_in_cat.append([img_name, img])
        img_dic[folder_name] = img_in_cat
    return img_dic
    
def create_sift_discription(train_data):

    description_bag = []
    sift_data_dic = {}
    
    for key, value in train_data.items():
        temp_list = []
        for i in range(len(value)):
            kp, des = dsift(value[i][1], step=[5, 5], fast=True)
            description_bag.extend(des)
            temp_list.append(des)
        sift_data_dic[key] = temp_list
    return description_bag, sift_data_dic

def knn(train_feats, train_labels, test_feats, test_labels):
    all_test_num, correct_test_num = 0, 0
    for test_feat, test_label in zip(test_feats, test_labels):
        minimum = 1e20
        ans_label = ""
        for train_feat, train_label in zip(train_feats, train_labels):
            dist = distance.euclidean(test_feat, train_feat)
            if(dist < minimum):
                minimum = dist
                ans_label = train_label
        if(ans_label == test_label):
            correct_test_num += 1
        all_test_num += 1

    return correct_test_num, all_test_num

def get_bags_of_sifts(data_dict, vocab):
    feats, labels = [], []
    for i, label in enumerate(data_dict.keys()):
        for descriptors in data_dict[label]:
            distances = distance.cdist(descriptors, vocab, 'euclidean')
            clusters = np.argmin(distances, axis=1)
            hist, bin_edges = np.histogram(clusters, bins=np.arange(len(vocab)+1), density=True)
            feats.append(hist)
            labels.append(i)
            assert len(hist) == (len(bin_edges)-1)
    return feats, labels

train_data_path = "./hw5_data/train/"
test_data_path = "./hw5_data/test/"

#format:{'catagory': ['img_name', img]}
train_data = load_img_from_folder(train_data_path)
test_data = load_img_from_folder(test_data_path)
print("finish load image...")

#sift description
description_bag, sift_data_dic = create_sift_discription(train_data)
_, test_sift_data_dic = create_sift_discription(test_data)
print("finish sift description...")

cluster_num = 200

bag_of_features = np.array(description_bag).astype(np.float32)
end = time.time()
if not os.path.exists("vocab_" + str(cluster_num) + ".npy"):
    vocab = kmeans(bag_of_features, cluster_num, initialization='PLUSPLUS')
    print("finish kmeans " + str(cluster_num) + ":", time.time()-end, "s")
    np.save("vocab_" + str(cluster_num) + ".npy", vocab)
else:
    vocab = np.load("vocab_" + str(cluster_num) + ".npy")

if not os.path.exists("train_hist_" + str(cluster_num) + ".npy"):
    end = time.time()
    train_feats, train_labels = get_bags_of_sifts(sift_data_dic, vocab)
    np.save("train_hist_" + str(cluster_num) + ".npy", np.array(train_feats))
    np.save("train_label_" + str(cluster_num) + ".npy", np.array(train_labels))
    print("finish training hist...", time.time() - end, "s")
else:
    train_feats = np.load("train_hist_" + str(cluster_num) + ".npy")
    train_labels = np.load("train_label_" + str(cluster_num) + ".npy")

test_feats, test_labels = get_bags_of_sifts(test_sift_data_dic, vocab)

train_x = [{(i+1): feat[i] for i in range(len(feat))} for feat in train_feats]
train_y = [(label+1) for label in train_labels]

prob = svm_problem(train_y, train_x)
param = svm_parameter('-t 0 -c 1e7 -b 1')
model = svm_train(prob, param)


test_x = [{(i+1): feat[i] for i in range(len(feat))} for feat in test_feats]
test_y = [(label+1) for label in test_labels]
#import pdb
#pdb.set_trace()
p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
