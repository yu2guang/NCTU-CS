import os, glob
from itertools import count
import random, math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from panoramic import *

def ssd(A, B):
    err = A - B
    return np.dot(err, err.T)

def compute_error(d1, d2s):
    results_index = []
    for d2 in d2s:
        results_index.append(ssd(d1, d2))
    results_index = np.argsort(results_index)
    ratio = ssd(d1, d2s[results_index[0]]) / ssd(d1, d2s[results_index[1]])
    return ratio, results_index[0]

def feature_matching(des1, des2, kp1, kp2, threshold = 0.2):
    matches = []
    for i in range(len(des1)):
        ratio, des2_index = compute_error(des1[i], des2)
        if ratio < threshold:
            matches.append([kp1[i].pt, kp2[des2_index].pt])
    return matches

def points_matching(img1, img2, gray_img1, gray_img2, target_path, img_name):
    # part1 points detection & feature description
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # part2 Feature matching by SIFT features
    match_points = feature_matching(des1, des2, kp1, kp2)

    # show drawing line image
    hA, wA, cA = img1.shape
    hB, wB, cB = img2.shape

    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

    vis[0:hA, 0:wA] = img1
    vis[0:hB, wA:] = img2

    for (corA, corB) in match_points:
        color = np.random.randint(0, high=255, size=3).tolist()

        int_corA = (int(corA[0]), int(corA[1]))
        int_corB = (int(corB[0]) + wA, int(corB[1]))

        cv2.line(vis, int_corA, int_corB, color, 1)

    cv2.imwrite('{}{}_feature.jpg'.format(target_path, img_name), vis)
    print('\n{}{}_feature.jpg'.format(target_path, img_name))

    return match_points

def homo_mat(img1_pts, img2_pts):
    """
    :param img1_pts:
    :param img2_pts:
    :return: the homography between two imgs
    """
    point_n = img1_pts.shape[0]
    P = np.zeros(shape=(2*point_n, 9))

    for i in range(point_n):
        Pi = np.zeros(shape=(2, 9))
        img2_pts_homo = np.array([img2_pts[i][0], img2_pts[i][1], 1])

        Pi[0, 0:3] += img2_pts_homo
        Pi[0, 6:9] += img2_pts_homo * img1_pts[i][0] * (-1)
        Pi[1, 3:6] += img2_pts_homo
        Pi[1, 6:9] += img2_pts_homo * img1_pts[i][1] * (-1)

        P[2*i:(2*i+2), 0:9] += Pi

    _, _, v_T = np.linalg.svd(P)
    m = v_T.T[:,-1]
    H_mat = m.reshape(3, 3)
    H_mat /= H_mat[2][2]

    return H_mat

def cal_outliers(points, N_points, H_mat, threshold):
    # turn to be homogeneous
    pts1, pts2 = points[:, 0, :], points[:, 1, :]
    pts1_homo = np.hstack((pts1, np.ones((N_points, 1))))
    pts2_homo = np.hstack((pts2, np.ones((N_points, 1))))

    # estimated points
    pts1_homo_ = (H_mat @ pts2_homo.T).T
    pts1_homo_ /= pts1_homo_[:, 2].reshape(-1, 1)

    # calculate the geometry distance
    distance = np.linalg.norm((pts1_homo - pts1_homo_), axis=1, keepdims=True)

    # score by threshold
    projected_pts = pts1_homo_[:, 0:2]
    outlier_index = np.where(distance > 20)[0]
    outlier = projected_pts[outlier_index]
    inlier_index = np.where(distance <= 20)[0]
    inlier = projected_pts[inlier_index]
    N_outliers = len(outlier_index)

    # min & max (for plot)
    x_min, x_max = min(projected_pts[:, 0]), max(projected_pts[:, 0])
    xd = (x_max - x_min) // 10
    y_min, y_max = min(projected_pts[:, 1]), max(projected_pts[:, 1])
    yd = (y_max - y_min) // 10
    x_y = [x_min, x_max, xd, y_min, y_max, yd]

    return N_outliers, outlier, inlier, x_y

def plot_homo(x_y, iteration, target_path, dist_threshold, outlier_ratio, outlier_pts, inlier_pts):

    in_x = inlier_pts[:, 0]
    in_y = inlier_pts[:, 1]
    out_x = outlier_pts[:, 0]
    out_y = outlier_pts[:, 1]

    x_min, x_max, xd, y_min, y_max, yd = x_y

    plt.figure(figsize=(8, 8))
    plt.title('Iteration '+str(iteration), fontsize=18)
    plt.xlim(x_min - xd / 4, x_max + xd / 4)
    plt.xticks(rotation=60)
    plt.xlabel(r'$\delta={:.2f},\ e={:.2f}$'.format(dist_threshold, outlier_ratio))
    plt.ylim(y_min - yd / 4, y_max + yd / 4)
    plt.scatter(in_x, in_y, label='inlier')
    plt.scatter(out_x, out_y, label='outlier')
    plt.legend(loc='upper left')
    plt.savefig(target_path+str(iteration)+'.jpg')
    print(target_path+str(iteration)+'.jpg')


def ransac(match_points, target_path, img_name):
    # parameters
    N_sample = 4           # homography least parameters
    N_iter = 100000000     # infinite large
    dist_threshold = 0.25
    correct_prob = 0.99    # hope model correct probability
    compute_N_iter = lambda p, e, s: int(math.log(1 - p) / math.log(1 - (1 - e) ** s))

    # initial
    N_points = match_points.shape[0]
    least_N_outliers = N_points
    best_H = np.zeros((3, 3))

    # mkdir homography
    homo_target_path = target_path+img_name+'_homo/'
    if not os.path.exists(homo_target_path):
        os.makedirs(homo_target_path)

    # iterate for N times
    for i in count(1):
        # sample S correspondences from the feature matching results
        sample_points = match_points[random.sample(range(N_points), N_sample)]
        img1_pts, img2_pts = sample_points[:, 0, :], sample_points[:, 1, :]

        # compute the homography matrix based on these sampled correspondences
        H_mat = homo_mat(img1_pts, img2_pts)

        # check the number of outliers by a threshold
        N_outliers, outlier_pts, inlier_pts, x_y = cal_outliers(match_points, N_points, H_mat, dist_threshold)

        # get the best homography matrix with smallest number of outliers
        if N_outliers < least_N_outliers:
            best_H = H_mat
            print('Homography:', best_H)
            least_N_outliers = N_outliers
            outlier_ratio = N_outliers / N_points
            plot_homo(x_y, i, homo_target_path, dist_threshold, outlier_ratio, outlier_pts, inlier_pts)
            if N_outliers == 0:
                break
            N_iter = compute_N_iter(correct_prob, outlier_ratio, N_sample)

        if i > N_iter:
            break

    return best_H


if __name__ == '__main__':

    # name
    src_path = "./data/"
    target_path = './results/'

    # mkdir target_path
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # get imgs
    imgs = glob.glob('{}*'.format(src_path))
    imgs.sort()

    for i in range(len(imgs) // 2):

        # read imgs
        img_name = imgs[2*i].split('/')[-1].split('.')[0].rstrip('1')
        img1 = cv2.imread(imgs[2*i])
        img2 = cv2.imread(imgs[2*i+1])

        gray_img1 = cv2.imread(imgs[2*i], cv2.IMREAD_GRAYSCALE)
        gray_img2 = cv2.imread(imgs[2*i+1], cv2.IMREAD_GRAYSCALE)

        # part 1 & 2
        match_points = points_matching(img1, img2, gray_img1, gray_img2, target_path, img_name)

        # part 3
        best_H = ransac(np.array(match_points), target_path, img_name)

        # part 4
        panoramic = Panoramic(img1, img2, best_H)
        # four type of blending algorithm
        blend_img = panoramic.normal_blend()
        cv2.imwrite('{}{}_normal_blend.jpg'.format(target_path, img_name), blend_img)
        blend_img = panoramic.average_blend()
        cv2.imwrite('{}{}_average_blend.jpg'.format(target_path, img_name), blend_img)
        blend_img = panoramic.weight_blend()
        cv2.imwrite('{}{}_weight_blend.jpg'.format(target_path, img_name), blend_img)
        blend_img = panoramic.multiband_blend()
        cv2.imwrite('{}{}_multiband_blend.jpg'.format(target_path, img_name), blend_img)



