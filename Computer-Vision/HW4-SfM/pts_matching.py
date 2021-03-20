import cv2
import numpy as np
from numba import njit

@njit(cache=True)
def ssd(A, B):
    err = A - B
    return np.dot(err, err.T)

@njit(cache=True)
def compute_error(d1, d2s):
    results_index = []
    for d2 in d2s:
        results_index.append(ssd(d1, d2))
    results_index = np.argsort(np.array(results_index))
    ratio = ssd(d1, d2s[results_index[0]]) / ssd(d1, d2s[results_index[1]])
    return ratio, results_index[0]

@njit(cache=True)
def feature_matching(des1, des2, kp1, kp2, threshold=0.2):
    matches = []
    for i in range(len(des1)):
        ratio, des2_index = compute_error(des1[i], des2)
        if ratio < threshold:
            matches.append([kp1[i], kp2[des2_index]])
    return matches

def points_matching(img1, img2, target_path, img_name):
    # convert to gray scale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # points detection & feature description
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # feature matching by SIFT features
    kp1_pt = np.array([p.pt for p in kp1])
    kp2_pt = np.array([p.pt for p in kp2])

    match_points = feature_matching(des1, des2, kp1_pt, kp2_pt)

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

    cv2.imwrite('{}{}_feature{}'.format(target_path, img_name[0], img_name[1]), vis)
    print('\n{}{}_feature{}'.format(target_path, img_name[0], img_name[1]))

    return match_points