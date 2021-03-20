import random, os, glob
import cv2
import numpy as np

from pts_matching import points_matching


def to_homogeneous(points):
    if points.shape[1] == 3:
        return points
    ones = np.ones(points.shape[0])
    homo_points = np.vstack((points.T, ones))
    return homo_points.T

def normalize_image_coordinates(img_shape, pts):
    h, w = img_shape[0], img_shape[1]
    ones = np.ones(pts.shape[0]).reshape(-1, 1)
    pts = np.concatenate((pts, ones), axis=1).T
    norm_mat = np.array([[2/w, 0,  -1],
                        [  0, 2/h, -1],
                        [  0,  0,   1]])
    normalized_pts = norm_mat @ pts
    
    return normalized_pts, norm_mat

def compute_fundamental_matrix(img1_pts, img2_pts):
    A = []
    points_num = img1_pts.shape[1]
    for i in range(points_num):
        x1 = img1_pts[0][i]
        y1 = img1_pts[1][i]
        x2 = img2_pts[0][i]
        y2 = img2_pts[1][i]

        A.append([x2*x1, x2*y1, x2,
                  y2*x1, y2*y1, y2,
                  x1   , y1   , 1])
    A = np.array(A)
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V
    return F

def cal_inliers(pts1, pts2, F_mat, threshold):

    #Multiple View Geometry in Computer Vision p287(305 in pdf)
    # sampson distance
    L1 = F_mat.T @ pts1
    L2 = F_mat @ pts2
    
    JJT = L1[0]**2+L1[1]**2+L2[0]**2+L2[1]**2
    err_arr = np.diag(pts2.T @ F_mat @ pts1)**2 / JJT

    num_inlier = 0
    inliers_mask = []
    # Compute num of inliner
    for j in range (len(err_arr)):
        if err_arr[j] <= threshold:
            num_inlier += 1
            inliers_mask.append(1)
        else:
            inliers_mask.append(0)
    return num_inlier, inliers_mask

def ransac_and_fundamental (pts1, pts2, mat1, mat2, n_iters=4000, threshold=5e-5):
    # parameters
    N_sample = 8

    # initial
    N_points = pts1.shape[1]
    least_N_inliers = 0
    best_F = np.zeros((3, 3))
    best_inliers_mask = []

    # iterate for N times
    for i in range(n_iters):
        # sample S correspondences from the feature matching results
        ram_id = random.sample(range(N_points), N_sample)
        sample_pts1 = pts1[:,ram_id]
        sample_pts2 = pts2[:,ram_id]

        # compute the fundamental matrix based on these sampled correspondences
        F_mat = compute_fundamental_matrix(sample_pts1, sample_pts2)

        # check the number of inliers by a threshold
        N_inliers, inliers_mask = cal_inliers(pts1, pts2, F_mat, threshold)

        if N_inliers > least_N_inliers:
            #denormalize
            best_F = mat2.T @ F_mat @ mat1
            best_inliers_mask = inliers_mask
            print('N_inliers: {}, N_iter: {}'.format(N_inliers, i))
            if N_inliers == N_points:
                break
            least_N_inliers = N_inliers

    return best_F/best_F[-1, -1], best_inliers_mask

def plot_epilines(img1, img2, inlier_mask, match_points, target_path, img_name):
    hA, wA = img1.shape[:2]
    hB, wB = img2.shape[:2]

    inlier_mask = np.array(inlier_mask)
    in_pts1 = match_points[:, 0][inlier_mask.ravel()==1]
    in_pts2 = match_points[:, 1][inlier_mask.ravel()==1]
    #cv2 epilines
    #lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    #lines1 = lines1.reshape(-1,3)
    pts2_homo = to_homogeneous(in_pts2)
    line2 = np.dot(pts2_homo, best_F)

    img1_tmp = img1.copy()
    img2_tmp = img2.copy()

    # plot epilines
    for (corA, corB, L2) in zip(in_pts1, in_pts2, line2):
        color = np.random.randint(0, high=255, size=3).tolist()

        x0,y0 = map(int, [0, -L2[2]/L2[1] ])
        x1,y1 = map(int, [wB, -(L2[2]+L2[0]*wB)/L2[1] ])
        img1_tmp = cv2.line(img1_tmp, (x0,y0), (x1,y1), color,1)
        img1_tmp = cv2.circle(img1_tmp,tuple(map(int, corA[:2])),5,color,-1)
        img2_tmp = cv2.circle(img2_tmp,tuple(map(int, corB[:2])),5,color,-1)
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

    vis[0:hA, 0:wA] = img1_tmp
    vis[0:hB, wA:] = img2_tmp
    # cv2.imshow('My Image1', cv2.resize(vis, (800, 400)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('{}{}_epipolar_lines.jpg'.format(target_path, img_name), vis)

def get_K(img_name):
    if img_name == 'Statue':
        K1 = np.array([[5426.566895,    0.678017, 330.096680],
                       [   0.000000, 5423.133301, 648.950012],
                       [   0.000000,    0.000000,   1.000000]])

        K2 = np.array([[5426.566895,    0.678017, 387.430023],
                       [   0.000000, 5423.133301, 620.616699],
                       [   0.000000,    0.000000,   1.000000]])
    elif img_name == 'Mesona':
        K1 = np.array([[1.4219, 0.0005, 0.5092],
                       [0.0000, 1.4219, 0.3802],
                       [0.0000, 0.0000, 0.0010]])

        K2 = K1
    elif img_name == 'deer':
        K1 = np.array([[ 1.23451531e+03, -2.30410376e+00,  5.64016232e+02],
                       [ 0.00000000e+00,  1.23420888e+03,  7.29026308e+02],
                       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        K2 = K1


    return K1, K2

def get_four_extrinsic_from_essential(E):
    # make sure E is rank 2
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                                
    r1 = np.dot(U, np.dot(W, V))
    r2 = np.dot(U, np.dot(W.T, V))
    if np.linalg.det(r1) < 0:
        r1 = -r1
    if np.linalg.det(r2) < 0:
        r2 = -r2

    t1 = np.expand_dims(U[:, 2], axis=1)
    t2 = np.expand_dims(-U[:, 2], axis=1)

    extrinsic_Bs =  [
        np.hstack((r1, t1)),
        np.hstack((r1, t2)),
        np.hstack((r2, t1)),
        np.hstack((r2, t2))
    ]

    return extrinsic_Bs

def cal_valid_point(project, X):
    r = project[:,0:3]
    t = project[:,3]
    count = 0
    for target in X:
        # left z-axis dot target in left camera coordination
        left_check = np.dot(target, np.array([0, 0, 1]))
        # right z-axis dot target in right camera coordination
        right_check = np.dot((target - t), np.expand_dims(r[-1, :], axis=1))
        if left_check > 0 and right_check > 0:
            count += 1
    return count

def out_3Dcsv(data, data_name):
    fp = open(data_name, 'w')
    for row in data:
        row_join = ','.join(map(str, row))
        fp.write(row_join)
        fp.write('\n')
    fp.close()
    print(data_name)


if __name__ == '__main__':

    # name
    src_path = "../CV2020_HW4/data/"
    target_path = '../CV2020_HW4/results/'

    # mkdir target_path
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # get imgs
    imgs = glob.glob('{}*'.format(src_path))
    imgs.sort()

    # fund fp
    fund_fp = open(target_path+'fund.txt', 'w')

    for i in range(len(imgs) // 2):
        # read imgs
        img_name = [imgs[2 * i].split('/')[-1].split('.')[0].rstrip('1'), '.' + imgs[2 * i].split('/')[-1].split('.')[-1]]
        if not os.path.exists(target_path + '/' + img_name[0]):
            os.makedirs(target_path + '/' + img_name[0])
        fund_fp.write(img_name[0] + '\n')
        img1 = cv2.imread(imgs[2 * i])
        img2 = cv2.imread(imgs[2 * i + 1])

        # 1. find out correspondence across images
        match_points = np.array(points_matching(img1, img2, target_path, img_name))

        # 2. estimate the fundamental matrix across images (normalized 8 points)
        # Do normalization
        match_points = np.array(match_points)
        img1_normalized_pts, img1_norm_mat = normalize_image_coordinates(img1.shape[:2], match_points[:, 0])
        img2_normalized_pts, img2_norm_mat = normalize_image_coordinates(img2.shape[:2], match_points[:, 1])

        # compute best fundamental matrix
        best_F, inlier_mask = ransac_and_fundamental(img1_normalized_pts, img2_normalized_pts, img1_norm_mat, img2_norm_mat)
        print("fundamental matrix:")
        print(best_F)
        fund_fp.write('Fundamental matrix:\n')
        fund_fp.write(str(best_F)+'\n')
        # cv2 fundamental matrix
        #match_points = np.array(match_points)
        #F_cv, mask_cv = cv2.findFundamentalMat(match_points[:, 0], match_points[:, 1], method=cv2.FM_8POINT + cv2.FM_RANSAC)
        #print(F_cv)

        # 3. draw the interest points on you found in step.1 in one image
        # and the corresponding epipolar lines in another
        plot_epilines(img1, img2, np.array(inlier_mask), match_points, target_path, img_name[0])

        # 4. & 5. get essential matrix
        intrinsic_A, intrinsic_B = get_K(img_name[0])
        E = np.dot(intrinsic_B.T, np.dot(best_F, intrinsic_A))
        fund_fp.write('Essential matrix:\n')
        fund_fp.write(str(E)+'\n\n')

        extrinsic_A = np.hstack((np.eye(3), np.zeros((3,1))))
        extrinsic_Bs = get_four_extrinsic_from_essential(E)

        project_A = np.dot(intrinsic_A, extrinsic_A)
        project_Bs = []
        for e in extrinsic_Bs:
            project_Bs.append(np.dot(intrinsic_B, e))
        project_Bs = np.array(project_Bs)

        # 6. apply triangulation to get 3D points
        correct_X = None
        max_point = 0
        for project_B in project_Bs:
            X = []
            inlier_mask = np.array(inlier_mask)
            in_pts1 = match_points[:, 0][inlier_mask.ravel()==1]
            in_pts2 = match_points[:, 1][inlier_mask.ravel()==1]
            for (u1, v1), (u2, v2) in zip(in_pts1, in_pts2):
                A = []
                A.append(v1 * project_A[2, :] - project_A[1, :])
                A.append(u1 * project_A[1, :] - v1 * project_A[0, :])
                A.append(v2 * project_B[2, :] - project_B[1, :])
                A.append(u2 * project_B[1, :] - v2 * project_B[0, :])
                A = np.array(A)
                A_U, A_S, A_V = np.linalg.svd(A)
                last_V = A_V[np.argmin(A_S)]
                X.append((last_V/last_V[3])[:3])

            valid_point_num = cal_valid_point(project_B, X)
            if valid_point_num > max_point:
                max_point = valid_point_num
                correct_X = X

        # output csv
        out_3Dcsv(project_A, target_path+img_name[0]+'/CameraMatrix.csv')
        out_3Dcsv(match_points[:, 0][inlier_mask.ravel()==1], target_path+img_name[0]+'/pts2D.csv')
        out_3Dcsv(correct_X, target_path+img_name[0]+'/pts3D.csv')

