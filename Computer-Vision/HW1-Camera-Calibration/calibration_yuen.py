import numpy as np
import cv2

def calibration(objpoints, imgpoints, img_n):

	Hi = []
	V = []
	for i in range(img_n):
		# 1. Use the points in each images to find Hi
		Hi_tmp = findHomography(objpoints[i], imgpoints[i])
		Hi.append(Hi_tmp)

		# 2. Use Hi to find out the intrinsic matrix K
		V_tmp = find_V(Hi_tmp)
		V.append(V_tmp)
	V = np.array(V).reshape(-1,6)

	mtx = find_K(V)
	K_inv = np.linalg.inv(mtx)

	# 3. Find out the extrensics matrix of each images.
	extrinsics = []
	for i in range(img_n):
		extrinsics_tmp = find_extrinsics(K_inv, Hi[i])
		extrinsics.append(extrinsics_tmp)
	extrinsics = np.array(extrinsics).reshape(-1,6)

	return mtx, extrinsics


def findHomography(objpoints, imgpoints):
	point_n = objpoints.shape[0]
	P = np.zeros(shape=(2*point_n,9))
	for i in range(point_n):
		Pi = find_P(objpoints[i], imgpoints[i])
		P[2*i:(2*i+2), 0:9] += Pi

	_, _, v_T = np.linalg.svd(P) 
	m = v_T.T[:,-1]
	Hi = m.reshape(3, 3)
	if(Hi[2][2]<0):
		Hi *= -1

	# Hi, _ = cv2.findHomography(objpoints, imgpoints)
	return Hi

def find_P(objp_i, imgp_i):
	P = np.zeros(shape=(2,9))
	objp_i[2] = 1

	P[0, 0:3] += objp_i
	P[0, 6:9] += objp_i*imgp_i[0]*(-1)
	P[1, 3:6] += objp_i
	P[1, 6:9] += objp_i*imgp_i[1]*(-1)

	return P

def find_V(Hi):
	h11, h12, h13 = Hi[0]
	h21, h22, h23 = Hi[1]
	h31, h32, h33 = Hi[2]

	V = np.zeros(shape=(2,6))
	V[0][0] = h11*h12
	V[0][1] = h21*h12+h11*h22
	V[0][2] = h31*h12+h11*h32
	V[0][3] = h21*h22
	V[0][4] = h31*h22+h21*h32
	V[0][5] = h31*h32
	V[1][0] = h11**2-h12**2
	V[1][1] = 2*(h11*h21-h12*h22)
	V[1][2] = 2*(h11*h31-h12*h32)
	V[1][3] = h21**2-h22**2
	V[1][4] = 2*(h21*h31-h22*h32)
	V[1][5] = h31**2-h32**2

	return V

def find_K(V):	
	# argmin Vb
	_, _, v_T = np.linalg.svd(V, full_matrices=False) 
	b = v_T.T[:,-1]*(-1)
	B = np.zeros(shape=(3,3))
	B[0][0] = b[0]
	B[0][1] = b[1]
	B[0][2] = b[2]
	B[1][0] = b[1]
	B[1][1] = b[3]
	B[1][2] = b[4]
	B[2][0] = b[2]
	B[2][1] = b[4]
	B[2][2] = b[5] 
	# B = K^(-T)*K^(-1)
	K_inv = np.linalg.cholesky(B).T
	K = np.linalg.inv(K_inv)
	K /= K[2][2]

	return K

def find_extrinsics(K_inv, Hi):
	h1, h2, h3 = Hi.T[0:3]
	lambda_term = 1/np.linalg.norm(K_inv@h1.T)

	r1 = lambda_term*(K_inv@h1.T)
	r2 = lambda_term*(K_inv@h2.T)
	r3 = np.cross(r1,r2)
	t = lambda_term*(K_inv@h3.T)

	r_rotate_mat = np.concatenate((r1.reshape(-1,1), r2.reshape(-1,1), r3.reshape(-1,1)), axis=1)
	r, _ = cv2.Rodrigues(r_rotate_mat)
	extrinsics = np.concatenate((r, t.reshape(3,1)), axis=0).reshape(6)

	return extrinsics