import math
import numpy as np
import cv2

cutoff = {'high': 24, 'low': 16}
# cutoff = {'high': 16, 'low': 6}

def centerT(img):
	imgCT = np.copy(img)
	rows, cols = imgCT.shape
	for x in range(rows):
		for y in range(cols):
			imgCT[x][y] = img[x][y] * ((-1)**(x+y))
	return imgCT

def generateFilter(size, filterType):
	centerI = round(size[0]/2)
	centerJ = round(size[1]/2)

	def gaussian(D0, D):
		passFilter = math.exp(-1*D**2/(2*D0**2))
		if(filterType[1]=='high'):
			passFilter = 1 - passFilter
		return passFilter

	def ideal(D0, D):
		passFilter = 1 if(D<=D0) else 0
		if(filterType[1]=='high'):
			passFilter = 1 - passFilter
		return passFilter

	D0 = cutoff[filterType[1]]
	outFilter = []
	for i in range(size[0]):
		tmp = []
		for j in range(size[1]):
			D = ((i-centerI)**2+(j-centerJ)**2)**(1/2)
			if(filterType[0]=='ideal'):
				tmp.append(ideal(D0, D))
			else:
				tmp.append(gaussian(D0, D))
		outFilter.append(tmp)

	return np.array(outFilter)

def FT(img, filterType):
	imgSize = img.shape

	# 1. Multiply the input image by (-1)x+y to center the transform.
	imgCT = centerT(img)

	# 2. Compute Fourier transformation of input image, i.e. F(u,v).	
	imgF = np.fft.fft2(imgCT)
	
	# 3. Multiply F(u,v) by a filter function H(u,v).
	H = generateFilter(imgSize, filterType)
	imgOut = imgF * H	

	return imgOut

def iFT(img):
	# 5. Compute the inverse Fourier transformation of the result in (4).
	imgIF = np.fft.ifft2(img)

	# 6. Obtain the real part of the result in (5).
	imgOut = np.real(imgIF)

	# 7. Multiply the result in (6) by (-1)x+y.
	return centerT(imgOut)


def hybrid(highImg, lowImg, igFilterType):	
	u_min = min(highImg.shape[0], lowImg.shape[0])
	v_min = min(highImg.shape[1], lowImg.shape[1])
	highImg = highImg[:u_min, :v_min]
	lowImg = lowImg[:u_min, :v_min]

	# 4. Add the results, highpass image and lowpass image, in (3) together.
	highImg_, lowImg_ = FT(highImg.astype(np.float32), [igFilterType, 'high']), FT(lowImg.astype(np.float32), [igFilterType, 'low'])
	imgOut = highImg_ + lowImg_

	imgOut = iFT(imgOut)

	return np.clip(imgOut, 0, 255).astype(int)

def hybridColor(highImg, lowImg, colorType, igFilterType):
	if(colorType=='gray'):
		highGrayImg = cv2.cvtColor(highImg, cv2.COLOR_BGR2GRAY)		
		lowGrayImg = cv2.cvtColor(lowImg, cv2.COLOR_BGR2GRAY)	
		return hybrid(highGrayImg, lowGrayImg, igFilterType)
	else:
		Bh, Gh, Rh= cv2.split(highImg)
		Bl, Gl, Rl= cv2.split(lowImg)
		B_hy = hybrid(Bh, Bl, igFilterType)
		G_hy = hybrid(Gh, Gl, igFilterType)
		R_hy = hybrid(Rh, Rl, igFilterType)
		return cv2.merge([B_hy, G_hy, R_hy])

	