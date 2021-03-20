import os
import glob
import cv2
import numpy as np

from task1_hybrid import hybridColor

def main():

	# path
	srcPath = '../../hw2_data/task1and2_hybrid_pyramid/'
	targetPath = '../../results/'
	# srcPath = '../../src4/'
	# targetPath = '../../src4_results3/'
	filterType = ['ideal', 'guassian'] 
	colorType = ['gray', 'color']
	if not os.path.exists(targetPath):
		os.makedirs(targetPath)	

	# task1 hybrid
	imgs = glob.glob('{}*'.format(srcPath))
	imgs.sort()

	for i in range(len(imgs)//2):
	# for i in range(1,4):
		for filter_i in filterType:
			for color_i in colorType:
				highImg = cv2.imread(imgs[2*i])
				lowImg = cv2.imread(imgs[2*i+1])					
				# highImg = cv2.imread(imgs[i])
				# lowImg = cv2.imread(imgs[0])
				hyImg = hybridColor(highImg, lowImg, color_i, filter_i)
				print('{}{}_{}_{}.jpg'.format(targetPath, i, filter_i, color_i))
				cv2.imwrite('{}{}_{}_{}.jpg'.format(targetPath, i, filter_i, color_i), hyImg)

if __name__ == '__main__':
	main()

