#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from my_functions import *
import glob
import ntpath


path = "../hw2_data/task3_colorizing/"
imgs = glob.glob('{}*'.format(path))

for imname in imgs:
	imgname = ntpath.basename(imname)
	#imname = path + imgname
	img=Image.open(imname)
	img=np.asarray(img)
	print("img name:", imgname, "img size:", img.shape)
	height, width =  img.shape
	one_third_height = int(height / 3)
	imgB = img[:one_third_height,:]
	imgG = img[one_third_height:one_third_height * 2,:]
	imgR = img[one_third_height * 2:one_third_height * 3,:]

	#crop image by 10%
	imgB = crop(imgB, 0.1)
	imgG = crop(imgG, 0.1)
	imgR = crop(imgR, 0.1)

	origin_imgB, origin_imgG, origin_imgR = imgB, imgG, imgR

	#do sobel
	sobel_x, sobel_y = get_sobel_filter()

	imgB_sobel_x = np.convolve(origin_imgB.flatten() , sobel_x.flatten() , 'same')
	imgB_sobel_y = np.convolve(origin_imgB.flatten() , sobel_y.flatten() , 'same')

	imgG_sobel_x = np.convolve(origin_imgG.flatten() , sobel_x.flatten() , 'same')
	imgG_sobel_y = np.convolve(origin_imgG.flatten() , sobel_y.flatten() , 'same')

	imgR_sobel_x = np.convolve(origin_imgR.flatten() , sobel_x.flatten() , 'same')
	imgR_sobel_y = np.convolve(origin_imgR.flatten() , sobel_y.flatten() , 'same')

	imgB_sobel = np.hypot(np.reshape(imgB_sobel_x, (-1, imgB.shape[1])), np.reshape(imgB_sobel_y, (-1, imgB.shape[1])))
	imgG_sobel = np.hypot(np.reshape(imgG_sobel_x, (-1, imgG.shape[1])), np.reshape(imgG_sobel_y, (-1, imgG.shape[1])))
	imgR_sobel = np.hypot(np.reshape(imgR_sobel_x, (-1, imgR.shape[1])), np.reshape(imgR_sobel_y, (-1, imgR.shape[1])))


	#allign raw image
	#final_img = (np.dstack((imgR,imgG,imgB))).astype(np.uint8)
	#plt.imshow(final_img)


	#use ssd to align small image
	if img.shape[0] < 5000:
		print("align small image")
		Gshift = ssdAlign(imgB_sobel, imgG_sobel, 20)
		Rshift = ssdAlign(imgB_sobel, imgR_sobel, 20)
		final_img = (np.dstack((np.roll(imgR,Rshift,axis=(0,1)),np.roll(imgG,Gshift,axis=(0,1)),imgB)))
		plt.imshow(final_img)
		save_result(imgname, final_img)

	#apply pyramid colorizing (large image)
	else:
		print("align large image")
		total_row_shift_g, total_col_shift_g, total_row_shift_r, total_col_shift_r = 0, 0, 0, 0

		height, width =  imgB_sobel.shape
		subsampled_scale = 16
		window_size = int((max(height, width) / subsampled_scale) / 5)
		while subsampled_scale >= 4:
			print("subsampled_scale", subsampled_scale,"window_size", window_size)
			imgB = subsampling(imgB_sobel, (int(height // subsampled_scale), int(width // subsampled_scale)))
			imgG = subsampling(imgG_sobel, (int(height // subsampled_scale), int(width // subsampled_scale)))
			imgR = subsampling(imgR_sobel, (int(height // subsampled_scale), int(width // subsampled_scale)))
			
			Gshift = ssdAlign(imgB, imgG, window_size)
			Rshift = ssdAlign(imgB, imgR, window_size)
			Gshift[0] *= subsampled_scale
			Gshift[1] *= subsampled_scale
			Rshift[0] *= subsampled_scale
			Rshift[1] *= subsampled_scale

			imgG_sobel = np.roll(imgG_sobel,Gshift,axis=(0,1))
			imgR_sobel = np.roll(imgR_sobel,Rshift,axis=(0,1))
			total_row_shift_g += int(Gshift[0])
			total_col_shift_g += int(Gshift[1])
			total_row_shift_r += int(Rshift[0])
			total_col_shift_r += int(Rshift[1])
			
			subsampled_scale = int(subsampled_scale / 2)
			window_size = int(window_size / 2)
			
		print("green image shift:", total_row_shift_g, total_col_shift_g)
		print("red image shift:", total_row_shift_r, total_col_shift_r)




		#align the origin image
		final_img = (np.dstack((np.roll(origin_imgR,(int(total_row_shift_r), int(total_col_shift_r)),axis=(0,1)),np.roll(origin_imgG,(int(total_row_shift_g), int(total_col_shift_g)),axis=(0,1)),origin_imgB)))
		#final_img = Image.fromarray(final_img)
		plt.imshow(final_img)
		save_result(imgname, final_img)



