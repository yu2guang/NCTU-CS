import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.fft import fft2, ifft2
import os
import glob

def get_gaussian_filter(kernel_size=(3,3), sigma=1):
    op = lambda x, y, sigma: (1/(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2) / (2.*sigma**2))
    gaussian_kernel = np.zeros(kernel_size)
    center_x = (kernel_size[0] + 1) // 2 - 1
    center_y = (kernel_size[1] + 1) // 2 - 1
    for u in range(kernel_size[0]):
        for v in range(kernel_size[1]):
            gaussian_kernel[v, u] = op(u - center_x, v - center_y, sigma)
    gaussian_kernel /= gaussian_kernel.sum()
    return gaussian_kernel
    
def zero_padding(img, size=(1,1)):
    r_pad, c_pad = size
    row, column, ch = img.shape
    img_pad = np.zeros((row + 2 * r_pad, column + 2 * c_pad, ch), dtype=np.uint8)
    img_pad[r_pad:-r_pad, c_pad:-c_pad] = img[:, :]
    return img_pad
    
def conv2D(img, kernel):
    k_y, k_x = kernel.shape
    
    p_x = (k_x - 1) // 2
    p_y = (k_y - 1) // 2
    img = zero_padding(img, size=(p_y,p_x))
    row, column, ch = img.shape
    
    output_img = np.copy(img)
    for y in range(p_y, row - p_y):
        for x in range(p_x, column - p_x):
            for c in range(ch):
                img_window = img[(y - p_y):(y + p_y) + 1:, (x - p_x):(x + p_x) + 1, c]
                output_img[y, x, c] = np.sum(img_window * kernel)
            
    output_img = output_img[p_y:-p_y, p_x:-p_x]
    return output_img

def subsampling(img, ratio):
    row, column, ch = img.shape
    img_sub = np.zeros((row // ratio, column // ratio, ch), dtype=np.uint8)
    for y in range(img_sub.shape[0]):
        for x in range(img_sub.shape[1]):
            r = min(y * ratio, row - 1)
            c = min(x * ratio, column - 1)
            img_sub[y][x] = img[int(r)][int(c)]
    return img_sub

def upsampling(img, size):
    row, column, ch = img.shape
    r_y = size[0] // row
    r_x = size[1] // column
    img_up = np.zeros(size, dtype=np.float64)
    for y in range(row):
        for x in range(column):
            y_low = y * r_y
            y_high = y_low + r_y
            x_low = x * r_x
            x_high = x_low + r_x
            img_up[y_low:y_high, x_low:x_high] = img[y, x] 
    return img_up

def image_pyramid(img, gaussian_filter, num_layers=5, ratio=2):
    gaussian_pyramid = [img]
    laplacian_pyramid = []
    for i in range(num_layers):
        # Gaussian Pyramid
        img_layer = gaussian_pyramid[-1]
        row, column, ch = img_layer.shape
        smooth_img = conv2D(img_layer, gaussian_filter)
        img_layer = subsampling(smooth_img, ratio)
        gaussian_pyramid.append(img_layer)

        # Laplacian Pyramid
        img_high = gaussian_pyramid[i]
        img_low = gaussian_pyramid[i+1]
        img_up = upsampling(img_low, img_high.shape)
        laplacian = img_high - img_up
        laplacian_pyramid.append(laplacian)

    return gaussian_pyramid, laplacian_pyramid

def get_magnitude(pyramid):
    magnitudes = [] 
    for i, img in enumerate(pyramid):
        row, column, ch = img.shape
        for c in range(ch):
            sub_img = img[:, :, c]
            dft = np.fft.fft2(sub_img)
            dft_shift = np.fft.fftshift(dft)
            magnitude = np.log(np.abs(dft_shift))
            magnitudes.append(magnitude)
    return magnitudes

def plot_magnitude(magnitudes, GRAY):
    plt.figure(figsize = (20,10))
    low = 1e+10
    high = -1e+10
    for magnitude in magnitudes:
        low = min(min(magnitude.reshape(-1)), low)
        high = max(max(magnitude.reshape(-1)), high)

    ch = len(magnitudes) // num_layers
    for i, magnitude in enumerate(magnitudes):
        norm_fourier = (magnitude - low) / (high - low) * 255
        plt.subplot(ch, num_layers, num_layers*(i%ch)+i//ch + 1), plt.axis('off'), 
        plt.imshow(norm_fourier, cmap='gray')
    plt.show()

def plot_pyramid(pyramid, GRAY, seperate=False):
    plt.figure(figsize = (20,10))
    for i, img in enumerate(pyramid):
        if GRAY:
            plt.subplot(1, num_layers, i+1), plt.axis('off')
            plt.imshow(img[:, :, 0], cmap='gray')
        else:
            if seperate:
                _, _, ch = img.shape
                for j in range(ch):
                    sub_img = img[:, :, j]
                    plt.subplot(ch, num_layers, num_layers*(j)+i + 1), plt.axis('off'), 
                    plt.imshow(sub_img, cmap='gray')
            else:
                plt.subplot(1, num_layers, i+1), plt.axis('off')
                plt.imshow(img)
    plt.show()


def plot_result(img, GRAY):
    gaussian_pyramid, laplacian_pyramid = image_pyramid(img, gaussian_filter, num_layers, ratio=2)

    # Plot Gaussian pyramid
    plot_pyramid(gaussian_pyramid[:-1], GRAY)
    # Plot Laplacian pyramid
    plot_pyramid(laplacian_pyramid, GRAY, False)

    # Plot Gaussian magnitude
    magnitudes = get_magnitude(gaussian_pyramid[:-1])
    plot_magnitude(magnitudes, GRAY)
    # Plot Laplacian magnitude
    magnitudes = get_magnitude(laplacian_pyramid)
    plot_magnitude(magnitudes, GRAY)


if __name__ == '__main__':
    #srcPath = '../hw2_data/task1and2_hybrid_pyramid/'
    srcPath = '../hw2_data/our/'
    targetPath = './results/'
    filterType = ['ideal', 'guassian'] 
    colorType = ['gray', 'color']
    if not os.path.exists(targetPath):
        os.makedirs(targetPath)	

    datas = glob.glob('{}*'.format(srcPath))
    for data_path in datas:
        img = cv2.imread(data_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gaussian_filter = get_gaussian_filter(kernel_size=(5,5), sigma=1)
        num_layers = 5

        img_gray = img_gray[:, :, None]
        # gray version
        # plot_result(img_gray, True)
        # color version
        plot_result(img, False)

