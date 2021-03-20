import numpy as np

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

