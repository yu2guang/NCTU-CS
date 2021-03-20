import numpy as np
import cv2
import os

def crop(img, scale):
    height, width = img.shape
    
    y1 = int(height * scale)
    y2 = height - int(height * scale)
    x1 = int(width * scale)
    x2 = width - int(width * scale)
    
    return img[y1:y2, x1:x2]


def get_sobel_filter():
    sobel_x = np.array(
        [[1,0,-1],
         [2,0,-2],
         [1,0,-1]])
    sobel_y = sobel_x.T
    return sobel_x, sobel_y


def padding_image(img, x_size, y_size):
    img_y = img.shape[0]
    img_x = img.shape[1]
    img_pad = np.zeros((img_y + 2 * y_size, img_x + 2 * x_size))
    img_pad[1:img_y + 1, 1:img_x + 1] = img
    return img_pad
    

def do_sobel(img, mfilter):
    mfilter_y, mfilter_x = mfilter.shape
    
    padding_size_x = int((mfilter_x - 1) / 2)
    padding_size_y = int((mfilter_y - 1) / 2)
    
    img = padding_image(img, padding_size_x, padding_size_y)
    
    img_y, img_x = img.shape

    img_result = np.zeros((img_y, img_x))
    print(img_y, img_x)
    
    for i in range(padding_size_y, img_y - padding_size_y):
        for j in range(padding_size_x, img_x - padding_size_x):
            img_window = img[i - padding_size_y: i + padding_size_y + 1, j - padding_size_x: j + padding_size_x + 1]
            img_result[i, j] = np.sum(img_window * mfilter)

    return img_result[padding_size_y: -padding_size_y][padding_size_x: -padding_size_x]
    
def ssd(A, B):
    return np.sum( (A-B) ** 2)


def ssdAlign(A, B, t):
    mat = np.zeros((t*2, t*2))
    ivalue=np.linspace(-t,t,2*t,dtype=int)
    jvalue=np.linspace(-t,t,2*t,dtype=int)
    for i in range(-t, t):
        for j in range(-t, t):
            ssd_diff = ssd(A,np.roll(B,(i,j),axis=(0,1)))
            mat[i+t, j+t] = ssd_diff
            
    lowest_point = mat.argmin()
    row = (lowest_point // (t * 2)) - t
    col = (lowest_point % (t * 2)) - t

    return [row, col]


def subsampling(img, size):
    m, n = img.shape
    sy = m / size[0]
    sx = n / size[1]
    img_sub = np.zeros(size)
    for row in range(size[0]):
        for col in range(size[1]):
            r = min(row*sy, m-1)
            c = min(col*sx, n-1)
            img_sub[row][col] = img[int(r)][int(c)]
    return img_sub
    
def save_result(imgname, final_img):
    # Save result
    save_path = "task3_result/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name  = save_path + imgname + "_result.png"
    print("Save result to (%s)\n\n"%save_name)
    final_img = final_img / final_img.max()
    final_img = (final_img*255.0*255.0).astype(np.uint16)
    img_bgr = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_name, img_bgr)