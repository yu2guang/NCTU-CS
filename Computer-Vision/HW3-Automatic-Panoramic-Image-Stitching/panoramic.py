import numpy as np

from image_pyramid import *
np.set_printoptions(suppress=True)

def crop_black_boundary(img):
    if len(img.shape) == 3:
        n_rows, n_cols, c = img.shape
    else:
        n_rows, n_cols = img.shape
    row_low, row_high = 0, n_rows
    col_low, col_high = 0, n_cols
    
    for row in range(n_rows):
        if np.count_nonzero(img[row]) > 0:
            row_low = row
            break
    for row in range(n_rows - 1, 0, -1):
        if np.count_nonzero(img[row]) > 0:
            row_high = row
            break
    for col in range(n_cols):
        if np.count_nonzero(img[:, col]) > 0:
            col_low = col
            break
    for col in range(n_cols - 1, 0, -1):
        if np.count_nonzero(img[:, col]) > 0:
            col_high = col
            break
    
    return img[row_low:row_high, col_low:col_high]

def bilinear_interpolate(img, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    h, w, c = img.shape

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1);
    x1 = np.clip(x1, 0, w - 1);
    y0 = np.clip(y0, 0, h - 1);
    y1 = np.clip(y1, 0, h - 1);

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id

def get_blending_mask(shape, ratio=1):
    h, w, c = shape
    blending_mask = np.zeros(shape)
    size = w * ratio
    left = max(0, int(w // 2 - size // 2))
    right = min(w, int(w // 2 + size // 2))
    blending_mask[:, :int(w // 2), :] = 1
    blending_mask[:, int(w // 2):, :] = 0
    
    for col in range(left, right):
        blending_mask[:, col, :] = 1 - (col - left) / (right - left) 
    return blending_mask
    
def multiband_blend(img1, img2):
    gaussian_filter = get_gaussian_filter(kernel_size=(5,5), sigma=1)
    _, laplacian1 = image_pyramid(img1, gaussian_filter, 6, ratio=2)
    _, laplacian2 = image_pyramid(img2, gaussian_filter, 6, ratio=2)
    
    # Now add left and right halves of images in each level
    laplacians = []
    ratios = [1, 0.8, 0.8, 0.6, 0.6, 0.4]
    for i, j, r in zip(laplacian1, laplacian2, ratios):
        alpha = get_blending_mask(i.shape, r)
        laplacians.append(i * alpha + j * (1 - alpha))

    blend = laplacians[-1]
    for i in range(len(laplacians) - 2, -1, -1):
        blend = upsampling(blend, laplacians[i].shape)
        blend += laplacians[i]
    return blend

def get_left_bound(mask):
    for col in range(mask.shape[1]):
        if not np.all(mask[:, col] == [False, False, False]):
            break
    return col

def overlap(point_a, point_b):
    return not (np.array_equal(point_a, [False, False, False]) or \
                np.array_equal(point_b, [False, False, False]))
class Panoramic():
    def __init__(self, img1, img2, homography):
        self.img1 = img1
        self.img2 = img2
        h1, w1, c = self.img1.shape
        h2, w2, c = self.img2.shape
        self.resSize = (min(h1,h2), w1+w2, c)

        self.img1_warp = self.get_warp(self.img1, None)
        self.img2_warp = self.get_warp(self.img2, homography)
        
        self.img1_mask = self.get_mask(self.img1_warp)
        self.img2_mask = self.get_mask(self.img2_warp)
    
    def get_warp(self, img, H=None):
        h, w, c = img.shape
        warp_img = np.zeros(self.resSize)

        if np.array_equal(H, None):
            warp_img[:h, :w] = img
        else:
            H_inv = np.linalg.inv(H)
            us = np.arange(warp_img.shape[1])
            vs = np.arange(warp_img.shape[0])
            us, vs = np.meshgrid(us, vs)
            uvs = np.concatenate((us.reshape(1, -1), vs.reshape(1, -1), np.ones((1, warp_img.shape[1] * warp_img.shape[0]))), axis=0)
            uvs_t = np.matmul(np.linalg.inv(H), uvs)
            uvs_t /= uvs_t[2]
            us_t = uvs_t[0]
            vs_t = uvs_t[1]
            valid = (us_t > 0) * (us_t < img.shape[1]) * (vs_t > 0) * (vs_t < img.shape[0])
            uvs_t = uvs_t[:, valid]
            uvs = uvs[:, valid]
            t = bilinear_interpolate(img, uvs_t[0], uvs_t[1])
            for i in range(uvs.shape[1]):
                warp_img[int(uvs[1][i]), int(uvs[0][i])] = t[i]
        return warp_img / 255


    def get_mask(self, img):
        mask = img[not np.array_equal(img, [0,0,0])][0]
        mask = np.array(mask, dtype=bool)
        return mask

    def normal_blend(self):
        h1, w1, c = self.img1.shape
        h2, w2, c = self.img2.shape
        blend_img = np.concatenate((self.img1_warp[:, :w1], self.img2_warp[:, w1:]), axis=1)
        return crop_black_boundary(blend_img * 255).astype(np.uint8)

    def average_blend(self):
        overlap = (self.img1_mask.astype(np.uint8) + self.img2_mask.astype(np.uint8))
        overlap[overlap == 0] = 1
        blend_img = self.img1_warp + self.img2_warp
        blend_img = blend_img / overlap
        return crop_black_boundary(blend_img * 255).astype(np.uint8)
    
    def multiband_blend(self):
        blend_img = self.img1_warp + self.img2_warp
        # calculate the region of the overlapping part
        right = self.img1.shape[1]
        left = get_left_bound(self.img2_mask)
        width = right - left + 1
        overlap1 = self.img1_warp[:, left:right]
        overlap2 = self.img2_warp[:, left:right]
        blend = multiband_blend(overlap1, overlap2)
        for col in range(left, right + 1):
            for row in range(self.img2_warp.shape[0]):
                if overlap(self.img1_mask[row, col], self.img2_mask[row, col]):
                    blend_img[row, col] = blend[row, col - left]
        return np.clip(crop_black_boundary(blend_img * 255), 0, 255).astype(np.uint8)
        
    def weight_blend(self):
        blend_img = self.img1_warp + self.img2_warp
        # calculate the region of the overlapping part
        right = self.img1.shape[1]
        left = get_left_bound(self.img2_mask)
        width = right - left + 1

        for col in range(left, right + 1):
            for row in range(self.img2_warp.shape[0]):
                alpha = (col - left) / (width)
                if overlap(self.img1_mask[row, col], self.img2_mask[row, col]):
                    blend_img[row, col] = (1 - alpha) * self.img1_warp[row, col] + \
                        alpha * self.img2_warp[row, col]
        return crop_black_boundary(blend_img * 255).astype(np.uint8)
