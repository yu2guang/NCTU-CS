import torch
from torch.utils import data
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np


class dataLoader(data.Dataset):
    def __init__(self, src_path: str, mode: str, img_size: int):
        """
        :param src_path (string): Source path of the dataset
        :param mode (string): Indicate procedure status (train/test)
        :param img_size (int): Image size put into the model
        """
        self.src_path = src_path + mode + '_images/'
        self.mode = mode
        self.img_size = img_size
        self.coco_data = COCO(f'{src_path}{self.mode}.json')
        self.coco_id = list(self.coco_data.imgs.keys())

        if self.mode == 'train':
            self.trans = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        else:
            self.trans = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        print("> {}: Found {} images".format(self.mode, len(self.coco_id)))

    def __len__(self):
        """return the size of dataset"""
        return len(self.coco_id)

    def __getitem__(self, index):
        """
            return processed image, targets, image id, and image path
        """
        img_id = self.coco_id[index]
        img_info = self.coco_data.loadImgs(ids=img_id)[0]
        path = self.src_path + img_info['file_name']
        anns = self.coco_data.loadAnns(self.coco_data.getAnnIds(imgIds=img_id))
        targets = {'labels': [], 'boxes': [], 'masks': []}

        # image: [C, H, W], 0-1, resize
        img_origin = Image.open(path, 'r').convert('RGB')
        img, pad, resize_factor = pad_img_to_square(img_origin, self.img_size)
        img = self.trans(img)

        for ann_i in anns:
            # labels (Int64Tensor[N])
            # label name: self.coco_data.cats[int(ann_i['category_id'])]['name']
            targets['labels'].append(int(ann_i['category_id'] - 1))

            # boxes (FloatTensor[N, 4]): [top left x position, top left y position, width, height] -> [x1, y1, x2, y2]
            x1, y1, w, h = ann_i['bbox']
            x2, y2 = x1 + w, y1 + h
            targets['boxes'].append(adjust_info_value([x1, y1, x2, y2], pad[0], pad[2], resize_factor))

            # masks (UInt8Tensor[N, H, W])
            mask = self.coco_data.annToMask(ann_i)
            mask = pad_mask_to_square(mask, self.img_size, pad)
            targets['masks'].append(mask.numpy())

        # to tensor
        targets['labels'] = torch.as_tensor(targets['labels'], dtype=torch.int64)
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32)
        targets['masks'] = torch.as_tensor(targets['masks'], dtype=torch.uint8)

        return img, targets, img_id, path


    def collate_fn(self, batch):
        imgs, targets, img_ids, paths = list(zip(*batch))
        imgs = torch.stack(imgs)

        return imgs, targets, img_ids, paths


def pad_img_to_square(img, re_size):
    """
        return square & resized image, padding, and resize factor
    """

    # to tensor
    img_tensor = transforms.ToTensor()(img)

    # padding
    _, h, w = img_tensor.shape
    diff = abs(w - h)
    pad = [0, 0, diff // 2, diff - diff // 2] if w >= h else [diff // 2, diff - diff // 2, 0, 0]
    img_pad = F.pad(img_tensor, pad, "constant")

    # resize
    img_resize = F.interpolate(img_pad.unsqueeze(0), size=re_size, mode="nearest").squeeze(0)
    resize_factor = re_size / w if w >= h else re_size / h

    return img_resize, pad, resize_factor


def adjust_info_value(origin_seq, w_pad, h_pad, factor):
    """
        return the value (e.g. bbox) adjusted for the square & resized image
    """
    origin_np = np.array(origin_seq).reshape(-1, 2)
    out_seq = np.zeros(origin_np.shape)
    out_seq[:, 0] = ((origin_np[:, 0] + w_pad) * factor).astype(int)
    out_seq[:, 1] = ((origin_np[:, 1] + h_pad) * factor).astype(int)

    return out_seq.reshape(-1).tolist()


def pad_mask_to_square(mask, re_size, pad):
    """
        return square & resized mask
    """
    if pad[0] != 0:  # w
        z1 = np.zeros((mask.shape[0], pad[0]))
        z2 = np.zeros((mask.shape[0], pad[1]))
        mask_pad = np.hstack((z1, mask, z2))
    else:  # h
        z1 = np.zeros((pad[2], mask.shape[1]))
        z2 = np.zeros((pad[3], mask.shape[1]))
        mask_pad = np.vstack((z1, mask, z2))

    mask_resize = torch.as_tensor(mask_pad, dtype=torch.uint8)
    mask_resize = F.interpolate(mask_resize.unsqueeze(0).unsqueeze(0), size=[re_size, re_size], mode="nearest").squeeze()

    return mask_resize


def rescale(w, h, re_size, origin_seq=None, origin_mask=None):
    """
        return the value (bbox & mask) rescaled for the origin image
    """
    diff = abs(w - h)
    pad = [0, 0, diff // 2, diff - diff // 2] if w >= h else [diff // 2, diff - diff // 2, 0, 0]
    factor = w / re_size if w >= h else h / re_size

    out_seq = origin_seq
    if origin_seq is not None:
        origin_np = np.array(origin_seq).reshape(-1, 2) * factor
        out_seq = np.zeros(origin_np.shape)
        out_seq[:, 0] = (origin_np[:, 0] - pad[0]).astype(int)
        out_seq[:, 1] = (origin_np[:, 1] - pad[2]).astype(int)
        out_seq = out_seq.reshape(-1).tolist()

    out_mask = origin_mask

    if origin_mask is not None:
        origin_size = w if w >= h else h
        mask_resize = F.interpolate(origin_mask.unsqueeze(0), size=[origin_size, origin_size], mode="nearest").squeeze()#.squeeze(0)
        out_mask = mask_resize[pad[2]:pad[2]+h, pad[0]:pad[0]+w]

    return out_seq, out_mask


def parse_info_cfg(src_path):
    """
        return the parsed config file info
    """
    fp = open(src_path, 'r')
    data_info = {}

    for line in fp.readlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        else:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            if key in ['n_cpu', 'epoch', 'batch', 'img_size', 'grad_accum', 'n_class']:
                value = int(value)
            elif key in ['lr', 'momentum', 'decay', 'iou_thres', 'conf_thres', 'nms_thres', 'thres']:
                value = float(value)
            elif key in ['detect', 'multiscale', 'save_img']:
                value = True if value == 'True' else False
            elif key in ['class_names']:
                value = value.strip('[]').split(', ')
                value = [v.strip('\'') for v in value]

            data_info[key] = value

    fp.close()
    return data_info
