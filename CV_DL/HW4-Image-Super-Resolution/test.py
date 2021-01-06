import os, glob, argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='./saved/SRCNN/epoch_x_xx.xx.pth')
    parser.add_argument('--images-dir', type=str, default='./data/testing_lr_images/')
    parser.add_argument('--outputs-dir', type=str, default='./saved/test_out/')
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()
    print(args)

    os.makedirs(args.outputs_dir, exist_ok=True)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    psnr_seq = []
    for image_path in sorted(glob.glob('{}*.png'.format(args.images_dir))):

        image = pil_image.open(image_path).convert('RGB')
        cur_w, cur_h = image.width * args.scale, image.height * args.scale
        image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        psnr = calc_psnr(y, preds)
        psnr_seq.append(psnr.cpu().item())
        print('{} PSNR: {:.2f}'.format(image_path.split('/')[-1], psnr))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(args.outputs_dir + image_path.split('/')[-1])

    print(f'Average PSNR: {np.mean(psnr_seq)}')
