import torch, torchvision, os, json, tqdm, random, argparse
import torch.utils.data as Data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from data_process import parse_info_cfg, dataLoader, rescale
from utils import binary_mask_to_rle


def plot_img(img_path, detections, re_size, saved_path, class_names):

    # Create plot
    img_np = np.array(Image.open(img_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    plt.axis("off")
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 100)]
    bbox_colors = random.sample(colors, len(detections['boxes']))
    for i in range(len(detections['labels'])):
        color = bbox_colors[i]

        # Create a Rectangle patch
        out_seq, out_mask = rescale(img_np.shape[1], img_np.shape[0], re_size, detections['boxes'][i], detections['masks'][i])
        x1, y1, x2, y2 = out_seq
        bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")

        # Add the bbox to the plot
        ax.add_patch(bbox)

        # Add label
        plt.text(
            x1,
            y1,
            s=class_names[int(detections['labels'][i])],
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )

        ax.imshow(out_mask, cmap='jet', alpha=0.3)

    # Save generated image with detections
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(saved_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def test(cfg_path, pretrain_imageNet=True):
    # info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_info = parse_info_cfg(cfg_path)
    print(test_info)
    os.makedirs(test_info['img_path'], exist_ok=True)

    # load data
    test_data = dataLoader(test_info['src_path'], test_info['mode'], test_info['img_size'])
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=test_info['batch'],
        shuffle=False,
        num_workers=test_info['n_cpu'],
        collate_fn=test_data.collate_fn
    )

    # load an instance segmentation model pre-trained on ImageNet
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=test_info['n_class'],
                                                               pretrained_backbone=pretrain_imageNet).to(device)

    # test
    submit_seq = []
    model.load_state_dict(torch.load(test_info['pkl_path'], map_location=device))
    model.eval()
    with torch.no_grad():
        for imgs, _, img_ids, paths in tqdm.tqdm(test_loader):
            imgs = imgs.to(device)
            detections = model(imgs)

            for detect_i, img_id_i, path in zip(detections, img_ids, paths):
                detect_i = {key: values.cpu() for key, values in detect_i.items()}

                img_np = np.array(Image.open(path).convert('RGB'))
                for j, l_j in enumerate(detect_i['labels']):
                    if detect_i['scores'][j] >= test_info['thres']:
                        detect_dict = {}
                        detect_dict['image_id'] = int(img_id_i)
                        detect_dict['score'] = float(detect_i['scores'][j])
                        detect_dict['category_id'] = int(l_j) + 1
                        _, out_mask = rescale(img_np.shape[1], img_np.shape[0], test_info['img_size'], origin_mask=detect_i['masks'][j])
                        out_mask = torch.round(out_mask)  # binary mask
                        detect_dict['segmentation'] = binary_mask_to_rle(out_mask.numpy())

                        submit_seq.append(detect_dict)

                        if test_info['save_img']:
                            plot_img(path, detect_i, test_info['img_size'], f'{test_info["img_path"]}plot_{path.split("/")[-1]}', test_info['class_names'])

    # dump json file
    json_path = test_info['saved_path'] + test_info['pkl_path'].split('/')[3].split('.')[0] + f'_{int(test_info["thres"]*100)}.json'
    print(json_path)
    with open(json_path, 'w') as outfile:
        json.dump(submit_seq, outfile)


if __name__ == '__main__':
    import sys, warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="test.cfg", help="path to the test config file")
    opt = parser.parse_args()
    print(opt)

    test(opt.cfg)
