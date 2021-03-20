import torch, torchvision, os, time, argparse
import torch.utils.data as Data

from data_process import parse_info_cfg, dataLoader


def lr_decay(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*0.99


def train(cfg_path, pretrain_imageNet=True):
    # info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_info = parse_info_cfg(cfg_path)
    print(train_info)
    os.makedirs(train_info['pkls_path'], exist_ok=True)

    # load data
    train_data = dataLoader(train_info['src_path'], 'train', train_info['img_size'])
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=train_info['batch'],
        shuffle=True,
        num_workers=train_info['n_cpu'],
        collate_fn=train_data.collate_fn
    )

    # load an instance segmentation model pre-trained on ImageNet
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=train_info['n_class'],
                                                               pretrained_backbone=pretrain_imageNet).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_info['lr'],
        momentum=train_info['momentum'],
        weight_decay=train_info['decay']
    )

    # train
    for epoch_i in range(1, train_info['epoch']+1):
        print('\n# --- {} Epoch {} --- #'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_i))
        model.train()
        for batch_i, (imgs, targets, _, _) in enumerate(train_loader, 1):
            imgs = imgs.to(device)
            for j in range(len(targets)):
                targets[j]['labels'] = targets[j]['labels'].to(device)
                targets[j]['boxes'] = targets[j]['boxes'].to(device)
                targets[j]['masks'] = targets[j]['masks'].to(device)

            # clear gradient
            optimizer.zero_grad()

            # forward & backward
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            print(losses.item())
            losses.backward()

            # update parameters
            optimizer.step()

        lr_decay(optimizer)

        # save model
        torch.save(model.state_dict(),
                   f'{train_info["pkls_path"]}maskRCNN_{epoch_i}.pkl')


if __name__ == '__main__':
    import sys, warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="train.cfg", help="path to the train config file")
    opt = parser.parse_args()
    print(opt)

    train(opt.cfg)
