import os, time, torch, random, argparse, torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet

from data_process import parse_info_cfg, AptosDataset


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def quadratic_kappa(y_hat, y, device: str='cuda:0'):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'), device=device)


def build_model(model_type):
    n_out = 1 if model_type[1] == 'reg' else 5
    criterion = nn.MSELoss() if model_type[1] == 'reg' else nn.CrossEntropyLoss()

    if model_type[0] == 'EfficientNet':
        model = EfficientNet.from_pretrained('efficientnet-b5')
        feature = model._fc.in_features
        model._fc = nn.Linear(in_features=feature, out_features=n_out, bias=True)
    else:
        model = torchvision.models.resnet50(pretrained=True)
        feature = model.fc.in_features
        model.fc = nn.Linear(in_features=feature, out_features=n_out, bias=True)

    return model, criterion


def train_model(model, model_type, optimizer, train_loader, device, criterion):
    model.train()
    predictions = []
    actual_labels = []
    avg_loss = 0.
    optimizer.zero_grad()
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.to(device), labels.float().to(device)
        output_train = model(imgs_train)
        if model_type[1] == 'class':
            loss = criterion(output_train, labels_train.long().squeeze())
            output_train = torch.max(output_train, 1)[1]
        else:
            loss = criterion(output_train, labels_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.item() / len(train_loader)
        predictions.extend(output_train.data.cpu())
        actual_labels.extend(labels_train.data.cpu())
    qk = quadratic_kappa(torch.Tensor(predictions), torch.Tensor(actual_labels))

    return avg_loss, qk


def test_model(model, model_type, valid_loader, device, criterion):
    model.eval()
    avg_val_loss = 0.
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(valid_loader):
            imgs_vaild, labels_vaild = imgs.to(device), labels.float().to(device)
            output_test = model(imgs_vaild)
            if model_type[1] == 'class':
                loss = criterion(output_test, labels_vaild.long().squeeze())
                output_test = torch.max(output_test, 1)[1]
            else:
                loss = criterion(output_test, labels_vaild)
            avg_val_loss += loss.item() / len(valid_loader)
            predictions.extend(output_test.data.cpu())
            actual_labels.extend(labels_vaild.data.cpu())
    qk = quadratic_kappa(torch.Tensor(predictions), torch.Tensor(actual_labels))

    return avg_val_loss, qk


def train(cfg_path):
    # info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_info = parse_info_cfg(cfg_path)
    train_info['saved_path'] += (train_info['model_type'][0] + '_' + train_info['model_type'][1])
    print(train_info)
    os.makedirs(train_info['saved_path'], exist_ok=True)
    seed_everything(train_info['random_seed'])

    # split train data into train and valid
    train_csv = pd.read_csv(f'{train_info["src_path"]}train.csv')
    train_df, valid_df = train_test_split(train_csv, test_size=train_info['valid_proportion'],
                                          random_state=2018, stratify=train_csv.diagnosis)

    train_data = AptosDataset(f'{train_info["train_img_path"]}', train_df, 'train', train_info['img_size'])
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
        batch_size=train_info['batch'],
        shuffle=True,
        num_workers=train_info['n_cpu'])

    valid_data = AptosDataset(f'{train_info["train_img_path"]}', valid_df, 'test', train_info['img_size'])
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
        batch_size=train_info['batch'],
        shuffle=False,
        num_workers=train_info['n_cpu'])

    # build model
    model, criterion = build_model(train_info['model_type'])
    if train_info['pretrain_path'] != 'None':
        model.load_state_dict(torch.load(train_info['pretrain_path'], map_location=device))
        print(f'Reload weight: {train_info["pretrain_path"]}')
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_info['lr'],
        weight_decay=train_info['w_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6,
                                                           threshold=0.9, min_lr=1e-6, factor=0.1, verbose=True)

    # train model
    best_qk = 0
    history = pd.DataFrame([], columns=['epoch', 'train_loss', 'val_loss', 'train_qk', 'qk', 'lr'])
    for epoch_i in range(1, train_info['epoch'] + 1):

        start_time = time.time()
        avg_loss, train_qk = train_model(model, train_info['model_type'], optimizer, train_loader, device, criterion)
        avg_val_loss, qk = test_model(model, train_info['model_type'], valid_loader, device, criterion)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t train_qk={:.4f} \tval_loss={:.4f} \t qk={:.4f} \t time={:.2f}s'.format(
            epoch_i, train_info['epoch'], avg_loss, train_qk, avg_val_loss, qk, elapsed_time))
        if epoch_i % 5 == 0:
            torch.save(model.state_dict(), f'{train_info["saved_path"]}/weight_{epoch_i}_{int(qk.item()*10000)}.pt')
            print(f'Save weight: {train_info["saved_path"]}/weight_{epoch_i}_{int(qk.item()*10000)}.pt')
        elif qk.item() > best_qk:
            best_qk = qk.item()
            torch.save(model.state_dict(), f'{train_info["saved_path"]}/weight_{epoch_i}_{int(qk.item()*10000)}.pt')
            print(f'Save weight: {train_info["saved_path"]}/weight_{epoch_i}_{int(qk.item()*10000)}.pt')

        # save train log
        row_dict = {}
        row_dict["epoch"] = epoch_i
        row_dict["val_loss"] = avg_val_loss
        row_dict["train_loss"] = avg_loss
        row_dict["qk"] = qk.item()
        row_dict["train_qk"] = train_qk.item()
        row_dict["lr"] = optimizer.param_groups[0]['lr']
        history = pd.concat([history, pd.DataFrame([row_dict])])
        scheduler.step(qk.item())
        history.to_csv(f'{train_info["saved_path"]}/history.csv')


if __name__ == '__main__':
    import sys, warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="train.cfg", help="path to the train config file")
    opt = parser.parse_args()
    print(opt)

    train(opt.cfg)