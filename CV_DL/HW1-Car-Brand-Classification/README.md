# HW1: Car Brand Classification

- 309554001 劉雨恩
- Github link
https://github.com/yu2guang/NCTU-CS/tree/master/CV_DL/HW1-Car-Brand-Classification
- Kaggle link
https://www.kaggle.com/c/cs-t0828-2020-hw1/overview

## 1. Reference

Kaiming He, Xiangyu Zhang, et al. Deep Residual Learning for Image Recognition. CVPR, 2016.
https://arxiv.org/abs/1512.03385

## 2. Introduction

本次實驗目的為架構 ResNet 神經網路藉由影像來分類 196 種車子廠牌。其步驟有三：做 data preprocessing、寫出 pretrained 的 ResNet 架構，以及將成果丟上 kaggle 算出 testing accuracy 來評估神經網路表現。需要分析比較的地方為調整 hyperparameters 跟 data preprocessing 方式，其目標是達到 testing accuracy 有 87% (baseline) 以上的表現。

## 3. Methodology

### 3.1 Data Preprocess

1. 找出在 training data 中總共有 196 個 label，並且將 label 編號 0~195
    ```python=
    # get the num of classify labels & mapping
    data = pd.read_csv(train_path + 'training_labels.csv')
    labels_name = data['label'].unique()
    label2num_dict = {name: i for i, name in enumerate(labels_name)}
    num2label_dict = {i: name for i, name in enumerate(labels_name)}
    print('> The num of classify labels is: {}'.format(len(labels_name)))
    ```
2. 輸出 testing data 所有 id 的 csv，由於 label 不會用到，因此那欄數值設為和 id 相同
    ```python=
    # output testing id csv
    img_names = glob.glob(test_path + '*.jpg')
    img_names = pd.Series(img_names).str.split('/', expand=True)[3] \
        .str.split('.', expand=True)[0]
    test_df = pd.DataFrame({
        'id': img_names,
        'label': img_names
    })
    test_df.to_csv(test_path + 'testing_labels.csv', index=False)
    print('> Testing csv saved\n')
    ```
3. Data augmentation
    - `transforms.Resize`：由於資料集裡 image 的尺寸各不相同，因此將所有資料都 resize 成 (250, 300)；如此也可訓練較大的 batch，增加對整體的準確度
    - `transforms.ToTensor`：將圖片的 [H, W, C] 轉成 [C, H, W]。 (H: Height, W: Width, C: Channel)
    - `transforms.Normalize`：歸一化成 [-1, 1] 之間，能增加模型精準度
    - `transforms.RandomPerspective`、`transforms.ColorJitter`、`transforms.RandomHorizontalFlip`：將訓練集資料做一半機率的透視變換、亮度、對比度和飽和度變換、水平翻轉；藉以產生新的資料，增加資料量以提升訓練強度，可有效避免 overfitting。而在實際實驗出來，的確能提升 testing 的準確率。 (註：在 testing 的時候要注意不要忘了把 data augmentation 拿掉，否則測出來就不準了!)
    - Training data
    ```python=
    transforms.Compose([transforms.RandomPerspective(),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((250, 300), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    ```
    - Testing data
    ```python=
    transforms.Compose([transforms.ToTensor(),
         transforms.Resize((250, 300), interpolation=Image.NEAREST),        
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    ```

### 3.2 Model (ResNet)

#### Net Arcitecture

![](https://i.imgur.com/2DEYuNo.png)

#### Solved Problem

![](https://i.imgur.com/8nhvukT.png)

ResNet 解決深層網路的問題: 

- Vanishing/Exploding gradients
- Degradation problem

添加 skip / shortcut connection 用來將 input 在經過一些 layer 後加到 output。比起直接學習 output 近似於 x，使 F(x)(=output-x)近似於 0 的計算複雜度減少許多。也因為多加了 這個 x，而使 vanishing/exploding gradients 的問題獲得解決(如下圖)。

![](https://i.imgur.com/2MW8LkH.png)

也因為直接將前面的 input 傳到較後面的 layer，能夠降低訊息的混亂或失誤；使得 ResNet 很有效的解決 degradation 的問題。

### 3.3 Hyperparameters

- Batch Size: 256
- Loss function: torch.nn.CrossEntropyLoss()
- Optimizer: Adam
- Learning rate: 1e-3 (decay 0.98 every epoch)
- Weight Decay: 1e-4
- Number of ResNet Layer: 18
- Size of ResNet Output: 196
- Pretrained: True
- Training data shuffle: True

## 4. Results

![](https://i.imgur.com/MjgnAd7.png)

epoch: 89, training accuracy: 0.996, testing accuracy: 0.58919

![](https://i.imgur.com/trI31jr.jpg)

一開始使用 ResNet 50 Layer 做為 model 訓練，效果比 baseline 還差。其可能原因如下：
- testind data 忘記做 normalize 跟 resize
- 由於網路較多層且 image resize 較大，礙於記憶體空間，batch size 只能 32，訓練不夠全面
- weight decay 不夠大，造成 overfitting

![](https://i.imgur.com/N9uBZqM.png)

epoch: 55, training accuracy: 0.998, testing accuracy: 0.88780

![](https://i.imgur.com/kusXzbk.jpg)

後來使用 ResNet 18 Layer 做為 model 訓練，有超過 baseline 了！而且 train 二三十個 epoch 就能達到 training accuracy 九十幾，模型收斂較快。在這裡做了以下改變：
- testind data 做 normalize 跟 resize
- batch size 較大，改為 256
- weight decay 較大，改為 1e-4

在這裡 data augmentation 的部分只有做 `transforms.RandomHorizontalFlip`，因此猜測可能要多做其他改變，讓訓練資料更全面

![](https://i.imgur.com/ywqIABy.png)

epoch: 646, training accuracy: 0.999, testing accuracy: 0.91140

![](https://i.imgur.com/mCvCb7v.jpg)

這裡只多做了 `transforms.RandomPerspective` 就提升了不少 testing accuracy，真不錯！

![](https://i.imgur.com/oTJpQ7T.png)

epoch: 268, training accuracy: 0.999, testing accuracy: 0.91000

![](https://i.imgur.com/zvIxhwQ.jpg)

這裡多做了 `transforms.RandomRotation(random.randrange(0, 60))` 跟將 weight decay 調大一點變 5e-4。
可能旋轉跟透視變換有些相近，所以對 testing accuracy 看起來沒什麼提升，因此在 code 的部分就先拿掉了。

![](https://i.imgur.com/q0Lx9IJ.png)

epoch: 867, training accuracy: 0.999, testing accuracy: 0.91180

![](https://i.imgur.com/tPXsKAA.jpg)

這裡多做了 `transforms.ColorJitter()` 跟將 weight decay 調回去變 1e-4。
雖然對 testing accuracy 看起來沒什麼提升，但卻是目前最好的結果。猜測提升原因為亮度、對比度和飽和度變換增加了一些之前沒看過的資料。

