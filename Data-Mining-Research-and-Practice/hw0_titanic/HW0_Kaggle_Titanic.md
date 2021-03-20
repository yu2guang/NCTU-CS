# HW0: Kaggle Titanic

309554001 劉雨恩

## 1. Data Evaluation

### 1.1 Null value

```python=
import pandas as pd

# read file
train_data = pd.read_csv(data_path+'train.csv')
test_data = pd.read_csv(data_path+'test.csv')

# info: before
train_data.info()
test_data.info()
```
| Train | Test |
| ----- | ---- |
|![](https://i.imgur.com/d2R7HaO.png)| ![](https://i.imgur.com/oPFm2G9.png)|


首先使用 info() 來檢視 dataset，發現在 training data 中乘客資料總共有 891 筆，其中在 Age、Cabin 和 Embark 這幾個欄位中有空值的存在；而在 testing data 中乘客資料總共有 418 筆，其中在 Age、Fare 和 Cabin 這幾個欄位中有空值的存在。

### 1.2 Pair Comparing

檢視各個 label 與 `Survived` 的關係

#### Survived

死亡率大概為六成，生存率大概為四成。

![](https://i.imgur.com/aetIQcA.png =50%x)

```python=
import seaborn as sns
sns.countplot(train_data['Survived'])
```

#### Pclass

生存率由高而低分別為：1 艙等、2 艙等、3 艙等。

![](https://i.imgur.com/t7FPlIR.png =50%x)

```python=
sns.countplot(train_data['Pclass'], hue=data['Survived'])
```

#### Name

將 `Name` 前面的稱謂拿出來看，除了 `Mr`, `Miss`, `Mrs`, 和 `Master`，其他的稱謂都偏少。

| Sex | Age |
| --- | --- |
|![](https://i.imgur.com/qBnyU6s.png =80%x)|![](https://i.imgur.com/R3aK2Qe.png =80%x)|

```python=
print(pd.crosstab(train_data['Title'], train_data['Sex']))
train_title_age_avg = train_data.groupby(['Title'])['Age'].mean()
print(train_title_age_avg)
```

將較少的稱謂依照年齡及性別歸類於 `Mr`, `Miss`, `Mrs`, 和 `Master`，可看出 `Mr` 的死亡率明顯偏高。

![](https://i.imgur.com/DTwSXBP.png)

```python=
sns.countplot(train_data['Title'], hue=data['Survived'])
```

#### Sex

男性乘客比女性數量多，但生存率卻比女性低很多。

![](https://i.imgur.com/L4k7WGt.png =50%x)

```python=
sns.countplot(train_data['Sex'], hue=data['Survived'])
```

#### Age

小孩與老人的生存率較高。

![](https://i.imgur.com/LtNViwZ.png =80%x)

```python=
g = sns.FacetGrid(train_data, col='Survived')
g.map(sns.distplot, 'Age', kde=False)
```

#### Family (SibSp + Parch)

依照原本的資料集來看，無論無 `SibSp` 或無 `Parch`，死亡率都偏高

![](https://i.imgur.com/bJZPZTu.png =45%x) ![](https://i.imgur.com/kt0cOJF.png =45%x)

```python=
sns.countplot(train_data['SibSp'], hue=data['Survived'])
sns.countplot(train_data['Parch'], hue=data['Survived'])
```

兩資料似乎相似性頗高，因此將兄弟姊妹、伴侶及父母的人數合起來作為家人人數，可看出無家人的死亡率偏高

![](https://i.imgur.com/7A27jmo.png =50%x)

```python=
sns.countplot(train_data['Family'], hue=data['Survived'])
```

由於無家人的資料仍屬壓倒性的多，因此單純以是否獨自登船作為標準，這樣可明顯看出獨身一人死亡率較高

![](https://i.imgur.com/tsmtywh.png =50%x)

```python=
sns.countplot(train_data['Alone'], hue=data['Survived'])
```

#### Fare

票價較低的，死亡率極高。

![](https://i.imgur.com/sgEhmxP.png =80%x)

```python=
g = sns.FacetGrid(train_data, col='Survived')
g.map(sns.distplot, 'Fare', kde=False)
```

#### Embarked

在 `S` 登船的，不只人數高、死亡率也高。

![](https://i.imgur.com/R6vLrWa.png =50%x)

```python=
sns.countplot(train_data['Embarked'], hue=data['Survived'])
```

## 2. Data Preprocessing

### 2.1 Fill NAN

#### Name

將 training/testing data `Name` 的稱謂提取出來建立新屬性 `Title`，並將較少的稱謂依照年齡及性別歸類於 `Mr`, `Miss`, `Mrs`, 和 `Master`。

| Mr (male)     | Miss (female) | Mrs  (female)     | Master  (male) |
| ------------- | ------------- | ----------------- | -------------- |
| Mr (32)       | Miss (22)     | Mrs (36)          | Master (5)     |
| Capt (70)     | Mlle (24)     | Lady (48)         |                |
| Col (58)      | Mme (21)      | the Countess (33) |                |
| Don (40)      | Ms (28)       |                   |                |
| Dr (42)       |               |                   |                |
| Jonkheer (38) |               |                   |                |
| Major (48)    |               |                   |                |
| Rev (43)      |               |                   |                |
| Sir (49)      |               |                   |                |

```python=
# training data
train_data['Title'] = train_data['Name'].str.split(',', expand=True)[1]
train_data['Title'] = train_data['Title'].str.split('.', expand=True)[0].str.strip()
train_data['Title'] = train_data['Title'].replace(
    ['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Mlle', 'Mme', 'Ms', 'Lady', 'the Countess'],
    ['Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Miss', 'Miss', 'Miss', 'Mrs', 'Mrs'])

# testing data    
test_data['Title'] = test_data['Name'].str.split(',', expand=True)[1]
test_data['Title'] = test_data['Title'].str.split('.', expand=True)[0].str.strip()
test_data['Title'] = test_data['Title'].replace(
    ['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Mlle', 'Mme', 'Ms', 'Lady', 'the Countess'],
    ['Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Miss', 'Miss', 'Miss', 'Mrs', 'Mrs'])
```

#### Age

將 training/testing data 依照 `Title`，用 training data 裡對應 `Title` 的平均填補 `Age` 的缺值。

```python=
# training data
train_data.loc[train_data['Title'] == 'Mr', 'Age'] = train_data.loc[train_data['Title'] == 'Mr', 'Age'].fillna(train_title_age_avg['Mr'])
train_data.loc[train_data['Title'] == 'Miss', 'Age'] = train_data.loc[train_data['Title'] == 'Miss', 'Age'].fillna(train_title_age_avg['Miss'])
train_data.loc[train_data['Title'] == 'Mrs', 'Age'] = train_data.loc[train_data['Title'] == 'Mrs', 'Age'].fillna(train_title_age_avg['Mrs'])
train_data.loc[train_data['Title'] == 'Master', 'Age'] = train_data.loc[train_data['Title'] == 'Master', 'Age'].fillna(train_title_age_avg['Master'])
    
# testing data
test_data.loc[test_data['Title'] == 'Mr', 'Age'] = test_data.loc[test_data['Title'] == 'Mr', 'Age'].fillna(train_title_age_avg['Mr'])
test_data.loc[test_data['Title'] == 'Miss', 'Age'] = test_data.loc[test_data['Title'] == 'Miss', 'Age'].fillna(train_title_age_avg['Miss'])
test_data.loc[test_data['Title'] == 'Mrs', 'Age'] = test_data.loc[test_data['Title'] == 'Mrs', 'Age'].fillna(train_title_age_avg['Mrs'])
test_data.loc[test_data['Title'] == 'Master', 'Age'] = test_data.loc[test_data['Title'] == 'Master', 'Age'].fillna(train_title_age_avg['Master'])
```

#### Family (SibSp + Parch)

將 training/testing data 建立 `Family` (`= SibSp + Parch`) 和 `Alone` (是否有 `Family`) 類別。

```python=
# training data
train_data['Family'] = train_data['SibSp'] + train_data['Parch']
train_data['Alone'] = 0
train_data.loc[train_data['Family'] == 0, 'Alone'] = 1

# testing data
test_data['Family'] = test_data['SibSp'] + test_data['Parch']
test_data['Alone'] = 0
test_data.loc[test_data['Family'] == 0, 'Alone'] = 1
```

#### Embarked

將 training data 的 `Embarked` 缺值以最多數量的 `S` 填補。

```python=
# training data
train_data['Embarked'] = train_data['Embarked'].fillna('S')
```

#### Fare

將 testing data 的 `Fare` 缺值以中位數填補。

```python=
# testing data
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].median())
```

### 2.2 Label encoder

將 training/testing data 裡的 `Sex`, `Title`, 和 `Embarked` encode 轉為數字。

```python=
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# training data
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
train_data['Title'] = label_encoder.fit_transform(train_data['Title'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'].astype(str))

# testing data
test_data['Sex'] = label_encoder.fit_transform(test_data['Sex'])
test_data['Title'] = label_encoder.fit_transform(test_data['Title'])
test_data['Embarked'] = label_encoder.fit_transform(test_data['Embarked'].astype(str))
```

### 2.3 Predictors

設立欲要參考的屬性 list，以減輕模型負擔。

```python=
# Submission 0
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Submission 1
predictors = ['Pclass', 'Sex', 'Age', 'Family', 'Fare', 'Embarked']

# Submission 2
predictors = ['Pclass', 'Sex', 'Embarked', 'Title', 'Alone', 'Age']

# Submission 3
predictors = ['Pclass', 'Sex', 'Embarked', 'Title']

# Submission 4
predictors = ['Pclass', 'Sex', 'Embarked']
```

### 2.4 Drop NAN

將 training data 裡仍有缺失值的資料丟棄。

```python=
train_data.dropna(axis=0, how='any', subset=predictors, inplace=True)
```

## 3. Model Training

使用 Random Forest Classifier 做為 model。

```python=
# random forest model
forest = RandomForestClassifier(oob_score=True)
forest.fit(train_data[predictors], train_data['Survived'])
print(forest.oob_score_)

# predict
preds = forest.predict(test_data[predictors])
```

## 4. Results

### Submission 0

由於 `Cabin` 資料量很少，和 `PassengerId`, `Name`, `Ticket` 較跟 `Survived` 沒什麼關係，因此將他們先從 model 排除。

- predictors: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- oob score: 0.7941176470588235
- Kaggle score: 0.74401

![](https://i.imgur.com/PVwk27A.png)

### Submission 1

將 `SibSp` 和 `Parch` 併成 `Family`，可看到 Kaggle score 大概上升了 0.02。

- predictors: `Pclass`, `Sex`, `Age`, `Family`, `Fare`, `Embarked`
- oob score: 0.7955182072829131
- Kaggle score: 0.76315

![](https://i.imgur.com/4Y37DsL.png)

### Submission 2

將 `Family` 換成 `Alone`、加上 `Title`，並排除一些相似的類別，可看到 Kaggle score 大幅下降。

- predictors: `Pclass`, `Sex`, `Embarked`, `Alone`, `Title`
- oob score: 0.8237934904601572
- Kaggle score: 0.72966

![](https://i.imgur.com/07dNbVZ.png)

### Submission 3

將 `Alone` 排除，可看到 Kaggle score 比之前大概上升了 0.025。

![](https://i.imgur.com/y1vVAGm.png)

- predictors: `Pclass`, `Sex`, `Embarked`, `Title`
- oob score: 0.8305274971941639
- Kaggle score: 0.76794

### Submission 4

然而將 `Title` 排除，竟可看到 Kaggle score 比之前大概上升了 0.03。

- predictors: `Pclass`, `Sex`, `Embarked`
- oob score: 0.8114478114478114
- Kaggle score: 0.77751

![](https://i.imgur.com/rUJ9IgY.png)

