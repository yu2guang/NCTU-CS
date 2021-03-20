# HW6: Spark

- Group 7
    - 309554001 劉雨恩
    - 309706013 黃柏元
    - 0613144 葉之晴
    - 309707007 黃如君

## 1. Hadoop & Spark 安裝

### Step1. 虛擬機安裝
![](https://i.imgur.com/dFkwdyz.png =80%x) 

![](https://i.imgur.com/pVJZzos.jpg =80%x)

### Step2. 安裝 Hadoop Single Node Cluster 並啟動 Datanode
![](https://i.imgur.com/5vhFfbC.png)

### Step3. 安裝 Scala
![](https://i.imgur.com/P8GT3iD.png)

### Step4. 安裝 Spark
![](https://i.imgur.com/5J8lIB2.png)

### Step5. 安裝 Anaconda
![](https://i.imgur.com/JZJeuhs.png)

## 2. Data Preprocessing

### 2.1 Read Data, drop unused columns, and fill nan value 

```python=
import pandas as pd

data = pd.read_csv("train.csv", sep = ',')
data = data.drop(columns = ['Descript', 'Resolution', 'Address'])
data.fillna(0, inplace=True)
data.info()
data
```

![](https://i.imgur.com/debpaNc.png =50%x)
![](https://i.imgur.com/Nv8q187.png =80%x)


### 2.2 Change `DayofWeek` & `PdDistrict` to dummy features

```python=
# Change 'DayofWeek' to dummy features
all_dum = pd.get_dummies(pd.Series(data['DayOfWeek']))
new_data = pd.concat([data, all_dum], axis=1).drop(columns=['DayOfWeek'])

# Change 'PdDistrict' to dummy features
all_dum = pd.get_dummies(pd.Series(data['PdDistrict']))
new_data = pd.concat([new_data, all_dum], axis=1) \
                .drop(columns=['PdDistrict'])
new_data.info()
```

![](https://i.imgur.com/n6Mkk7r.png =50%x)

### 2.3 Take hour from `Dates` & change to dummy features

```python=
new_data['Dates'] = new_data['Dates'].str[11:-6]
all_dum = pd.get_dummies(pd.Series(new_data['Dates']))
new_data = pd.concat([new_data, all_dum], axis=1).drop(columns=['Dates'])
new_data.info()

# output preprocessed data csv
new_data.to_csv('data/train.csv')
```

![](https://i.imgur.com/CemWPC4.png =50%x)

## 3. Model Training

### Spark

訓練模型 `DecisionTree.trainClassifier`，並計算 `Accuracy`、`Precision` 和 `Recall`。

1. 將處理好的資料 (`data/train.csv`) load 進來
```python=
print(" Load  Data... ")
rawDataWithHeader = sc.textFile(Path+"data/train.csv")
header = rawDataWithHeader.first()
rawData = rawDataWithHeader.filter(lambda x:x != header)
rData = rawData.map(lambda x: x.replace("\"",""))
lines = rData.map(lambda x: x.split(","))
print("共計：" + str(lines.count()) + "筆")
lines.take(10)
```
![](https://i.imgur.com/aR6wHtI.png)

2. Construct RDD[LabeledPoint]

```python=
labelpointRDD = lines.map(lambda r:LabeledPoint(extract_label(r), \
                                        extract_features(r,len(r) - 1)))
print ("labelpointRDD = ",labelpointRDD.first(),"\n")
```
![](https://i.imgur.com/5upO1AC.png)

3. Randomly divide the data into 3 parts: train (80%), validation (10%), and test (10%)
```python=
(trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
print(" trainData : " + str(trainData.count()) + 
      " validationData : " + str(validationData.count()) +
      " testData : " + str(testData.count()))
```
![](https://i.imgur.com/7umvHNM.png)

4. 訓練模型 `pyspark.mllib.tree.DecisionTree.trainClassifier`，並計算 `Accuracy`、`Precision` 和 `Recall`
```python=
# train model
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
model = DecisionTree.trainClassifier(trainData, numClasses=39, 
                                     categoricalFeaturesInfo={}, 
                                     impurity="entropy", 
                                     maxDepth=10, maxBins=10)

# evaluate model
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = MulticlassMetrics(scoreAndLabels)
    accuracy = metrics.accuracy
    recall = metrics.recall()
    precision = metrics.precision()
    print "Accuracy = " , str(accuracy)
    print "Recall = ", str(recall)
    print "Precision = ", str(precision)

evaluateModel(model, validationData)
```

### One computer

1. Randomly split to traing (75%) and test (25%) dataset

```python=
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
                                new_data.drop(columns=['Category']), 
                                new_data['Category'], test_size=0.25, 
                                random_state=42)
```

2. 訓練模型 `sklearn.tree.DecisionTreeClassifier`，並計算 `Accuracy`、`Precision` 和 `Recall`

```python=
from sklearn import tree

# train model & predict
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_test_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calculate & plot the Confusion Matrix
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average='macro')
print('Accuracy: {}, Precision: {}, Recall: {}'.format(acc, prec, recall))
```


## 4. Results


| Value     | Spark          | One computer        |
| --------- | -------------- | ------------------- |
| Accuracy  | 0.229124771481 | 0.24598543138675158 |
| Precision | 0.229124771481 | 0.10776022682136717 |
| Recall    | 0.229124771481 | 0.10627600568659873 |