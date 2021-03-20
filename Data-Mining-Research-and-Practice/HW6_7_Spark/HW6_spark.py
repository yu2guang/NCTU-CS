
# coding: utf-8


# Set Path
global Path
if sc.master[0:5] == "local":
    Path = "file:/home/hduser/pythonwork/PythonProject/"
else:
    Path = "hdfs://master:9000/user/hduser/"



def convert_float(x):
    return (0 if x=="?" else float(x))



def extract_label(record):
    label=(record[-1])
    return float(label)-1

In[78]:

def extract_features(record, featureEnd):
    numericalFeatures = [convert_float(field)  for  field in record[0: featureEnd]]    
    return  numericalFeatures




#----------------------1. Load Data-------------
print(" Load  Data... ")
rawDataWithHeader = sc.textFile(Path+"data/train.csv")
header = rawDataWithHeader.first()
rawData = rawDataWithHeader.filter(lambda x:x != header)
rData = rawData.map(lambda x: x.replace("\"",""))
lines = rData.map(lambda x: x.split(","))
print("共計：" + str(lines.count()) + "筆")



lines.take(10)




#----------------------2. Construct RDD[LabeledPoint]-------------
labelpointRDD = lines.map( lambda r:LabeledPoint(extract_label(r), extract_features(r,len(r) - 1)))
print ("labelpointRDD = ",labelpointRDD.first(),"\n")




#----------------------3. Randomly divide the data into 3 parts -------------
(trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
print(" trainData : " + str(trainData.count()) + 
          " validationData : " + str(validationData.count()) +
          " testData : " + str(testData.count()))




trainData.persist()
validationData.persist()
testData.persist()




import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics




#----------------------4. Train Model -------------
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
model = DecisionTree.trainClassifier(trainData,numClasses=39,categoricalFeaturesInfo={},impurity="entropy",maxDepth=10,maxBins=10)


#----------------------5. Accuracy, Recall, Preciosion -------------
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






