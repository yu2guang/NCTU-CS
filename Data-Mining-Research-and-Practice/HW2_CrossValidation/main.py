import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def K_fold_CV(k, x, y):
    sub_size = len(y)//k
    last_remain = len(y)%k

    train_acc, test_acc = 0, 0
    for i in range(k):
        if i < (k-1):
            idx_seq = [j for j in range(sub_size*i, sub_size*(i+1))]
        else:
            idx_seq = [j for j in range(sub_size*i, sub_size*(i+1)+last_remain)]

        test_x, test_y = x[idx_seq[0]:idx_seq[-1]+1], y[idx_seq[0]:idx_seq[-1]+1]
        train_x, train_y = x.drop(idx_seq, axis=0), np.delete(y, idx_seq, axis=0)

        # random forest model
        forest = RandomForestClassifier(oob_score=True)
        forest.fit(train_x, train_y)
        train_acc += forest.oob_score_

        # predict
        pred_y = forest.predict(test_x)
        test_acc += accuracy_score(test_y, pred_y)

    return train_acc / k, test_acc / k

def plot_pair_count(data, x_label, y_label='income', pics_path = './pics/', i=0):
    plt.clf()

    fig = plt.figure()
    if x_label == y_label:
        sns.countplot(data[x_label])
    elif x_label == 'age' or x_label == 'fnlwgt' or x_label == 'education_num' or x_label == 'capital_total' or x_label == 'hours_per_week':
        g = sns.FacetGrid(data, col=y_label)
        g.map(sns.distplot, x_label, kde=False)
    else:
        sns.countplot(data[x_label], hue=data[y_label])

    fig.autofmt_xdate()
    plt.savefig(pics_path + str(i) + '_' + x_label + '.png')

def main():

    # read data
    data = pd.read_csv('HW2data.csv')
    data.info()

    # data evaluation
    data['capital_total'] = data['capital_gain'] - data['capital_loss']
    data = data.drop(columns=['capital_gain', 'capital_loss'])
    for i, label in enumerate(data.columns):
        print(label)
        plot_pair_count(data, label, i=i)

    # data preprocessing
    x_data = data.drop(columns=['fnlwgt', 'income', 'workclass', 'education_num', 'race',
                                'hours_per_week', 'native_country', 'capital_total'])
    y_data = data['income']

    # encode: string -> int
    label_encoder = preprocessing.LabelEncoder()
    x_data['education'] = label_encoder.fit_transform(x_data['education'])
    x_data['marital_status'] = label_encoder.fit_transform(x_data['marital_status'])
    x_data['occupation'] = label_encoder.fit_transform(x_data['occupation'])
    x_data['relationship'] = label_encoder.fit_transform(x_data['relationship'])
    x_data['sex'] = label_encoder.fit_transform(x_data['sex'])
    y_data = label_encoder.fit_transform(y_data)

    # average accuracy
    train_acc, test_acc = K_fold_CV(10, x_data, y_data)
    print(train_acc, test_acc)

if __name__ == '__main__':
    main()