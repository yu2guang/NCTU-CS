import os
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


def plot_pair_count(target_path, data, label):
    plt.clf()

    if label == 'Survived':
        sns.countplot(data[label])
    elif label == 'Age' or label == 'Fare':
        g = sns.FacetGrid(data, col='Survived')
        g.map(sns.distplot, label, kde=False)
    else:
        sns.countplot(data[label], hue=data['Survived'])

    plt.savefig(target_path + label + '.png')


if __name__ == '__main__':

    # info
    data_path = './titanic_data/'
    target_path = './saved/'
    predictors = ['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Family', 'Alone', 'Fare', 'Embarked']

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # read file
    train_data = pd.read_csv(data_path+'train.csv')
    test_data = pd.read_csv(data_path+'test.csv')

    # info: before
    train_data.info()
    test_data.info()

    # data evaluation
    plot_pair_count(target_path, train_data, 'Survived')
    plot_pair_count(target_path, train_data, 'Pclass')
    plot_pair_count(target_path, train_data, 'Sex')
    plot_pair_count(target_path, train_data, 'Embarked')
    plot_pair_count(target_path, train_data, 'Age')
    plot_pair_count(target_path, train_data, 'Fare')
    plot_pair_count(target_path, train_data, 'SibSp')
    plot_pair_count(target_path, train_data, 'Parch')
    train_data['Family'] = train_data['SibSp'] + train_data['Parch']
    plot_pair_count(target_path, train_data, 'Family')
    train_data['Alone'] = 0
    train_data.loc[train_data['Family'] == 0, 'Alone'] = 1
    plot_pair_count(target_path, train_data, 'Alone')

    # data preprocessing: encoding
    label_encoder = preprocessing.LabelEncoder()

    # Title: name's title, replace less title
    train_data['Title'] = train_data['Name'].str.split(',', expand=True)[1]
    train_data['Title'] = train_data['Title'].str.split('.', expand=True)[0].str.strip()
    print(train_data['Title'].value_counts())
    print(pd.crosstab(train_data['Title'], train_data['Sex']))
    train_title_age_avg = train_data.groupby(['Title'])['Age'].mean()
    print(train_title_age_avg)
    train_data['Title'] = train_data['Title'].replace(
        ['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Mlle', 'Mme', 'Ms', 'Lady', 'the Countess'],
        ['Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Miss', 'Miss', 'Miss', 'Mrs', 'Mrs'])
    print(train_data['Title'].value_counts())
    plot_pair_count(target_path, train_data, 'Title')

    # Age: fill the nan with the average age of the same title
    train_data.loc[train_data['Title'] == 'Mr', 'Age'] = train_data.loc[train_data['Title'] == 'Mr', 'Age'].fillna(train_title_age_avg['Mr'])
    train_data.loc[train_data['Title'] == 'Miss', 'Age'] = train_data.loc[train_data['Title'] == 'Miss', 'Age'].fillna(train_title_age_avg['Miss'])
    train_data.loc[train_data['Title'] == 'Mrs', 'Age'] = train_data.loc[train_data['Title'] == 'Mrs', 'Age'].fillna(train_title_age_avg['Mrs'])
    train_data.loc[train_data['Title'] == 'Master', 'Age'] = train_data.loc[train_data['Title'] == 'Master', 'Age'].fillna(train_title_age_avg['Master'])
    train_data['Embarked'] = train_data['Embarked'].fillna('S')
    train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
    train_data['Title'] = label_encoder.fit_transform(train_data['Title'])
    train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'].astype(str))
    train_data.dropna(axis=0, how='any', subset=predictors, inplace=True)

    test_data['Title'] = test_data['Name'].str.split(',', expand=True)[1]
    test_data['Title'] = test_data['Title'].str.split('.', expand=True)[0].str.strip()
    test_data['Title'] = test_data['Title'].replace(
        ['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Mlle', 'Mme', 'Ms', 'Lady', 'the Countess'],
        ['Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Miss', 'Miss', 'Miss', 'Mrs', 'Mrs'])
    test_data.loc[test_data['Title'] == 'Mr', 'Age'] = test_data.loc[test_data['Title'] == 'Mr', 'Age'].fillna(train_title_age_avg['Mr'])
    test_data.loc[test_data['Title'] == 'Miss', 'Age'] = test_data.loc[test_data['Title'] == 'Miss', 'Age'].fillna(train_title_age_avg['Miss'])
    test_data.loc[test_data['Title'] == 'Mrs', 'Age'] = test_data.loc[test_data['Title'] == 'Mrs', 'Age'].fillna(train_title_age_avg['Mrs'])
    test_data.loc[test_data['Title'] == 'Master', 'Age'] = test_data.loc[test_data['Title'] == 'Master', 'Age'].fillna(train_title_age_avg['Master'])
    test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].median())
    test_data['Sex'] = label_encoder.fit_transform(test_data['Sex'])
    test_data['Title'] = label_encoder.fit_transform(test_data['Title'])
    test_data['Embarked'] = label_encoder.fit_transform(test_data['Embarked'].astype(str))
    test_data['Family'] = test_data['SibSp'] + test_data['Parch']
    test_data['Alone'] = 0
    test_data.loc[test_data['Family'] == 0, 'Alone'] = 1

    # info: after
    train_data[predictors].info()
    test_data[predictors].info()

    # random forest model
    forest = RandomForestClassifier(oob_score=True)
    forest.fit(train_data[predictors], train_data['Survived'])
    print(forest.oob_score_)

    # predict
    preds = forest.predict(test_data[predictors])
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": preds
    })
    submission.to_csv(target_path+'submission.csv', index=False)


