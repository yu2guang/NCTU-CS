import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def main():
    # info
    data_path = './data/'
    target_path = './saved/'

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # read file
    death_data = pd.read_csv(data_path + 'character-deaths.csv')
    death_data.info()

    # 1. Fill null value
    death_data.fillna(value=0, inplace=True)

    # 2. Create 'Death' feature
    death_data['Death'] = death_data['Death Year'] + death_data['Book of Death'] + death_data['Death Chapter']
    death_data.loc[death_data['Death'] != 0, 'Death'] = 1

    # 3. Change 'Allegiances' to dummy features
    all_dum = pd.get_dummies(pd.Series(death_data['Allegiances']))
    new_death_data = pd.concat([death_data, all_dum], axis=1).drop(
        columns=['Death Year', 'Book of Death', 'Death Chapter', 'Allegiances', 'Name'])
    new_death_data.info()

    # 4. Randomly split to training (75%) and testing (25%) dataset
    x_train, x_test, y_train, y_test = train_test_split(new_death_data.drop(columns=['Death']), new_death_data['Death'], test_size=0.25, random_state=42)

    # 5. Training model & Predict
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)

    plt.figure(figsize=(35, 10))
    tree.plot_tree(clf, feature_names=new_death_data.drop(columns=['Death']).columns, class_names=['Alive', 'Death'],
                   filled=True, max_depth=4, fontsize=10)
    plt.savefig(target_path + 'decistion_tree.png')

    # 6. Calculate & plot the Confusion Matrix
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    print('Accuracy: {}, Precision: {}, Recall: {}'.format(acc, prec, recall))

    plt.figure()
    cm = confusion_matrix(y_test, y_test_pred)
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sns.heatmap(df_cm, annot=True, cmap='GnBu')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig(target_path + 'conf_mat.png')


if __name__ == '__main__':
    main()