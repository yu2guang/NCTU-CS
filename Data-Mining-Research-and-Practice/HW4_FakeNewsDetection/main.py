import spacy, joblib, os
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def data_preprocess(data_path, mode, stop_words):
    # read data
    fp = open(f'{data_path}{mode}.csv')
    data_lines = fp.readlines()
    fp.close()

    # data preprocess
    data_df = pd.DataFrame([], columns=['id', 'text', 'label'])
    if mode == 'train':
        for i, l_i in enumerate(data_lines):
            if i == 0:
                continue
            try:
                dict_i = {}
                text_i, label = l_i.split('\t')

                dict_i['id'] = [str(i)]
                dict_i['text'] = [text_i.strip()]
                dict_i['label'] = [int(label.strip())]
                data_df = pd.concat([data_df, pd.DataFrame.from_dict(dict_i, orient='columns')])
            except:
                pass
    else:
        label_data = pd.read_csv(f'{data_path}sample_submission.csv')

        for i, (test_li, label_i) in enumerate(zip(data_lines, label_data['label'])):
            if i == 0:
                continue
            dict_i = {}
            id, text_i = test_li.split('\t')

            dict_i['id'] = [id.strip()]
            dict_i['text'] = [text_i.strip()]
            dict_i['label'] = [int(label_i)]
            data_df = pd.concat([data_df, pd.DataFrame.from_dict(dict_i, orient='columns')])

    data_df.to_csv(f'{data_path}{mode}_df.csv', index=False)

    # drop stop words
    data_df = pd.read_csv(f'{data_path}{mode}_df.csv')
    tv = TfidfVectorizer(stop_words=stop_words, max_features=10000)
    X_tf = tv.fit_transform(data_df['text']).toarray()

    joblib.dump(X_tf, f'{data_path}{mode}_tf')

    return data_df, X_tf


def fit_model(target_path, model_type, x_train, y_train, x_test, y_test, test_id):
    # model predict
    if model_type == 'XGBoost':
        model = XGBClassifier(n_estimators=100, max_features=100, max_depth=5, learning_rate=0.1)
    elif model_type == 'GBDT':
        model = GradientBoostingClassifier(n_estimators=100, max_features=100, max_depth=5, learning_rate=0.1)
    elif model_type == 'LightGBM':
        model = LGBMClassifier(n_estimators=100, num_leaves=100, max_depth=5, learning_rate=0.1)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # evaluate model
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_measure = f1_score(y_test, y_pred)
    eval_str = f'Accuracy: {acc}, Precision: {prec}, Recall: {recall}, F-measure: {f_measure}\n'
    print(eval_str)

    # submission
    submission = pd.DataFrame({
        "id": test_id,
        "label": y_pred
    })
    sub_path = f'{target_path}pred_{model_type}.csv'
    print(sub_path)
    submission.to_csv(sub_path, index=False)

    return eval_str


def main():
    # info
    data_path = './data/'
    target_path = './saved/'
    os.makedirs(target_path, exist_ok=True)

    # --- data preprocessing --- #
    spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    train_df, train_tfarray = data_preprocess(data_path, 'train', spacy_stopwords)
    test_df, test_tfarray = data_preprocess(data_path, 'test', spacy_stopwords)

    # --- model training --- #
    model_type = ['XGBoost', 'GBDT', 'LightGBM']
    fp = open(f'{target_path}evaluate.txt', 'w')
    for model_i in model_type:
        mode_str = f'\n### {model_i} ###\n'
        print(mode_str, end='')
        fp.write(mode_str)

        y_train = np.array(train_df['label'].values)
        y_test = np.array(test_df['label'].values)
        eval_str = fit_model(target_path, model_i, train_tfarray, y_train,
                             test_tfarray, y_test, test_df['id'])
        fp.write(eval_str)
    fp.close()



    pass


if __name__ == '__main__':
    import sys, warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    main()