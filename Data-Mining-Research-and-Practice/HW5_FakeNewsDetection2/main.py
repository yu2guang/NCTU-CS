import spacy, os, re, keras
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM
import matplotlib.pyplot as plt


def data_preprocess(data_path, mode, stop_words):

    def rm_tags_stops(text, stop_words):
        # remove tags
        re_tag = re.compile(r'<[^>]+>')
        out_text = re_tag.sub(' ', text.lower())

        # remove stop words
        out_text = (" ").join([word for word in out_text.split(' ') if not word in stop_words])

        return out_text

    # read data
    fp = open(f'{data_path}{mode}.csv')
    data_lines = fp.readlines()
    fp.close()

    # data preprocess
    data_df = pd.DataFrame([], columns=['id', 'text', 'label'])
    avg_text_len = 0
    if mode == 'train':
        for i, l_i in enumerate(data_lines):
            if i == 0:
                continue
            try:
                dict_i = {}
                text_i, label = l_i.split('\t')
                text_i = rm_tags_stops(text_i.strip(), stop_words)
                avg_text_len += len(text_i.split(' '))

                dict_i['id'] = [str(i)]
                dict_i['text'] = [text_i]
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
            text_i = rm_tags_stops(text_i.strip(), stop_words)
            avg_text_len += len(text_i.split(' '))

            dict_i['id'] = [id.strip()]
            dict_i['text'] = [text_i]
            dict_i['label'] = [int(label_i)]
            data_df = pd.concat([data_df, pd.DataFrame.from_dict(dict_i, orient='columns')])
    print(f'{mode} avg text len: {avg_text_len / len(data_df)}')

    return data_df['text'], np.array(data_df['label'].values)


def train(target_path, model_type, params, x_train, y_train, x_test, y_test):
    # set up model
    model = Sequential()
    model.add(Embedding(output_dim=128,
                        input_dim=params['dict_len'],
                        input_length=params['content_len']))
    model.add(Dropout(params['drop_out']))
    if model_type == 'RNN':
        model.add(SimpleRNN(units=128))
    else:
        model.add(LSTM(units=128))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(params['drop_out']))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()

    # define model
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])

    # training
    fp_train_loss = open(f'{target_path}{model_type}_train_loss.txt', 'w')
    fp_train_acc = open(f'{target_path}{model_type}_train_acc.txt', 'w')
    fp_test_acc = open(f'{target_path}{model_type}_test_acc.txt', 'w')

    for epoch_i in range(1, params['epoch'] + 1):
        train_history = model.fit(x_train, y_train,
                                  epochs=1, batch_size=params['batch_size'],
                                  shuffle=True, workers=12,
                                  verbose=0)
        train_loss = train_history.history["loss"][0]
        train_acc = train_history.history["accuracy"][0]
        print(f'\nEpoch {epoch_i}/{params["epoch"]}')
        print(f'Loss: {train_loss}, Accuracy: {train_acc}')
        fp_train_loss.write(f'{train_loss}\n')
        fp_train_acc.write(f'{train_acc}\n')

        score = model.evaluate(x_test, y_test, verbose=0, workers=12)
        print(f'Testing accuracy: {score[1]}')
        fp_test_acc.write(f'{score[1]}\n')

    fp_train_loss.close()
    fp_train_acc.close()
    fp_test_acc.close()


def plt_train_process(target_path, model_type):
    # read file
    fp = open(f'{target_path}{model_type}_train_loss.txt', 'r')
    train_loss_lines = fp.readlines()
    fp.close()
    fp = open(f'{target_path}{model_type}_train_acc.txt', 'r')
    train_acc_lines = fp.readlines()
    fp.close()
    fp = open(f'{target_path}{model_type}_test_acc.txt', 'r')
    test_acc_lines = fp.readlines()
    fp.close()

    # to float
    train_loss_lines = [float(loss.strip()) for loss in train_loss_lines]
    train_acc_lines = [float(acc.strip())*100 for acc in train_acc_lines]
    test_acc_lines = [float(acc.strip())*100 for acc in test_acc_lines]
    x = [int(i) for i in range(len(train_acc_lines))]

    # plot loss
    fig = plt.figure()

    plt.plot(x, train_loss_lines, color='#005ab5')

    plt.title('Training Loss', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    fig.savefig(f'{target_path}{model_type}_train_loss.jpg')
    print(f'{target_path}{model_type}_train_loss.jpg')

    # plot accuarcy
    fig = plt.figure()

    plt.plot(x, train_acc_lines, label='Train')
    plt.plot(x, test_acc_lines, label='Test')

    plt.title('Accuarcy', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Accuarcy (%)')
    plt.legend(loc='upper left')

    fig.savefig(f'{target_path}{model_type}_accuarcy.jpg')
    print(f'{target_path}{model_type}_accuarcy.jpg')


def main():
    # info
    data_path = './data/'
    target_path = './saved/'
    os.makedirs(target_path, exist_ok=True)

    hyper_params = {'epoch': 100, 'batch_size': 4987, 'drop_out': 0.7,
              'dict_len': 3800, 'content_len': 380}

    # --- data preprocessing --- #
    # remove stop words
    spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    text_train, y_train = data_preprocess(data_path, 'train', spacy_stopwords)
    text_test, y_test = data_preprocess(data_path, 'test', spacy_stopwords)

    # create token
    token = Tokenizer(num_words=hyper_params['dict_len'])
    token.fit_on_texts(text_train.values)
    x_train = token.texts_to_sequences(text_train.values)
    x_train = sequence.pad_sequences(x_train, maxlen=hyper_params['content_len'])
    x_test = token.texts_to_sequences(text_test.values)
    x_test = sequence.pad_sequences(x_test, maxlen=hyper_params['content_len'])

    # --- model training --- #
    model_type = ['RNN', 'LSTM']
    for model_i in model_type:
        train(target_path, model_i, hyper_params, x_train, y_train, x_test, y_test)
        plt_train_process(target_path, model_i)


if __name__ == '__main__':
    main()