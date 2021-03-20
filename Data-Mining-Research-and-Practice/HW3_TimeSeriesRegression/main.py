import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def main():
    # --- read data --- #
    fp = open(r'./新竹_2019.csv', encoding='gb18030')
    data = pd.read_csv(fp)
    data.info()
    fp.close()

    # --- data preprocessing --- #

    # drop unused data
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop([0], axis=0, inplace=True)

    # drop blank
    data.columns = data.columns.str.strip()
    for label in data.columns:
        data[label] = data[label].str.strip()

    # rename Chinese to English
    data.rename(columns={data.columns[0]: 'date', data.columns[1]: 'test_item'}, inplace=True)
    data = data.reset_index(drop=True)

    # drop month 1~9
    data = data.loc[data['date'] >= '2019/10/01']
    print(data.head(10))

    # process missing value
    data = data.replace(to_replace=r'.*[0-9]+(#|\*|x|A)$', value=np.nan, regex=True)
    for i, label in enumerate(data.columns):
        if i > 1:
            data[label] = data[label].astype(float)

    data_num = data[data.columns[2:]]
    data_num.interpolate(method='linear', axis=1, inplace=True, limit_direction='both')
    data[data.columns[2:]] = data_num

    # 10, 11 -> train data, 12 -> test data
    train_data = data.loc[data['date'] < '2019/12/01']
    train_data.drop(data.columns[0], axis=1, inplace=True)
    test_data = data.loc[data['date'] >= '2019/12/01']
    test_data.drop(data.columns[0], axis=1, inplace=True)

    # group by 18 classes
    new_train_data = pd.DataFrame()
    for name, group in train_data.groupby(['test_item']):
        new_train_data[name] = np.array(group.iloc[:, 1:]).reshape(-1)

    new_test_data = pd.DataFrame()
    for name, group in test_data.groupby(['test_item']):
        new_test_data[name] = np.array(group.iloc[:, 1:]).reshape(-1)

    new_train_data.to_csv('train_data.csv', index=False)
    new_test_data.to_csv('test_data.csv', index=False)

    # --- model training --- #
    new_train_data.info()
    new_test_data.info()

    train_len, test_len = len(new_train_data), len(new_test_data)
    train_np, test_np = np.array(new_train_data), np.array(new_test_data)
    train25_np, test25_np = np.array(new_train_data['PM2.5']), np.array(new_test_data['PM2.5'])
    slice_num = 6

    for future_i in [1, 6]:
        # training data
        all_train_x, all_train_y = [], []
        PM25_train_x, PM25_train_y = [], []
        for row_i in range(train_len - slice_num - future_i + 1):
            all_train_x.append(train_np[row_i:row_i + slice_num].T)
            all_train_y.append(train_np[row_i + slice_num])
            PM25_train_x.append(train25_np[row_i:row_i + slice_num])
            PM25_train_y.append(train25_np[row_i + slice_num])
        all_train_x = np.array(all_train_x).reshape(-1, 6)
        all_train_y = np.array(all_train_y).reshape(-1)
        PM25_train_x = np.array(PM25_train_x).reshape(-1, 6)
        PM25_train_y = np.array(PM25_train_y).reshape(-1)

        # testing data
        all_test_x, all_test_y = [], []
        PM25_test_x, PM25_test_y = [], []
        for row_i in range(test_len - slice_num - future_i + 1):
            all_test_x.append(test_np[row_i:row_i + slice_num].T)
            all_test_y.append(test_np[row_i + slice_num])
            PM25_test_x.append(test25_np[row_i:row_i + slice_num])
            PM25_test_y.append(test25_np[row_i + slice_num])
        all_test_x = np.array(all_test_x).reshape(-1, 6)
        all_test_y = np.array(all_test_y).reshape(-1)
        PM25_test_x = np.array(PM25_test_x).reshape(-1, 6)
        PM25_test_y = np.array(PM25_test_y).reshape(-1)

        for PM25_i in [True, False]:

            if PM25_i:
                train_x = PM25_train_x
                train_y = PM25_train_y
                test_x = PM25_test_x
                test_y = PM25_test_y
            else:
                train_x = all_train_x
                train_y = all_train_y
                test_x = all_test_x
                test_y = all_test_y


            # Linear Regression
            Lreg = LinearRegression().fit(train_x, train_y)
            Lreg_score = Lreg.score(train_x, train_y)
            Lreg_y = Lreg.predict(test_x)
            Lreg_MAE = mean_absolute_error(test_y, Lreg_y)

            # Random Forest Regression
            RFreg = RandomForestRegressor(oob_score=True)
            RFreg.fit(train_x, train_y)
            RFreg_score = RFreg.oob_score_
            RFreg_y = RFreg.predict(test_x)
            RFreg_MAE = mean_absolute_error(test_y, RFreg_y)

            print('\nFuture {} hour after & PM2.5 only {}'.format(future_i, PM25_i))
            print('Linear Regression: {}(score), {}(MAE)'.format(Lreg_score, Lreg_MAE))
            print('Random Forest Regression: {}(score), {}(MAE)'.format(RFreg_score, RFreg_MAE))

if __name__ == '__main__':
    main()