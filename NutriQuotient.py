#!/usr/bin/python

import sys
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from pandas.tools.plotting import scatter_matrix
import pickle
from numpy import array

clf = KNeighborsClassifier()


def load_dataset(file, type, sep=','):
    if type == 'csv':
        df = pd.read_csv(file, sep)
    elif type == 'excel':
        df = pd.read_excel(file, 'ABBREV', index_col=None, na_values=['NA'])
    else:
        print('Only CSV/Excel files in input please!!')
    df.dropna(axis=1, how='any')
    # df.replace(np.nan, 0.0, regex=True)
    return df


def cols_to_extract(df):
    macro_list = ['Shrt_Desc', 'Category', 'Protein_(g)', 'Carbohydrt_(g)', 'Lipid_Tot_(g)', 'Sodium_(mg)',
                  'Fiber_TD_(g)', 'Energ_Kcal']
    df_w_macros = df[[i for i in macro_list]]
    return df_w_macros


def cat_wise_mean(df, col):
    item_catg_mean = {}
    df.groupby(lambda x: x.iloc[0]).mean()
    print(df.head())


def load_data_knn(df):
    X = np.array(df.drop(['Shrt_Desc', 'Category'], 1))
    y = np.array(df['Category'])

    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.2)
    # np.set_printoptions(threshold=np.nan)
    # print(X_train.dtype)

    clf.fit(X_train, y_train)

    expected = y_test
    predicted = clf.predict(X_test)

    accuracy = clf.score(X_test, y_test)
    print('---------------------------------------------')
    print('Accuracy on the test dataset (20%) is {0:.2f}'.format(accuracy * 100) + "%")
    print('---------------------------------------------')


def predict_diet_cat(nut_array=[23.78, 0.57, 8.06, 534, 0.0, 352]):
    source_df = load_dataset('C:/Users/abagchee/Desktop/Hackathon_Works/food_facts.xlsx', 'excel')
    macro_df = cols_to_extract(source_df)
    # macro_df.head(30)
    labelled_df = load_data_knn(macro_df)

    prediction = clf.predict(nut_array)
    print('***********************************************************')
    print('THE MACRO QUOTIENT FOR THE INPUT FOOD ITEM IS: ', prediction)
    print('***********************************************************')


def main():
    # print command line arguments
    z = []
    for arg in sys.argv[1:]:
        z.append(float(arg))
    predict_diet_cat(array(z).reshape(1,-1))


if __name__ == "__main__":
    main()
