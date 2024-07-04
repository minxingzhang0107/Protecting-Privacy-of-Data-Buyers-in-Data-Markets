import pandas as pd
import numpy as np


def read_raw_data():
    file_path = 'adult.data'
    df = pd.read_csv(file_path, header=None, sep=', ', engine='python')
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                  'income']
    print(df.head())
    print(df.shape)
    # save to csv
    df.to_csv('adult_raw.csv', index=False)


def continuous2categorical():
    # read the csv file
    file_path = 'adult_raw.csv'
    df = pd.read_csv(file_path, header=0, sep=',', engine='python')
    print(df.head())
    # for the age column, divide it into 'childhood' (age under 18), 'young adult' (age between 18 and 24),
    # 'working adult' (age between 25 and 62), 'retirement' (62 and above)
    df['age'] = pd.cut(df['age'], bins=[0, 17, 24, 61, np.inf], labels=['childhood', 'young adult', 'working adult',
                                                                        'retirement'], right=True)
    print(df.head())
    # print the distribution of age
    print(df['age'].value_counts())

    # print the range of "capital gain" and "capital loss"
    print(df['capital-gain'].describe())
    print(df['capital-loss'].describe())
    # for capital gain, divide it into "small gain" (0 - 9999), "medium gain" (10000 - 49999), "large gain" (50000
    # and above)
    df['capital-gain'] = pd.cut(df['capital-gain'], bins=[-1, 9999, 49999, np.inf], labels=['small gain', 'medium gain',
                                                                                            'large gain'], right=True)
    # print the distribution of capital gain
    print(df['capital-gain'].value_counts())

    # working hours per week
    print(df['hours-per-week'].describe())
    # divide it into "part-time" (0 - 34), "full-time" (35 - 40), "overtime" (41 and above)
    df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=[-1, 34, 40, np.inf], labels=['part-time', 'full-time',
                                                                                   'overtime'], right=True)
    # print the distribution of working hours per week
    print(df['hours-per-week'].value_counts())

    # remove fnlwgt, education-num, capital-gain, capital-loss
    df = df.drop(['fnlwgt', 'education-num', 'capital-gain', 'capital-loss'], axis=1)
    # remove the rows with missing values for any column
    df = df.dropna(axis=0, how='any')
    # remove the rows with '?' for any column
    df = df[(df != '?').all(axis=1)]

    # save to csv
    df.to_csv('adult_processed.csv', index=False)

    # select different combinations of attributes
    # df = df[['age', 'education', 'marital-status', 'race', 'sex']]
    df = df[['age', 'race', 'sex', 'hours-per-week', 'income']]
    # df = df[['age', 'relationship', 'race', 'sex', 'income']]
    # df = df[['age', 'workclass', 'race', 'sex', 'income']]
    # df = df[['age', 'marital-status', 'race', 'sex', 'income']]
    # df = df[['relationship', 'race', 'sex', 'hours-per-week', 'income']]
    # df = df[['marital-status', 'relationship', 'race', 'sex', 'income']]
    # df = df[['marital-status', 'race', 'sex', 'hours-per-week', 'income']]
    # df = df[['workclass', 'race', 'sex', 'hours-per-week', 'income']]
    # df = df[['age', 'relationship', 'sex', 'hours-per-week', 'income']]
    # save to csv in 'processed_data_different_combinations' folder
    df.to_csv('processed_data_different_combinations/adult_processed_5_attributes_1.csv', index=False)


if __name__ == '__main__':
    # read_raw_data()
    read_raw_data()
    # change the continuous variables to categorical variables
    continuous2categorical()


