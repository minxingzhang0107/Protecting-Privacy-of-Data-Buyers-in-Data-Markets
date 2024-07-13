import pandas as pd
import numpy as np


def df2data_cube(df):
    print(df.head())
    # get the column names of the dataframe
    column_names = df.columns.values.tolist()
    dimension_1 = column_names[0]
    dimension_2 = column_names[1]
    dimension_3 = column_names[2]
    dimension_4 = column_names[3]
    dimension_5 = column_names[4]
    dimension_6 = column_names[5]
    dimension_7 = column_names[6]
    dimension_8 = column_names[7]
    dimension_9 = column_names[8]
    dimension_10 = column_names[9]
    dimension_11 = column_names[10]
    # get the unique values of each column
    dimension_1_unique = df[dimension_1].unique().tolist()
    dimension_2_unique = df[dimension_2].unique().tolist()
    dimension_3_unique = df[dimension_3].unique().tolist()
    dimension_4_unique = df[dimension_4].unique().tolist()
    dimension_5_unique = df[dimension_5].unique().tolist()
    dimension_6_unique = df[dimension_6].unique().tolist()
    dimension_7_unique = df[dimension_7].unique().tolist()
    dimension_8_unique = df[dimension_8].unique().tolist()
    dimension_9_unique = df[dimension_9].unique().tolist()
    dimension_10_unique = df[dimension_10].unique().tolist()
    dimension_11_unique = df[dimension_11].unique().tolist()
    # get the number of unique values of each column
    dimension_1_size = len(dimension_1_unique)
    dimension_2_size = len(dimension_2_unique)
    dimension_3_size = len(dimension_3_unique)
    dimension_4_size = len(dimension_4_unique)
    dimension_5_size = len(dimension_5_unique)
    dimension_6_size = len(dimension_6_unique)
    dimension_7_size = len(dimension_7_unique)
    dimension_8_size = len(dimension_8_unique)
    dimension_9_size = len(dimension_9_unique)
    dimension_10_size = len(dimension_10_unique)
    dimension_11_size = len(dimension_11_unique)
    # create a data cube
    data_cube = np.zeros((dimension_1_size, dimension_2_size, dimension_3_size, dimension_4_size, dimension_5_size,
                          dimension_6_size, dimension_7_size, dimension_8_size, dimension_9_size, dimension_10_size,
                          dimension_11_size), dtype=int)
    # store the data into the data cube
    for i in range(len(df)):
        print(i)
        data_cube[dimension_1_unique.index(df[dimension_1][i])][dimension_2_unique.index(df[dimension_2][i])][
            dimension_3_unique.index(df[dimension_3][i])][dimension_4_unique.index(df[dimension_4][i])][
            dimension_5_unique.index(df[dimension_5][i])][dimension_6_unique.index(df[dimension_6][i])][
            dimension_7_unique.index(df[dimension_7][i])][dimension_8_unique.index(df[dimension_8][i])][
            dimension_9_unique.index(df[dimension_9][i])][dimension_10_unique.index(df[dimension_10][i])][
            dimension_11_unique.index(df[dimension_11][i])] += 1
    # flatten the data cube and plot the distribution
    data_cube_flatten = data_cube.flatten()
    # data_cube_flatten = data_cube_flatten[data_cube_flatten != 0]
    # describe the distribution of the data cube
    print('The mean of the data cube is: ', np.mean(data_cube_flatten))
    print('The median of the data cube is: ', np.median(data_cube_flatten))
    print('The mode of the data cube is: ', np.argmax(np.bincount(data_cube_flatten)))
    print('The standard deviation of the data cube is: ', np.std(data_cube_flatten))
    print('The minimum value of the data cube is: ', np.min(data_cube_flatten))
    print('The maximum value of the data cube is: ', np.max(data_cube_flatten))
    print('The 25th percentile of the data cube is: ', np.percentile(data_cube_flatten, 25))
    print('The 75th percentile of the data cube is: ', np.percentile(data_cube_flatten, 75))
    print("The proportion of records that is 0 is: ", np.sum(data_cube_flatten == 0) / len(data_cube_flatten))
    print("The proportion of records that is 1 is: ", np.sum(data_cube_flatten == 1) / len(data_cube_flatten))
    print("The proportion of records that is 2 is: ", np.sum(data_cube_flatten == 2) / len(data_cube_flatten))
    print("The proportion of records that is 3 is: ", np.sum(data_cube_flatten == 3) / len(data_cube_flatten))
    print("The proportion of records that is 4 is: ", np.sum(data_cube_flatten == 4) / len(data_cube_flatten))
    print("The proportion of records that is 5 is: ", np.sum(data_cube_flatten == 5) / len(data_cube_flatten))
    print("The proportion of records that is 6 is: ", np.sum(data_cube_flatten == 6) / len(data_cube_flatten))
    print("The proportion of records that is 7 is: ", np.sum(data_cube_flatten == 7) / len(data_cube_flatten))
    print("The proportion of records that is 8 is: ", np.sum(data_cube_flatten == 8) / len(data_cube_flatten))
    print("The proportion of records that is 9 is: ", np.sum(data_cube_flatten == 9) / len(data_cube_flatten))
    print("The proportion of records that is 10 is: ", np.sum(data_cube_flatten == 10) / len(data_cube_flatten))
    # proportion between 10 and 20
    print("The proportion of records that is between 10 and 20 is: ",
            np.sum((data_cube_flatten >= 10) & (data_cube_flatten < 20)) / len(data_cube_flatten))
    # proportion between 20 and 50
    print("The proportion of records that is between 20 and 50 is: ",
            np.sum((data_cube_flatten >= 20) & (data_cube_flatten < 50)) / len(data_cube_flatten))
    # proportion between 50 and 100
    print("The proportion of records that is between 50 and 100 is: ",
            np.sum((data_cube_flatten >= 50) & (data_cube_flatten < 100)) / len(data_cube_flatten))
    # proportion between 100 and 330
    print("The proportion of records that is between 100 and 330 is: ",
            np.sum((data_cube_flatten >= 100) & (data_cube_flatten < 330)) / len(data_cube_flatten))
    unique_values_on_each_dimension = [dimension_1_unique, dimension_2_unique, dimension_3_unique, dimension_4_unique,
                                       dimension_5_unique, dimension_6_unique, dimension_7_unique, dimension_8_unique,
                                       dimension_9_unique, dimension_10_unique, dimension_11_unique]
    return data_cube, unique_values_on_each_dimension


def df2data_cube_five_attributes(df):
    print(df.head())
    # get the column names of the dataframe
    column_names = df.columns.values.tolist()
    dimension_1 = column_names[0]
    dimension_2 = column_names[1]
    dimension_3 = column_names[2]
    dimension_4 = column_names[3]
    dimension_5 = column_names[4]
    # get the unique values of each column
    dimension_1_unique = df[dimension_1].unique().tolist()
    dimension_2_unique = df[dimension_2].unique().tolist()
    dimension_3_unique = df[dimension_3].unique().tolist()
    dimension_4_unique = df[dimension_4].unique().tolist()
    dimension_5_unique = df[dimension_5].unique().tolist()
    # get the number of unique values of each column
    dimension_1_size = len(dimension_1_unique)
    dimension_2_size = len(dimension_2_unique)
    dimension_3_size = len(dimension_3_unique)
    dimension_4_size = len(dimension_4_unique)
    dimension_5_size = len(dimension_5_unique)
    # create a data cube
    data_cube = np.zeros((dimension_1_size, dimension_2_size, dimension_3_size, dimension_4_size, dimension_5_size),
                         dtype=int)
    # store the data into the data cube
    for i in range(len(df)):
        print(i)
        data_cube[dimension_1_unique.index(df[dimension_1][i])][dimension_2_unique.index(df[dimension_2][i])][
            dimension_3_unique.index(df[dimension_3][i])][dimension_4_unique.index(df[dimension_4][i])][
            dimension_5_unique.index(df[dimension_5][i])] += 1
    # flatten the data cube and plot the distribution
    data_cube_flatten = data_cube.flatten()
    # data_cube_flatten = data_cube_flatten[data_cube_flatten != 0]
    # describe the distribution of the data cube
    print('The mean of the data cube is: ', np.mean(data_cube_flatten))
    print('The median of the data cube is: ', np.median(data_cube_flatten))
    print('The mode of the data cube is: ', np.argmax(np.bincount(data_cube_flatten)))
    print('The standard deviation of the data cube is: ', np.std(data_cube_flatten))
    print('The minimum value of the data cube is: ', np.min(data_cube_flatten))
    print('The maximum value of the data cube is: ', np.max(data_cube_flatten))
    print('The 25th percentile of the data cube is: ', np.percentile(data_cube_flatten, 25))
    print('The 75th percentile of the data cube is: ', np.percentile(data_cube_flatten, 75))
    print("The proportion of records that is 0 is: ", np.sum(data_cube_flatten == 0) / len(data_cube_flatten))
    print("The proportion of records that is 1 is: ", np.sum(data_cube_flatten == 1) / len(data_cube_flatten))
    print("The proportion of records that is 2 is: ", np.sum(data_cube_flatten == 2) / len(data_cube_flatten))
    print("The proportion of records that is 3 is: ", np.sum(data_cube_flatten == 3) / len(data_cube_flatten))
    print("The proportion of records that is 4 is: ", np.sum(data_cube_flatten == 4) / len(data_cube_flatten))
    print("The proportion of records that is 5 is: ", np.sum(data_cube_flatten == 5) / len(data_cube_flatten))
    print("The proportion of records that is 6 is: ", np.sum(data_cube_flatten == 6) / len(data_cube_flatten))
    print("The proportion of records that is 7 is: ", np.sum(data_cube_flatten == 7) / len(data_cube_flatten))
    print("The proportion of records that is 8 is: ", np.sum(data_cube_flatten == 8) / len(data_cube_flatten))
    print("The proportion of records that is 9 is: ", np.sum(data_cube_flatten == 9) / len(data_cube_flatten))
    print("The proportion of records that is 10 is: ", np.sum(data_cube_flatten == 10) / len(data_cube_flatten))
    # proportion between 10 and 20
    print("The proportion of records that is between 10 and 20 is: ",
            np.sum((data_cube_flatten >= 10) & (data_cube_flatten < 20)) / len(data_cube_flatten))
    # proportion between 20 and 50
    print("The proportion of records that is between 20 and 50 is: ",
            np.sum((data_cube_flatten >= 20) & (data_cube_flatten < 50)) / len(data_cube_flatten))
    # proportion between 50 and 100
    print("The proportion of records that is between 50 and 100 is: ",
            np.sum((data_cube_flatten >= 50) & (data_cube_flatten < 100)) / len(data_cube_flatten))
    # proportion between 100 and 330
    print("The proportion of records that is between 100 and 330 is: ",
            np.sum((data_cube_flatten >= 100) & (data_cube_flatten < 330)) / len(data_cube_flatten))
    unique_values_on_each_dimension = [dimension_1_unique, dimension_2_unique, dimension_3_unique, dimension_4_unique,
                                       dimension_5_unique]
    return data_cube, unique_values_on_each_dimension


def generate_unit_cost_data_cube(data_cube, lower_bound, upper_bound, random_seed):
    # generate a np array with the same shape as the data cube, while the number of each element is randomized with
    # random seed
    np.random.seed(random_seed)
    unit_cost_data_cube = np.random.randint(lower_bound, upper_bound, size=data_cube.shape)
    return unit_cost_data_cube


def generate_unit_cost_data_cube_same_price(data_cube):
    # generate a np array with the same shape as the data cube and all elements are 1
    unit_cost_data_cube = np.ones(data_cube.shape)
    return unit_cost_data_cube


if __name__ == '__main__':
    # read the csv file
    # file_path = 'adult_processed.csv'
    file_path = 'processed_data_different_combinations/adult_processed_5_attributes_1.csv'
    df = pd.read_csv(file_path, header=0, sep=',', engine='python')
    data_cube, unique_values_on_each_dimension = df2data_cube_five_attributes(df)

