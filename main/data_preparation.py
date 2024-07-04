import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import sys
from scipy.stats import truncnorm
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adult_data'))
from adult_data import helper


def create_true_intent_random(unique_values_on_each_dimension, random_seed):
    true_intent = []
    for i in range(len(unique_values_on_each_dimension)):
        if i == 1:
            # skip the second dimension by appending everything
            current_dimension_true_intent = unique_values_on_each_dimension[i]
            true_intent.append(current_dimension_true_intent)
            continue
        current_dimension_unique_values = unique_values_on_each_dimension[i]
        total_length = len(current_dimension_unique_values)
        np.random.seed(random_seed)
        # k equals a random number between 1 and total_length
        k = np.random.randint(1, total_length)
        # k = 1
        # randomly select k unique values from current_dimension_unique_values
        current_dimension_true_intent = np.random.choice(current_dimension_unique_values, k, replace=False)
        true_intent.append(current_dimension_true_intent)
    return true_intent


def create_true_intent_deliberate(unique_values_on_each_dimension):
    true_intent = []
    for i in range(len(unique_values_on_each_dimension)):
        if i == 0:
            # working adult
            current_dimension_true_intent = ['working adult']
            true_intent.append(current_dimension_true_intent)
        if i == 1:
            # Black or Asian-Pac-Islander
            current_dimension_true_intent = ['Black', 'Asian-Pac-Islander']
            true_intent.append(current_dimension_true_intent)
            continue
        if i == 2:
            # Female
            current_dimension_true_intent = ['Female']
            true_intent.append(current_dimension_true_intent)
        if i == 3:
            # full-time
            current_dimension_true_intent = ['full-time']
            true_intent.append(current_dimension_true_intent)
        if i == 4:
            # >50K
            current_dimension_true_intent = ['>50K']
            true_intent.append(current_dimension_true_intent)
    return true_intent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify the attack type')
    parser.add_argument('--data_type', type=str, default='real_data',
                        help='Specify the data type: real_data, synthetic_data')
    parser.add_argument('--true_intent_size', type=int, default=1,
                        help='Specify the true intent size: 1, 2')
    if parser.parse_args().data_type == 'real_data' and parser.parse_args().true_intent_size == 1:
        current_iteration = 1
    elif parser.parse_args().data_type == 'real_data' and parser.parse_args().true_intent_size == 2:
        current_iteration = 2
    elif parser.parse_args().data_type == 'synthetic_data' and parser.parse_args().true_intent_size == 2:
        current_iteration = 3
    elif parser.parse_args().data_type == 'synthetic_data' and parser.parse_args().true_intent_size == 1:
        current_iteration = 4
    else:
        raise ValueError("Invalid data_type and true_intent_size")


    ## real_data
    if current_iteration == 1 or current_iteration == 2:
        # Get the path of the current directory
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the data file
        data_file_path = os.path.join(current_directory, '..', 'adult_data', 'processed_data_different_combinations',
                                      'adult_processed_5_attributes_1.csv')
        # Read the data file
        adult_data = pd.read_csv(data_file_path)
        print(adult_data.head())
        print(adult_data.shape)
        data_cube, unique_values_on_each_dimension = helper.df2data_cube_five_attributes(adult_data)
        cost_cube = helper.generate_unit_cost_data_cube_same_price(data_cube)
        true_intent = create_true_intent_deliberate(unique_values_on_each_dimension)
        # change to a list of lists
        true_intent = [list(x) for x in true_intent]
        if current_iteration == 1:
            # change true intent to only include 1 cell
            race_dimension = true_intent[1].copy()
            race_dimension = [race_dimension[0]]
            true_intent[1] = race_dimension
        # print the sum of the data cube
        print("Sum of the data cube: ", np.sum(data_cube))
        # save everything under running_data/iteration_iteration_number
        running_data_directory = os.path.join(current_directory, 'running_data')
        if not os.path.exists(running_data_directory):
            os.makedirs(running_data_directory)
        iteration_directory = os.path.join(running_data_directory, 'iteration_' + str(current_iteration))
        if not os.path.exists(iteration_directory):
            os.makedirs(iteration_directory)
        data_cube_file_path = os.path.join(iteration_directory, 'data_cube.npy')
        np.save(data_cube_file_path, data_cube)
        cost_cube_file_path = os.path.join(iteration_directory, 'cost_cube.npy')
        np.save(cost_cube_file_path, cost_cube)
        true_intent_file_path = os.path.join(iteration_directory, 'true_intent.csv')
        with open(true_intent_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(true_intent)
        unique_values_on_each_dimension_file_path = os.path.join(iteration_directory, 'unique_values_on_each_dimension.csv')
        with open(unique_values_on_each_dimension_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(unique_values_on_each_dimension)


    ## synthetic data
    if current_iteration == 3 or current_iteration == 4:
        # Get the path of the current directory
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the data file
        data_file_path = os.path.join(current_directory, '..', 'adult_data', 'processed_data_different_combinations',
                                      'adult_processed_5_attributes_1.csv')
        # Read the data file
        adult_data = pd.read_csv(data_file_path)
        print(adult_data.head())
        print(adult_data.shape)
        data_cube, unique_values_on_each_dimension = helper.df2data_cube_five_attributes(adult_data)
        # set every element in the data cube to be 0
        data_cube = np.zeros(data_cube.shape)
        cost_cube = np.zeros(data_cube.shape)
        np.random.seed(4)
        # impose Gaussian distribution on the data cube
        shape = data_cube.shape
        mean_data = 1000
        std_dev_data = 300
        lower_bound_data = 0
        upper_bound_data = np.inf
        a_data = (lower_bound_data - mean_data) / std_dev_data
        b_data = (upper_bound_data - mean_data) / std_dev_data
        trunc_normal_data = truncnorm(a_data, b_data, loc=mean_data, scale=std_dev_data)
        data_cube = trunc_normal_data.rvs(size=shape)
        # ensure integer values
        data_cube = data_cube.astype(int)
        mean_cost = 20
        std_dev_cost = 5
        lower_bound_cost = 1
        upper_bound_cost = np.inf
        a_cost = (lower_bound_cost - mean_cost) / std_dev_cost
        b_cost = (upper_bound_cost - mean_cost) / std_dev_cost
        trunc_normal_cost = truncnorm(a_cost, b_cost, loc=mean_cost, scale=std_dev_cost)
        cost_cube = trunc_normal_cost.rvs(size=shape)
        # # generate unit cost data cube
        # cost_cube = helper.generate_unit_cost_data_cube_same_price(data_cube)
        # ensure integer values
        cost_cube = cost_cube.astype(int)
        true_intent = create_true_intent_deliberate(unique_values_on_each_dimension)
        # change to a list of lists
        true_intent = [list(x) for x in true_intent]
        if current_iteration == 4:
            # change true intent to only include 1 cell
            race_dimension = true_intent[1].copy()
            race_dimension = [race_dimension[0]]
            true_intent[1] = race_dimension
        # print the sum of the data cube
        print("Sum of the data cube: ", np.sum(data_cube))
        # save everything under running_data/iteration_iteration_number
        running_data_directory = os.path.join(current_directory, 'running_data')
        if not os.path.exists(running_data_directory):
            os.makedirs(running_data_directory)
        iteration_directory = os.path.join(running_data_directory, 'iteration_' + str(current_iteration))
        if not os.path.exists(iteration_directory):
            os.makedirs(iteration_directory)
        data_cube_file_path = os.path.join(iteration_directory, 'data_cube.npy')
        np.save(data_cube_file_path, data_cube)
        cost_cube_file_path = os.path.join(iteration_directory, 'cost_cube.npy')
        np.save(cost_cube_file_path, cost_cube)
        true_intent_file_path = os.path.join(iteration_directory, 'true_intent.csv')
        with open(true_intent_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(true_intent)
        unique_values_on_each_dimension_file_path = os.path.join(iteration_directory, 'unique_values_on_each_dimension.csv')
        with open(unique_values_on_each_dimension_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(unique_values_on_each_dimension)