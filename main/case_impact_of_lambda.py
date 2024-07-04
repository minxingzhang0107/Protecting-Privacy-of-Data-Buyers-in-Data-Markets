import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'defense'))
from defense import expansion


def compute_total_cost(data_cube, cost_cube, published_intent, unique_values_on_each_dimension, alpha):
    PI = list(product(*published_intent))
    total_num_records_for_each_cell_in_PI = []
    num_records_bought_for_each_cell_in_PI = []
    total_cost_for_each_cell_in_PI = []
    total_cost = 0.0
    for record in PI:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        current_unit_price = cost_cube[record_index]
        num_records = data_cube[record_index]
        adjusted_num_records = round(num_records * alpha)
        current_cost = current_unit_price * adjusted_num_records
        total_cost += current_cost
        total_num_records_for_each_cell_in_PI.append(num_records)
        num_records_bought_for_each_cell_in_PI.append(adjusted_num_records)
        total_cost_for_each_cell_in_PI.append(current_cost)
    return total_cost, total_num_records_for_each_cell_in_PI, num_records_bought_for_each_cell_in_PI, \
        total_cost_for_each_cell_in_PI


def compute_total_num_records(data_cube, unique_values_on_each_dimension, published_intent):
    PI = list(product(*published_intent))
    total_num_records = 0
    for record in PI:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        num_records = data_cube[record_index]
        total_num_records += num_records
    return total_num_records, PI


def compute_stat_PRI(cost_cube, unique_values_on_each_dimension, pseudo_PI, TI, purchased_set_of_records, is_string):
    total_cost_of_PI = 0.0
    total_cost_of_TI = 0.0
    num_records_in_PI = 0
    num_records_in_TI = 0
    pseudo_PI_tuple = [tuple(x) for x in pseudo_PI]
    pseudo_PI_tuple_as_str = np.array([str(tuple_) for tuple_ in pseudo_PI_tuple])
    TI_tuple = [tuple(x) for x in TI]
    TI_tuple_as_str = np.array([str(tuple_) for tuple_ in TI_tuple])
    # if element in purchased_set_of_records is not in str format, convert it to str format
    if is_string == False:
        purchased_set_of_records_tuple = [tuple(x) for x in purchased_set_of_records]
        purchased_set_of_records_as_str = np.array([str(tuple_) for tuple_ in purchased_set_of_records_tuple])
    else:
        purchased_set_of_records_as_str = purchased_set_of_records
    # for each cell in pseudo_PI, count the number of records in the sampled_records
    counts = np.array([np.sum(purchased_set_of_records_as_str == tuple_) for tuple_ in pseudo_PI_tuple_as_str])
    for i in range(len(pseudo_PI)):
        current_cell = pseudo_PI[i]
        corresponding_count = counts[i]
        if corresponding_count == 0:
            continue
        current_cell_index = []
        for j in range(len(current_cell)):
            current_feature_value = current_cell[j]
            current_feature_value_index = unique_values_on_each_dimension[j].index(current_feature_value)
            current_cell_index.append(current_feature_value_index)
        current_cell_index = tuple(current_cell_index)
        current_unit_price = cost_cube[current_cell_index]
        # check if the current cell is in TI
        if pseudo_PI_tuple_as_str[i] in TI_tuple_as_str:
            total_cost_of_TI += current_unit_price * corresponding_count
            num_records_in_TI += corresponding_count
        total_cost_of_PI += current_unit_price * corresponding_count
        num_records_in_PI += corresponding_count
    return total_cost_of_PI, total_cost_of_TI, num_records_in_PI, num_records_in_TI


if __name__ == '__main__':
    ## read data
    current_iteration = 2
    # read the data cube
    current_directory = os.path.dirname(os.path.abspath(__file__))
    iteration_directory = os.path.join(current_directory, 'running_data', 'iteration_' + str(current_iteration))
    data_cube_file_path = os.path.join(iteration_directory, 'data_cube.npy')
    data_cube = np.load(data_cube_file_path)
    # read the cost cube
    cost_cube_file_path = os.path.join(iteration_directory, 'cost_cube.npy')
    cost_cube = np.load(cost_cube_file_path)
    # read the true intent
    true_intent_file_path = os.path.join(iteration_directory, 'true_intent.csv')
    true_intent = []
    with open(true_intent_file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            true_intent.append(row)
    # read the unique values on each dimension
    unique_values_on_each_dimension_file_path = os.path.join(iteration_directory, 'unique_values_on_each_dimension.csv')
    unique_values_on_each_dimension = []
    with open(unique_values_on_each_dimension_file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            unique_values_on_each_dimension.append(row)


    ## Impact of lambda on PI-uniform attack
    lambda_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    attack_type = 'PI_uniform_attack'
    alpha = 1
    w_1 = 1
    w_2 = 1 - w_1
    num_records_in_TI_list = []
    num_records_in_PI_list = []
    for lambda_value in lambda_list:
        published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, attack_type, lambda_value,
                                                                 alpha)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, alpha)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_PI_uniform_attack,
                                                             unique_values_on_each_dimension, alpha)
        num_records_in_TI_list.append(TI_total_cost)
        num_records_in_PI_list.append(PI_total_cost)


    ## Impact of lambda on EM attack
    lambda_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alpha = 1
    w_1 = 0.6
    w_2 = 1 - w_1
    attack_type_2 = 'EM_attack'
    background_knowledge = 'both_f_d_and_cost'
    num_records_in_TI_list_EM = []
    num_records_in_PI_list_EM = []
    for lambda_value in lambda_list:
        published_intent_EM_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                         true_intent, w_1, w_2, attack_type_2, lambda_value, alpha)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, alpha)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_EM_attack,
                                                             unique_values_on_each_dimension, alpha)
        num_records_in_TI_list_EM.append(TI_total_cost)
        num_records_in_PI_list_EM.append(PI_total_cost)


    ## Plotting
    total_number_of_records = 30162.0
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    lambda_list = [x * 100 for x in lambda_list]
    lambda_list = [int(x) for x in lambda_list]
    lambda_list = [str(x) + '%' for x in lambda_list]
    num_records_in_TI_list = [x / total_number_of_records * 100 for x in num_records_in_TI_list]
    num_records_in_PI_list = [x / total_number_of_records * 100 for x in num_records_in_PI_list]
    num_records_in_PI_list_EM = [x / total_number_of_records * 100 for x in num_records_in_PI_list_EM]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(lambda_list, num_records_in_TI_list, marker='s', markersize=10, label='True Intent')
    plt.plot(lambda_list, num_records_in_PI_list, marker='o', markersize=10,
             label='Published Intent (PI-uniform Attack)')
    plt.plot(lambda_list, num_records_in_PI_list_EM, marker='^', markersize=10,
             label='Published Intent (Efficiency Maximization Attack)')
    # keep 1 decimal
    num_records_in_TI_list = [round(x, 1) for x in num_records_in_TI_list]
    num_records_in_PI_list = [round(x, 1) for x in num_records_in_PI_list]
    num_records_in_PI_list_EM = [round(x, 1) for x in num_records_in_PI_list_EM]
    for x, y, label in zip(lambda_list, num_records_in_TI_list, num_records_in_TI_list):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15, color='blue')
    for x, y, label in zip(lambda_list, num_records_in_PI_list, num_records_in_PI_list):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=15, color='orange')
    for x, y, label in zip(lambda_list, num_records_in_PI_list_EM, num_records_in_PI_list_EM):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 17), ha='center', fontsize=15, color='green')
    plt.xlabel('Privacy Threshold $\lambda$')
    plt.ylabel('Percentage of Records Purchased (%)')
    plt.xticks(lambda_list, fontsize=15)
    plt.legend()
    plt.show()



