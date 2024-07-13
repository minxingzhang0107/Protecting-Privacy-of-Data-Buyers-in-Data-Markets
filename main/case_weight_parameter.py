import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'defense'))
from defense import expansion


def compute_total_cost(data_cube, cost_cube, published_intent, unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid):
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
        adjusted_num_records = round(num_records * percent_of_records_purchased_in_each_grid)
        current_cost = current_unit_price * adjusted_num_records
        total_cost += current_cost
        total_num_records_for_each_cell_in_PI.append(num_records)
        num_records_bought_for_each_cell_in_PI.append(adjusted_num_records)
        total_cost_for_each_cell_in_PI.append(current_cost)
    return total_cost, total_num_records_for_each_cell_in_PI, num_records_bought_for_each_cell_in_PI, \
        total_cost_for_each_cell_in_PI


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


    ## Impact of weight on PI-uniform attack
    weight_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    attack_type = 'PI_uniform_attack'
    percent_of_records_purchased_in_each_grid = 1
    lambda_value = 0.3
    num_records_in_TI_list_PI_uniform = []
    num_records_in_PI_list_PI_uniform = []
    for weight in weight_list:
        w_1 = weight
        w_2 = 1 - weight
        published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, attack_type, lambda_value,
                                                                 percent_of_records_purchased_in_each_grid)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_PI_uniform_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        TI_total_cost = int(TI_total_cost)
        PI_total_cost = int(PI_total_cost)
        num_records_in_TI_list_PI_uniform.append(TI_total_cost)
        num_records_in_PI_list_PI_uniform.append(PI_total_cost)
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(weight_list, num_records_in_TI_list_PI_uniform, marker='s', markersize=10, label='True Intent')
    plt.plot(weight_list, num_records_in_PI_list_PI_uniform, marker='^', markersize=10,
             label='Published Intent (PI-uniform)')
    plt.xlabel('Weight Parameter Alpha')
    plt.ylabel('Number of Records')
    plt.legend(loc='best')
    plt.show()


    ## Impact of weight on EM attack
    weight_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    percent_of_records_purchased_in_each_grid = 1
    lambda_value = 0.3
    attack_type_2 = 'EM_attack'
    background_knowledge = 'both_f_d_and_cost'
    num_records_in_TI_list_EM_attack = []
    num_records_in_PI_list_EM_attack = []
    for weight in weight_list:
        w_1 = weight
        w_2 = 1 - weight
        published_intent_EM_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                         true_intent, w_1, w_2, attack_type_2, lambda_value, percent_of_records_purchased_in_each_grid)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_EM_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        TI_total_cost = int(TI_total_cost)
        PI_total_cost = int(PI_total_cost)
        num_records_in_TI_list_EM_attack.append(TI_total_cost)
        num_records_in_PI_list_EM_attack.append(PI_total_cost)
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(weight_list, num_records_in_TI_list_PI_uniform, marker='s', markersize=10, label='True Intent')
    plt.plot(weight_list, num_records_in_PI_list_EM_attack, marker='o', markersize=10, label='Published Intent (EM)')
    plt.xlabel('Weight Parameter Alpha')
    plt.ylabel('Number of Records')
    plt.legend(loc='best')
    plt.show()


    ## Plot
    total_number_of_records = 30162.0
    num_records_in_TI_list_PI_uniform = [x / total_number_of_records * 100 for x in num_records_in_TI_list_PI_uniform]
    num_records_in_PI_list_PI_uniform = [x / total_number_of_records * 100 for x in num_records_in_PI_list_PI_uniform]
    num_records_in_PI_list_EM_attack = [x / total_number_of_records * 100 for x in num_records_in_PI_list_EM_attack]
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(weight_list, num_records_in_TI_list_PI_uniform, marker='s', markersize=10, label='True Intent')
    plt.plot(weight_list, num_records_in_PI_list_PI_uniform, marker='^', markersize=10,
             label='Published Intent (PI-uniform Attack)')
    plt.plot(weight_list, num_records_in_PI_list_EM_attack, marker='o', markersize=10,
             label='Published Intent (Efficiency Maximization Attack)')
    # take 1 decimal place
    num_records_in_TI_list_PI_uniform = [round(x, 1) for x in num_records_in_TI_list_PI_uniform]
    num_records_in_PI_list_PI_uniform = [round(x, 1) for x in num_records_in_PI_list_PI_uniform]
    num_records_in_PI_list_EM_attack = [round(x, 1) for x in num_records_in_PI_list_EM_attack]
    for x, y, label in zip(weight_list, num_records_in_PI_list_EM_attack, num_records_in_PI_list_EM_attack):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15, color='green')
    for x, y, label in zip(weight_list, num_records_in_TI_list_PI_uniform, num_records_in_TI_list_PI_uniform):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15, color='blue')
    for x, y, label in zip(weight_list, num_records_in_PI_list_PI_uniform, num_records_in_PI_list_PI_uniform):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15, color='orange')
    plt.xlabel('Weight Parameter $\\alpha$')
    plt.ylabel('Percentage of Records Purchased (%)')
    plt.legend(loc='best')
    plt.show()