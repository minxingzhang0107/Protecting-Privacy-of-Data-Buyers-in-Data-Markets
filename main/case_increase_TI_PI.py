import argparse
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import sys
from matplotlib.ticker import PercentFormatter

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'attack'))
from attack import PI_attack_analysis

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
        current_unit_price = cost_cube[record_index[0]][record_index[1]][record_index[2]][record_index[3]][
            record_index[4]]
        num_records = data_cube[record_index[0]][record_index[1]][record_index[2]][record_index[3]][record_index[4]]
        adjusted_num_records = round(num_records * percent_of_records_purchased_in_each_grid)
        current_cost = current_unit_price * adjusted_num_records
        total_cost += current_cost
        total_num_records_for_each_cell_in_PI.append(num_records)
        num_records_bought_for_each_cell_in_PI.append(adjusted_num_records)
        total_cost_for_each_cell_in_PI.append(current_cost)
    return total_cost, total_num_records_for_each_cell_in_PI, num_records_bought_for_each_cell_in_PI, \
        total_cost_for_each_cell_in_PI


def compute_size(published_intent):
    PI = list(product(*published_intent))
    return len(PI)


if __name__ == '__main__':
    ## specify the parameters
    parser = argparse.ArgumentParser(description='Specify the dimension to expand')
    parser.add_argument('--dimension_name', type=str, default='age',
                        help='Specify the dimension to expand')

    ## load data
    current_iteration = 1
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

    ## set parameter
    lambda_value = 0.3
    attack_type = 'PI_uniform_attack'
    percent_of_records_purchased_in_each_grid = 1
    # adjust the true intent
    race_dimension = true_intent[1].copy()
    race_dimension = [race_dimension[0]]
    true_intent[1] = race_dimension
    dimension_names = ['age', 'race', 'sex', 'hours-per-week', 'income']


    ## expand on age
    if parser.parse_args().dimension_name == 'age':
        w_1 = 0.5
        w_2 = 1 - w_1
        confidence_list_lower = []
        confidence_list_upper = []
        PI_total_cost_list = []
        TI_total_cost_list = []
        PI_size_list = []
        TI_size_list = []
        num_cells_in_PI_list = []
        num_cells_in_TI_list = []
        expanded_feature_name_list = []
        published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, attack_type, lambda_value,
                                                                 percent_of_records_purchased_in_each_grid)
        confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
        confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent, published_intent_PI_uniform_attack)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_PI_uniform_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        confidence_list_lower.append(confidence_lower)
        confidence_list_upper.append(confidence_upper)
        PI_total_cost_list.append(PI_total_cost)
        TI_total_cost_list.append(TI_total_cost)
        num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
        num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
        PI_size_list.append(num_records_bought_in_PI)
        TI_size_list.append(num_records_bought_in_TI)
        num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
        num_cells_in_TI = compute_size(true_intent)
        num_cells_in_PI_list.append(num_cells_in_PI)
        num_cells_in_TI_list.append(num_cells_in_TI)
        # find the index of age in dimension_names
        age_index = dimension_names.index('age')
        current_unique_values_on_dimension = unique_values_on_each_dimension[age_index]
        current_value_in_true_intent = true_intent[age_index]
        expanded_feature_name_list.append(current_value_in_true_intent[0])
        unexpended_values_on_current_dimension = [value for value in current_unique_values_on_dimension if value not in
                                                  current_value_in_true_intent]
        for unexpended_value in unexpended_values_on_current_dimension:
            if unexpended_value == 'young adult':
                w_1_new = 1
                w_2_new = 1 - w_1_new
            elif unexpended_value == 'retirement':
                w_1_new = 0.5
                w_2_new = 1 - w_1_new
            else:
                w_1_new = 0.5
                w_2_new = 1 - w_1_new
            true_intent_current_dimension = true_intent[age_index].copy()
            true_intent_current_dimension.append(unexpended_value)
            true_intent[age_index] = true_intent_current_dimension
            published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube,
                                                                     unique_values_on_each_dimension,
                                                                     true_intent, w_1_new, w_2_new, attack_type,
                                                                     lambda_value, percent_of_records_purchased_in_each_grid)
            confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
            confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent,
                                                                         published_intent_PI_uniform_attack)
            TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
                TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
                PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube,
                                                                 published_intent_PI_uniform_attack,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            confidence_list_lower.append(confidence_lower)
            confidence_list_upper.append(confidence_upper)
            PI_total_cost_list.append(PI_total_cost)
            TI_total_cost_list.append(TI_total_cost)
            num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
            num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
            PI_size_list.append(num_records_bought_in_PI)
            TI_size_list.append(num_records_bought_in_TI)
            num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
            num_cells_in_TI = compute_size(true_intent)
            num_cells_in_PI_list.append(num_cells_in_PI)
            num_cells_in_TI_list.append(num_cells_in_TI)
            expanded_feature_name_list.append(str(unexpended_value))
        # plot
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
        sns.set_context("paper")
        sns.set(font_scale=1.2)
        plt.figure()
        plt.plot(expanded_feature_name_list, confidence_list_lower, marker='^', markersize=10,
                 label='Lower Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_lower, confidence_list_lower):
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, confidence_list_upper, marker='s', markersize=10,
                 label='Upper Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_upper, confidence_list_upper):
            # round y to three decimal places and then add % sign
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        # second plot for total number of records bought
        plt.figure()
        plt.plot(expanded_feature_name_list, PI_size_list, marker='s', markersize=10,
                 label='# Records (PI)')
        for x, y, label in zip(expanded_feature_name_list, PI_size_list, PI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, TI_size_list, marker='^', markersize=10,
                 label='# Records (TI)')
        for x, y, label in zip(expanded_feature_name_list, TI_size_list, TI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()
        # third plot for intent size
        plt.figure()
        plt.plot(expanded_feature_name_list, num_cells_in_PI_list, marker='s', markersize=10,
                 label='Size (PI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_PI_list, num_cells_in_PI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, num_cells_in_TI_list, marker='^', markersize=10,
                 label='Size (TI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_TI_list, num_cells_in_TI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()


    ## expand on ethnicity
    if parser.parse_args().dimension_name == 'ethnicity':
        w_1 = 0.5
        w_2 = 1 - w_1
        confidence_list_lower = []
        confidence_list_upper = []
        PI_total_cost_list = []
        TI_total_cost_list = []
        PI_size_list = []
        TI_size_list = []
        num_cells_in_PI_list = []
        num_cells_in_TI_list = []
        expanded_feature_name_list = []
        published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, attack_type, lambda_value,
                                                                 percent_of_records_purchased_in_each_grid)
        confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
        confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent, published_intent_PI_uniform_attack)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_PI_uniform_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        confidence_list_lower.append(confidence_lower)
        confidence_list_upper.append(confidence_upper)
        PI_total_cost_list.append(PI_total_cost)
        TI_total_cost_list.append(TI_total_cost)
        num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
        num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
        PI_size_list.append(num_records_bought_in_PI)
        TI_size_list.append(num_records_bought_in_TI)
        num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
        num_cells_in_TI = compute_size(true_intent)
        num_cells_in_PI_list.append(num_cells_in_PI)
        num_cells_in_TI_list.append(num_cells_in_TI)
        # find the index of race in dimension_names
        age_index = dimension_names.index('race')
        current_unique_values_on_dimension = unique_values_on_each_dimension[age_index]
        current_value_in_true_intent = true_intent[age_index]
        expanded_feature_name_list.append(current_value_in_true_intent[0])
        unexpended_values_on_current_dimension = [value for value in current_unique_values_on_dimension if value not in
                                                  current_value_in_true_intent]
        for unexpended_value in unexpended_values_on_current_dimension:
            if unexpended_value == 'White':
                w_1_new = 1
                w_2_new = 1 - w_1_new
            elif unexpended_value == 'Asian-Pac-Islander':
                w_1_new = 1
                w_2_new = 1 - w_1_new
            elif unexpended_value == 'Amer-Indian-Eskimo':
                w_1_new = 1
                w_2_new = 1 - w_1_new
            else:
                w_1_new = 1
                w_2_new = 1 - w_1_new
            true_intent_current_dimension = true_intent[age_index].copy()
            true_intent_current_dimension.append(unexpended_value)
            true_intent[age_index] = true_intent_current_dimension
            published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube,
                                                                     unique_values_on_each_dimension,
                                                                     true_intent, w_1_new, w_2_new,
                                                                     attack_type, lambda_value, percent_of_records_purchased_in_each_grid)
            confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
            confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent,
                                                                         published_intent_PI_uniform_attack)
            TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
                TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
                PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube,
                                                                 published_intent_PI_uniform_attack,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            confidence_list_lower.append(confidence_lower)
            confidence_list_upper.append(confidence_upper)
            PI_total_cost_list.append(PI_total_cost)
            TI_total_cost_list.append(TI_total_cost)
            num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
            num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
            PI_size_list.append(num_records_bought_in_PI)
            TI_size_list.append(num_records_bought_in_TI)
            num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
            num_cells_in_TI = compute_size(true_intent)
            num_cells_in_PI_list.append(num_cells_in_PI)
            num_cells_in_TI_list.append(num_cells_in_TI)
            expanded_feature_name_list.append(str(unexpended_value))
        # plot
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
        sns.set_context("paper")
        sns.set(font_scale=1.2)
        plt.figure()
        plt.plot(expanded_feature_name_list, confidence_list_lower, marker='^', markersize=10,
                 label='Lower Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_lower, confidence_list_lower):
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, -18), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, confidence_list_upper, marker='s', markersize=10,
                 label='Upper Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_upper, confidence_list_upper):
            # round y to three decimal places and then add % sign
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.xticks(fontsize=9)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        # second plot for total number of records bought
        plt.figure()
        plt.plot(expanded_feature_name_list, PI_size_list, marker='s', markersize=10,
                 label='# Records (PI)')
        for x, y, label in zip(expanded_feature_name_list, PI_size_list, PI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, TI_size_list, marker='^', markersize=10,
                 label='# Records (TI)')
        for x, y, label in zip(expanded_feature_name_list, TI_size_list, TI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.xticks(fontsize=9)
        plt.show()
        # third plot for intent size
        plt.figure()
        plt.plot(expanded_feature_name_list, num_cells_in_PI_list, marker='s', markersize=10,
                 label='Size (PI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_PI_list, num_cells_in_PI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, num_cells_in_TI_list, marker='^', markersize=10,
                 label='Size (TI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_TI_list, num_cells_in_TI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.xticks(fontsize=9)
        plt.show()


    ## expand on gender
    if parser.parse_args().dimension_name == 'gender':
        w_1 = 0.5
        w_2 = 1 - w_1
        confidence_list_lower = []
        confidence_list_upper = []
        PI_total_cost_list = []
        TI_total_cost_list = []
        PI_size_list = []
        TI_size_list = []
        num_cells_in_PI_list = []
        num_cells_in_TI_list = []
        expanded_feature_name_list = []
        published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, attack_type, lambda_value,
                                                                 percent_of_records_purchased_in_each_grid)
        confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
        confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent, published_intent_PI_uniform_attack)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_PI_uniform_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        confidence_list_lower.append(confidence_lower)
        confidence_list_upper.append(confidence_upper)
        PI_total_cost_list.append(PI_total_cost)
        TI_total_cost_list.append(TI_total_cost)
        num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
        num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
        PI_size_list.append(num_records_bought_in_PI)
        TI_size_list.append(num_records_bought_in_TI)
        num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
        num_cells_in_TI = compute_size(true_intent)
        num_cells_in_PI_list.append(num_cells_in_PI)
        num_cells_in_TI_list.append(num_cells_in_TI)
        # find the index of sex in dimension_names
        age_index = dimension_names.index('sex')
        current_unique_values_on_dimension = unique_values_on_each_dimension[age_index]
        current_value_in_true_intent = true_intent[age_index]
        expanded_feature_name_list.append(current_value_in_true_intent[0])
        unexpended_values_on_current_dimension = [value for value in current_unique_values_on_dimension if value not in
                                                  current_value_in_true_intent]
        for unexpended_value in unexpended_values_on_current_dimension:
            if unexpended_value == 'Male':
                w_1_new = 1
                w_2_new = 1 - w_1_new
            true_intent_current_dimension = true_intent[age_index].copy()
            true_intent_current_dimension.append(unexpended_value)
            true_intent[age_index] = true_intent_current_dimension
            published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube,
                                                                     unique_values_on_each_dimension,
                                                                     true_intent, w_1_new, w_2_new, attack_type,
                                                                     lambda_value, percent_of_records_purchased_in_each_grid)
            confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
            confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent,
                                                                         published_intent_PI_uniform_attack)
            TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
                TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
                PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube,
                                                                 published_intent_PI_uniform_attack,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            confidence_list_lower.append(confidence_lower)
            confidence_list_upper.append(confidence_upper)
            PI_total_cost_list.append(PI_total_cost)
            TI_total_cost_list.append(TI_total_cost)
            num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
            num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
            PI_size_list.append(num_records_bought_in_PI)
            TI_size_list.append(num_records_bought_in_TI)
            num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
            num_cells_in_TI = compute_size(true_intent)
            num_cells_in_PI_list.append(num_cells_in_PI)
            num_cells_in_TI_list.append(num_cells_in_TI)
            expanded_feature_name_list.append(str(unexpended_value))
        # plot
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
        sns.set_context("paper")
        sns.set(font_scale=1.2)
        plt.figure()
        plt.plot(expanded_feature_name_list, confidence_list_lower, marker='^', markersize=10,
                 label='Lower Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_lower, confidence_list_lower):
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=12)
        plt.plot(expanded_feature_name_list, confidence_list_upper, marker='s', markersize=10,
                 label='Upper Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_upper, confidence_list_upper):
            # round y to three decimal places and then add % sign
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=9)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        # second plot for total number of records bought
        plt.figure()
        plt.plot(expanded_feature_name_list, PI_size_list, marker='s', markersize=10,
                 label='# Records (PI)')
        for x, y, label in zip(expanded_feature_name_list, PI_size_list, PI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=12)
        plt.plot(expanded_feature_name_list, TI_size_list, marker='^', markersize=10,
                 label='# Records (TI)')
        for x, y, label in zip(expanded_feature_name_list, TI_size_list, TI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=9)
        plt.show()
        # third plot for intent size
        plt.figure()
        plt.plot(expanded_feature_name_list, num_cells_in_PI_list, marker='s', markersize=10,
                 label='Size (PI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_PI_list, num_cells_in_PI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=12)
        plt.plot(expanded_feature_name_list, num_cells_in_TI_list, marker='^', markersize=10,
                 label='Size (TI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_TI_list, num_cells_in_TI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=9)
        plt.show()


    ## expand on hours-per-week
    if parser.parse_args().dimension_name == 'hours-per-week':
        w_1 = 0.5
        w_2 = 1 - w_1
        confidence_list_lower = []
        confidence_list_upper = []
        PI_total_cost_list = []
        TI_total_cost_list = []
        PI_size_list = []
        TI_size_list = []
        num_cells_in_PI_list = []
        num_cells_in_TI_list = []
        expanded_feature_name_list = []
        published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, attack_type, lambda_value,
                                                                 percent_of_records_purchased_in_each_grid)
        confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
        confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent, published_intent_PI_uniform_attack)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_PI_uniform_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        confidence_list_lower.append(confidence_lower)
        confidence_list_upper.append(confidence_upper)
        PI_total_cost_list.append(PI_total_cost)
        TI_total_cost_list.append(TI_total_cost)
        num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
        num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
        PI_size_list.append(num_records_bought_in_PI)
        TI_size_list.append(num_records_bought_in_TI)
        num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
        num_cells_in_TI = compute_size(true_intent)
        num_cells_in_PI_list.append(num_cells_in_PI)
        num_cells_in_TI_list.append(num_cells_in_TI)
        # find the index of hours-per-week in dimension_names
        age_index = dimension_names.index('hours-per-week')
        current_unique_values_on_dimension = unique_values_on_each_dimension[age_index]
        current_value_in_true_intent = true_intent[age_index]
        expanded_feature_name_list.append(current_value_in_true_intent[0])
        unexpended_values_on_current_dimension = [value for value in current_unique_values_on_dimension if value not in
                                                  current_value_in_true_intent]
        for unexpended_value in unexpended_values_on_current_dimension:
            if unexpended_value == 'part-time':
                w_1_new = 1
                w_2_new = 1 - w_1_new
            else:
                w_1_new = 0.5
                w_2_new = 0
            true_intent_current_dimension = true_intent[age_index].copy()
            true_intent_current_dimension.append(unexpended_value)
            true_intent[age_index] = true_intent_current_dimension
            published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube,
                                                                     unique_values_on_each_dimension,
                                                                     true_intent, w_1_new, w_2_new, attack_type,
                                                                     lambda_value, percent_of_records_purchased_in_each_grid)
            confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
            confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent, published_intent_PI_uniform_attack)
            TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
                TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
                PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube,
                                                                 published_intent_PI_uniform_attack,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            confidence_list_lower.append(confidence_lower)
            confidence_list_upper.append(confidence_upper)
            PI_total_cost_list.append(PI_total_cost)
            TI_total_cost_list.append(TI_total_cost)
            num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
            num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
            PI_size_list.append(num_records_bought_in_PI)
            TI_size_list.append(num_records_bought_in_TI)
            num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
            num_cells_in_TI = compute_size(true_intent)
            num_cells_in_PI_list.append(num_cells_in_PI)
            num_cells_in_TI_list.append(num_cells_in_TI)
            expanded_feature_name_list.append(str(unexpended_value))
        # plot
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
        sns.set_context("paper")
        sns.set(font_scale=1.2)
        plt.figure()
        plt.plot(expanded_feature_name_list, confidence_list_lower, marker='^', markersize=10,
                 label='Lower Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_lower, confidence_list_lower):
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, confidence_list_upper, marker='s', markersize=10,
                 label='Upper Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_upper, confidence_list_upper):
            # round y to three decimal places and then add % sign
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        # second plot for total number of records bought
        plt.figure()
        plt.plot(expanded_feature_name_list, PI_size_list, marker='s', markersize=10,
                 label='# Records (PI)')
        for x, y, label in zip(expanded_feature_name_list, PI_size_list, PI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, TI_size_list, marker='^', markersize=10,
                 label='# Records (TI)')
        for x, y, label in zip(expanded_feature_name_list, TI_size_list, TI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()
        # third plot for intent size
        plt.figure()
        plt.plot(expanded_feature_name_list, num_cells_in_PI_list, marker='s', markersize=10,
                 label='Size (PI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_PI_list, num_cells_in_PI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=15)
        plt.plot(expanded_feature_name_list, num_cells_in_TI_list, marker='^', markersize=10,
                 label='Size (TI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_TI_list, num_cells_in_TI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -17), ha='center', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()


    ## expand on income
    if parser.parse_args().dimension_name == 'income':
        w_1 = 0.5
        w_2 = 1 - w_1
        confidence_list_lower = []
        confidence_list_upper = []
        PI_total_cost_list = []
        TI_total_cost_list = []
        PI_size_list = []
        TI_size_list = []
        num_cells_in_PI_list = []
        num_cells_in_TI_list = []
        expanded_feature_name_list = []
        published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, attack_type, lambda_value,
                                                                 percent_of_records_purchased_in_each_grid)
        confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
        confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent, published_intent_PI_uniform_attack)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_PI_uniform_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        confidence_list_lower.append(confidence_lower)
        confidence_list_upper.append(confidence_upper)
        PI_total_cost_list.append(PI_total_cost)
        TI_total_cost_list.append(TI_total_cost)
        num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
        num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
        PI_size_list.append(num_records_bought_in_PI)
        TI_size_list.append(num_records_bought_in_TI)
        num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
        num_cells_in_TI = compute_size(true_intent)
        num_cells_in_PI_list.append(num_cells_in_PI)
        num_cells_in_TI_list.append(num_cells_in_TI)
        # find the index of income in dimension_names
        age_index = dimension_names.index('income')
        current_unique_values_on_dimension = unique_values_on_each_dimension[age_index]
        current_value_in_true_intent = true_intent[age_index]
        expanded_feature_name_list.append(current_value_in_true_intent[0])
        unexpended_values_on_current_dimension = [value for value in current_unique_values_on_dimension if value not in
                                                  current_value_in_true_intent]
        for unexpended_value in unexpended_values_on_current_dimension:
            if unexpended_value == '<=50K':
                w_1_new = 0.5
                w_2_new = 1 - w_1_new
            true_intent_current_dimension = true_intent[age_index].copy()
            true_intent_current_dimension.append(unexpended_value)
            true_intent[age_index] = true_intent_current_dimension
            published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube,
                                                                     unique_values_on_each_dimension,
                                                                     true_intent, w_1_new, w_2_new, attack_type,
                                                                     lambda_value, percent_of_records_purchased_in_each_grid)
            confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
            confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent,
                                                                         published_intent_PI_uniform_attack)
            TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
                TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
                PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube,
                                                                 published_intent_PI_uniform_attack,
                                                                 unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
            confidence_list_lower.append(confidence_lower)
            confidence_list_upper.append(confidence_upper)
            PI_total_cost_list.append(PI_total_cost)
            TI_total_cost_list.append(TI_total_cost)
            num_records_bought_in_PI = np.sum(PI_num_records_bought_for_each_cell)
            num_records_bought_in_TI = np.sum(TI_num_records_bought_for_each_cell)
            PI_size_list.append(num_records_bought_in_PI)
            TI_size_list.append(num_records_bought_in_TI)
            num_cells_in_PI = compute_size(published_intent_PI_uniform_attack)
            num_cells_in_TI = compute_size(true_intent)
            num_cells_in_PI_list.append(num_cells_in_PI)
            num_cells_in_TI_list.append(num_cells_in_TI)
            expanded_feature_name_list.append(str(unexpended_value))
        # plot
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
        sns.set_context("paper")
        sns.set(font_scale=1.2)
        plt.figure()
        plt.plot(expanded_feature_name_list, confidence_list_lower, marker='^', markersize=10,
                 label='Lower Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_lower, confidence_list_lower):
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=12)
        plt.plot(expanded_feature_name_list, confidence_list_upper, marker='s', markersize=10,
                 label='Upper Confidence Bound')
        for x, y, label in zip(expanded_feature_name_list, confidence_list_upper, confidence_list_upper):
            # round y to three decimal places and then add % sign
            y = round(y, 3)
            formatted_label = f'{y:.1%}'
            plt.annotate(formatted_label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=9)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        # second plot for total number of records bought
        plt.figure()
        plt.plot(expanded_feature_name_list, PI_size_list, marker='s', markersize=10,
                 label='# Records (PI)')
        for x, y, label in zip(expanded_feature_name_list, PI_size_list, PI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=12)
        plt.plot(expanded_feature_name_list, TI_size_list, marker='^', markersize=10,
                 label='# Records (TI)')
        for x, y, label in zip(expanded_feature_name_list, TI_size_list, TI_size_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=9)
        plt.show()
        # third plot for intent size
        plt.figure()
        plt.plot(expanded_feature_name_list, num_cells_in_PI_list, marker='s', markersize=10,
                 label='Size (PI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_PI_list, num_cells_in_PI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=12)
        plt.plot(expanded_feature_name_list, num_cells_in_TI_list, marker='^', markersize=10,
                 label='Size (TI)')
        for x, y, label in zip(expanded_feature_name_list, num_cells_in_TI_list, num_cells_in_TI_list):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=9)
        plt.show()





