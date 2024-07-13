import numpy as np
import os
import csv
from itertools import product
import sys

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


def compute_size(published_intent):
    PI = list(product(*published_intent))
    return len(PI)


if __name__ == '__main__':
    ## load data
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


    ## reduce dimensionality
    lambda_value = 0.3
    w_1 = 1
    w_2 = 1 - w_1
    attack_type = 'PI_uniform_attack'
    percent_of_records_purchased_in_each_grid = 1
    published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                             true_intent, w_1, w_2, attack_type, lambda_value, percent_of_records_purchased_in_each_grid)
    confidence_lower_parent = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack)
    confidence_upper_parent = PI_attack_analysis.confidence_upper_bound(true_intent, published_intent_PI_uniform_attack)
    total_number_of_dimensions = len(unique_values_on_each_dimension)
    dimension_names = ['age', 'race', 'sex', 'hours-per-week', 'income']
    for i in range(total_number_of_dimensions):
        print("We reduce the dimensionality by removing the dimension: ", dimension_names[i])
        published_intent_on_dimension_i = published_intent_PI_uniform_attack[i]
        unique_values_on_dimension_i = unique_values_on_each_dimension[i]
        index_of_published_intent_on_dimension_i = []
        for value in published_intent_on_dimension_i:
            index_of_published_intent_on_dimension_i.append(unique_values_on_dimension_i.index(value))
        # for data_cube, only keep the slice on dimension i where the published intent is located
        data_cube_reduced = np.take(data_cube, index_of_published_intent_on_dimension_i, axis=i)
        data_cube_reduced = np.sum(data_cube_reduced, axis=i)
        cost_cube_reduced = np.sum(cost_cube, axis=i)
        # change every element in the cost cube to 1
        cost_cube_reduced = np.ones(cost_cube_reduced.shape)
        unique_values_on_each_dimension_reduced = unique_values_on_each_dimension.copy()
        unique_values_on_each_dimension_reduced.pop(i)
        true_intent_reduced = true_intent.copy()
        true_intent_reduced.pop(i)
        published_intent_PI_uniform_attack_reduced = published_intent_PI_uniform_attack.copy()
        published_intent_PI_uniform_attack_reduced.pop(i)
        # compute the attacker's confidence
        confidence_lower = PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack_reduced)
        confidence_upper = PI_attack_analysis.confidence_upper_bound(true_intent_reduced,
                                                                     published_intent_PI_uniform_attack_reduced)
        print("The attacker's confidence upper bound is: ", confidence_upper)
        print("The attacker's confidence lower bound is: ", confidence_lower)
        print("The change of confidence upper bound is: ",  confidence_upper_parent - confidence_upper)
        print("The change of confidence lower bound is: ", confidence_lower_parent - confidence_lower)
        print("----------------------------------------")
        TI_total_cost_reduced, TI_total_num_records_for_each_cell_reduced, \
            TI_num_records_bought_for_each_cell_reduced, TI_total_cost_for_each_cell_reduced = \
            compute_total_cost(data_cube_reduced, cost_cube_reduced, true_intent_reduced,
                               unique_values_on_each_dimension_reduced, percent_of_records_purchased_in_each_grid)
        PI_total_cost_reduced, PI_total_num_records_for_each_cell_reduced, \
            PI_num_records_bought_for_each_cell_reduced, PI_total_cost_for_each_cell_reduced = \
            compute_total_cost(data_cube_reduced, cost_cube_reduced, published_intent_PI_uniform_attack_reduced,
                               unique_values_on_each_dimension_reduced, percent_of_records_purchased_in_each_grid)
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost_reduced)
        print("The total cost spent by the buyer on the true intent is: ", TI_total_cost_reduced)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell_reduced))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(TI_num_records_bought_for_each_cell_reduced))
        size_of_published_intent = compute_size(published_intent_PI_uniform_attack_reduced)
        print("The size of the published intent is: ", size_of_published_intent)
        print("----------------------------------------")
        print("Then, we conduct expansion to find the new published intent after projection.")
        if i == 0:
            w_1_reduced = 1
            w_2_reduced = 1 - w_1_reduced
        elif i == 1:
            w_1_reduced = 1
            w_2_reduced = 1 - w_1_reduced
        elif i == 2:
            w_1_reduced = 1
            w_2_reduced = 1 - w_1_reduced
        elif i == 3:
            w_1_reduced = 0.8
            w_2_reduced = 1 - w_1_reduced
        elif i == 4:
            w_1_reduced = 1
            w_2_reduced = 1 - w_1_reduced
        new_published_intent_after_projection = expansion.expansion(
            data_cube_reduced, cost_cube_reduced, unique_values_on_each_dimension_reduced, true_intent_reduced,
            w_1_reduced, w_2_reduced, attack_type, lambda_value, percent_of_records_purchased_in_each_grid)
        TI_total_cost_modified, TI_total_num_records_for_each_cell_modified, \
            TI_num_records_bought_for_each_cell_modified, TI_total_cost_for_each_cell_modified = \
            compute_total_cost(data_cube_reduced, cost_cube_reduced, true_intent_reduced,
                               unique_values_on_each_dimension_reduced, percent_of_records_purchased_in_each_grid)
        PI_total_cost_modified, PI_total_num_records_for_each_cell_modified, \
            PI_num_records_bought_for_each_cell_modified, PI_total_cost_for_each_cell_modified = \
            compute_total_cost(data_cube_reduced, cost_cube_reduced, new_published_intent_after_projection,
                               unique_values_on_each_dimension_reduced, percent_of_records_purchased_in_each_grid)
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost_modified)
        print("The total cost spent by the buyer on the true intent is: ", TI_total_cost_modified)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell_modified))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(TI_num_records_bought_for_each_cell_modified))
        size_of_published_intent_modified = compute_size(new_published_intent_after_projection)
        print("The size of the published intent is: ", size_of_published_intent_modified)
        print("----------------------------------------")
        print("----------------------------------------")