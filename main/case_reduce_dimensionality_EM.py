import numpy as np
import os
import csv
from itertools import product
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'attack'))
from attack import EM_attack_analysis

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


def confidence_upper_bound_only_f_d(data_cube, published_intent, target_record, unique_values_on_each_dimension):
    """
    :param data_cube: a np-array with 5 dimensions, each dimension contains the feature values on that dimension
    :param published_intent: a list containing 5 small lists, each list contains the feature values of one dimension
    :param target_record: a list containing 5 feature values for each dimension
    :param unique_values_on_each_dimension: a list containing 5 small lists, each list contains the unique feature on
    that dimension
    :return:
    """
    data_cube_sum = np.sum(data_cube)
    # get the index of the target record in the data cube
    index = []
    for i in range(len(target_record)):
        current_feature_value = target_record[i]
        current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
        index.append(current_feature_value_index)
    index = tuple(index)
    # get the number of records in the data cube based on the index
    f_d_x = data_cube[index]
    if f_d_x == 0:
        print("Regarding record ", target_record, " the f_d_x is 0")
        return 0
    f_d_x = float(f_d_x)
    f_d_x = f_d_x / data_cube_sum
    record_in_published_intent = list(product(*published_intent))
    records_in_PI_f_d_summation = 0.0
    for record in record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        tmp = data_cube[record_index]
        tmp = float(tmp)
        records_in_PI_f_d_summation += tmp / data_cube_sum
    return f_d_x / records_in_PI_f_d_summation


def confidence_upper_bound_only_cost(cost_data_cube, published_intent, target_record, unique_values_on_each_dimension):
    """
    :param cost_data_cube: a np-array with 11 dimensions, each dimension contains the feature values on that dimension,
    the cube stores the cost of each record
    :param published_intent: a list containing 11 small lists, each list contains the feature values of one dimension
    :param target_record: a list containing 11 feature values for each dimension
    :param unique_values_on_each_dimension: a list containing 11 small lists, each list contains the unique feature on
    that dimension
    :return:
    """
    # get the index of the target record in the cost data cube
    index = []
    for i in range(len(target_record)):
        current_feature_value = target_record[i]
        current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
        index.append(current_feature_value_index)
    index = tuple(index)
    # get the cost of the target record in the cost data cube
    cost_x = cost_data_cube[index]
    record_in_published_intent = list(product(*published_intent))
    records_in_PI_cost_summation = 0.0
    for record in record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        records_in_PI_cost_summation += cost_data_cube[record_index]
    return cost_x / records_in_PI_cost_summation


def confidence_upper_bound_both_f_d_and_cost(data_cube, cost_data_cube, published_intent, target_record,
                                             unique_values_on_each_dimension):
    """
    :param data_cube: a np-array with 11 dimensions, each dimension contains the feature values on that dimension
    :param cost_data_cube: a np-array with 11 dimensions, each dimension contains the feature values on that dimension,
    the cube stores the cost of each record
    :param published_intent: a list containing 11 small lists, each list contains the feature values of one dimension
    :param target_record: a list containing 11 feature values for each dimension
    :param unique_values_on_each_dimension: a list containing 11 small lists, each list contains the unique feature on
    that dimension
    :return:
    """
    data_cube_sum = np.sum(data_cube)
    # get the index of the target record in the data cube
    index = []
    for i in range(len(target_record)):
        current_feature_value = target_record[i]
        current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
        index.append(current_feature_value_index)
    # get the number of records in the data cube based on the index
    index = tuple(index)
    f_d_x = data_cube[index]
    f_d_x = float(f_d_x)
    # get the cost of the target record in the cost data cube
    cost_x = cost_data_cube[index]
    record_x_multiplication = (f_d_x / data_cube_sum) * cost_x
    record_in_published_intent = list(product(*published_intent))
    records_in_PI_f_d_cost_multiplication_summation = 0.0
    for record in record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        f_d_t = data_cube[record_index]
        cost_t = cost_data_cube[record_index]
        records_in_PI_f_d_cost_multiplication_summation += (f_d_t / data_cube_sum) * cost_t
    return record_x_multiplication / records_in_PI_f_d_cost_multiplication_summation


def confidence_upper_bound_generalization(data_cube, cost_data_cube, published_intent, true_intent,
                                          unique_values_on_each_dimension, background_knowledge):
    confidence_list = []
    record_in_true_intent = list(product(*true_intent))
    for record in record_in_true_intent:
        if background_knowledge == 'only_f_d':
            confidence = confidence_upper_bound_only_f_d(data_cube, published_intent, record,
                                                         unique_values_on_each_dimension)
        elif background_knowledge == 'both_f_d_and_cost':
            confidence = confidence_upper_bound_both_f_d_and_cost(data_cube, cost_data_cube, published_intent, record,
                                                                  unique_values_on_each_dimension)
        elif background_knowledge == 'only_cost':
            confidence = confidence_upper_bound_only_cost(cost_data_cube, published_intent, record,
                                                          unique_values_on_each_dimension)
        else:
            raise ValueError('background knowledge is not valid')
        confidence_list.append(confidence)
    return np.max(confidence_list)


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


    ## reduce the dimensionality
    w_1 = 0.6
    w_2 = 1 - w_1
    alpha = 1
    lambda_value = 0.3
    attack_type_2 = 'EM_attack'
    published_intent_EM_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                     true_intent, w_1, w_2, attack_type_2, lambda_value, alpha)
    print("We compute the attacker's confidence based on the new published intent")
    background_knowledge_3 = 'both_f_d_and_cost'
    attacker_confidence_parent = EM_attack_analysis.confidence_upper_bound_generalization(
        data_cube, cost_cube, published_intent_EM_attack, true_intent, unique_values_on_each_dimension,
        background_knowledge_3)
    print("Given both f_d and cost as background knowledge, the confidence's upper bound is: ",
          attacker_confidence_parent)
    total_number_of_dimensions = len(unique_values_on_each_dimension)
    dimension_names = ['age', 'race', 'sex', 'hours-per-week', 'income']
    for i in range(total_number_of_dimensions):
        print("We reduce the dimensionality by removing the dimension: ", dimension_names[i])
        published_intent_on_dimension_i = published_intent_EM_attack[i]
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
        published_intent_EM_attack_reduced = published_intent_EM_attack.copy()
        published_intent_EM_attack_reduced.pop(i)
        # compute the attacker's confidence
        background_knowledge_3 = 'both_f_d_and_cost'
        confidence_after_projection = confidence_upper_bound_generalization(
            data_cube_reduced, cost_cube_reduced, published_intent_EM_attack_reduced, true_intent_reduced,
            unique_values_on_each_dimension_reduced, background_knowledge_3)
        print("Given both f_d and cost as background knowledge, the confidence's upper bound is: ",
              confidence_after_projection)
        print("The change of confidence is: ", attacker_confidence_parent - confidence_after_projection)
        print("----------------------------------------")
        TI_total_cost_reduced, TI_total_num_records_for_each_cell_reduced, \
            TI_num_records_bought_for_each_cell_reduced, TI_total_cost_for_each_cell_reduced = \
            compute_total_cost(data_cube_reduced, cost_cube_reduced, true_intent_reduced,
                               unique_values_on_each_dimension_reduced, alpha)
        PI_total_cost_reduced, PI_total_num_records_for_each_cell_reduced, \
            PI_num_records_bought_for_each_cell_reduced, PI_total_cost_for_each_cell_reduced = \
            compute_total_cost(data_cube_reduced, cost_cube_reduced, published_intent_EM_attack_reduced,
                               unique_values_on_each_dimension_reduced, alpha)
        PI_size_reduced = compute_size(published_intent_EM_attack_reduced)
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost_reduced)
        print("The total cost spent by the buyer on the true intent is: ", TI_total_cost_reduced)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell_reduced))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(TI_num_records_bought_for_each_cell_reduced))
        print("The size of the published intent is: ", PI_size_reduced)
        print("----------------------------------------")
        print("Then, we conduct expansion to form a modified published intent")
        # w_1 and w_2 varies among different dimensions
        if dimension_names[i] == 'age':
            w_1_reduced = 0.6
            w_2_reduced = 1 - w_1_reduced
        if dimension_names[i] == 'race':
            w_1_reduced = 0.5
            w_2_reduced = 1 - w_1_reduced
        if dimension_names[i] == 'sex':
            w_1_reduced = 0.5
            w_2_reduced = 1 - w_1_reduced
        if dimension_names[i] == 'hours-per-week':
            w_1_reduced = 0.6
            w_2_reduced = 1 - w_1_reduced
        if dimension_names[i] == 'income':
            w_1_reduced = 0.4
            w_2_reduced = 1 - w_1_reduced
        new_published_intent_after_projection = expansion.expansion(
            data_cube_reduced, cost_cube_reduced, unique_values_on_each_dimension_reduced, true_intent_reduced,
            w_1_reduced, w_2_reduced, attack_type_2, lambda_value, alpha)
        TI_total_cost_modified, TI_total_num_records_for_each_cell_modified, \
            TI_num_records_bought_for_each_cell_modified, TI_total_cost_for_each_cell_modified = \
            compute_total_cost(data_cube_reduced, cost_cube_reduced, true_intent_reduced,
                               unique_values_on_each_dimension_reduced, alpha)
        PI_total_cost_modified, PI_total_num_records_for_each_cell_modified, \
            PI_num_records_bought_for_each_cell_modified, PI_total_cost_for_each_cell_modified = \
            compute_total_cost(data_cube_reduced, cost_cube_reduced, new_published_intent_after_projection,
                               unique_values_on_each_dimension_reduced, alpha)
        PI_size_modified = compute_size(new_published_intent_after_projection)
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost_modified)
        print("The total cost spent by the buyer on the true intent is: ", TI_total_cost_modified)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell_modified))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(TI_num_records_bought_for_each_cell_modified))
        print("The size of the published intent is: ", PI_size_modified)
        print("----------------------------------------")
        print("----------------------------------------")



