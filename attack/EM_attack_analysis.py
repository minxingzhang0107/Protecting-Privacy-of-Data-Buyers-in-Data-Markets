import numpy as np
from itertools import product


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
    index = tuple(index)
    # get the number of records in the data cube based on the index
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


def lambda_privacy_published_intent_lower_bound(lambda_value, true_intent, data_cube, cost_cube,
                                                unique_values_on_each_dimension):
    data_cube_sum = np.sum(data_cube)
    record_in_true_intent = list(product(*true_intent))
    records_in_TI_f_d_cost_multiplication_maximum_list = []
    for record in record_in_true_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        f_d_t = data_cube[record_index]
        f_d_t = float(f_d_t)
        cost_t = cost_cube[record_index]
        records_in_TI_f_d_cost_multiplication_maximum_list.append(f_d_t / data_cube_sum * cost_t)
    records_in_TI_f_d_cost_multiplication_maximum = max(records_in_TI_f_d_cost_multiplication_maximum_list)
    return records_in_TI_f_d_cost_multiplication_maximum / lambda_value, records_in_TI_f_d_cost_multiplication_maximum


def lambda_privacy_published_intent_lower_bound_only_f_d(lambda_value, true_intent, data_cube,
                                                         unique_values_on_each_dimension):
    data_cube_sum = np.sum(data_cube)
    record_in_true_intent = list(product(*true_intent))
    records_in_TI_f_d_maximum_list = []
    for record in record_in_true_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        f_d_t = data_cube[record_index]
        f_d_t = float(f_d_t)
        records_in_TI_f_d_maximum_list.append(f_d_t / data_cube_sum)
    records_in_TI_f_d_maximum = max(records_in_TI_f_d_maximum_list)
    return records_in_TI_f_d_maximum / lambda_value, records_in_TI_f_d_maximum


def lambda_privacy_published_intent_lower_bound_only_cost(lambda_value, true_intent, cost_cube,
                                                          unique_values_on_each_dimension):
    record_in_true_intent = list(product(*true_intent))
    records_in_TI_cost_maximum_list = []
    for record in record_in_true_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        cost_t = cost_cube[record_index]
        records_in_TI_cost_maximum_list.append(cost_t)
    records_in_TI_cost_maximum = max(records_in_TI_cost_maximum_list)
    return records_in_TI_cost_maximum / lambda_value, records_in_TI_cost_maximum


def compute_records_in_PI_f_d_and_cost_multiplication_summation(data_cube, cost_cube, published_intent,
                                                                unique_values_on_each_dimension):
    data_cube_sum = np.sum(data_cube)
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
        f_d_t = float(f_d_t)
        cost_t = cost_cube[record_index]
        records_in_PI_f_d_cost_multiplication_summation += f_d_t / data_cube_sum * cost_t
    return records_in_PI_f_d_cost_multiplication_summation


def compute_records_in_PI_f_d_summation(data_cube, published_intent, unique_values_on_each_dimension):
    data_cube_sum = np.sum(data_cube)
    record_in_published_intent = list(product(*published_intent))
    records_in_PI_f_d_summation = 0.0
    for record in record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        f_d_t = data_cube[record_index]
        f_d_t = float(f_d_t)
        records_in_PI_f_d_summation += f_d_t / data_cube_sum
    return records_in_PI_f_d_summation


def compute_records_in_PI_cost_summation(cost_cube, published_intent, unique_values_on_each_dimension):
    record_in_published_intent = list(product(*published_intent))
    records_in_PI_cost_summation = 0.0
    for record in record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        cost_t = cost_cube[record_index]
        records_in_PI_cost_summation += cost_t
    return records_in_PI_cost_summation




