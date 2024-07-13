import numpy as np
import os
import math
from itertools import product
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'attack'))
from attack import PI_attack_analysis
from attack import EM_attack_analysis
from sklearn.preprocessing import MinMaxScaler


def compute_best_dimension_and_feature(feature_list, corres_dimension_list, total_cost_list, increase_list,
                                       w_1, w_2, attack_type):
    # change to np.array
    total_cost_list = np.array(total_cost_list)
    increase_list = np.array(increase_list)
    corres_dimension_list = np.array(corres_dimension_list)
    feature_list = np.array(feature_list)
    if attack_type == 'PI_uniform_attack':
        # check if total_cost_list contains 0, if so, return the corresponding dimension and feature
        if 0 in total_cost_list:
            index = np.where(total_cost_list == 0)[0][0]
            return corres_dimension_list[index], feature_list[index]
        # convert everything in total cost to max(total_cost_list) - current_cost
        total_cost_list = np.max(total_cost_list) - total_cost_list
        # min-max normalization
        scaler = MinMaxScaler()
        total_cost_list = scaler.fit_transform(total_cost_list.reshape(-1, 1))
        increase_list = scaler.fit_transform(increase_list.reshape(-1, 1))
        # compute final score
        final_score_list = w_1 * total_cost_list + w_2 * increase_list
        # find the index of the max score
        index = np.argmax(final_score_list)
        return corres_dimension_list[index], feature_list[index]
    elif attack_type == 'EM_attack':
        # remove all the 0s in total_cost_list
        zero_index_list = np.where(total_cost_list == 0)[0]
        total_cost_list = np.delete(total_cost_list, zero_index_list)
        increase_list = np.delete(increase_list, zero_index_list)
        corres_dimension_list = np.delete(corres_dimension_list, zero_index_list)
        feature_list = np.delete(feature_list, zero_index_list)
        # convert everything in total cost to max(total_cost_list) - current_cost
        total_cost_list = np.max(total_cost_list) - total_cost_list
        # min-max normalization
        scaler = MinMaxScaler()
        total_cost_list = scaler.fit_transform(total_cost_list.reshape(-1, 1))
        increase_list = scaler.fit_transform(increase_list.reshape(-1, 1))
        # compute final score
        final_score_list = w_1 * total_cost_list + w_2 * increase_list
        # find the index of the max score
        index = np.argmax(final_score_list)
        return corres_dimension_list[index], feature_list[index]


def compute_best_dimension_and_feature_only_cost(feature_list, corres_dimension_list, total_cost_list, increase_list,
                                                 w_1, w_2):
    total_cost_list = np.array(total_cost_list)
    increase_list = np.array(increase_list)
    corres_dimension_list = np.array(corres_dimension_list)
    feature_list = np.array(feature_list)
    # check if total_cost_list contains 0, if so, return the corresponding dimension and feature
    if 0 in total_cost_list:
        index = np.where(total_cost_list == 0)[0][0]
        return corres_dimension_list[index], feature_list[index]
    # convert everything in total cost to max(total_cost_list) - current_cost
    total_cost_list = np.max(total_cost_list) - total_cost_list
    # min-max normalization
    scaler = MinMaxScaler()
    total_cost_list = scaler.fit_transform(total_cost_list.reshape(-1, 1))
    increase_list = scaler.fit_transform(increase_list.reshape(-1, 1))
    # compute final score
    final_score_list = w_1 * total_cost_list + w_2 * increase_list
    # find the index of the max score
    index = np.argmax(final_score_list)
    return corres_dimension_list[index], feature_list[index]


def compute_total_cost(feature_ij, corresponding_dimension, data_cube, cost_cube, unique_values_on_each_dimension,
                       published_intent, percent_of_records_purchased_in_each_grid):
    adjusted_published_intent = published_intent.copy()
    # replace unique_values_on_dimension_i in published_intent with [feature_ij]
    adjusted_published_intent[corresponding_dimension] = [feature_ij]
    newly_added_record_in_published_intent = list(product(*adjusted_published_intent))
    total_cost = 0.0
    for record in newly_added_record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        current_unit_price = cost_cube[record_index]
        num_records = data_cube[record_index]
        adjusted_num_records = round(num_records * percent_of_records_purchased_in_each_grid)
        total_cost += current_unit_price * adjusted_num_records
    return total_cost


def compute_increase_PI_attack(feature_ij, corresponding_dimension, published_intent):
    adjusted_published_intent = published_intent.copy()
    # replace unique_values_on_dimension_i in published_intent with [feature_ij]
    adjusted_published_intent[corresponding_dimension] = [feature_ij]
    newly_added_record_in_published_intent = list(product(*adjusted_published_intent))
    return len(newly_added_record_in_published_intent)


def compute_increase_EM_attack(data_cube, cost_cube, feature_ij, corresponding_dimension, published_intent,
                               unique_values_on_each_dimension):
    data_cube_sum = np.sum(data_cube)
    adjusted_published_intent = published_intent.copy()
    # replace unique_values_on_dimension_i in published_intent with [feature_ij]
    adjusted_published_intent[corresponding_dimension] = [feature_ij]
    newly_added_record_in_published_intent = list(product(*adjusted_published_intent))
    increase = 0.0
    for record in newly_added_record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        num_records = data_cube[record_index]
        num_records = float(num_records)
        unit_price = cost_cube[record_index]
        increase += (num_records / data_cube_sum) * unit_price
    return increase


def compute_increase_EM_attack_only_f_d(data_cube, feature_ij, corresponding_dimension, published_intent,
                                        unique_values_on_each_dimension):
    data_cube_sum = np.sum(data_cube)
    adjusted_published_intent = published_intent.copy()
    # replace unique_values_on_dimension_i in published_intent with [feature_ij]
    adjusted_published_intent[corresponding_dimension] = [feature_ij]
    newly_added_record_in_published_intent = list(product(*adjusted_published_intent))
    increase = 0.0
    for record in newly_added_record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        num_records = data_cube[record_index]
        num_records = float(num_records)
        increase += (num_records / data_cube_sum)
    return increase


def compute_increase_EM_attack_only_cost(cost_cube, feature_ij, corresponding_dimension, published_intent,
                                         unique_values_on_each_dimension):
    adjusted_published_intent = published_intent.copy()
    # replace unique_values_on_dimension_i in published_intent with [feature_ij]
    adjusted_published_intent[corresponding_dimension] = [feature_ij]
    newly_added_record_in_published_intent = list(product(*adjusted_published_intent))
    increase = 0.0
    for record in newly_added_record_in_published_intent:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        unit_price = cost_cube[record_index]
        increase += unit_price
    return increase


def pairwise_distance_age_hours_per_week():
    age_feature_list = ['childhood', 'young adult', 'working adult','retirement']
    # compute pairwise distance between two features and store in a dictionary
    age_feature_distance = {}
    for i in range(len(age_feature_list)):
        for j in range(i + 1, len(age_feature_list)):
            # here the distance is simple, it is the difference between the index of two features
            distance = abs(i - j)
            age_feature_distance[(age_feature_list[i], age_feature_list[j])] = distance
            age_feature_distance[(age_feature_list[j], age_feature_list[i])] = distance
    hours_per_week_feature_list = ['part-time', 'full-time', 'overtime']
    hours_per_week_feature_distance = {}
    for i in range(len(hours_per_week_feature_list)):
        for j in range(i + 1, len(hours_per_week_feature_list)):
            distance = abs(i - j)
            hours_per_week_feature_distance[(hours_per_week_feature_list[i], hours_per_week_feature_list[j])] = distance
            hours_per_week_feature_distance[(hours_per_week_feature_list[j], hours_per_week_feature_list[i])] = distance
    return age_feature_distance, hours_per_week_feature_distance


def compute_nearest_neighbor(current_published_intent, age_feature_distance, hours_per_week_feature_distance,
                             current_dimension_name, deduction):
    # age
    nearest_distance = np.inf
    if current_dimension_name == 'age':
        for unexplored_feature in deduction:
            accumulated_distance = 0
            for i in range(len(current_published_intent)):
                current_feature = current_published_intent[i]
                accumulated_distance += age_feature_distance[(current_feature, unexplored_feature)]
            if accumulated_distance < nearest_distance:
                nearest_distance = accumulated_distance
                nearest_neighbor = unexplored_feature
    # hours per week
    elif current_dimension_name == 'hours-per-week':
        for unexplored_feature in deduction:
            accumulated_distance = 0
            for i in range(len(current_published_intent)):
                current_feature = current_published_intent[i]
                accumulated_distance += hours_per_week_feature_distance[(current_feature, unexplored_feature)]
            if accumulated_distance < nearest_distance:
                nearest_distance = accumulated_distance
                nearest_neighbor = unexplored_feature
    return nearest_neighbor


def expansion(data_cube, cost_cube, unique_values_on_each_dimension, true_intent, w_1, w_2, attack_type, lambda_value,
              percent_of_records_purchased_in_each_grid):
    dimension_name_list = ['age', 'race', 'sex', 'hours-per-week', 'income']
    # here we consider dimension reduction as a case study
    if len(unique_values_on_each_dimension) == 4:
        if ['working adult', 'young adult', 'retirement', 'childhood'] not in unique_values_on_each_dimension:
            dimension_name_list.remove('age')
        elif ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'] not in \
                unique_values_on_each_dimension:
            dimension_name_list.remove('race')
        elif ['Male', 'Female'] not in unique_values_on_each_dimension:
            dimension_name_list.remove('sex')
        elif ['full-time', 'part-time', 'overtime'] not in unique_values_on_each_dimension:
            dimension_name_list.remove('hours-per-week')
        else:
            dimension_name_list.remove('income')
    age_feature_distance, hours_per_week_feature_distance = pairwise_distance_age_hours_per_week()
    published_intent = true_intent.copy()
    if attack_type == 'PI_uniform_attack':
        # minimum size of PI
        privacy_threshold = PI_attack_analysis.lambda_privacy_published_intent_lower_bound(lambda_value, true_intent)
        privacy_threshold = math.ceil(privacy_threshold)
        PI_size = PI_attack_analysis.compute_PI_size(published_intent)
        while PI_size < privacy_threshold:
            potential_feature_for_expansion = []
            # deduct the feature that has been used in published intent from unique_values_on_each_dimension
            for i in range(len(published_intent)):
                current_dimension_name = dimension_name_list[i]
                current_published_intent = published_intent[i]
                current_unique_values_on_dimension_i = unique_values_on_each_dimension[i]
                deduction = [feature_tmp for feature_tmp in current_unique_values_on_dimension_i if feature_tmp
                             not in current_published_intent]
                if len(deduction) == 0 or len(deduction) == 1:
                    potential_feature_for_expansion.append(deduction)
                else:
                    # age dimension and hours_per_week dimension are special because we have pairwise distance
                    if current_dimension_name == 'age':
                        nearest_neighbor = compute_nearest_neighbor(current_published_intent, age_feature_distance,
                                                                    hours_per_week_feature_distance,
                                                                    current_dimension_name, deduction)
                        potential_feature_for_expansion.append([nearest_neighbor])
                    elif current_dimension_name == 'hours-per-week':
                        nearest_neighbor = compute_nearest_neighbor(current_published_intent, age_feature_distance,
                                                                    hours_per_week_feature_distance,
                                                                    current_dimension_name, deduction)
                        potential_feature_for_expansion.append([nearest_neighbor])
                    else:
                        potential_feature_for_expansion.append(deduction)
            # compute score for each feature
            feature_list = []
            corres_dimension_list = []
            total_cost_list = []
            increase_list = []
            for i in range(len(published_intent)):
                current_dimension = potential_feature_for_expansion[i]
                # if current_dimension is empty, skip
                if len(current_dimension) == 0:
                    continue
                for feature in current_dimension:
                    total_cost_tmp = compute_total_cost(feature, i, data_cube, cost_cube,
                                                        unique_values_on_each_dimension, published_intent, percent_of_records_purchased_in_each_grid)
                    increase_tmp = compute_increase_PI_attack(feature, i, published_intent)
                    feature_list.append(feature)
                    corres_dimension_list.append(i)
                    total_cost_list.append(total_cost_tmp)
                    increase_list.append(increase_tmp)
            best_dimension, best_feature = compute_best_dimension_and_feature(feature_list, corres_dimension_list,
                                                                              total_cost_list, increase_list,
                                                                              w_1, w_2, attack_type)
            # add best_feature to published intent
            published_intent_current_dimension = published_intent[best_dimension].copy()
            published_intent_current_dimension.append(best_feature)
            published_intent[best_dimension] = published_intent_current_dimension
            PI_size = PI_attack_analysis.compute_PI_size(published_intent)
        return published_intent
    elif attack_type == 'EM_attack':
        privacy_threshold, _ = EM_attack_analysis.lambda_privacy_published_intent_lower_bound(lambda_value, true_intent,
                                                  data_cube, cost_cube, unique_values_on_each_dimension)
        current_summation = EM_attack_analysis.compute_records_in_PI_f_d_and_cost_multiplication_summation(data_cube,
                                                         cost_cube, published_intent, unique_values_on_each_dimension)
        while current_summation < privacy_threshold:
            potential_feature_for_expansion = []
            # deduct the feature that has been used in published intent from unique_values_on_each_dimension
            for i in range(len(published_intent)):
                current_dimension_name = dimension_name_list[i]
                current_published_intent = published_intent[i]
                current_unique_values_on_dimension_i = unique_values_on_each_dimension[i]
                deduction = [feature_tmp for feature_tmp in current_unique_values_on_dimension_i if feature_tmp not in
                             current_published_intent]
                if len(deduction) == 0 or len(deduction) == 1:
                    potential_feature_for_expansion.append(deduction)
                else:
                    # age dimension and hours_per_week dimension are special because we have pairwise distance
                    if current_dimension_name == 'age':
                        nearest_neighbor = compute_nearest_neighbor(current_published_intent, age_feature_distance,
                                                                    hours_per_week_feature_distance,
                                                                    current_dimension_name, deduction)
                        potential_feature_for_expansion.append([nearest_neighbor])
                    elif current_dimension_name == 'hours-per-week':
                        nearest_neighbor = compute_nearest_neighbor(current_published_intent, age_feature_distance,
                                                                    hours_per_week_feature_distance,
                                                                    current_dimension_name, deduction)
                        potential_feature_for_expansion.append([nearest_neighbor])
                    else:
                        potential_feature_for_expansion.append(deduction)
            # if potential_feature_for_expansion contains all empty list, then stop
            if all(len(current_dimension) == 0 for current_dimension in potential_feature_for_expansion):
                print("There is no feature to expand")
                break
            feature_list = []
            corres_dimension_list = []
            total_cost_list = []
            increase_list = []
            for i in range(len(published_intent)):
                current_dimension = potential_feature_for_expansion[i]
                # if current_dimension is empty, skip
                if len(current_dimension) == 0:
                    continue
                for feature in current_dimension:
                    total_cost_tmp = compute_total_cost(feature, i, data_cube, cost_cube,
                                                        unique_values_on_each_dimension, published_intent, percent_of_records_purchased_in_each_grid)
                    increase_tmp = compute_increase_EM_attack(data_cube, cost_cube, feature, i, published_intent,
                                                              unique_values_on_each_dimension)
                    feature_list.append(feature)
                    corres_dimension_list.append(i)
                    total_cost_list.append(total_cost_tmp)
                    increase_list.append(increase_tmp)
            best_dimension, best_feature = compute_best_dimension_and_feature(feature_list, corres_dimension_list,
                                                                              total_cost_list, increase_list,
                                                                              w_1, w_2, attack_type)
            # add best_feature to published intent
            published_intent_current_dimension = published_intent[best_dimension].copy()
            published_intent_current_dimension.append(best_feature)
            published_intent[best_dimension] = published_intent_current_dimension
            current_summation = EM_attack_analysis.compute_records_in_PI_f_d_and_cost_multiplication_summation(
                data_cube, cost_cube, published_intent, unique_values_on_each_dimension)
        return published_intent


def expansion_EMF_EMC(data_cube, cost_cube, unique_values_on_each_dimension, true_intent, w_1, w_2,
                      background_knowledge, lambda_value, percent_of_records_purchased_in_each_grid):
    dimension_name_list = ['age', 'race', 'sex', 'hours-per-week', 'income']
    age_feature_distance, hours_per_week_feature_distance = pairwise_distance_age_hours_per_week()
    published_intent = true_intent.copy()
    if background_knowledge == 'only_f_d':
        privacy_threshold, _ = EM_attack_analysis.lambda_privacy_published_intent_lower_bound_only_f_d(
            lambda_value, true_intent, data_cube, unique_values_on_each_dimension)
        current_summation = EM_attack_analysis.compute_records_in_PI_f_d_summation(
            data_cube, published_intent, unique_values_on_each_dimension)
        while current_summation < privacy_threshold:
            potential_feature_for_expansion = []
            # deduct the feature that has been used in published intent from unique_values_on_each_dimension
            for i in range(len(published_intent)):
                current_dimension_name = dimension_name_list[i]
                current_published_intent = published_intent[i]
                current_unique_values_on_dimension_i = unique_values_on_each_dimension[i]
                deduction = [feature_tmp for feature_tmp in current_unique_values_on_dimension_i
                             if feature_tmp not in current_published_intent]
                if len(deduction) == 0 or len(deduction) == 1:
                    potential_feature_for_expansion.append(deduction)
                else:
                    # age dimension and hours_per_week dimension are special because we have pairwise distance
                    if current_dimension_name == 'age':
                        nearest_neighbor = compute_nearest_neighbor(current_published_intent, age_feature_distance,
                                                                    hours_per_week_feature_distance,
                                                                    current_dimension_name, deduction)
                        potential_feature_for_expansion.append([nearest_neighbor])
                    elif current_dimension_name == 'hours-per-week':
                        nearest_neighbor = compute_nearest_neighbor(current_published_intent, age_feature_distance,
                                                                    hours_per_week_feature_distance,
                                                                    current_dimension_name, deduction)
                        potential_feature_for_expansion.append([nearest_neighbor])
                    else:
                        potential_feature_for_expansion.append(deduction)
            # compute score for each feature
            feature_list = []
            corres_dimension_list = []
            total_cost_list = []
            increase_list = []
            for i in range(len(published_intent)):
                current_dimension = potential_feature_for_expansion[i]
                # if current_dimension is empty, skip
                if len(current_dimension) == 0:
                    continue
                for feature in current_dimension:
                    total_cost_tmp = compute_total_cost(feature, i, data_cube, cost_cube,
                                                        unique_values_on_each_dimension, published_intent, percent_of_records_purchased_in_each_grid)
                    increase_tmp = compute_increase_EM_attack_only_f_d(
                        data_cube, feature, i, published_intent, unique_values_on_each_dimension)
                    feature_list.append(feature)
                    corres_dimension_list.append(i)
                    total_cost_list.append(total_cost_tmp)
                    increase_list.append(increase_tmp)
            best_dimension, best_feature = compute_best_dimension_and_feature(feature_list, corres_dimension_list,
                                                                              total_cost_list, increase_list,
                                                                              w_1, w_2, 'EM_attack')
            # add best_feature to published intent
            published_intent_current_dimension = published_intent[best_dimension].copy()
            published_intent_current_dimension.append(best_feature)
            published_intent[best_dimension] = published_intent_current_dimension
            current_summation = EM_attack_analysis.compute_records_in_PI_f_d_summation(
                data_cube, published_intent, unique_values_on_each_dimension)
        return published_intent
    elif background_knowledge == 'only_cost':
        privacy_threshold, _ = EM_attack_analysis.lambda_privacy_published_intent_lower_bound_only_cost(
            lambda_value, true_intent, cost_cube, unique_values_on_each_dimension)
        current_summation = EM_attack_analysis.compute_records_in_PI_cost_summation(
            cost_cube, published_intent, unique_values_on_each_dimension)
        while current_summation < privacy_threshold:
            potential_feature_for_expansion = []
            # deduct the feature that has been used in published intent from unique_values_on_each_dimension
            for i in range(len(published_intent)):
                current_dimension_name = dimension_name_list[i]
                current_published_intent = published_intent[i]
                current_unique_values_on_dimension_i = unique_values_on_each_dimension[i]
                deduction = [feature_tmp for feature_tmp in current_unique_values_on_dimension_i if feature_tmp not in
                             current_published_intent]
                if len(deduction) == 0 or len(deduction) == 1:
                    potential_feature_for_expansion.append(deduction)
                else:
                    # age dimension and hours_per_week dimension are special because we have pairwise distance
                    if current_dimension_name == 'age':
                        nearest_neighbor = compute_nearest_neighbor(current_published_intent, age_feature_distance,
                                                                    hours_per_week_feature_distance,
                                                                    current_dimension_name, deduction)
                        potential_feature_for_expansion.append([nearest_neighbor])
                    elif current_dimension_name == 'hours-per-week':
                        nearest_neighbor = compute_nearest_neighbor(current_published_intent, age_feature_distance,
                                                                    hours_per_week_feature_distance,
                                                                    current_dimension_name, deduction)
                        potential_feature_for_expansion.append([nearest_neighbor])
                    else:
                        potential_feature_for_expansion.append(deduction)
            feature_list = []
            corres_dimension_list = []
            total_cost_list = []
            increase_list = []
            for i in range(len(published_intent)):
                current_dimension = potential_feature_for_expansion[i]
                # if current_dimension is empty, skip
                if len(current_dimension) == 0:
                    continue
                for feature in current_dimension:
                    total_cost_tmp = compute_total_cost(feature, i, data_cube, cost_cube,
                                                        unique_values_on_each_dimension, published_intent, percent_of_records_purchased_in_each_grid)
                    increase_tmp = compute_increase_EM_attack_only_cost(cost_cube, feature, i, published_intent,
                                                                        unique_values_on_each_dimension)
                    feature_list.append(feature)
                    corres_dimension_list.append(i)
                    total_cost_list.append(total_cost_tmp)
                    increase_list.append(increase_tmp)
            # if all the features in potential_feature_for_expansion are empty, break
            if len(feature_list) == 0:
                print('No feature can be added to published intent')
                break
            best_dimension, best_feature = compute_best_dimension_and_feature_only_cost(
                feature_list, corres_dimension_list, total_cost_list, increase_list, w_1, w_2)
            # add best_feature to published intent
            published_intent_current_dimension = published_intent[best_dimension].copy()
            published_intent_current_dimension.append(best_feature)
            published_intent[best_dimension] = published_intent_current_dimension
            # update current_summation
            current_summation = EM_attack_analysis.compute_records_in_PI_cost_summation(
                cost_cube, published_intent, unique_values_on_each_dimension)
        return published_intent





