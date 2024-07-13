import numpy as np
from itertools import combinations


def utility_comp(parents_list, TI_index, PI, PI_tuple_as_str, num_purchased_records, cost_cube,
                 unique_values_on_each_dimension):
    proportion_of_records_in_target_cell_list = []
    for parent in parents_list:
        parent_tuple_as_str = np.array([str(tuple(tuple_)) for tuple_ in parent])
        counts = np.array([np.sum(parent_tuple_as_str == tuple_) for tuple_ in PI_tuple_as_str])
        proportion_of_records_in_target_cell = np.sum(counts[TI_index])
        proportion_of_records_in_target_cell_list.append(proportion_of_records_in_target_cell / num_purchased_records)
    proportion_of_records_in_target_cell_list = np.array(proportion_of_records_in_target_cell_list)
    utility_list = proportion_of_records_in_target_cell_list
    return utility_list


def selection_by_utility(parents_list, TI_index, PI, PI_tuple_as_str, num_purchased_records,
                         cost_cube, unique_values_on_each_dimension, top_parents_num):
    num_of_parents = len(parents_list)
    # num_of_parents_selected = int(num_of_parents * proportion)
    num_of_parents_selected = top_parents_num
    if num_of_parents == 2 or num_of_parents_selected < 2:
        return parents_list
    utility_list = utility_comp(parents_list, TI_index, PI, PI_tuple_as_str, num_purchased_records, cost_cube,
                                unique_values_on_each_dimension)
    utility_list = np.array(utility_list)
    # select the top parents
    top_parents_index = np.argsort(utility_list)[-num_of_parents_selected:]
    parents_list = np.array(parents_list)
    top_parents = parents_list[top_parents_index]
    return top_parents


def crossover(parents_list):
    children_list = []
    for parent_1, parent_2 in combinations(parents_list, 2):
        crossover_point = np.random.randint(1, len(parent_1))
        child_1 = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))
        child_2 = np.concatenate((parent_2[:crossover_point], parent_1[crossover_point:]))
        children_list.extend([child_1, child_2])
    return children_list


def filter_by_privacy(children_list, PI_tuple_as_str, TI, saved_list, lambda_threshold, sample_time):
    confidence_list = []
    for child in children_list:
        # change everything in child to tuple
        child = [tuple(tuple_) for tuple_ in child]
        sampled_records_as_str = np.array([str(tuple_) for tuple_ in child])
        proportion_list = []
        for i in range(len(saved_list)):
            current_true_intent = TI[i]
            associated_ml_simulation_result = saved_list[i]
            number_of_current_records = np.sum(sampled_records_as_str == str(current_true_intent))
            # get the proportion in associated_ml_simulation_result that is equal or greater than
            # number_of_current_records
            current_proportion = np.sum(associated_ml_simulation_result >= number_of_current_records) / sample_time
            proportion_list.append(current_proportion)
        minimum_proportion = np.min(proportion_list)
        confidence = 1 - minimum_proportion
        confidence_list.append(confidence)
    confidence_list = np.array(confidence_list)
    children_list = np.array(children_list)
    children_list = children_list[confidence_list <= lambda_threshold]
    return children_list


def Genetic_sampling(data_cube, cost_cube, unique_values_on_each_dimension, num_purchased_records,
                     PI, TI, lambda_value, num_of_generations, saved_list, lambda_threshold,
                     top_parents_selected, sample_time):
    PI_tuple_as_str = np.array([str(tuple_) for tuple_ in PI])
    TI_tuple_as_str = np.array([str(tuple_) for tuple_ in TI])
    # for each cell in TI, find the index in PI
    TI_index = []
    for cell in TI_tuple_as_str:
        TI_index.append(np.where(PI_tuple_as_str == cell)[0][0])
    sampling_pool_new = []
    for cell in PI:
        index = []
        for i in range(len(cell)):
            current_feature_value = cell[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            index.append(current_feature_value_index)
        index = tuple(index)
        number_of_current_records = data_cube[index]
        tuple_cell = np.array([cell])
        tuple_cell = np.repeat(tuple_cell, number_of_current_records, axis=0)
        sampling_pool_new.extend(tuple_cell)
    # sampling
    parents_list = []
    num_of_parents = top_parents_selected * 5
    while len(parents_list) <= 1:
        parents_list = []
        corresponding_confidence_list = []
        for i in range(num_of_parents):
            sampled_records = np.random.choice(len(sampling_pool_new), num_purchased_records, replace=False)
            sampled_records = np.array(sampling_pool_new)[sampled_records]
            sampled_records = [tuple(x) for x in sampled_records]
            sampled_records_as_str = np.array([str(tuple_) for tuple_ in sampled_records])
            proportion_list = []
            for i in range(len(saved_list)):
                current_true_intent = TI[i]
                associated_ml_simulation_result = saved_list[i]
                number_of_current_records = np.sum(sampled_records_as_str == str(current_true_intent))
                # get the proportion in associated_ml_simulation_result that is equal or greater than
                # number_of_current_records
                current_proportion = np.sum(associated_ml_simulation_result >= number_of_current_records) / sample_time
                proportion_list.append(current_proportion)
            minimum_proportion = np.min(proportion_list)
            confidence = 1 - minimum_proportion
            parents_list.append(sampled_records)
            corresponding_confidence_list.append(confidence)
        # only keep parents that satisfy the privacy
        parents_list = np.array(parents_list)
        corresponding_confidence_list = np.array(corresponding_confidence_list)
        parents_list = parents_list[corresponding_confidence_list < lambda_value]
    # select parents that have high utility
    parents_list_top_utility_initial = selection_by_utility(parents_list, TI_index, PI, PI_tuple_as_str,
                                                            num_purchased_records, cost_cube,
                                                            unique_values_on_each_dimension, top_parents_selected)
    parents_list_top_utility = parents_list_top_utility_initial
    # children generation
    for i in range(num_of_generations):
        print("generation: ", i)
        children_list = crossover(parents_list_top_utility)
        # if it is the last iteration, we need to filter the children by privacy
        if i == num_of_generations - 1:
            children_list_after_filtering = filter_by_privacy(children_list, PI_tuple_as_str, TI, saved_list,
                                                              lambda_threshold, sample_time)
            if len(children_list_after_filtering) == 0:
                children_list_after_filtering = parents_list_top_utility_initial
            break
        children_list_high_utility = selection_by_utility(children_list, TI_index, PI, PI_tuple_as_str,
                                                          num_purchased_records, cost_cube,
                                                          unique_values_on_each_dimension, top_parents_selected)
        parents_list_top_utility = children_list_high_utility
    utility_list_final = utility_comp(children_list_after_filtering, TI_index, PI, PI_tuple_as_str,
                                      num_purchased_records, cost_cube, unique_values_on_each_dimension)
    optimal_solution = children_list_after_filtering[np.argmax(utility_list_final)]
    # compute confidence
    optimal_solution = [tuple(tuple_) for tuple_ in optimal_solution]
    optimal_solution_as_str = np.array([str(tuple_) for tuple_ in optimal_solution])
    proportion_list = []
    for i in range(len(saved_list)):
        current_true_intent = TI[i]
        associated_ml_simulation_result = saved_list[i]
        number_of_current_records = np.sum(optimal_solution_as_str == str(current_true_intent))
        # get the proportion in associated_ml_simulation_result that is equal or greater than
        # number_of_current_records
        current_proportion = np.sum(associated_ml_simulation_result >= number_of_current_records) / sample_time
        proportion_list.append(current_proportion)
    minimum_proportion = np.min(proportion_list)
    confidence = 1 - minimum_proportion
    return optimal_solution, confidence


