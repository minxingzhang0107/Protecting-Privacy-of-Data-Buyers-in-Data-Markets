import numpy as np


def utility_computation(sampled_records_as_str, TI, PI, unique_values_on_each_dimension, cost_data_cube):
    PI_tuple_as_str = np.array([str(tuple_) for tuple_ in PI])
    TI_tuple_as_str = np.array([str(tuple_) for tuple_ in TI])
    # for each cell in TI, find the index in PI
    TI_index = []
    for cell in TI_tuple_as_str:
        TI_index.append(np.where(PI_tuple_as_str == cell)[0][0])
    counts = np.array([np.sum(sampled_records_as_str == tuple_) for tuple_ in PI_tuple_as_str])
    num_records_in_TI = np.sum(counts[TI_index])
    total_cost = 0.0
    return total_cost, num_records_in_TI


def find_optimal_solution(total_cost_list, num_records_in_TI_list, solution_list, num_purchased_records):
    proportion_of_records_in_TI_list = np.array(num_records_in_TI_list) / num_purchased_records
    utility_list = proportion_of_records_in_TI_list
    optimal_index = np.argmax(utility_list)
    optimal_solution = solution_list[optimal_index]
    return optimal_solution


def MC_simulation(data_cube, cost_data_cube, unique_values_on_each_dimension, num_purchased_records,
                  TI, PI, lambda_value, iteration_num, saved_list, sample_time):
    PI_tuple_as_str = np.array([str(tuple_) for tuple_ in PI])
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
    # MC simulation
    total_cost_list = []
    num_records_in_TI_list = []
    solution_list = []
    for i in range(iteration_num):
        # shuffle the sampling pool
        np.random.shuffle(sampling_pool_new)
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
        # if satisfy privacy threshold
        if confidence <= lambda_value:
            total_cost, num_records_in_TI = utility_computation(sampled_records_as_str, TI, PI,
                                                                unique_values_on_each_dimension, cost_data_cube)
            total_cost_list.append(total_cost)
            num_records_in_TI_list.append(num_records_in_TI)
            solution_list.append(sampled_records)
        else:
            continue
    if len(total_cost_list) == 0:
        return None, None
    else:
        optimal_solution = find_optimal_solution(total_cost_list, num_records_in_TI_list, solution_list,
                                                 num_purchased_records)
        # compute attack confidence
        sampled_records_as_str = np.array([str(tuple_) for tuple_ in optimal_solution])
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
        final_confidence = 1 - minimum_proportion
        return optimal_solution, final_confidence