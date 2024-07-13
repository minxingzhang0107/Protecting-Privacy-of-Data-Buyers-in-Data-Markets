import numpy as np


def utility_comp(counts, TI_index, PI, unique_values_on_each_dimension, cost_data_cube):
    num_records_purchased = np.sum(counts)
    # proportion of records in the target cell
    TI_records_count = counts[TI_index]
    TI_records_count = np.array(TI_records_count)
    TI_records_count = np.sum(TI_records_count)
    proportion_of_records_in_target_cell = TI_records_count / num_records_purchased
    # total cost
    total_cost = 0.0
    return total_cost, proportion_of_records_in_target_cell


def compute_minimum(proportion_of_records_in_target_cell_current_state,
                    proportion_of_records_in_target_cell_new_state):
    # if proportion is 0, replace it with a tiny number
    if proportion_of_records_in_target_cell_current_state == 0:
        proportion_of_records_in_target_cell_current_state = 1e-10
    if proportion_of_records_in_target_cell_new_state == 0:
        proportion_of_records_in_target_cell_new_state = 1e-10
    utility_current_state = proportion_of_records_in_target_cell_current_state
    utility_new_state = proportion_of_records_in_target_cell_new_state
    return min(1, utility_new_state / utility_current_state)


def MCMC(data_cube, cost_data_cube, unique_values_on_each_dimension, num_purchased_records, TI, PI, lambda_value,
         saved_list, sample_time, diff2threshold_MCMC):
    PI = [tuple(x) for x in PI]
    TI = [tuple(x) for x in TI]
    PI_tuple_as_str = np.array([str(tuple_) for tuple_ in PI])
    TI_tuple_as_str = np.array([str(tuple_) for tuple_ in TI])
    # for each cell in TI, find the index in PI
    TI_index = []
    for cell in TI_tuple_as_str:
        TI_index.append(np.where(PI_tuple_as_str == cell)[0][0])
    # disguise index is the remaining index in PI
    disguise_index = np.delete(np.arange(len(PI)), TI_index)
    sampling_pool_new = []
    total_number_of_records_in_TI_ground_truth = 0
    for cell in PI:
        index = []
        for i in range(len(cell)):
            current_feature_value = cell[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            index.append(current_feature_value_index)
        index = tuple(index)
        number_of_current_records = data_cube[index]
        if str(tuple(cell)) in TI_tuple_as_str:
            total_number_of_records_in_TI_ground_truth += number_of_current_records
        tuple_cell = np.array([cell])
        tuple_cell = np.repeat(tuple_cell, number_of_current_records, axis=0)
        sampling_pool_new.extend(tuple_cell)
    # sampling
    confidence = np.inf
    while confidence > 0.3:
        sampled_records = np.random.choice(len(sampling_pool_new), num_purchased_records, replace=False)
        sampled_records = np.array(sampling_pool_new)[sampled_records]
        sampled_records = [tuple(x) for x in sampled_records]
        sampled_records_as_str = np.array([str(tuple_) for tuple_ in sampled_records])
        counts = np.array([np.sum(sampled_records_as_str == tuple_) for tuple_ in PI_tuple_as_str])
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
    disguise_cell_counts = np.sum(counts[disguise_index])
    target_cell_counts = np.sum(counts[TI_index])
    while disguise_cell_counts > 0 and (lambda_value - confidence) > diff2threshold_MCMC and target_cell_counts < \
            total_number_of_records_in_TI_ground_truth:
        confidence_old = confidence
        # randomly choose 2 cells from PI_tuple_as_str
        target_cell_index, disguise_cell_index = np.random.choice(len(PI), 2, replace=False)
        target_cell = PI[target_cell_index]
        disguise_cell = PI[disguise_cell_index]
        disguise_cell_as_str = str(tuple(disguise_cell))
        if disguise_cell_as_str in sampled_records_as_str:
            total_cost_current_state, proportion_of_records_in_target_cell_current_state = utility_comp(
                counts, TI_index, PI, unique_values_on_each_dimension, cost_data_cube)
            # as we only need to remove one disguise cell, we can choose the first one
            sampled_records_as_str = np.delete(sampled_records_as_str,
                                               np.where(sampled_records_as_str == disguise_cell_as_str)[0][0])
            # add one target cell to the sampled records
            sampled_records_as_str = np.append(sampled_records_as_str, str(tuple(target_cell)))
            # update the counts
            counts[target_cell_index] += 1
            counts[disguise_cell_index] -= 1
            if target_cell_index in TI_index:
                # find the index of the target cell in TI
                target_cell_index_ref = np.where(TI_index == target_cell_index)[0][0]
                current_num = counts[target_cell_index]
                associated_ml_simulation_result = saved_list[target_cell_index_ref]
                current_proportion = np.sum(associated_ml_simulation_result >= current_num) / sample_time
                confidence = 1 - current_proportion
                # if exceed the privacy threshold
                if confidence > lambda_value:
                    sampled_records_as_str = np.delete(sampled_records_as_str, np.where(sampled_records_as_str ==
                                                                                        str(tuple(target_cell)))[0][0])
                    sampled_records_as_str = np.append(sampled_records_as_str, str(tuple(disguise_cell)))
                    counts[target_cell_index] -= 1
                    counts[disguise_cell_index] += 1
                    confidence = confidence_old
                # if not exceed the privacy threshold
                else:
                    total_cost_new_state, proportion_of_records_in_target_cell_new_state = utility_comp(
                        counts, TI_index, PI, unique_values_on_each_dimension, cost_data_cube)
                    min_tmp = compute_minimum(proportion_of_records_in_target_cell_current_state,
                                              proportion_of_records_in_target_cell_new_state)
                    if min_tmp == 1:
                        if target_cell_index in TI_index and disguise_cell_index in disguise_index:
                            disguise_cell_counts -= 1
                            target_cell_counts += 1
                        elif target_cell_index in disguise_index and disguise_cell_index in TI_index:
                            disguise_cell_counts += 1
                            target_cell_counts -= 1
                    else:
                        random_num = np.random.uniform(0, 1)
                        if random_num <= min_tmp:
                            if target_cell_index in TI_index and disguise_cell_index in disguise_index:
                                disguise_cell_counts -= 1
                                target_cell_counts += 1
                            elif target_cell_index in disguise_index and disguise_cell_index in TI_index:
                                disguise_cell_counts += 1
                                target_cell_counts -= 1
                        else:
                            sampled_records_as_str = np.delete(sampled_records_as_str,
                                                               np.where(sampled_records_as_str ==
                                                                        str(tuple(target_cell)))[0][0])
                            sampled_records_as_str = np.append(sampled_records_as_str, str(tuple(disguise_cell)))
                            counts[target_cell_index] -= 1
                            counts[disguise_cell_index] += 1
                            confidence = confidence_old
        else:
            continue
    return sampled_records_as_str, confidence