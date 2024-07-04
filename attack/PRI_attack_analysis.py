import numpy as np


def modeless_simulation_updated(data_cube, cost_data_cube, pseudo_PI, true_intent, unique_values_on_each_dimension,
                                num_purchased_records, sample_time):
    sampling_pool = []
    for cell in pseudo_PI:
        index = []
        for i in range(len(cell)):
            current_feature_value = cell[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            index.append(current_feature_value_index)
        index = tuple(index)
        number_of_current_records = data_cube[index]
        tuple_cell = np.array([cell])
        tuple_cell = np.repeat(tuple_cell, number_of_current_records, axis=0)
        sampling_pool.extend(tuple_cell)
    saved_list = []
    for cell in true_intent:
        tmp_list = []
        for i in range(sample_time):
            print("Iteration: ", i)
            sampled_records = np.random.choice(len(sampling_pool), num_purchased_records, replace=False)
            sampled_records = np.array(sampling_pool)[sampled_records]
            # change each row of sampled_records to tuple
            sampled_records = [tuple(x) for x in sampled_records]
            # Convert the tuples to hashable objects (tuples are not hashable)
            sampled_records_as_str = np.array([str(tuple_) for tuple_ in sampled_records])
            # check how many times the cell appears in the sampled records
            counts = np.sum(sampled_records_as_str == str(cell))
            tmp_list.append(counts)
        saved_list.append(tmp_list)
    return saved_list




