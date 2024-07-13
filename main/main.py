import numpy as np
import os
import csv
from itertools import product
import sys
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adult_data'))
from adult_data import helper

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'attack'))
from attack import PI_attack_analysis
from attack import EM_attack_analysis
from attack import PRI_attack_analysis

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'defense'))
from defense import expansion
from defense import G_MCMC
from defense import MC_simulation
from defense import MCMC
from defense import Genetic_sampling


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


def compute_total_num_records(data_cube, unique_values_on_each_dimension, published_intent):
    PI = list(product(*published_intent))
    total_num_records = 0
    for record in PI:
        record_index = []
        for i in range(len(record)):
            current_feature_value = record[i]
            current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
            record_index.append(current_feature_value_index)
        record_index = tuple(record_index)
        num_records = data_cube[record_index]
        total_num_records += num_records
    return total_num_records, PI


def compute_stat_PRI(cost_cube, unique_values_on_each_dimension, pseudo_PI, TI, purchased_set_of_records, is_string):
    total_cost_of_PI = 0.0
    total_cost_of_TI = 0.0
    num_records_in_PI = 0
    num_records_in_TI = 0
    pseudo_PI_tuple = [tuple(x) for x in pseudo_PI]
    pseudo_PI_tuple_as_str = np.array([str(tuple_) for tuple_ in pseudo_PI_tuple])
    TI_tuple = [tuple(x) for x in TI]
    TI_tuple_as_str = np.array([str(tuple_) for tuple_ in TI_tuple])
    # if element in purchased_set_of_records is not in str format, convert it to str format
    if is_string == False:
        purchased_set_of_records_tuple = [tuple(x) for x in purchased_set_of_records]
        purchased_set_of_records_as_str = np.array([str(tuple_) for tuple_ in purchased_set_of_records_tuple])
    else:
        purchased_set_of_records_as_str = purchased_set_of_records
    # for each cell in pseudo_PI, count the number of records in the sampled_records
    counts = np.array([np.sum(purchased_set_of_records_as_str == tuple_) for tuple_ in pseudo_PI_tuple_as_str])
    for i in range(len(pseudo_PI)):
        current_cell = pseudo_PI[i]
        corresponding_count = counts[i]
        if corresponding_count == 0:
            continue
        current_cell_index = []
        for j in range(len(current_cell)):
            current_feature_value = current_cell[j]
            current_feature_value_index = unique_values_on_each_dimension[j].index(current_feature_value)
            current_cell_index.append(current_feature_value_index)
        current_cell_index = tuple(current_cell_index)
        current_unit_price = cost_cube[current_cell_index]
        # check if the current cell is in TI
        if pseudo_PI_tuple_as_str[i] in TI_tuple_as_str:
            total_cost_of_TI += current_unit_price * corresponding_count
            num_records_in_TI += corresponding_count
        total_cost_of_PI += current_unit_price * corresponding_count
        num_records_in_PI += corresponding_count
    return total_cost_of_PI, total_cost_of_TI, num_records_in_PI, num_records_in_TI


if __name__ == '__main__':
    ## specify the parameters
    parser = argparse.ArgumentParser(description='Specify the parameters')
    parser.add_argument('--attack_type', type=str, default='PI_uniform_attack',
                        help='Specify the attack type: PI_uniform_attack, EM_attack, PRI_attack')
    parser.add_argument('--data_type', type=str, default='real_data',
                        help='Specify the data type: real_data, synthetic_data')
    parser.add_argument('--true_intent_size', type=int, default=2,
                        help='Specify the true intent size: 1, 2')
    # specify a defense strategy for purchased record inference attack
    parser.add_argument('--protect_method_PRI', type=str, default='MC_simulation',
                        help='Specify a protection method for Purchased Record Inference Attack: G_MCMC, '
                             'MC_simulation, MCMC, Genetic_sampling')
    if parser.parse_args().data_type == 'real_data' and parser.parse_args().true_intent_size == 1:
        current_iteration = 1
    elif parser.parse_args().data_type == 'real_data' and parser.parse_args().true_intent_size == 2:
        current_iteration = 2
    elif parser.parse_args().data_type == 'synthetic_data' and parser.parse_args().true_intent_size == 2:
        current_iteration = 3
    elif parser.parse_args().data_type == 'synthetic_data' and parser.parse_args().true_intent_size == 1:
        current_iteration = 4
    else:
        raise ValueError("Invalid data_type and true_intent_size")


    ## load the generated data based on the specified iteration
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


    ## PI-uniform attack
    if parser.parse_args().attack_type == 'PI_uniform_attack':
        print("PI-uniform attack")
        if current_iteration == 1:
            w_1 = 0.5
        elif current_iteration == 2:
            w_1 = 1
        elif current_iteration == 3:
            w_1 = 0.6
        elif current_iteration == 4:
            w_1 = 0.8
        else:
            raise ValueError("Invalid current_iteration")
        w_2 = 1 - w_1
        percent_of_records_purchased_in_each_grid = 1
        lambda_value = 0.3
        print("If there is no disguise for PI uniform attack, published intent is the same as true intent.")
        print("The confidence's lower bound is: ", PI_attack_analysis.confidence_lower_bound(true_intent))
        print("The confidence's upper bound is: ", PI_attack_analysis.confidence_upper_bound(true_intent, true_intent))
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost)
        print("The total cost spent by the buyer on the true intent is: ", PI_total_cost)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("-----------------------")
        print("Given lambda value: ", lambda_value)
        print("To protect against PI-uniform attack, the size of the published intent should be: ",
              np.round(PI_attack_analysis.lambda_privacy_published_intent_lower_bound(lambda_value, true_intent)))
        print("Given w_1 (the weight controlling total cost): ", w_1, " and w_2 (the weight controlling increase in"
                                                                      " privacy): ", w_2)
        print("To defend against PI-uniform attack, we leverage expansion method")
        attack_type = 'PI_uniform_attack'
        start_time = time.time()
        published_intent_PI_uniform_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, attack_type, lambda_value, percent_of_records_purchased_in_each_grid)
        end_time = time.time()
        print("Time taken for expansion method to defend against PI-uniform attack: ", end_time - start_time)
        # print("The resulted published intent is: ", published_intent_PI_uniform_attack)
        # print("The true intent is: ", true_intent)
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_PI_uniform_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        TI_tuple_form = list(product(*true_intent))
        PI_tuple_form = list(product(*published_intent_PI_uniform_attack))
        print("The attacker's confidence based on the new published intent")
        print("The confidence's lower bound is: ",
              PI_attack_analysis.confidence_lower_bound(published_intent_PI_uniform_attack))
        print("The confidence's upper bound is: ",
              PI_attack_analysis.confidence_upper_bound(true_intent, published_intent_PI_uniform_attack))
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost)
        print("The total cost spent by the buyer on the true intent is: ", TI_total_cost)
        # print("The total number of records in the true intent is: ", np.sum(TI_total_num_records_for_each_cell))
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(TI_num_records_bought_for_each_cell))


    ## EM attack
    if parser.parse_args().attack_type == 'EM_attack':
        print("EM attack")
        lambda_value = 0.3
        percent_of_records_purchased_in_each_grid = 1
        attack_type_2 = 'EM_attack'
        print("-----------------------")
        print("-----------------------")
        print("Given only data distribution f_d as attacker's background knowledge")
        background_knowledge = 'only_f_d'
        print("If there is no disguise for EM uniform attack, published intent is the same as true intent.")
        print("Given only f_d as background knowledge, the confidence's upper bound is: ",
              EM_attack_analysis.confidence_upper_bound_generalization(data_cube, cost_cube, true_intent, true_intent,
                                                                       unique_values_on_each_dimension,
                                                                       background_knowledge))
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost)
        print("The total cost spent by the buyer on the true intent is: ", PI_total_cost)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        if current_iteration == 1:
            w_1 = 0.5
        elif current_iteration == 2:
            w_1 = 0.6
        elif current_iteration == 3:
            w_1 = 0.4
        elif current_iteration == 4:
            w_1 = 0.4
        w_2 = 1 - w_1
        print("Given lambda value: ", lambda_value)
        print("Given w_1 (the weight controlling total cost):", w_1, " and w_2 (the weight controlling increase in"
                                                                     " privacy): ", w_2)
        start_time = time.time()
        published_intent_EM_attack = expansion.expansion_EMF_EMC(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, background_knowledge,
                                                                 lambda_value, percent_of_records_purchased_in_each_grid)
        print("The time spent on expansion is: ", time.time() - start_time)
        print("The attacker confidence's upper bound after protection via expansion method is: ",
              EM_attack_analysis.confidence_upper_bound_generalization(data_cube, cost_cube, published_intent_EM_attack,
                                                                       true_intent, unique_values_on_each_dimension,
                                                                       background_knowledge))
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_EM_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        TI_tuple_form = list(product(*true_intent))
        PI_tuple_form = list(product(*published_intent_EM_attack))
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost)
        print("The total cost spent by the buyer on the true intent is: ", TI_total_cost)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(TI_num_records_bought_for_each_cell))
        print("-----------------------")
        print("-----------------------")
        print("Given only cost as attacker's background knowledge")
        background_knowledge_2 = 'only_cost'
        print("If there is no disguise for PI uniform attack, published intent is the same as true intent.")
        print("Given only cost as background knowledge, the confidence's upper bound is: ",
              EM_attack_analysis.confidence_upper_bound_generalization(data_cube, cost_cube, true_intent, true_intent,
                                                                       unique_values_on_each_dimension,
                                                                       background_knowledge_2))
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost)
        print("The total cost spent by the buyer on the true intent is: ", PI_total_cost)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        if current_iteration == 1:
            w_1 = 0.5
        elif current_iteration == 2:
            w_1 = 0.5
        elif current_iteration == 3:
            w_1 = 0.7
        elif current_iteration == 4:
            w_1 = 0.8
        w_2 = 1 - w_1
        print("Given lambda value: ", lambda_value)
        print("Given w_1 (the weight controlling total cost): ", w_1, " and w_2 (the weight controlling increase in"
                                                                      " privacy): ", w_2)
        start_time = time.time()
        published_intent_EM_attack = expansion.expansion_EMF_EMC(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                 true_intent, w_1, w_2, background_knowledge_2,
                                                                 lambda_value, percent_of_records_purchased_in_each_grid)
        print("The time spent on expansion is: ", time.time() - start_time)
        print("The attacker confidence's upper bound after protection via expansion method is: ",
              EM_attack_analysis.confidence_upper_bound_generalization(data_cube, cost_cube, published_intent_EM_attack,
                                                                       true_intent, unique_values_on_each_dimension,
                                                                       background_knowledge_2))
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_EM_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        TI_tuple_form = list(product(*true_intent))
        PI_tuple_form = list(product(*published_intent_EM_attack))
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost)
        print("The total cost spent by the buyer on the true intent is: ", TI_total_cost)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(TI_num_records_bought_for_each_cell))
        print("-----------------------")
        print("-----------------------")
        print("Case - given both f_d and cost as attacker's background knowledge")
        background_knowledge_3 = 'both_f_d_and_cost'
        print("If there is no disguise for PI uniform attack, published intent is the same as true intent.")
        print("Given both f_d and cost as background knowledge, the attacker confidence's upper bound is: ",
              EM_attack_analysis.confidence_upper_bound_generalization(data_cube, cost_cube, true_intent, true_intent,
                                                                       unique_values_on_each_dimension,
                                                                       background_knowledge_3))
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost)
        print("The total cost spent by the buyer on the true intent is: ", PI_total_cost)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        if current_iteration == 1:
            w_1 = 0.5
        elif current_iteration == 2:
            w_1 = 0.6
        elif current_iteration == 3:
            w_1 = 0.5
        elif current_iteration == 4:
            w_1 = 0.4
        w_2 = 1 - w_1
        print("Given lambda value: ", lambda_value)
        print("Given w_1 (the weight controlling total cost): ", w_1, " and w_2 (the weight controlling increase in"
                                                                      " privacy): ", w_2)
        records_in_TI_f_d_cost_multiplication_maximum_divided_by_lambda, records_in_TI_f_d_cost_multiplication_maximum = \
            EM_attack_analysis.lambda_privacy_published_intent_lower_bound(lambda_value, true_intent, data_cube,
                                                                           cost_cube, unique_values_on_each_dimension)
        print("To protect against EM attack, the summation of f_d(t) * cost(t) for all t in published intent should be: ",
              records_in_TI_f_d_cost_multiplication_maximum_divided_by_lambda)
        print("The maximum of f_d(t) * cost(t) for all t in true intent is: ",
              records_in_TI_f_d_cost_multiplication_maximum)
        start_time = time.time()
        published_intent_EM_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                         true_intent, w_1, w_2, attack_type_2, lambda_value, percent_of_records_purchased_in_each_grid)
        print("The time spent on expansion is: ", time.time() - start_time)
        print("The attacker confidence's upper bound after protection via expansion method is: ",
              EM_attack_analysis.confidence_upper_bound_generalization(data_cube, cost_cube, published_intent_EM_attack,
                                                                       true_intent, unique_values_on_each_dimension,
                                                                       background_knowledge_3))
        TI_total_cost, TI_total_num_records_for_each_cell, TI_num_records_bought_for_each_cell, \
            TI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, true_intent,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        PI_total_cost, PI_total_num_records_for_each_cell, PI_num_records_bought_for_each_cell, \
            PI_total_cost_for_each_cell = compute_total_cost(data_cube, cost_cube, published_intent_EM_attack,
                                                             unique_values_on_each_dimension, percent_of_records_purchased_in_each_grid)
        TI_tuple_form = list(product(*true_intent))
        PI_tuple_form = list(product(*published_intent_EM_attack))
        print("The total cost spent by the buyer on the published intent is: ", PI_total_cost)
        print("The total cost spent by the buyer on the true intent is: ", TI_total_cost)
        print("The number of records bought by the buyer in the published intent is: ",
              np.sum(PI_num_records_bought_for_each_cell))
        print("The number of records bought by the buyer in the true intent is: ",
              np.sum(TI_num_records_bought_for_each_cell))

    ## Purchased record inference attack
    if parser.parse_args().attack_type == 'PRI_attack':
        print("PRI attack")
        print("Given only data distribution f_d as attacker's background knowledge")
        if current_iteration == 1:
            w_1 = 0.5
        elif current_iteration == 2:
            w_1 = 0.6
        elif current_iteration == 3:
            w_1 = 0.4
        elif current_iteration == 4:
            w_1 = 0.4
        w_2 = 1 - w_1
        percent_of_records_purchased_in_each_grid = 1
        lambda_value = 0.3
        sample_time = 100000
        iteration_num = 100000
        attack_type_2 = 'EM_attack'
        published_intent_EM_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                         true_intent, w_1, w_2, attack_type_2, lambda_value, percent_of_records_purchased_in_each_grid)
        TI_num_records, TI = compute_total_num_records(data_cube, unique_values_on_each_dimension, true_intent)
        PI_num_records, PI = compute_total_num_records(data_cube, unique_values_on_each_dimension,
                                                       published_intent_EM_attack)
        TI_size = len(TI)
        PI_size = len(PI)
        print("Before allocation, we first determine the published intent via expansion method")
        print("The size of the true intent is: ", TI_size)
        print("The size of the published intent is: ", PI_size)
        purchased_set_of_records = []
        for cell in PI:
            index = []
            for i in range(len(cell)):
                current_feature_value = cell[i]
                current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
                index.append(current_feature_value_index)
            index = tuple(index)
            number_of_current_records = data_cube[index]
            # get every records in true intent under the case of no disguise
            if cell in TI:
                tuple_cell = np.array([cell])
                tuple_cell = np.repeat(tuple_cell, number_of_current_records, axis=0)
                purchased_set_of_records.extend(tuple_cell)
            # only get one record in PI that is not TI under the case of no disguise
            else:
                if number_of_current_records > 0:
                    tuple_cell = np.array([cell])
                    purchased_set_of_records.extend(tuple_cell)
                else:
                    continue
        num_purchased_records = len(purchased_set_of_records)
        print("The number of records bought by the buyer is: ", num_purchased_records)

        ## -----------------------------------------------
        # # modeless simulation (note that this part requires some time to run, so I save the result, which would be
        # # loaded directly in the next time)
        # saved_list = PRI_attack_analysis.modeless_simulation_updated(
        #     data_cube, cost_cube, PI, TI, unique_values_on_each_dimension,
        #     num_purchased_records, sample_time)
        # # save the list
        # running_data_directory = os.path.join(current_directory, 'running_data')
        # if not os.path.exists(running_data_directory):
        #     os.makedirs(running_data_directory)
        # iteration_directory = os.path.join(running_data_directory, 'iteration_' + str(current_iteration))
        # if not os.path.exists(iteration_directory):
        #     os.makedirs(iteration_directory)
        # np.save(os.path.join(iteration_directory, 'ml_simulation_result_true_intent.npy'), saved_list)
        ## -----------------------------------------------

        print("If there is no protection for the buyer")
        print("Suppose the buyer buys all the records in the true intent, while just 1 record in each cell in the "
              "disguise")
        # load the ml_simulation_result_true_intent.npy
        saved_list = np.load(os.path.join(iteration_directory, 'ml_simulation_result_true_intent.npy'))
        proportion_list = []
        for i in range(len(saved_list)):
            current_true_intent = TI[i]
            associated_ml_simulation_result = saved_list[i]
            index = []
            for i in range(len(current_true_intent)):
                current_feature_value = current_true_intent[i]
                current_feature_value_index = unique_values_on_each_dimension[i].index(current_feature_value)
                index.append(current_feature_value_index)
            index = tuple(index)
            number_of_current_records = data_cube[index]
            # get the proportion in associated_ml_simulation_result that is equal or greater than
            # number_of_current_records
            current_proportion = np.sum(associated_ml_simulation_result >= number_of_current_records) / sample_time
            proportion_list.append(current_proportion)
        maximum_proportion = np.max(proportion_list)
        print("The attack's confidence (no protection) is: ", 1-maximum_proportion)
        total_cost_of_PI, total_cost_of_TI, num_records_in_PI, num_records_in_TI = compute_stat_PRI(
            cost_cube, unique_values_on_each_dimension, PI, TI, purchased_set_of_records, False)
        print("The total cost spent by the buyer on the published intent is: ", total_cost_of_PI)
        print("The total cost spent by the buyer on the true intent is: ", total_cost_of_TI)
        print("The number of records bought by the buyer in the published intent is: ", num_records_in_PI)
        print("The number of records bought by the buyer in the true intent is: ", num_records_in_TI)


        ## Purchased record inference attack defense
        # load the ml_simulation_result_true_intent.npy
        saved_list = np.load(os.path.join(iteration_directory, 'ml_simulation_result_true_intent.npy'))
        published_intent_EM_attack = expansion.expansion(data_cube, cost_cube, unique_values_on_each_dimension,
                                                         true_intent, w_1, w_2, attack_type_2, lambda_value, percent_of_records_purchased_in_each_grid)
        TI_num_records, TI = compute_total_num_records(data_cube, unique_values_on_each_dimension, true_intent)
        PI_num_records, PI = compute_total_num_records(data_cube, unique_values_on_each_dimension,
                                                       published_intent_EM_attack)
        # if PRI_protect_method is G_MCMC
        if parser.parse_args().protect_method_PRI == 'G_MCMC':
            print("-----------------------")
            print("Using Greedy Markov Chain Monte Carlo to allocate records in the pseudo published intent")
            print("We run G-MCMC for 10 times")
            if current_iteration == 1:
                diff2threshold = 0.07
            elif current_iteration == 2:
                diff2threshold = 0.01
            elif current_iteration == 3:
                diff2threshold = 0.01
            elif current_iteration == 4:
                diff2threshold = 0.005
            else:
                raise ValueError("Invalid current_iteration")
            time_list = []
            confidence_list = []
            total_cost_of_PI_list = []
            total_cost_of_TI_list = []
            num_records_in_PI_list = []
            num_records_in_TI_list = []
            proportion_cost_TI_list = []
            proportion_records_TI_list = []
            for i in range(10):
                print("The ", i, "th time")
                start_time = time.time()
                sampled_records_as_str, confidence = G_MCMC.G_MCMC(data_cube, cost_cube, unique_values_on_each_dimension,
                                                                   num_purchased_records, TI, PI, lambda_value,
                                                                   saved_list, sample_time, diff2threshold)
                end_time = time.time()
                print("The running time for G-MCMC is: ", end_time - start_time)
                print("The attack's confidence (G-MCMC) is: ", confidence)
                total_cost_of_PI, total_cost_of_TI, num_records_in_PI, num_records_in_TI = compute_stat_PRI(
                    cost_cube, unique_values_on_each_dimension, PI, TI, sampled_records_as_str, True)
                print("The total cost spent by the buyer on the published intent is: ", total_cost_of_PI)
                print("The total cost spent by the buyer on the true intent is: ", total_cost_of_TI)
                print("The number of records bought by the buyer in the published intent is: ", num_records_in_PI)
                print("The number of records bought by the buyer in the true intent is: ", num_records_in_TI)
                print("-----------------------")
                time_list.append(end_time - start_time)
                confidence_list.append(confidence)
                total_cost_of_PI_list.append(total_cost_of_PI)
                total_cost_of_TI_list.append(total_cost_of_TI)
                num_records_in_PI_list.append(num_records_in_PI)
                num_records_in_TI_list.append(num_records_in_TI)
                proportion_cost_TI_list.append(total_cost_of_TI / total_cost_of_PI)
                proportion_records_TI_list.append(num_records_in_TI / num_records_in_PI)
            print("The average running time for G-MCMC is: ", np.mean(time_list))
            print("The average attack's confidence (G-MCMC) is: ", np.mean(confidence_list))
            print("The average total cost spent by the buyer on the published intent is: ",
                  np.mean(total_cost_of_PI_list))
            print("The average total cost spent by the buyer on the true intent is: ", np.mean(total_cost_of_TI_list))
            print("The average number of records bought by the buyer in the published intent is: ",
                  np.mean(num_records_in_PI_list))
            print("The average number of records bought by the buyer in the true intent is: ",
                  np.mean(num_records_in_TI_list))
            print("The average proportion of cost spent by the buyer on the true intent is: ",
                  np.mean(proportion_cost_TI_list))
            print("The average proportion of records bought by the buyer in the true intent is: ",
                  np.mean(proportion_records_TI_list))
            # standard deviation
            print("The standard deviation of running time for G-MCMC is: ", np.std(time_list))
            print("The standard deviation of attack's confidence (G-MCMC) is: ", np.std(confidence_list))
            print("The standard deviation of total cost spent by the buyer on the published intent is: ",
                    np.std(total_cost_of_PI_list))
            print("The standard deviation of total cost spent by the buyer on the true intent is: ",
                    np.std(total_cost_of_TI_list))
            print("The standard deviation of number of records bought by the buyer in the published intent is: ",
                    np.std(num_records_in_PI_list))
            print("The standard deviation of number of records bought by the buyer in the true intent is: ",
                    np.std(num_records_in_TI_list))
            print("The standard deviation of proportion of cost spent by the buyer on the true intent is: ",
                    np.std(proportion_cost_TI_list))
            print("The standard deviation of proportion of records bought by the buyer in the true intent is: ",
                    np.std(proportion_records_TI_list))
        # if PRI_protect_method is G_MCMC
        if parser.parse_args().protect_method_PRI == 'MC_simulation':
            print("-----------------------")
            print("Using MC simulation to allocate records in the pseudo published intent")
            print("We run MC simulation for 10 times")
            time_list = []
            confidence_list = []
            total_cost_of_PI_list = []
            total_cost_of_TI_list = []
            num_records_in_PI_list = []
            num_records_in_TI_list = []
            proportion_cost_TI_list = []
            proportion_records_TI_list = []
            for i in range(10):
                print("The ", i, "th time")
                start_time = time.time()
                optimal_solution, final_confidence = MC_simulation.MC_simulation(
                    data_cube, cost_cube, unique_values_on_each_dimension, num_purchased_records,
                    TI, PI, lambda_value, iteration_num, saved_list, sample_time)
                end_time = time.time()
                print("The running time for MC simulation is: ", end_time - start_time)
                print("The attacker's confidence (MC simulation) is: ", final_confidence)
                total_cost_of_PI, total_cost_of_TI, num_records_in_PI, num_records_in_TI = compute_stat_PRI(
                    cost_cube, unique_values_on_each_dimension, PI, TI, optimal_solution, False)
                print("The total cost spent by the buyer on the published intent is: ", total_cost_of_PI)
                print("The total cost spent by the buyer on the true intent is: ", total_cost_of_TI)
                print("The number of records bought by the buyer in the published intent is: ", num_records_in_PI)
                print("The number of records bought by the buyer in the true intent is: ", num_records_in_TI)
                proportion_cost_TI = total_cost_of_TI / total_cost_of_PI
                proportion_records_TI = num_records_in_TI / num_records_in_PI
                print("The proportion of cost spent by the buyer on the true intent is: ", proportion_cost_TI)
                print("The proportion of records bought by the buyer in the true intent is: ", proportion_records_TI)
                time_list.append(end_time - start_time)
                confidence_list.append(final_confidence)
                total_cost_of_PI_list.append(total_cost_of_PI)
                total_cost_of_TI_list.append(total_cost_of_TI)
                num_records_in_PI_list.append(num_records_in_PI)
                num_records_in_TI_list.append(num_records_in_TI)
                proportion_cost_TI_list.append(proportion_cost_TI)
                proportion_records_TI_list.append(proportion_records_TI)
                print("-----------------------")
            print("The average running time for MC simulation is: ", np.mean(time_list))
            print("The average attack's confidence (MC simulation) is: ", np.mean(confidence_list))
            print("The average total cost spent by the buyer on the published intent is: ",
                  np.mean(total_cost_of_PI_list))
            print("The average total cost spent by the buyer on the true intent is: ", np.mean(total_cost_of_TI_list))
            print("The average number of records bought by the buyer in the published intent is: ",
                  np.mean(num_records_in_PI_list))
            print("The average number of records bought by the buyer in the true intent is: ",
                  np.mean(num_records_in_TI_list))
            print("The average proportion of cost spent by the buyer on the true intent is: ",
                  np.mean(proportion_cost_TI_list))
            print("The average proportion of records bought by the buyer in the true intent is: ",
                  np.mean(proportion_records_TI_list))
            # standard deviation
            print("The standard deviation of running time for MC simulation is: ", np.std(time_list))
            print("The standard deviation of attack's confidence (MC simulation) is: ", np.std(confidence_list))
            print("The standard deviation of total cost spent by the buyer on the published intent is: ",
                  np.std(total_cost_of_PI_list))
            print("The standard deviation of total cost spent by the buyer on the true intent is: ",
                  np.std(total_cost_of_TI_list))
            print("The standard deviation of number of records bought by the buyer in the published intent is: ",
                  np.std(num_records_in_PI_list))
            print("The standard deviation of number of records bought by the buyer in the true intent is: ",
                  np.std(num_records_in_TI_list))
            print("The standard deviation of proportion of cost spent by the buyer on the true intent is: ",
                  np.std(proportion_cost_TI_list))
            print("The standard deviation of proportion of records bought by the buyer in the true intent is: ",
                  np.std(proportion_records_TI_list))
        if parser.parse_args().protect_method_PRI == 'MCMC':
            print("-----------------------")
            print("Using Markov Chain Monte Carlo to allocate records in the pseudo published intent")
            print("We run MCMC for 10 times")
            if current_iteration == 1:
                diff2threshold_MCMC = 0.001
            elif current_iteration == 2:
                diff2threshold_MCMC = 0.001
            elif current_iteration == 3:
                diff2threshold_MCMC = 0.001
            elif current_iteration == 4:
                diff2threshold_MCMC = 0.001
            else:
                raise ValueError("Invalid current_iteration")
            time_list = []
            confidence_list = []
            total_cost_of_PI_list = []
            total_cost_of_TI_list = []
            num_records_in_PI_list = []
            num_records_in_TI_list = []
            proportion_cost_TI_list = []
            proportion_records_TI_list = []
            for i in range(10):
                print("The ", i, "th time")
                start_time = time.time()
                sampled_records_as_str, confidence = MCMC.MCMC(
                    data_cube, cost_cube, unique_values_on_each_dimension, num_purchased_records, TI, PI, lambda_value,
                    saved_list, sample_time, diff2threshold_MCMC)
                end_time = time.time()
                print("The running time for MCMC is: ", end_time - start_time)
                print("The attacker's confidence (MCMC) is: ", confidence)
                total_cost_of_PI, total_cost_of_TI, num_records_in_PI, num_records_in_TI = compute_stat_PRI(
                    cost_cube, unique_values_on_each_dimension, PI, TI, sampled_records_as_str, True)
                print("The total cost spent by the buyer on the published intent is: ", total_cost_of_PI)
                print("The total cost spent by the buyer on the true intent is: ", total_cost_of_TI)
                print("The number of records bought by the buyer in the published intent is: ", num_records_in_PI)
                print("The number of records bought by the buyer in the true intent is: ", num_records_in_TI)
                proportion_cost_TI = total_cost_of_TI / total_cost_of_PI
                proportion_records_TI = num_records_in_TI / num_records_in_PI
                print("The proportion of cost spent by the buyer on the true intent is: ", proportion_cost_TI)
                print("The proportion of records bought by the buyer in the true intent is: ", proportion_records_TI)
                time_list.append(end_time - start_time)
                confidence_list.append(confidence)
                total_cost_of_PI_list.append(total_cost_of_PI)
                total_cost_of_TI_list.append(total_cost_of_TI)
                num_records_in_PI_list.append(num_records_in_PI)
                num_records_in_TI_list.append(num_records_in_TI)
                proportion_cost_TI_list.append(proportion_cost_TI)
                proportion_records_TI_list.append(proportion_records_TI)
                print("-----------------------")
            print("The average running time for MCMC is: ", np.mean(time_list))
            print("The average attack's confidence (MCMC) is: ", np.mean(confidence_list))
            print("The average total cost spent by the buyer on the published intent is: ",
                  np.mean(total_cost_of_PI_list))
            print("The average total cost spent by the buyer on the true intent is: ", np.mean(total_cost_of_TI_list))
            print("The average number of records bought by the buyer in the published intent is: ",
                  np.mean(num_records_in_PI_list))
            print("The average number of records bought by the buyer in the true intent is: ",
                  np.mean(num_records_in_TI_list))
            print("The average proportion of cost spent by the buyer on the true intent is: ",
                  np.mean(proportion_cost_TI_list))
            print("The average proportion of records bought by the buyer in the true intent is: ",
                  np.mean(proportion_records_TI_list))
            # standard deviation
            print("The standard deviation of running time for MCMC is: ", np.std(time_list))
            print("The standard deviation of attack's confidence (MCMC) is: ", np.std(confidence_list))
            print("The standard deviation of total cost spent by the buyer on the published intent is: ",
                  np.std(total_cost_of_PI_list))
            print("The standard deviation of total cost spent by the buyer on the true intent is: ",
                  np.std(total_cost_of_TI_list))
            print("The standard deviation of number of records bought by the buyer in the published intent is: ",
                  np.std(num_records_in_PI_list))
            print("The standard deviation of number of records bought by the buyer in the true intent is: ",
                  np.std(num_records_in_TI_list))
            print("The standard deviation of proportion of cost spent by the buyer on the true intent is: ",
                  np.std(proportion_cost_TI_list))
            print("The standard deviation of proportion of records bought by the buyer in the true intent is: ",
                  np.std(proportion_records_TI_list))
        if parser.parse_args().protect_method_PRI == 'Genetic_sampling':
            print("-----------------------")
            print("Using Genetic Sampling to allocate records in the pseudo published intent")
            print("We run Genetic Sampling for 10 times")
            num_of_generations = 30
            top_parents_selected = 10
            time_list = []
            confidence_list = []
            total_cost_of_PI_list = []
            total_cost_of_TI_list = []
            num_records_in_PI_list = []
            num_records_in_TI_list = []
            proportion_cost_TI_list = []
            proportion_records_TI_list = []
            for i in range(10):
                print("The ", i, "th time")
                start_time = time.time()
                optimal_solution, confidence = Genetic_sampling.Genetic_sampling(data_cube, cost_cube,
                             unique_values_on_each_dimension, num_purchased_records,
                             PI, TI, lambda_value, num_of_generations, saved_list, lambda_value,
                             top_parents_selected, sample_time)
                end_time = time.time()
                time_list.append(end_time - start_time)
                total_cost_of_PI, total_cost_of_TI, num_records_in_PI, num_records_in_TI = compute_stat_PRI(
                    cost_cube, unique_values_on_each_dimension, PI, TI, optimal_solution, False)
                print("The total cost spent by the buyer on the published intent is: ", total_cost_of_PI)
                print("The total cost spent by the buyer on the true intent is: ", total_cost_of_TI)
                print("The number of records bought by the buyer in the published intent is: ", num_records_in_PI)
                print("The number of records bought by the buyer in the true intent is: ", num_records_in_TI)
                proportion_cost_TI = total_cost_of_TI / total_cost_of_PI
                proportion_records_TI = num_records_in_TI / num_records_in_PI
                print("The proportion of cost spent by the buyer on the true intent is: ", proportion_cost_TI)
                print("The proportion of records bought by the buyer in the true intent is: ", proportion_records_TI)
                print("The confidence of the attack is: ", confidence)
                confidence_list.append(confidence)
                total_cost_of_PI_list.append(total_cost_of_PI)
                total_cost_of_TI_list.append(total_cost_of_TI)
                num_records_in_PI_list.append(num_records_in_PI)
                num_records_in_TI_list.append(num_records_in_TI)
                proportion_cost_TI_list.append(proportion_cost_TI)
                proportion_records_TI_list.append(proportion_records_TI)
                print("-----------------------")
            print("The average running time for Genetic Sampling is: ", np.mean(time_list))
            print("The average attack's confidence (Genetic Sampling) is: ", np.mean(confidence_list))
            print("The average total cost spent by the buyer on the published intent is: ",
                  np.mean(total_cost_of_PI_list))
            print("The average total cost spent by the buyer on the true intent is: ", np.mean(total_cost_of_TI_list))
            print("The average number of records bought by the buyer in the published intent is: ",
                  np.mean(num_records_in_PI_list))
            print("The average number of records bought by the buyer in the true intent is: ",
                  np.mean(num_records_in_TI_list))
            print("The average proportion of cost spent by the buyer on the true intent is: ",
                  np.mean(proportion_cost_TI_list))
            print("The average proportion of records bought by the buyer in the true intent is: ",
                  np.mean(proportion_records_TI_list))
            # standard deviation
            print("The standard deviation of running time for Genetic Sampling is: ", np.std(time_list))
            print("The standard deviation of attack's confidence (Genetic Sampling) is: ", np.std(confidence_list))
            print("The standard deviation of total cost spent by the buyer on the published intent is: ",
                  np.std(total_cost_of_PI_list))
            print("The standard deviation of total cost spent by the buyer on the true intent is: ",
                  np.std(total_cost_of_TI_list))
            print("The standard deviation of number of records bought by the buyer in the published intent is: ",
                  np.std(num_records_in_PI_list))
            print("The standard deviation of number of records bought by the buyer in the true intent is: ",
                  np.std(num_records_in_TI_list))
            print("The standard deviation of proportion of cost spent by the buyer on the true intent is: ",
                  np.std(proportion_cost_TI_list))
            print("The standard deviation of proportion of records bought by the buyer in the true intent is: ",
                  np.std(proportion_records_TI_list))

