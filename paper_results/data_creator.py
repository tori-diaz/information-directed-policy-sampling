"""Script used to generate results for the paper."""

import pickle
import time
import pandas as pd
import multiprocessing
from pathlib import Path
from operator import itemgetter
import src.utils.prep_funcs as prep_funcs
from src.incomplete_information_mdp.setting import Setting
from src.incomplete_information_mdp.algorithm_applicator import AlgorithmApplicator
from src.incomplete_information_mdp.algorithms import PosteriorSampling
from src.incomplete_information_mdp.algorithms import InformationDirectedPolicySampling
from src.incomplete_information_mdp.incomplete_information_mdp import IncompleteInformationMDP


def create_directories(directory_path_list):
    '''Make all the directories in directory_path_list. 

    If a directory already exists, then don't do anything. 
    If other folders in the directory's path doesn't exist, then create them.'''

    for directory_path in directory_path_list:

        directory_path.mkdir(parents=True, exist_ok=True)


def create_and_save_an_incomplete_information_mdp(mdp_name, mdp_kind, mdp_param_values, main_path):
    '''Creates an incomplete_information_mdp.pkl file in the 
    data/mdp_name/mdp_kind directory'''

    # Create the mdps and optimal polices
    mdps = prep_funcs.create_mdps(mdp_name, mdp_kind, mdp_param_values)
    optimal_policies = prep_funcs.create_optimal_policies(mdps)

    # Create the incomplete_information_mdp
    incomplete_information_mdp = IncompleteInformationMDP(
        mdps, optimal_policies, main_path)

    # Save the incomplete_information_mdp
    with open(main_path/'incomplete_information_mdp.pkl', 'wb') as file:
        pickle.dump(incomplete_information_mdp, file)

    # Save information about the incomplete_information_mdp
    print(f'File {main_path/'incomplete_information_mdp.pkl'} was created.')


def apply_algorithms_for_one_iteration(main_path, true_mdp_index_path, iteration_num, num_of_episodes, true_mdp_index):

    # NOTE: mdp_name & mdp_kind are not needed as the incomplete_information_mdp has already been created

    for algorithm in [PosteriorSampling, InformationDirectedPolicySampling]:

        # Load the incomplete_information_mdp
        with open(main_path/'incomplete_information_mdp.pkl', 'rb') as file:
            incomplete_information_mdp = pickle.load(file)

        # Create the setting
        setting = Setting(incomplete_information_mdp, true_mdp_index)

        # Apply the algorithm
        algorithm_applicator = AlgorithmApplicator(
            setting, incomplete_information_mdp, num_of_episodes, algorithm)
        algorithm_decisions = algorithm_applicator.get_decisions_from_algorithm()

        # Determine the rewards & regrets
        rewards = setting.get_rewards(algorithm_decisions)
        regrets = setting.get_regrets(algorithm_decisions)

        # Save the rewards & regrets
        pd.DataFrame(rewards).transpose().to_csv(true_mdp_index_path/f'{
            algorithm.acronym}_rewards__iteration_{iteration_num}.csv', header=False, index=False)
        pd.DataFrame(regrets).transpose().to_csv(true_mdp_index_path/f'{
            algorithm.acronym}_regrets__iteration_{iteration_num}.csv', header=False, index=False)


def apply_algorithms(main_path,
                     true_mdp_index_path,
                     num_of_iterations,
                     num_of_episodes,
                     true_mdp_index,
                     use_multiprocessing=False):

    if use_multiprocessing:

        pool = multiprocessing.Pool(processes=6)

        params = [(main_path,
                   true_mdp_index_path,
                   iteration_num,
                   num_of_episodes,
                   true_mdp_index) for iteration_num in range(num_of_iterations)]

        try:

            pool.starmap(apply_algorithms_for_one_iteration, params)

        except Exception as e:

            print(e)

        pool.close()

    else:

        for iteration_num in range(num_of_iterations):

            try:

                apply_algorithms_for_one_iteration(
                    main_path, true_mdp_index_path, iteration_num, num_of_episodes, true_mdp_index)

            except:

                continue


if __name__ == '__main__':

    import paper_results.scenarios_config as scenarios_config

    root_path = Path().resolve()
    data_path = root_path/'data'

    scenarios = scenarios_config.machine_repair__bernoulli__scenarios + \
        scenarios_config.machine_repair__truncated_geometric__scenarios + \
        scenarios_config.dynamic_pricing__binomial__scenarios + \
        scenarios_config.dynamic_pricing__poisson__scenarios + \
        scenarios_config.queue_control__binomial__scenarios + \
        scenarios_config.queue_control__poisson__scenarios + \
        scenarios_config.queue_control__uncertain_service_probs__scenarios

    start_total_time = time.time()
    start_total_perf_counter = time.perf_counter()

    for scenario in scenarios:

        start_scenario_time = time.time()
        start_scenario_perf_counter = time.perf_counter()

        # Unpack information about the scenario
        mdp_name, mdp_kind, mdp_param_values = itemgetter(
            'mdp_name', 'mdp_kind', 'mdp_param_values')(scenario)
        num_of_episodes, num_of_iterations, true_mdp_index = itemgetter(
            'num_of_episodes', 'num_of_iterations', 'true_mdp_index')(scenario)

        # Create the directory paths & directories that will contain the results
        main_path = data_path/mdp_name/mdp_kind
        true_mdp_index_path = data_path/mdp_name / \
            mdp_kind/f'true_mdp_index_{true_mdp_index}'
        create_directories([main_path, true_mdp_index_path])

        # Document the scenario
        scenario['mdp_param_values'] = str(scenario['mdp_param_values'])
        pd.DataFrame(scenario, index=[0]).transpose().to_csv(
            main_path/'scenario.csv', header=None)

        if (main_path/'incomplete_information_mdp.pkl').is_file():
            print(
                'NOTE: Some results from this scenario already exist. They will be used.')
        else:
            create_and_save_an_incomplete_information_mdp(
                mdp_name, mdp_kind, mdp_param_values, main_path)

        apply_algorithms(main_path, true_mdp_index_path, num_of_iterations,
                         num_of_episodes, true_mdp_index, use_multiprocessing=True)

        f = open(f"{data_path}/computational_times.txt", "a")
        f.write(f"\n\n{mdp_name}, {mdp_kind}, {true_mdp_index=}")
        f.write(f'\nTotal time = {
                (time.time() - start_scenario_time) / 60:,.2f} minutes')
        f.write(f'\nPerf counter = {
            (time.perf_counter() - start_scenario_perf_counter) / 60:,.2f} minutes')
        f.close()

    print(f'Total time = {(time.time() - start_total_time) / 60:,.2f} minutes')
    print(f'Perf counter = {
          (time.perf_counter() - start_total_perf_counter) / 60:,.2f} minutes')
