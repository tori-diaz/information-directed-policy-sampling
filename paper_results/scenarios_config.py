import copy


# Define the scenarios, except for the true_mdp_index key
machine_repair__bernoulli = {'mdp_name': 'machine_repair',
                             'mdp_kind': 'bernoulli',
                             'mdp_param_values': [0.2, 0.4, 0.6, 0.8],
                             'num_of_episodes': 40,
                             'num_of_iterations': 50}

machine_repair__truncated_geometric = {'mdp_name': 'machine_repair',
                                       'mdp_kind': 'truncated_geometric',
                                       'mdp_param_values': [0.15, 0.20, 0.45, 0.60, 0.75, 0.9],
                                       'num_of_episodes': 80,
                                       'num_of_iterations': 50}

dynamic_pricing__binomial = {'mdp_name': 'dynamic_pricing',
                             'mdp_kind': 'binomial',
                             'mdp_param_values': [(10, 0.10), (9, 0.20), (10, 0.30), (9, 0.40)],
                             'num_of_episodes': 60,
                             'num_of_iterations': 50}

dynamic_pricing__poisson = {'mdp_name': 'dynamic_pricing',
                            'mdp_kind': 'poisson',
                            'mdp_param_values': [(9, 0.15), (10, 0.20), (9, 0.25), (10, 0.30)],
                            'num_of_episodes': 60,
                            'num_of_iterations': 50}

queue_control__poisson = {'mdp_name': 'queue_control',
                          'mdp_kind': 'poisson',
                          'mdp_param_values': [0.2, 0.4, 0.6, 0.8],
                          'num_of_episodes': 60,
                          'num_of_iterations': 50}

queue_control__binomial = {'mdp_name': 'queue_control',
                           'mdp_kind': 'binomial',
                           'mdp_param_values': [(3, 0.25), (3, 0.75), (5, 0.3), (5, 0.9)],
                           'num_of_episodes': 50,
                           'num_of_iterations': 50}

queue_control__uncertain_service_probs = {'mdp_name': 'queue_control',
                                          'mdp_kind': 'finite_set',  # 'uncertain_service_probabilities',
                                          'mdp_param_values': [(0.3, 0.8), (0.2, 0.9), (0.4, 0.8)],
                                          'num_of_episodes': 100,
                                          'num_of_iterations': 50}

# Define the scenarios with the true MDP index


def create_scenarios_for_all_true_mdp_indices(incomplete_scenario):
    """
    Create a list of all complete scenarios. 

    A complete scenario is a dictionary with the following keys: 
        'mdp_name', 'mdp_kind', 'mdp_param_values', 'num_of_episodes', 
        'num_of_iterations', 'true_mdp_index'

    Employs deepcopy. 

    Parameters
    ----------
    incomplete_scenario : dictionary
        Contains the following keys: 'mdp_name', 'mdp_kind', 'mdp_param_values', 'num_of_episodes', 'num_of_iterations'

    Returns
    -------
    list
        A list of all complete scenarios for one specific incomplete scenario.
        Specifically, a list of dictionaries that all share the same (key, value)'s from incomplete_scenario
        and have one additional (key, value) pair: ('true_mdp_index': true_mdp_index). 
        The number of elements is the number of unique mdp_param_values. 
    """

    result = []

    num_of_true_mdp_indices = len(incomplete_scenario['mdp_param_values'])

    for true_mdp_index in range(num_of_true_mdp_indices):

        complete_scenario = copy.deepcopy(incomplete_scenario)
        complete_scenario['true_mdp_index'] = true_mdp_index

        result.append(complete_scenario)

    return result


machine_repair__bernoulli__scenarios = create_scenarios_for_all_true_mdp_indices(
    machine_repair__bernoulli)
machine_repair__truncated_geometric__scenarios = create_scenarios_for_all_true_mdp_indices(
    machine_repair__truncated_geometric)

dynamic_pricing__binomial__scenarios = create_scenarios_for_all_true_mdp_indices(
    dynamic_pricing__binomial)
dynamic_pricing__poisson__scenarios = create_scenarios_for_all_true_mdp_indices(
    dynamic_pricing__poisson)

queue_control__binomial__scenarios = create_scenarios_for_all_true_mdp_indices(
    queue_control__binomial)
queue_control__poisson__scenarios = create_scenarios_for_all_true_mdp_indices(
    queue_control__poisson)
queue_control__uncertain_service_probs__scenarios = create_scenarios_for_all_true_mdp_indices(
    queue_control__uncertain_service_probs)
