"""Script used to summarize the results displayed in the paper.

IDPS and PS are applied to a given IncompleteInformationMDP and Setting. There are 50 independent iterations of this task. The result of each iteration is stored in a separate csv file. 
This script combines the results and performs hypothesis testing.


    Common parameters used in this script.
    ----------
    data_path : pathlib.WindowsPath object
        Path to the data folder. 
        Example: Path('c:/Users/../idps/data')

    mdp_name : The name of the MDP example
        Options: 'machine_repair', 'dynamic_pricing', 'queue_control'

    mdp_kind : The specific MDP example
        Options when 'mdp_name' == 'machine_repair': 
            'bernoulli', 'truncated_geometric' 
        Options when 'mdp_name' == 'dynamic_pricing': 
            'binomial', 'poisson' 
        Options when 'mdp_name' == 'queue_control': 
            'poisson', 'binomial', 'finite_set'

    true_mdp_index : int
        The index of the true MDP in the IncompleteInformationMDP. 

    algorithm_acronym : str 
        The acronym of the considered algorithm. 
        Options: 'ps', 'idps'

    num_of_iterations : int
        The number of independent iterations ran by the algorithm.

    num_of_episodes : int
        The number of episodes considered by the algorithm.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from operator import itemgetter


def get_dataframe_with_cumulative_regrets(data_path, mdp_name, mdp_kind, true_mdp_index, algorithm_acronym, num_of_iterations, num_of_episodes):
    """Return a pd.DataFrame with the cumulative regret from all iterations 
    of an algorithm. 

    Parameters
    ----------
    Outlined at the beginning of this script. 

    Returns
    -------
    pd.DataFrame
        Each row corresponds to an independent iteration. 
        Each column corresponds to an episode. 

        Example
            The (i,j)th element contains the sum of the regrets from 
            episodes 1, 2, .., j from iteration i. 
    """

    specific_data_path = data_path/mdp_name / \
        mdp_kind/f'true_mdp_index_{true_mdp_index}'

    # Combine information from all iterations
    rows = []
    for iteration_num in range(num_of_iterations):
        file_name = specific_data_path / \
            f'{algorithm_acronym}_regrets__iteration_{iteration_num}.csv'

        if Path(file_name).is_file():

            row = pd.read_csv(file_name, header=None)

            if row.shape[1] != num_of_episodes:
                print(f'ERROR: There are episodes missing from iteration {iteration_num} for {mdp_name}, {mdp_kind}, and {
                      true_mdp_index=}.\n There are supposed to be {num_of_episodes} episodes, but there are only {row.shape[1]} episodes.')

            # Get the cumulative verison of the data
            row = list(np.cumsum(row.iloc[0]))
            rows.append(row)

        else:
            break

    result = pd.DataFrame(rows)

    # Checks
    if len(result) != num_of_iterations:
        print(f'ERROR: More iterations are needed.')

    return result


def conduct_statistical_test(idps_df, ps_df, alpha=0.05):
    """Conduct a statistical test to determine whether the (mean cumulative regret when IDPS is used) 
    is less than the (mean cumulative regret when PS is used). 

    Parameters
    ----------
    idps_df : pd.DataFrame
        The last column has the cumulative regret of all iterations.

    ps_df : pd.DataFrame
        The last columns has the cumulative regret of all iterations.

    alpha : float, optional
        Significance level for the hypothesis test, by default 0.05

    Returns
    -------
    List 
        A list with the following elements: 
        1. IDPS: Average
        2. IDPS: Standard error of the mean 
        3. PS: Average 
        4. PS: Standard error of the mean 
        5. p-value of the hypothesis test 
        6. Whether the IDPS test statistic is less than the PS test statistic
        7. Whether the results are statistically significant 
    """

    # Get the cumulative regrets at the end of the last episode
    idps_data = idps_df.iloc[:, -1]
    ps_data = ps_df.iloc[:, -1]

    # Calculuate means and standard error of the means
    idps_mean = np.mean(idps_data).round(4)
    idps_standard_error = stats.sem(idps_data).round(4)

    ps_mean = np.mean(ps_data).round(4)
    ps_standard_error = stats.sem(ps_data).round(4)

    # Conduct the statistical test
    p_value = stats.ttest_ind(
        idps_data, ps_data, equal_var=False, alternative='less')[1]

    is_significant = (p_value < alpha)
    is_test_reasonable = (idps_mean < ps_mean)

    return [idps_mean, idps_standard_error, ps_mean, ps_standard_error, p_value, is_test_reasonable, is_significant]


def read_scenario(data_path, mdp_name, mdp_kind):

    scenario_path = data_path/mdp_name/mdp_kind/'scenario.csv'

    # Load the data
    scenario_path = data_path/mdp_name/mdp_kind/'scenario.csv'
    scenario_df = pd.read_csv(scenario_path, header=None)

    # Convert the dataframe to a dictionary
    keys = scenario_df.iloc[:, 0]
    values = scenario_df.iloc[:, 1]
    scenario = dict(zip(keys, values))

    # Remove information that is already known
    del scenario['mdp_name']
    del scenario['mdp_kind']

    # Change the data types
    for k, v in scenario.items():
        scenario[k] = eval(v)

    # Unpack information about the scenario
    mdp_param_values, num_of_episodes = itemgetter(
        'mdp_param_values', 'num_of_episodes')(scenario)
    num_of_iterations, true_mdp_index = itemgetter(
        'num_of_iterations', 'true_mdp_index')(scenario)

    return mdp_param_values, num_of_episodes, num_of_iterations, true_mdp_index


def get_cumulative_regrets_df(data_path, mdp_name, mdp_kind, algorithm_acronym, chosen_true_mdp_index=None):

    mpd_param_values, num_of_episodes, num_of_iterations, true_mdp_index = read_scenario(
        data_path, mdp_name, mdp_kind)

    if not chosen_true_mdp_index is None:

        true_mdp_index = chosen_true_mdp_index

    if num_of_iterations != 50:
        print('ERROR: The number of iterations is not 50. The results in the paper used 50 iterations.')

    df = get_dataframe_with_cumulative_regrets(
        data_path, mdp_name, mdp_kind, true_mdp_index, algorithm_acronym, num_of_iterations, num_of_episodes)

    assert df.shape == (num_of_iterations, num_of_episodes)

    return df


if __name__ == '__main__':

    import paper_results.scenarios_config as scenarios_config

    root_path = Path().resolve()
    data_path = root_path/'data'
    tables_path = data_path/'summarizing_tables'
    tables_path.mkdir(parents=True, exist_ok=True)

    print_results = True
    save_results = True
    only_include_info_for_the_paper = True

    data_to_loop_over = [('machine_repair', 'bernoulli',
                          scenarios_config.machine_repair__bernoulli['mdp_param_values']),
                         ('machine_repair', 'truncated_geometric',
                          scenarios_config.machine_repair__truncated_geometric['mdp_param_values']),
                         ('dynamic_pricing', 'binomial',
                          scenarios_config.dynamic_pricing__binomial['mdp_param_values']),
                         ('dynamic_pricing', 'poisson',
                          scenarios_config.dynamic_pricing__poisson['mdp_param_values']),
                         ('queue_control', 'binomial',
                          scenarios_config.queue_control__binomial['mdp_param_values']),
                         ('queue_control', 'poisson',
                          scenarios_config.queue_control__poisson['mdp_param_values']),
                         ('queue_control', 'finite_set',
                          scenarios_config.queue_control__uncertain_service_probs['mdp_param_values'])]

    for (mdp_name, mdp_kind, mdp_param_values) in data_to_loop_over:

        # Rows for the dataframe
        rows = []

        num_of_true_mdp_indices = len(mdp_param_values)
        for true_mdp_index in range(num_of_true_mdp_indices):

            try:
                # Get the cumulative regret dataframes
                idps_df = get_cumulative_regrets_df(
                    data_path, mdp_name, mdp_kind, algorithm_acronym='idps', chosen_true_mdp_index=true_mdp_index)
                ps_df = get_cumulative_regrets_df(
                    data_path, mdp_name, mdp_kind, algorithm_acronym='ps', chosen_true_mdp_index=true_mdp_index)
                statistical_test_results = conduct_statistical_test(
                    idps_df, ps_df)
            except:
                statistical_test_results = [np.nan] * 7

            # Get the statistical test results
            row = [true_mdp_index] + statistical_test_results
            rows.append(row)

        column_names = ['true_mdp_index',
                        'IDPS mean', 'IDPS standard error',
                        'PS mean', 'PS standard error',
                        'p value',
                        'IDPS mean <= PS mean?',
                        'statistically significant?']

        df = pd.DataFrame(rows,
                          columns=column_names)

        df['True parameter value'] = mdp_param_values

        if only_include_info_for_the_paper:

            df = df.rename(columns={'IDPS mean': 'Average cumulative regret, IDPS',
                                    'PS mean': 'Average cumulative regret, PS',
                                    'statistically significant?': 'Statistically significant?'})

            df = df[['True parameter value',
                    'Average cumulative regret, IDPS',
                     'Average cumulative regret, PS', 'Statistically significant?']]

        if print_results:
            print(f'** {mdp_name}, {mdp_kind} **'.replace('_', ' ').title())
            print(df)

        if save_results:

            file_name = f'{tables_path}/{mdp_name}__{mdp_kind}.csv'
            df.to_csv(
                file_name, index=False)
            print(f'Results have been saved to {file_name}\n')
