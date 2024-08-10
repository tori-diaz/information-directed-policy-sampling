import logging
import datetime
import pandas as pd
import numpy as np
from src.mdp.mdp import MDP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    filename=datetime.datetime.now().strftime(
                        'logs/%m_%d_%Y__%H_%M__incomplete_information_mdp.log'),
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(funcName)20s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class IncompleteInformationMDP(MDP):
        
    """
    Parameters
    ----------
    mdps : list
        List of MDP objects.

    optimal_policies : list
        List of optimal policies.

    dir_for_data : pathlib.WindowsPath
        Path used to save state_trajectory_probabilities_df.csv

    tolerance : float, optional
        Probabilities less than the tolerance are set to 0, by default 1e-6
    """
    
    def __init__(self, mdps, optimal_policies, dir_for_data, tolerance=1e-6):

        self.mdps = mdps
        self.optimal_policies = optimal_policies
        self.dir_for_data = dir_for_data
        self.tolerance = tolerance

        self.num_of_optimal_policies = len(optimal_policies)
        self.num_of_mdps = len(mdps)

        first_mdp = mdps[0]
        super().__init__(first_mdp.state_space,
                         first_mdp.action_space,
                         first_mdp.horizon,
                         first_mdp.initial_state_pmf,
                         first_mdp.get_next_state_probabilities,
                         first_mdp.objective,
                         first_mdp.J_terminal)

        self.value_matrix = self.create_value_matrix()
        self.state_trajectory_probabilities_df = self.create_state_trajectory_probabilities_df()
        self.save_state_trajectory_probabilities_df()  

    def __str__(self):

        info = f'>> Incomplete Information MDP <<' + \
            f'\nNumber of optimal policies: {self.num_of_optimal_policies}' + \
            f'\nNumber of MDPs: {self.num_of_mdps}' + \
            f'\nValue matrix: {self.value_matrix}' +\
            f"\nMDP's objective: {self.objective}"

        return info

    def create_value_matrix(self):
        """Return the value matrix.

        Specifically, 
            value_matrix[i,j] 
                = the value of executing the (policy which is optimal to the i-th MDP) in the j-th MDP"""

        value_matrix = np.zeros((len(self.optimal_policies), len(self.mdps)))

        for mdp_index, mdp in enumerate(self.mdps):

            for optimal_policy_index, optimal_policy in enumerate(self.optimal_policies):

                value_matrix[optimal_policy_index,
                             mdp_index] = mdp.get_value_of_policy(optimal_policy)

        return value_matrix

    def get_all_possible_state_trajectories(self):

        # Create a list of state trajectories from all MDPs.
        # There will be a significant number of duplicates.
        all_possible_state_trajectories = []

        for mdp in self.mdps:

            all_possible_state_trajectories += mdp.state_trajectories

        # Remove all the duplicates before returning.
        return list(set(all_possible_state_trajectories))

    def save_state_trajectory_probabilities_df(self):
        """Save a csv with all state trajectory probabilities."""

        self.state_trajectory_probabilities_df.to_csv(
            self.dir_for_data/'state_trajectory_probabilities_df.csv')

    def create_state_trajectory_probabilities_df(self, print_info=True):
        '''Return the state_trajectory_probabilities_df. 

        Probabilities that are less than the tolerance level are excluded. 
        '''

        # Get all possible state trajectories.
        state_trajectories = self.get_all_possible_state_trajectories()

        # Get an initial version of state_trajectory_probabilities_df.
        # This DataFrame can include probabilities that are less than the tolerance.
        state_trajectory_probabilities_df = self.create_state_trajectory_probabilities_df_aux(
            state_trajectories)

        logging.debug('state_trajectory_probabilities_df has been created')

        if print_info:

            # Record information to print later
            original_num_of_state_trajectories = len(state_trajectories)
            original_num_of_probabilities = len(
                state_trajectory_probabilities_df)
            small_prob_cond = (
                state_trajectory_probabilities_df['probability'] <= self.tolerance)
            original_num_of_small_probabilities = len(
                state_trajectory_probabilities_df[small_prob_cond])

        # Only keep the entries where the probability is at least as large as the tolerance
        cond = (
            state_trajectory_probabilities_df['probability'] >= self.tolerance)
        state_trajectory_probabilities_df = state_trajectory_probabilities_df[cond]

        if print_info:

            # Record information to print later
            final_num_of_state_trajectories = len(state_trajectories)
            final_num_of_probabilities = len(
                state_trajectory_probabilities_df)
            small_prob_cond = (
                state_trajectory_probabilities_df['probability'] <= self.tolerance)
            final_num_of_small_probabilities = len(
                state_trajectory_probabilities_df[small_prob_cond])

            num_of_state_trajectories_removed = original_num_of_state_trajectories - \
                final_num_of_state_trajectories
            num_of_probabilities_removed = original_num_of_probabilities - \
                final_num_of_probabilities
            num_of_small_probabilities_removed = original_num_of_small_probabilities - \
                final_num_of_small_probabilities

            info = '\n\nBefore incorporating the tolerance, we have: ' +\
                f'\nNumber of state trajectories: {original_num_of_state_trajectories:,}' +\
                f"\nNumber of probabilities: {original_num_of_probabilities:,}" +\
                f"\nNumber of probabilities that are less than or equal to the tolerance ({
                    self.tolerance}): {original_num_of_small_probabilities:,}"

            info += '\n\nWhen applying the tolerance, the following were removed:' +\
                f'\nNumber of state trajectories: {num_of_state_trajectories_removed:,}' +\
                f"\nNumber of probabilities: {num_of_probabilities_removed:,}" +\
                f"\nNumber of probabilities that are less than or equal to the tolerance ({
                    self.tolerance}): {num_of_small_probabilities_removed:,}"

            info += '\n\nAfter incorporating the tolerance, we have: ' +\
                f'\nNumber of state trajectories: {final_num_of_state_trajectories:,}' +\
                f"\nNumber of probabilities: {final_num_of_probabilities:,}" +\
                f"\nNumber of probabilities that are less than or equal to the tolerance ({
                    self.tolerance}): {final_num_of_small_probabilities:,}\n\n"
            print(info)

            logging.info(info)

        state_trajectory_probabilities_df = state_trajectory_probabilities_df.reset_index()

        return state_trajectory_probabilities_df

    def create_state_trajectory_probabilities_df_aux(self, state_trajectories):
        '''Return state_trajectory_probabilities_df. 

        There might be probabilites that are less than the tolerance level.
        '''

        logging.debug(f'num of optimal_policies: {
                      self.num_of_optimal_policies}')

        logging.debug(f'num of mdps: {self.num_of_mdps}')
        logging.debug(f'num of state trajectories: {len(state_trajectories)}')

        mdp_indices = []
        optimal_policy_indices = []
        state_trajectory_list = []
        probabilities = []

        for mdp_index, mdp in enumerate(self.mdps):

            for optimal_policy_index, optimal_policy in enumerate(self.optimal_policies):

                for state_trajectory in state_trajectories:

                    mdp_indices.append(mdp_index)
                    optimal_policy_indices.append(optimal_policy_index)
                    state_trajectory_list.append(state_trajectory)

                    probabilities.append(mdp.get_prob_of_state_trajectory_given_policy(
                        state_trajectory, optimal_policy))

        state_trajectory_probabilities_df = pd.DataFrame({'mdp_index': mdp_indices, 'optimal_policy_index': optimal_policy_indices,
                                                         'state_trajectory': state_trajectory_list, 'probability': probabilities})
        logging.debug(f'size of state_trajectory_probabilities_df: {
                      state_trajectory_probabilities_df.shape}')

        return state_trajectory_probabilities_df

    def get_random_state_trajectory(self, mdp, policy):
        """Return a random state trajectory due to implementing the policy in the MDP.

        Parameters
        ----------
        mdp : MDP
            MDP in which to implement the policy
        policy : np.ndarray
            Policy to implement

        Returns
        -------
        tuple
            Random state trajectory
        """

        return mdp.get_random_state_trajectory(policy)

    def get_probabilities_using_state_trajectory(self, optimal_policy_index, state_trajectory):
        ''' Returns a list of probabilities. 

        probabilities[i] = probability of observing the state trajectory when the policy located at
                           optimal_policy_index is used and the underlying MDP is self.mdps[i]

        The new_method is better practice. However, old_method some times performs slightly faster 
        since the max number of rows appended is 3. 
        '''

        mdp_indices = [*range(self.num_of_mdps)]

        optimal_policy_index_cond = (
            self.state_trajectory_probabilities_df['optimal_policy_index'] == optimal_policy_index)

        state_trajectory_cond = (
            self.state_trajectory_probabilities_df['state_trajectory'] == state_trajectory)

        temp_df = self.state_trajectory_probabilities_df.loc[
            optimal_policy_index_cond & state_trajectory_cond, ['mdp_index', 'probability']].reset_index(drop=True)

        # fill in missing values
        missing_mdp_indices = set(mdp_indices) - \
            set(temp_df['mdp_index'])

        for missing_mdp_index in [*missing_mdp_indices]:

            temp_df.loc[len(temp_df)] = [missing_mdp_index, 0]

        # sort it properly
        sorted_temp_df = temp_df.sort_values(
            by='mdp_index')

        probabilities = sorted_temp_df['probability'].tolist()

        return probabilities

    def get_probabilities_using_state_trajectories(self, mdp_index, optimal_policy_index, relevant_state_trajectories):
        '''Return a list of probabilities. 

        For every state trajectory in relevant state trajectories, 
            calculate the probability of the state trajectory given the specified MDP and policy. 

        Probabilities are returned in the same order as relevant_state_tracjectories.
        '''

        mdp_index_cond = (
            self.state_trajectory_probabilities_df['mdp_index'] == mdp_index)
        optimal_policy_index_cond = (
            self.state_trajectory_probabilities_df['optimal_policy_index'] == optimal_policy_index)

        temp_df = self.state_trajectory_probabilities_df.loc[mdp_index_cond & optimal_policy_index_cond, [
            'state_trajectory', 'probability']]

        # Only data related to the state trajectories in relevant_state_trajectories is relevant
        temp_df = temp_df[temp_df['state_trajectory'].isin(
            relevant_state_trajectories)]

        # There might be some state trajectories that are in relevant_state_trajectories that have
        # 0 probability and thus are not in temp_df.
        # These must be included back in.
        unaccounted_state_trajectories = set(
            relevant_state_trajectories) - set(temp_df['state_trajectory'].tolist())

        unaccounted_state_trajectories_df = pd.DataFrame({'state_trajectory': list(unaccounted_state_trajectories),
                                                          'probability': [
            0] * len(unaccounted_state_trajectories)})

        temp_df = pd.concat([temp_df,
                            unaccounted_state_trajectories_df], axis=0)

        # Reorder temp_df so that it has the order of relevant_state_trajectories.
        # This is important because the result will be used in an inner product.
        temp_df = temp_df.set_index('state_trajectory')
        temp_df = temp_df.reindex(relevant_state_trajectories)
        probabilities = temp_df['probability'].tolist()

        if len(probabilities) != len(relevant_state_trajectories):
            print('ERROR: The number of probabilities returned is incorrect.')

        return probabilities

    def get_policy(self, policy_index):
        """Return the policy at the given index.

        Parameters
        ----------
        policy_index : int
            Index of the policy to return.

        Returns
        -------
        np.ndarray
            The policy at index policy_index.
        """

        return self.optimal_policies[policy_index]

    def get_policy_index(self, policy):
        """Return the index of the first policy that matches the given policy.

        Returns
        -------
        int
            The index of policy.
        """

        return [np.array_equal(policy, self.optimal_policies[i]) for i in range(len(self.optimal_policies))].index(True)

    def get_mdp(self, mdp_index):
        """
        Return the MDP at the given index

        Parameters
        ----------
        mdp_index : int
            Index of the MDP to return.

        Returns
        -------
        MDP
            MDP at the given index.
        """

        return self.mdps[mdp_index]

    def get_mdp_index(self, mdp):
        """
        Return the index of the MDP.

        Parameters
        ----------
        mdp : MDP
            MDP whose index will be returned.

        Returns
        -------
        int
            Index of MDP.
        """

        return self.mdps.index(mdp)

    def get_value_matrix_information(self, decimals=2):
        """
        Print information about the IncompleteInformationMDP.

        For each MDP, print the
        - number of unqiue policy values
        - minimum policy value
        - average policy values
        - maximum policy value
        - range of policy values

        Parameters
        ----------
        decimals : int, optional
            Number of decimals to display, by default 2

        Returns
        -------
        pd.DataFrame
            DataFrame with information about the IncompleteInformationMDP
        """

        num_of_columns = self.num_of_mdps

        max_values = []
        min_values = []
        avg_values = []
        nunique_values = []

        for column_index in range(num_of_columns):

            column = self.value_matrix[:, column_index]

            max_values.append(max(column))
            min_values.append(min(column))
            avg_values.append(np.mean(column))
            nunique_values.append(len(np.unique(column)))

        df = pd.DataFrame({'Index of the true MDP': range(num_of_columns),
                           'Num of unique values': nunique_values,
                          'Min policy value': min_values,
                           'Avg policy value': avg_values,
                           'Max policy value': max_values})

        df['Range of policy values'] = df['Max policy value'] - \
            df['Min policy value']

        return df.round(decimals)
