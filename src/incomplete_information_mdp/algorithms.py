import random
import logging
import traceback
import itertools
import numpy as np
import pandas as pd
import cvxpy as cp
from abc import ABC, abstractmethod
from src.utils import common_funcs


class IncompleteInformationMDPAlgorithm(ABC):

    '''Abstract class outlining methods needed for all algorithms that will be 
    applied to an IncompleteInformationMDP.'''

    def __init__(self, incomplete_information_mdp, belief_pmf):

        self.incomplete_information_mdp = incomplete_information_mdp
        self.belief_pmf = belief_pmf

    @abstractmethod
    def get_next_policy_index(self):

        pass


class PosteriorSampling(IncompleteInformationMDPAlgorithm):

    '''PosterSampling formalizes the poster sampling algorithm in the space of 
    incomplete information MDPs.'''

    acronym = 'ps'

    def __init__(self, incomplete_information_mdp, belief_pmf):

        super().__init__(incomplete_information_mdp, belief_pmf)

        logging.info('POSTERIOR SAMPLING')

    def __repr__(self):

        return f'ps: {id(self)}'

    @common_funcs.timeit
    def get_next_policy_index(self):

        logging.debug(f'get_next_policy_index(), weights: {self.belief_pmf}')

        return random.choices(range(self.incomplete_information_mdp.num_of_optimal_policies), weights=self.belief_pmf)[0]


class InformationDirectedPolicySampling(IncompleteInformationMDPAlgorithm):

    acronym = 'idps'

    def __init__(self, incomplete_information_mdp, belief_pmf):

        super().__init__(incomplete_information_mdp, belief_pmf)

        logging.info('INFORMATION DIRECTED POLICY SAMPLING')

    def __repr__(self):

        return f'idps: {id(self)}'

    def get_probability_of_state_trajectory_given_policy(self, state_trajectory: tuple, policy: np.ndarray) -> float:
        '''Return the probabilitiy of a state trajectory given a policy. 

        This includes an expectation with respect to the belief.'''

        optimal_policy_index = self.incomplete_information_mdp.get_policy_index(
            policy)

        # Get the probability of the state trajectory given every possible underlying MDP
        probabilities = self.incomplete_information_mdp.get_probabilities_using_state_trajectory(
            optimal_policy_index, state_trajectory)

        # Take the expectation with respect to the belief
        result = np.inner(self.belief_pmf, probabilities)

        return result

    def get_regret_of_policy(self, true_mdp, optimal_policy: np.ndarray, policy: np.ndarray) -> float:
        '''Return the regret of policy when the MDP is known.

        This depends on whether the MDP's objective is to maximize a reward or minimize
        a cost.
        '''

        if self.incomplete_information_mdp.objective == 'maximization':

            regret_of_policy = (true_mdp.get_value_of_policy(
                optimal_policy) - true_mdp.get_value_of_policy(policy))

        elif self.incomplete_information_mdp.objective == 'minimization':

            regret_of_policy = true_mdp.get_value_of_policy(
                policy) - true_mdp.get_value_of_policy(optimal_policy)

        return regret_of_policy

    def get_expected_regret_of_policy(self, policy: np.ndarray) -> float:
        '''Return the expected regret of a policy.'''

        expected_regret_of_policy = 0

        for true_mdp_index, true_mdp in enumerate(self.incomplete_information_mdp.mdps):

            # Get the optimal policy for the true MDP
            optimal_policy = self.incomplete_information_mdp.optimal_policies[true_mdp_index]

            # Get the regret of the policy given the true MDP
            regret_of_policy = self.get_regret_of_policy(
                true_mdp, optimal_policy, policy)

            # Compute the expected regret
            expected_regret_of_policy += self.belief_pmf[true_mdp_index] * (
                regret_of_policy)

        return expected_regret_of_policy

    @common_funcs.timeit
    def get_expected_regret_list(self) -> list:
        '''Return a list containing the expected regret of each potential optimal policy.'''

        expected_regrets = list(map(
            self.get_expected_regret_of_policy, self.incomplete_information_mdp.optimal_policies))
        logging.debug(f'{expected_regrets=}')

        return expected_regrets

    @staticmethod
    def get_computation_for_information_gain(list_a: list, list_b: list) -> float:
        '''Return a term used for calculating the information gain of a policy. 

        The most basic part of the computation has form: a*log(a/b)'''

        vector_a = np.array(list_a, dtype=float)
        vector_b = np.array(list_b, dtype=float)

        vector_c = np.divide(vector_a, vector_b, out=np.zeros_like(
            vector_a), where=vector_b != 0)

        inner_sum_values = vector_a * \
            np.log(vector_c, out=np.zeros_like(
                vector_c, dtype=np.float64), where=(vector_c != 0))

        return inner_sum_values.sum()

    def get_information_gain_of_policy(self, policy: np.ndarray, method = 'fast') -> float:
        '''Return the information gain of a policy
        
        method: str 
            Options: 'fast': Computes the information gain quickly 
                     'basic': Computes the information gain inuitively
        '''

        information_gain_of_policy = 0

        # Get the index of the policy
        policy_index = self.incomplete_information_mdp.get_policy_index(
            policy)

        # Get all relevant state trajectories.
        # These are state trajectories where the policy of observing such a state trajectory given the policy is
        # nonzero for at least one of the potential MDPs.
        cond = (
            self.incomplete_information_mdp.state_trajectory_probabilities_df['optimal_policy_index'] == policy_index)
        df_relevant_to_policy_index = self.incomplete_information_mdp.state_trajectory_probabilities_df[
            cond]
        relevant_state_trajectories = df_relevant_to_policy_index[
            'state_trajectory'].unique().tolist()

        if method == 'fast': 

            # Create temp_df
            num_of_mdps = self.incomplete_information_mdp.num_of_mdps
            temp_df_data = [[[i, j] for i in range(num_of_mdps)]
                            for j in relevant_state_trajectories]
            temp_df_data = list(itertools.chain.from_iterable(temp_df_data))
            temp_df = pd.DataFrame(temp_df_data, columns=[
                                   'mdp_index', 'state_trajectory'])

            # Create subset_df
            subset_df = self.incomplete_information_mdp.state_trajectory_probabilities_df

            optimal_policy_index_cond = (
                self.incomplete_information_mdp.state_trajectory_probabilities_df['optimal_policy_index'] == policy_index)

            state_trajectory_cond = (
                self.incomplete_information_mdp.state_trajectory_probabilities_df['state_trajectory'].isin(relevant_state_trajectories))

            subset_df = self.incomplete_information_mdp.state_trajectory_probabilities_df.loc[
                optimal_policy_index_cond & state_trajectory_cond, ['state_trajectory', 'mdp_index', 'probability']].reset_index(drop=True)

            # Create result_df
            result_df = pd.merge(subset_df, temp_df, on=[
                                 'state_trajectory', 'mdp_index'], how='right').replace(np.nan, 0)

            num_of_relevant_state_trajectories = len(
                relevant_state_trajectories)

            list_b = [np.inner(self.belief_pmf, result_df['probability'].iloc[i * num_of_mdps:(
                i+1) * num_of_mdps]) for i in range(num_of_relevant_state_trajectories)]

        if method == 'basic': 

            list_b = [self.get_probability_of_state_trajectory_given_policy(
                state_trajectory, policy) for state_trajectory in relevant_state_trajectories]

        for true_mdp_index, _ in enumerate(self.incomplete_information_mdp.mdps):

            list_a = self.incomplete_information_mdp.get_probabilities_using_state_trajectories(
                true_mdp_index, policy_index, relevant_state_trajectories)

            inner_sum_value = self.get_computation_for_information_gain(
                list_a, list_b)

            information_gain_of_policy += self.belief_pmf[true_mdp_index] * \
                inner_sum_value

        logging.debug(f'The information gain of the policy is {
            information_gain_of_policy}')

        return information_gain_of_policy

    @common_funcs.timeit
    def get_information_gain_list(self):
        '''Return the information gain of each policy.'''

        information_gains = list(map(
            self.get_information_gain_of_policy, self.incomplete_information_mdp.optimal_policies))
        logging.debug(f'{information_gains=}')

        return information_gains

    @staticmethod
    def get_convex_problem_solution(n, a, b, verbose=False):
        '''Return the solution to the convex problem OR return None, if there 
        was an issue solving the problem.

        Example 
            n = 3
            a = np.asarray([1,2,3])
            b = np.asarray([4,5,6])
            solution = idps.get_convex_problem_solution(n, a, b)

        Example of invalid input
            a = np.asarray([2, 3, 2])
            b = np.asarray([-2, -3, -2])
        '''

        try:

            # Define and solve the convex optimization problem
            x = cp.Variable(n, nonneg=True)
            problem = cp.Problem(cp.Minimize(
                cp.quad_over_lin(a@x, b@x)), [sum(x) == 1])
            problem.solve(solver=cp.ECOS, verbose=verbose)
            solution = x.value

            logging.debug(f'Optimal solution = {solution}; Sum of optimal solution = {
                sum(solution)}; Optimal value = {problem.value}.')

        except Exception as e:

            logging.exception(e)
            logging.debug(traceback.format_exc())
            logging.error('Error in solving the minimization problem.')

            solution = None

        return solution

    def remove_indices_corresponding_to_zero_info_gain(self):
        '''Return
            expected_regrets: The expected regret of each policy, except those with indices in zero_info_gain_policies
                List of length <= num_of_optimal_policies
            information_gains: The information gain of each policy, except those with indices in zero_info_gain_policies
                List of length <= num_of_optimal_policies
            zero_info_gain_policies: The indices of the optimal policies with 0 information gain
        '''

        expected_regrets = self.get_expected_regret_list()
        information_gains = self.get_information_gain_list()

        # Find indices where the information gain is 0
        zero_info_gain_indices = [index for index, info_gain in enumerate(
            information_gains) if info_gain == 0.0]
        num_of_opt_policies_with_zero_info_gain = len(zero_info_gain_indices)
        logging.debug(f'Number of policies with zero information gain: {
                      num_of_opt_policies_with_zero_info_gain}')

        # Case 1: All elements of information_gain are zero
        if num_of_opt_policies_with_zero_info_gain == self.incomplete_information_mdp.num_of_optimal_policies:
            print('ERROR: The information gain of each optimal policy is zero.')
            logging.critical(
                'The information gain of EVERY optimal policy is zero.')

        # Case 2: Some elements of information_gain are zero
        elif num_of_opt_policies_with_zero_info_gain > 0:

            # Remove the elements at these indices
            common_funcs.remove_indices_from_list(
                expected_regrets, zero_info_gain_indices)
            common_funcs.remove_indices_from_list(
                information_gains, zero_info_gain_indices)

        # Case 3: No elements of information_gain are zero
        else:
            pass

        return expected_regrets, information_gains, zero_info_gain_indices

    def get_information_ratio_minimizer(self, expected_regrets, information_gains, rescale=True):
        '''Return the solution to the information ratio minimization problem. 

        There is an option to rescale the expected_regrets and information_gains. 
        '''

        # Convert to numpy arrays
        expected_regrets = np.asarray(expected_regrets)
        information_gains = np.asarray(information_gains)

        if rescale:

            logging.debug('Before rescaling..')
            logging.debug(f'Information gain: {information_gains}')

            information_gains = information_gains * \
                (np.linalg.norm(expected_regrets, 2) /
                 np.linalg.norm(information_gains, 2))

            logging.debug('After rescaling..')
            logging.debug(f'Information gain: {information_gains}')

        else:

            logging.debug(f'Information gain: {information_gains}')

        n = len(expected_regrets)
        solution = self.get_convex_problem_solution(
            n, expected_regrets, information_gains)
        logging.debug(f'Optimal solution to convex problem = {solution}')

        return solution

    def reinclude_indices_corresponding_to_zero_gain(self, solution, zero_info_gain_indices):
        '''Modify the solution by including zeroes for policies where the information gain is zero.'''

        common_funcs.include_zeroes(
            list_to_be_modified=solution, index_list=zero_info_gain_indices)

        if len(solution) != self.incomplete_information_mdp.num_of_optimal_policies:
            logging.error(
                'ERROR: The length of the optimal solution is incorrect.')

        return None

    def get_next_policy_index(self):

        # Get data for the optimization problem
        expected_regrets, information_gains, zero_info_gain_indices = self.remove_indices_corresponding_to_zero_info_gain()

        # Solve the optimization problem
        solution = self.get_information_ratio_minimizer(
            expected_regrets, information_gains)

        # Postprocess the solution
        self.reinclude_indices_corresponding_to_zero_gain(
            solution, zero_info_gain_indices)

        logging.debug(f'get_next_policy_index(), weights: {solution}')

        return random.choices(range(self.incomplete_information_mdp.num_of_optimal_policies), weights=solution)[0]

