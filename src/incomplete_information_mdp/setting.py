import numpy as np


class Setting:

    """
    When an algorithm is applied to an Incomplete Information MDP, the chosen policies produce state trajectories determined by the true underlying MDP. 

    A Setting represents the situation under which algorithms will be applied. Thus, knowing the Setting is equivalent to knowing the true underlying MDP. 

    The purpose of defining a Setting is so that algorithms cannot access information about the Setting. For if an algorithm had access to attributes in the Setting, then it would unrealistically be privy to certain information.

    Parameters
    ----------
    incomplete_information_mdp : IncompleteInformationMDP
        The Incomplete Information MDP which an algorithm will be applied to.

    true_mdp_index : int
        The index of the true underlying MDP in the IncompleteInformationMDP.

    Attributes 
    ----------
    true_mdp : MDP 
        The true underlying MDP

    true_optimal_policy_index : int 
        The index of the optimal policy for the true underlying MDP

    value_of_optimal_policy_in_true_mdp : float
        The value of the optimal policy for the MDP at index true_mdp_index evaluated in the MDP at index true_mdp_index. 
    """

    def __init__(self,
                 incomplete_information_mdp,
                 true_mdp_index):

        self.incomplete_information_mdp = incomplete_information_mdp
        self.true_mdp_index = true_mdp_index

        self.true_mdp = incomplete_information_mdp.mdps[true_mdp_index]
        self.true_optimal_policy_index = true_mdp_index
        self.value_of_optimal_policy_in_true_mdp = self.incomplete_information_mdp.value_matrix[
            self.true_optimal_policy_index, self.true_mdp_index]

    def get_rewards(self, decisions_from_algorithm: list) -> list:
        '''Return a list of rewards corresponding to the given list of decisions from an algorithm.'''

        return [self.incomplete_information_mdp.value_matrix[policy_index, self.true_mdp_index] for policy_index in decisions_from_algorithm]

    def get_regrets(self, decisions_from_algorithm: list) -> list:
        '''Return a list of regrets corresponding to the given list of decisions from an algorithm.'''

        rewards = self.get_rewards(decisions_from_algorithm)

        if self.incomplete_information_mdp.objective == 'minimization':

            return [reward - self.value_of_optimal_policy_in_true_mdp for reward in rewards]

        elif self.incomplete_information_mdp.objective == 'maximization':

            return [self.value_of_optimal_policy_in_true_mdp - reward for reward in rewards]

    def get_cumulative_regrets(self, decisions_from_algorithm: list) -> list:
        '''Return a list of cumulative regrets corresponding to the given list of decisions from an algorithm. 

        The i-th element is: (regret from the first decision) + .. + (regret from the (i+1)-th decision).'''

        regrets = self.get_regrets(decisions_from_algorithm)

        return list(np.cumsum(regrets))

    def get_cumulative_rewards(self, decisions_from_algorithm: list) -> list:
        '''Return a list of cumulative rewwards corresponding to the given list of decisions from an algorithm. 

        The i-th element is: (reward from the first decision) + .. + (reward from the (i+1)-th decision).'''

        rewards = self.get_rewards(decisions_from_algorithm)

        return list(np.cumsum(rewards))
