import logging
import numpy as np
from src.utils import common_funcs
import datetime


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    filename=datetime.datetime.now().strftime(
                        f'logs/%m_%d_%Y__%H_%M__algorithm_applicator.log'),
                    filemode='w',
                    format='%(asctime)s %(levelname)-4s [%(filename)s, %(funcName)20s] \n%(message)s', datefmt='%d-%b-%y %H:%M:%S')


class AlgorithmApplicator:

    """
    An AlgorithmApplicator applies an algorithm to a Setting. It allows the algorithm to receive 
    feedback based on the Setting without having access to the information in the Setting.

    For instance, it allows the PosteriorSamplingAlgorithm to be applied to an incomplete information MDP 
    in the setting that the true optimal policy is the first one (of the IncompleteInformationMDP). The AlgorithmApplicator makes it so that the PosteriorSamplingAlgorithm cannot access this information. 

    Parameters
    ----------
    setting : Setting
        The setting in which the algorithm will be applied.

    incomplete_information_mdp : IncompleteInformationMDP
        The IncompleteInformationMDP that 

    num_of_episodes : int
        Number of episodes for which the algorithm will be implemented
        
    algorithm : IncompleteInformationMDPAlgorithm
        An algorithm designed for IncompleteInformationMDPs
    """

    def __init__(self, setting, incomplete_information_mdp, num_of_episodes, algorithm):

        self.setting = setting
        self.incomplete_information_mdp = incomplete_information_mdp
        self.num_of_episodes = num_of_episodes

        belief_pmf = self.create_initial_belief_pmf()
        self.algorithm = algorithm(incomplete_information_mdp, belief_pmf)

    def create_initial_belief_pmf(self):
        """Return a pmf over the true MDP."""

        return common_funcs.create_uniform_pmf(self.incomplete_information_mdp.num_of_mdps)

    def get_updated_belief_pmf(self, state_trajectory, policy):
        """
        Compute the decision-maker's updated belief after observing the state trajectory after implementing the policy.

        Parameters
        ----------
        state_trajectory : tuple
            The state trajectory observed after implementing the policy.

        policy : numpy.ndarray with shape (num_of_states, horizon + 1)
            The policy that was implemented.

        Returns
        -------
        list[float]
            A pmf over the true MDP. 
        """
        
        unnormalized_belief_pmf = []

        for mdp_index, mdp in enumerate(self.incomplete_information_mdp.mdps):

            unnormalized_belief_pmf.append(
                self.algorithm.belief_pmf[mdp_index] * mdp.get_prob_of_state_trajectory_given_policy(state_trajectory, policy))

        updated_belief_pmf = common_funcs.normalize(unnormalized_belief_pmf)

        logging.debug(f'Rounded updated belief pmf at the end of the episode: {
            list(np.around(np.array(updated_belief_pmf), 2))}')

        return updated_belief_pmf

    def get_decisions_from_algorithm(self) -> list:
        """Return a list of all the algorithm's decisions. 
        
        The i-th element contains the index of the policy chosen by the algorithm in the i-th episode."""

        policy_indices = []

        for episode in range(self.num_of_episodes):

            # Choose a policy
            policy_index = self.algorithm.get_next_policy_index()
            policy = self.incomplete_information_mdp.get_policy(policy_index)

            # Execute the policy
            state_trajectory = self.incomplete_information_mdp.get_random_state_trajectory(
                self.setting.true_mdp, policy)

            # Update the belief over the true optimal policy
            self.algorithm.belief_pmf = self.get_updated_belief_pmf(
                state_trajectory, policy)

            # Record the policy_index
            policy_indices.append(policy_index)

            # Log info
            logging.info(f'EPISODE {episode}; Chosen policy index={
                policy_index}')

        return policy_indices
