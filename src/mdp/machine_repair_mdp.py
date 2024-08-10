import numpy as np
import functools

from src.mdp.mdp import MDP
from src.utils import common_funcs


class MachineRepairMDP(MDP):

    """
    MachineRepairMDP is the MDP formulation of the machine repair problem.

    Parameters
    ----------
    num_of_states : int
        Number of possible conditions for the machine.

    horizon : int
        Time horizon for the problem.

    repair_cost : float
        Cost of repairing the machine.

    get_operating_cost : Callable[[int], float]
        Function that returns the cost of operating the machine in a given state.

    get_next_state_probabilities : Callable[[int, int], list[float]]
        Function that returns a pmf over the next state given the current state and action.
    """

    def __init__(self, num_of_states, horizon, repair_cost, get_operating_cost, get_next_state_probabilities):

        self.num_of_states = num_of_states
        self.repair_cost = repair_cost
        self.get_operating_cost = get_operating_cost
        self.get_next_state_probabilities = get_next_state_probabilities

        # Create parameter values for the MDP
        state_space = self._create_state_space()
        action_space = self._create_action_space()
        initial_state_pmf = self._create_initial_state_pmf()
        objective = 'minimization'
        J_terminal = self.create_J_terminal()
        super().__init__(state_space, action_space, horizon, initial_state_pmf,
                         get_next_state_probabilities, objective, J_terminal)

    def __str__(self):

        return 'Information about the underlying MDP: \n' + super().__str__()

    def _create_state_space(self):
        """Return the state space: 0, .., num_of_states. 

        State 0: Machine is in the best condition 
        State num_of_states: Machine is in the worst condition.
        """

        return range(self.num_of_states)

    def _create_action_space(self):
        """Return the action space 

        Action 0: Do not repair the machine 
        Action 1: Do repair the machine
        """

        return [0, 1]

    def _create_initial_state_pmf(self):
        """Return a pmf over the initial state space."""

        return common_funcs.create_uniform_pmf(self.num_of_states)

    def create_J_terminal(self) -> np.ndarray:
        """Return the terminal cost for each state."""

        return np.zeros((self.num_of_states))

    def get_Q_value(self, J, time_stage, state, action):
        """Return the Q value at a given time stage for a (state, action) pair."""

        next_state_pmf = self.get_next_state_probabilities(state, action)

        # Do not repair the machine
        if action == 0:
            Q_value = self.get_operating_cost(
                state) + np.dot(next_state_pmf, J[:, time_stage + 1])

        # Repair the machine
        elif action == 1:
            Q_value = self.repair_cost + \
                self.get_operating_cost(0) + J[0, time_stage + 1]

        return Q_value

    @classmethod
    def bernoulli_example(cls, p=0.7, num_of_states=4, horizon=5, repair_cost=4.5):
        """Return an object instance for the Bernoulli example.

        Use num_of_states >= 3 for interesting cases."""

        kind = 'bernoulli'
        get_operating_cost = cls.get_operating_cost_for_examples
        get_next_state_probabilities = functools.partial(cls.get_next_state_probabilities_for_examples,
                                                         num_of_states=num_of_states,
                                                         p=p,
                                                         kind=kind)

        return cls(num_of_states, horizon, repair_cost, get_operating_cost, get_next_state_probabilities)

    @classmethod
    def truncated_geometric_example(cls, p=0.7, num_of_states=4, horizon=5, repair_cost=8):
        """Return an object instance for the truncated geometric example."""

        kind = 'truncated_geometric'
        get_operating_cost = cls.get_operating_cost_for_examples
        get_next_state_probabilities = functools.partial(cls.get_next_state_probabilities_for_examples,
                                                         num_of_states=num_of_states,
                                                         p=p,
                                                         kind=kind)

        return cls(num_of_states, horizon, repair_cost, get_operating_cost, get_next_state_probabilities)

    @classmethod
    def example(cls, p, kind):
        """Return an object instance for either the Bernoulli or truncated geometric example. 

        kind: str
            Options: 'bernoulli', 'truncated_geometric'
        """

        if kind == 'bernoulli':
            return MachineRepairMDP.bernoulli_example(p)

        elif kind == 'truncated_geometric':
            return MachineRepairMDP.truncated_geometric_example(p)

    @staticmethod
    def get_operating_cost_for_examples(state):
        """Return the cost of operating the machine in a given state given 
        a linear function.

        Note: This is in terms of the state. It is not in terms of the state's index.
        """

        return state

    @staticmethod
    def get_next_state_probabilities_for_examples(state, action, num_of_states, p, kind):
        """Return a pmf over the state space where the i-th element 
        is the probability the next state will be state i. 

        p: float 
            Probability of staying in the same state.

        kind: 'bernoulli', 'truncated_geo'
        """
        # Repair the machine
        if action == 1:

            # The machine moves the best condition
            pmf = [1] + [0] * (num_of_states - 1)

        # Do not repair the machine
        elif action == 0:

            # If the machine is in its worst condition
            if state == num_of_states - 1:

                # The machine will stay in its current condition
                pmf = (num_of_states - 1) * [0] + [1]

            # If the machine is not in the worst condition
            else:

                if kind == 'bernoulli':

                    # The machine may deteriorate further by one state
                    pmf = [0]*state + [p, 1-p] + [0] * \
                        (num_of_states - state - 2)

                elif kind == 'truncated_geometric':

                    # The machine may deteriorate further
                    unnormalized_pmf = [0]*state + \
                        [p**i for i in range(num_of_states-state)]
                    pmf = common_funcs.normalize(unnormalized_pmf)

        return pmf


if __name__ == "__main__":

    ## Examples of how to use MachineRepairMDP

    p = 0.7

    # Create an object instance using a predefined example
    mdp = MachineRepairMDP.example(p, 'bernoulli')
    mdp = MachineRepairMDP.example(p, 'truncated_geometric')

    mdp = MachineRepairMDP.bernoulli_example()
    mdp = MachineRepairMDP.bernoulli_example(p=0.7)

    mdp = MachineRepairMDP.truncated_geometric_example()
    mdp = MachineRepairMDP.truncated_geometric_example(p=0.7)

    # Print out information about the MachineRepairMDP
    mdp_name_pretty = 'MachineRepairMDP'
    mdp.print_mdp_information(mdp_name_pretty)

    # Save some basic information to a file
    mdp.write_basic_mdp_information(
        'machine_repair_mdp_info.txt', ['Machine repair MDP.', 'Truncated geometric case with p = 0.7.', ''])

    # Other examples

    # Get the state at a given index
    state_index = 0
    state = mdp.get_state(state_index)
    print(state)

    # Print the state trajectories
    print(mdp.state_trajectories)

    # Get the probability of each state trajectory under a specific policy
    policy = mdp.get_an_optimal_policy()
    probabilities = mdp.get_all_state_trajectory_probabilities(policy)
    print(probabilities)

    # Get a random state trajectory
    random_state_trajectory = mdp.get_random_state_trajectory(policy)
    print(random_state_trajectory)

    # Get a random state trajectory and its probabilities under a specific policy
    random_state_trajectory, prob = mdp.get_random_state_trajectory_and_its_prob(
        policy)

    state_trajectory = random_state_trajectory
    prob = mdp.get_prob_of_state_trajectory_given_policy(
        state_trajectory, policy)
    print(f'The probability of state trajectory {
        state_trajectory} under policy {policy} is {prob}.')

    state_trajectory_index = mdp.get_state_trajectory_index(
        state_trajectory)
    print(f'The index of state trajectory {
        state_trajectory} is {state_trajectory_index}.')

    state_trajectory = mdp.state_trajectories[state_trajectory_index]
    print(f'The state trajectory at index {
        state_trajectory_index} is {state_trajectory}.')