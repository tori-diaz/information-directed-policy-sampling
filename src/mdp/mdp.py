import numpy as np
import random
import itertools


class MDP:

    """
    A model of a Markov Decision Process (MDP).

    Parameters
    ----------
    state_space : list
        The state space.

    action_space : list
        The action space.

    horizon : int
        The number of time-stages in the MDP.

    initial_state_pmf : list[float]
        A probability mass function (pmf) over the initial state.

    get_next_state_probabilities : Function of the form (state, action) -> pmf over the next state
        A function which returns a pmf over the next state given the current state and chosen action.

    objective : str
        'minimization', if the objective is to minimize cost
        'maximization', if the objective is to maximize reward

    J_terminal : numpy.ndarray with shape (number of states,)
        The terminal rewards (or costs), depending on whether the objective is minimization or maximization.

    Attributes
    ----------
    num_of_states: int
        The number of states.

    num_of_actions: int
        The number of actions.

    state_trajectories: list[tuple]
        All possible state trajectories for the MDP.

    num_of_state_trajectories: int
        The number of all possible state trajectories for the MDP.
    """

    def __init__(self, state_space, action_space, horizon, initial_state_pmf, get_next_state_probabilities, objective, J_terminal):

        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon
        self.initial_state_pmf = initial_state_pmf
        self.get_next_state_probabilities = get_next_state_probabilities
        self.objective = objective
        self.J_terminal = J_terminal

        # Create the attributes
        self.num_of_states = len(state_space)
        self.num_of_actions = len(action_space)
        self.state_trajectories = self.create_state_trajectories()
        self.num_of_state_trajectories = len(self.state_trajectories)

    def __str__(self):

        info_str = f'State space = {self.state_space}' + \
            f'\nNumber of states = {self.num_of_states}' + \
            f'\nAction space = {self.action_space}' + \
            f'\nNumber of actions = {self.num_of_actions}' + \
            f'\nHorizon = {self.horizon}' + \
            f'\nInitial state pmf = {self.initial_state_pmf}' + \
            f'\nNumber of possible state trajectories = {
                self.num_of_state_trajectories}'

        return info_str

    def get_an_optimal_policy(self):
        """
        Compute an optimal policy for the MDP.

        Returns
        -------
        numpy.ndarray with shape (num_of_states, horizon + 1)
            Specifically, optimal_policy[state_index, time_stage] = optimal action.
        """

        optimal_policy = np.zeros((self.num_of_states, self.horizon + 1))
        Jstar = np.zeros((self.num_of_states, self.horizon + 1))

        # Incorporate the terminal values
        Jstar[:, self.horizon] = self.J_terminal

        # Fill in the optimal policy with actions
        for time_stage in range(self.horizon - 1, -1, -1):

            for state_index, state in enumerate(self.state_space):

                Q = [self.get_Q_value(Jstar, time_stage, state, action)
                     for action in self.action_space]

                if self.objective == 'minimization':

                    Jstar[state_index, time_stage] = min(Q)
                    action_index = Q.index(min(Q))
                    optimal_policy[state_index,
                                   time_stage] = self.get_action(action_index)

                elif self.objective == 'maximization':

                    Jstar[state_index, time_stage] = max(Q)
                    action_index = Q.index(max(Q))
                    optimal_policy[state_index,
                                   time_stage] = self.get_action(action_index)

        # Actions in the optimal policy must have the same type as actions in the action space
        action_type = type(self.action_space[0])
        optimal_policy = optimal_policy.astype(action_type)

        return optimal_policy

    def get_value_of_policy(self, policy):
        """
        Compute the value of the policy.

        Parameters
        ----------
        policy : numpy.ndarray with shape (num_of_states, horizon + 1)
            A policy for the MDP.

        Returns
        -------
        float
            The value of the policy.
        """

        J = np.zeros((self.num_of_states, self.horizon + 1))

        # Incorporate the terminal values.
        J[:, self.horizon] = self.J_terminal

        # Step 1: Determine the J matrix
        for time_stage in range(self.horizon - 1, -1, -1):

            for state_index, state in enumerate(self.state_space):

                # Get the action from the policy
                action = policy[state_index, time_stage]

                J[state_index, time_stage] = self.get_Q_value(
                    J, time_stage, state, action)

        # Step 2: Take the dot product to find the value of the policy
        value_of_policy = np.dot(J[:, 0], self.initial_state_pmf)

        return value_of_policy

    def get_state(self, state_index):
        """
        Get the state at state_index.

        Parameters
        ----------
        state_index : int
            Index of the desired state in the state space.

        Returns
        -------
        dtype of the MDP's states
            The state at index state_index in the state space.
        """

        return self.state_space[state_index]

    def get_state_index(self, state):
        """
        Get the index of the given state in the state space.

        Parameters
        ----------
        state : dtype of the MDP's states
            The state whose index in state_space will be obtained.

        Returns
        -------
        int
            The index of the state in the state space.
        """

        return self.state_space.index(state)

    def get_states(self, state_indices) -> list:
        """
        Create a list of states that correspond to the list of the states' indices in the state space. 

        Parameters
        ----------
        state_indices : list[int]
            A list containing the indices of states in the state space.

        Returns
        -------
        list
            A list of states at the indices in state_indices.
        """

        states = [self.get_state(state_index) for state_index in state_indices]

        return states

    def get_action(self, action_index):
        """Return the action."""

        return self.action_space[action_index]

    def get_action_index(self, action):
        """Return the index of the action in the action space."""

        return self.action_space.index(action)

    def get_next_possible_states_aux(self, state) -> list:
        """Return a list of unique, sorted states that have a nonzero probability of becoming the next state.

        This is an auxiliary method for include_next_state_aux().
        """

        next_possible_states = []

        # For every action, get the next possible states
        for action in self.action_space:

            next_state_pmf = self.get_next_state_probabilities(state, action)

            indices_with_positive_probs = np.flatnonzero(
                np.array(next_state_pmf) > 0).tolist()

            next_possible_states += self.get_states(
                indices_with_positive_probs)

        # Make sure there are no duplicated states & then sort them
        next_possible_states = list(set(next_possible_states))
        next_possible_states.sort()

        return next_possible_states

    def include_next_state_aux(self, partial_state_trajectory) -> list:
        """Given a list of partial state trajectories, append a possible next state to each trajectory.
        Return the result.

        This is an auxiliary method for create_state_trajectories()."""

        latest_state = partial_state_trajectory[-1]

        next_possible_states = self.get_next_possible_states_aux(latest_state)

        return [partial_state_trajectory + [next_state] for next_state in next_possible_states]

    def create_state_trajectories(self) -> list:
        """
        Create a list of possible state trajectories. 

        Returns
        -------
        list[tuple]
            A list of state trajectories possible to the MDP.
        """

        # Initialize the trajectories with the initial state
        partial_state_trajectories = [[state] for state in self.state_space]

        for time_stage in range(self.horizon):

            nested_partial_state_trajectories = list(
                map(self.include_next_state_aux, partial_state_trajectories))
            partial_state_trajectories = list(
                itertools.chain(*nested_partial_state_trajectories))

        state_trajectories = partial_state_trajectories

        return [tuple(state_trajectory) for state_trajectory in state_trajectories]

    def get_all_state_trajectory_probabilities(self, policy) -> list:
        """Return a list with the probabilities of each state trajectory given the policy.

        The i-th element will have the probability of the i-th state trajectory
        when the policy is executed.
        Policies is in terms of actions. It does not contain have action indices.
        """

        all_state_trajectory_probabilities = []

        for state_trajectory in self.state_trajectories:

            probability = self.get_prob_of_state_trajectory_given_policy(
                state_trajectory, policy)

            all_state_trajectory_probabilities.append(probability)

        return all_state_trajectory_probabilities

    def get_random_state_trajectory_and_its_prob(self, policy) -> list:
        """Return a list with a random state trajectory and its probability given the policy."""

        # Get the probability of each state trajectory, given the policy
        all_state_trajectory_probabilities = self.get_all_state_trajectory_probabilities(
            policy)

        state_trajectory = random.choices(
            self.state_trajectories, weights=all_state_trajectory_probabilities)[0]
        index_of_state_trajectory = self.state_trajectories.index(
            state_trajectory)
        prob = all_state_trajectory_probabilities[index_of_state_trajectory]

        return [state_trajectory, prob]

    def get_random_state_trajectory(self, policy) -> list:
        """Return a randon state trajectory given the policy."""

        random_state_trajectory, _ = self.get_random_state_trajectory_and_its_prob(
            policy)

        return random_state_trajectory

    def get_prob_of_state_trajectory_given_policy(self, state_trajectory, policy) -> float:
        """Return the probability of a state trajectory given a policy."""

        probability = 1

        for time_stage, state, next_state in zip(range(self.horizon + 1), state_trajectory, state_trajectory[1:]):

            state_index = self.get_state_index(state)
            next_state_index = self.get_state_index(next_state)

            action = policy[state_index, time_stage]

            next_state_pmf = self.get_next_state_probabilities(
                state, action)

            probability *= next_state_pmf[next_state_index]

            if probability == 0.0:
                break

        return probability

    def get_state_trajectory_index(self, state_trajectory):
        """Return the index of the state trajectory."""

        return self.state_trajectories.index(state_trajectory)

    def print_mdp_information(self, mdp_name_pretty):

        print(f'\n> MDP: {mdp_name_pretty}')

        print(f'\n> {str(self)}')

        # Get an optimal policy & its value
        optimal_policy = self.get_an_optimal_policy()
        value_of_optimal_policy = self.get_value_of_policy(optimal_policy)
        print(f'\n> The value of the optimal policy is {
            value_of_optimal_policy:.2f} and the optimal policy is \n{optimal_policy}')

        # Get the indices of the next possible state
        state = 0
        next_possible_states = self.get_next_possible_states_aux(state)
        print(f'\n> When state = {state}, the next possible states are:', ', '.join(
            map(str, next_possible_states)) + '.')

        # Create all possible state trajectories
        print(f'\n> Possible state trajectories: {
            self.state_trajectories[0:4]}, ..')

        # Get the probabilities of each state trajectory, given a policy
        all_state_trajectory_probabilities = self.get_all_state_trajectory_probabilities(
            optimal_policy)
        print(f'\n> The probabilities of the aforementioned state trajectories:',
              ', '.join(map(str, all_state_trajectory_probabilities[0:4])) + '.')

        # Get a random state trajectory, given a policy
        print(f'\n> Example of a random state trajectory given the optimal policy: {
            self.get_random_state_trajectory(optimal_policy)}' + '.')

    def write_basic_mdp_information(self, file_name, initial_lines=['']):

        info = []

        if initial_lines != ['']:
            for line in initial_lines:

                info.append(line)
        info.append(f'State space: {self.state_space}')
        info.append(f'Number of states: {self.num_of_states}')
        info.append(f'Action state: {self.action_space}')
        info.append(f'Number of actions: {self.num_of_actions}')
        info.append(f'Horizon: {self.horizon}')
        info.append(f'Initial state pmf: {self.initial_state_pmf}')
        info.append(f'Number of state trajectories: {
                    self.num_of_state_trajectories}')

        # Get an optimal policy & its value
        optimal_policy = self.get_an_optimal_policy()
        value_of_optimal_policy = self.get_value_of_policy(optimal_policy)
        info.append(f'The value of the optimal policy is {
            value_of_optimal_policy:.2f}')
        info.append(f'An optimal policy: \n{optimal_policy}')

        with open(file_name, 'w') as f:
            for line in info:
                f.write(f"{line}\n")

        return info