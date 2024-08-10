import numpy as np
import functools
from scipy.stats import binom, poisson

from src.mdp.mdp import MDP
from src.utils import common_funcs


class DynamicPricingMDP(MDP):

    '''DynamicPricingMDP is the MDP formulation of the dynamic pricing problem.

    State represents the current inventory.
    '''

    def __init__(self, holding_cost, unmet_demand_cost, prices,
                 max_inventory,
                 get_probability_of_demand_given_price,
                 get_next_state_probabilities, horizon,
                 demand_type_and_params=(None, None)):

        self.holding_cost = holding_cost
        self.unmet_demand_cost = unmet_demand_cost
        self.prices = prices
        self.max_inventory = max_inventory
        self.get_probability_of_demand_given_price = get_probability_of_demand_given_price
        self.get_next_state_probabilities = get_next_state_probabilities

        # Create parameter values for the MDP
        state_space = self._create_state_space()
        action_space = self._create_action_space()
        initial_state_pmf = self._create_initial_state_pmf()

        # Required by self.create_J_terminal()
        self.num_of_states = len(state_space)
        objective = 'maximization'
        J_terminal = self.create_J_terminal()
        super().__init__(state_space, action_space, horizon, initial_state_pmf,
                         get_next_state_probabilities, objective, J_terminal)

        self.demand_type, self.demand_params = demand_type_and_params
        # demand_type, demand_params options:
        #   ('binomial', (n, alpha))
        #   ('poisson', (n, gamma))

    def __str__(self):

        return 'Information about the underlying MDP: \n' + super().__str__()

    def _create_state_space(self):

        return range(self.max_inventory + 1)

    def _create_action_space(self):

        return self.prices

    def _create_initial_state_pmf(self):
        """Return a pmf over the initial state space."""

        return common_funcs.create_uniform_pmf(self.max_inventory + 1)

    def create_J_terminal(self):

        return np.zeros((self.num_of_states))

    @staticmethod
    def get_num_sold(current_inventory, demand):

        return min(current_inventory, demand)

    @staticmethod
    def get_num_leftover(current_inventory, demand):

        return current_inventory - DynamicPricingMDP.get_num_sold(current_inventory, demand)

    @staticmethod
    def get_surplus_demand(current_inventory, demand):

        return demand - DynamicPricingMDP.get_num_sold(current_inventory, demand)

    def get_Q_value(self, J, time_stage, state, action):
        """Return the Q value at a given time stage for a (state, action) pair."""

        # Use the terminology for this setting
        current_inventory = state
        price = action

        Q_value = 0

        if self.demand_type == 'binomial':

            # NOTE max_demand > max_inventory for all numerical results
            max_demand = self.demand_params[0]

            # Situation 1: Buyer's demand is satisfied, ie, demand <= current_inventory
            for demand in range(current_inventory + 1):

                probability = self.get_probability_of_demand_given_price(
                    demand, price)
                num_sold = self.get_num_sold(current_inventory, demand)
                num_leftover = self.get_num_leftover(current_inventory, demand)

                Q_value += probability * \
                    ((price * num_sold) - self.holding_cost *
                     (num_leftover) + J[num_leftover, time_stage + 1])

            # Situation 2: Buyer's demand is not satisfied, ie, demand > current_inventory
            for demand in range(current_inventory + 1, max_demand + 1):

                probability = self.get_probability_of_demand_given_price(
                    demand, price)
                num_sold = self.get_num_sold(current_inventory, demand)
                num_surplus_demand = self.get_surplus_demand(
                    current_inventory, demand)

                Q_value += probability * \
                    (price * num_sold - self.unmet_demand_cost *
                     num_surplus_demand + J[0, time_stage + 1])

        elif self.demand_type == 'poisson':

            gamma, alpha = self.demand_params
            poisson_mean = gamma * np.exp(-1 * alpha * price)

            for demand in range(0, current_inventory):

                probability = self.get_probability_of_demand_given_price(
                    demand, price)

                Q_value += probability * (demand * (price + self.holding_cost + self.unmet_demand_cost) -
                                          self.holding_cost * current_inventory + J[current_inventory - demand, time_stage + 1])

            Q_value += (1 - poisson.cdf(current_inventory-1, poisson_mean)) * (current_inventory *
                                                                               (price + self.unmet_demand_cost) + J[0, time_stage + 1])

            Q_value -= self.unmet_demand_cost * poisson_mean

        return Q_value

    @classmethod
    def binomial_example(cls, 
                         demand_params=(10, 0.10),
                         max_inventory=4, 
                         holding_cost=0.5,
                          unmet_demand_cost=1, 
                         horizon=5, 
                         prices=[5, 10, 15, 20]):

        demand_type_and_params = ('binomial', demand_params)

        get_probability_of_demand_given_price = functools.partial(
                                                                cls.get_probability_of_demand_given_price_for_examples,
                                                                kind='binomial',
                                                                demand_params=demand_params)
        
        get_next_state_probabilities = functools.partial(cls.get_next_state_probabilities_for_examples,
                                                         max_inventory=max_inventory,
                                                         kind='binomial',
                                                         demand_params=demand_params)

        return cls(holding_cost, unmet_demand_cost, prices,
                   max_inventory,
                   get_probability_of_demand_given_price,
                   get_next_state_probabilities, horizon, demand_type_and_params)

    @classmethod
    def poisson_example(cls, 
                        demand_params=(9, 0.15), 
                        max_inventory=4, 
                        holding_cost=0.5, 
                        unmet_demand_cost=1, 
                        horizon=5,
                        prices=[5, 10, 15, 20]):

        demand_type_and_params = ('poisson', demand_params)

        get_probability_of_demand_given_price = functools.partial(
                                                        cls.get_probability_of_demand_given_price_for_examples,
                                                        kind='poisson',
                                                        demand_params=demand_params)
        
        get_next_state_probabilities = functools.partial(cls.get_next_state_probabilities_for_examples,
                                                         max_inventory=max_inventory,
                                                         kind='poisson',
                                                         demand_params=demand_params)

        return cls(holding_cost, unmet_demand_cost, prices,
                   max_inventory,
                   get_probability_of_demand_given_price,
                   get_next_state_probabilities, horizon, demand_type_and_params)

    @classmethod
    def example(cls, demand_params, kind):
        """Return an object instance for either the Binomial or Poisson example. 

        kind: str; 
            Options: 'binomial', 'poisson'
        """

        if kind == 'binomial':
            return DynamicPricingMDP.binomial_example(demand_params)

        elif kind == 'poisson':
            return DynamicPricingMDP.poisson_example(demand_params)

    @staticmethod
    def get_next_state_probabilities_for_examples(state, action, max_inventory, kind, demand_params):
        '''Return a pmf for the next state. 

        The probability of staying at the current inventory level is the probability that demand = 0. 
        The probability that the next inventory level goes down by 1 is the probability that demand = 1. Etc.
        The probability that the next inventory level is 0 is 1 - (sum of all other probabilities). 

        Note: The inventory level cannot increase.'''

        # Use the terminology for this example
        current_inventory = state
        price = action

        pmf = [0] * (max_inventory + 1)

        for demand, inventory in enumerate(range(current_inventory, 0, -1)):

            pmf[inventory] = DynamicPricingMDP.get_probability_of_demand_given_price_for_examples(
                demand, price, kind, demand_params)

        pmf[0] = 1 - sum(pmf)

        return pmf

    @staticmethod
    def get_probability_of_demand_given_price_for_examples(demand, price, kind, demand_params):

        if kind == 'binomial':

            n, alpha = demand_params
            p = np.exp(-1 * alpha * price)

            return binom.pmf(k=demand, n=n, p=p)

        elif kind == 'poisson':

            gamma, alpha = demand_params
            poisson_mean = gamma * np.exp(-1 * alpha * price)

            return poisson.pmf(k=demand, mu=poisson_mean)
