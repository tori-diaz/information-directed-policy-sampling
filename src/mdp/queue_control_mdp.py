import numpy as np
import functools
import math
from scipy.stats import binom, poisson

from src.mdp.mdp import MDP
from src.utils import common_funcs


class QueueControlMDP(MDP):

    def __init__(self,
                 max_num_of_customers,
                 horizon,
                 service_completion_probabilities,
                 service_operating_costs,
                 get_queue_length_cost,
                 get_terminal_cost,
                 get_next_state_probabilities):

        self.max_num_of_customers = max_num_of_customers
        self.horizon = horizon
        self.service_completion_probabilities = service_completion_probabilities
        self.service_operating_costs = service_operating_costs
        self.get_queue_length_cost = get_queue_length_cost
        self.state_space = self._create_state_space()
        self.get_terminal_cost = get_terminal_cost
        self.get_next_state_probabilities = get_next_state_probabilities

        # Create parameter values for the MDP
        state_space = self.state_space  
        action_space = self._create_action_space()
        initial_state_pmf = self._create_initial_state_pmf()
        objective = 'minimization'
        J_terminal = self.create_J_terminal()
        super().__init__(state_space, action_space, horizon, initial_state_pmf,
                         get_next_state_probabilities, objective, J_terminal)

    def __str__(self):

        return 'Information about the underlying MDP: \n' + super().__str__()

    def _create_state_space(self):

        return range(self.max_num_of_customers + 1)

    def _create_action_space(self):
        '''Return the action space. 

        0 = slow service 
        1 = fast service'''

        return [0, 1]

    def _create_initial_state_pmf(self):

        return common_funcs.create_uniform_pmf(self.max_num_of_customers + 1)

    def create_J_terminal(self):

        return np.array([self.get_terminal_cost_for_examples(state) for state in self.state_space])

    def get_Q_value(self, J, time_stage, state, action):
        """Return the Q value at a given time stage for a (state, action) pair."""

        next_state_pmf = self.get_next_state_probabilities(state, action)

        Q_value = self.get_queue_length_cost(
            state) + self.service_operating_costs[action] + np.dot(next_state_pmf, J[:, time_stage + 1])

        return Q_value

    @classmethod
    def binomial_example(cls, 
                         max_num_of_customers=4,
                         horizon=5,
                         service_completion_probabilities=[0.3, 0.8],
                         service_operating_costs=[1, 3],
                         customer_arrival_params=(3, 0.25)):

        kind = 'binomial'

        get_queue_length_cost = functools.partial(cls.get_queue_length_cost_for_examples, 
                                                  kind=kind)
        get_terminal_cost = cls.get_terminal_cost_for_examples
        queue_customer_arrival_probabilities = cls.get_queue_customer_arrival_probabilities_for_examples(
                                                        kind, 
                                                        max_num_of_customers, 
                                                        customer_arrival_params)
        get_next_state_probabilities = functools.partial(cls.get_next_state_probabilities_for_examples,
                                                    max_num_of_customers=max_num_of_customers,
                                                    customer_arrival_params=customer_arrival_params,
                                                    service_completion_probabilities=service_completion_probabilities,
                                                    queue_customer_arrival_probabilities=queue_customer_arrival_probabilities)

        return cls(max_num_of_customers,
                   horizon,
                   service_completion_probabilities,
                   service_operating_costs,
                   get_queue_length_cost,
                   get_terminal_cost,
                   get_next_state_probabilities)

    @classmethod
    def poisson_example(cls, max_num_of_customers=4,
                        horizon=5,
                        service_completion_probabilities=[0.3, 0.8],
                        service_operating_costs=[3, 4],
                        customer_arrival_params=0.2):
        
        kind = 'poisson'
        get_queue_length_cost = functools.partial(
            cls.get_queue_length_cost_for_examples, kind=kind)
        get_terminal_cost = cls.get_terminal_cost_for_examples
        queue_customer_arrival_probabilities = cls.get_queue_customer_arrival_probabilities_for_examples(
            kind, max_num_of_customers, customer_arrival_params)
        get_next_state_probabilities = functools.partial(cls.get_next_state_probabilities_for_examples,
                                                         max_num_of_customers=max_num_of_customers,
                                                         customer_arrival_params=customer_arrival_params,
                                                         service_completion_probabilities=service_completion_probabilities,
                                                         queue_customer_arrival_probabilities=queue_customer_arrival_probabilities)

        return cls(max_num_of_customers,
                   horizon,
                   service_completion_probabilities,
                   service_operating_costs,
                   get_queue_length_cost,
                   get_terminal_cost,
                   get_next_state_probabilities)

    @classmethod
    def finite_set_example(cls, max_num_of_customers=4,
                           horizon=5,
                           service_completion_probabilities=[0.3, 0.8],
                           service_operating_costs=[4, 5],
                           customer_arrival_kind='poisson',
                           customer_arrival_params=1):

        kind = 'uncertain_service_probabilities'

        get_queue_length_cost = functools.partial(
            cls.get_queue_length_cost_for_examples, kind=kind)
        get_terminal_cost = cls.get_terminal_cost_for_examples
        queue_customer_arrival_probabilities = cls.get_queue_customer_arrival_probabilities_for_examples(
            kind=customer_arrival_kind, max_num_of_customers=max_num_of_customers, customer_arrival_params=customer_arrival_params)
        get_next_state_probabilities = functools.partial(cls.get_next_state_probabilities_for_examples,
                                            max_num_of_customers=max_num_of_customers,
                                            customer_arrival_params=customer_arrival_params,
                                            service_completion_probabilities=service_completion_probabilities,
                                            queue_customer_arrival_probabilities=queue_customer_arrival_probabilities)
        
        return cls(max_num_of_customers,
                   horizon,
                   service_completion_probabilities,
                   service_operating_costs,
                   get_queue_length_cost,
                   get_terminal_cost,
                   get_next_state_probabilities)

    @classmethod
    def example(cls, kind, customer_arrival_params=None, service_probabilities=None):

        if kind == 'binomial':

            return QueueControlMDP.binomial_example(customer_arrival_params=customer_arrival_params)

        elif kind == 'poisson':

            return QueueControlMDP.poisson_example(customer_arrival_params=customer_arrival_params)

        elif kind == 'finite_set':

            return QueueControlMDP.finite_set_example(service_completion_probabilities=service_probabilities)

    @staticmethod
    def get_terminal_cost_for_examples(state):

        return state

    @staticmethod
    def get_queue_length_cost_for_examples(state, kind):

        if kind == 'binomial':
            return 3 * state

        elif kind == 'poisson':
            return 2 * state

        elif kind == 'uncertain_service_probabilities':
            return state

    @staticmethod
    def get_queue_customer_arrival_probabilities_for_examples(kind, max_num_of_customers, customer_arrival_params):
        '''Return a pmf over the number of new customers arriving to the queue. 

        This pmf will be over the values 1, 2, ..., max number of customers

        This is slightly different than the number of new customers that arrive since the queue has a max number 
        of customers.
        '''

        if kind == 'binomial':

            B, p = customer_arrival_params
            queue_customer_arrival_probabilities = [
                binom.pmf(k, n=B, p=p) for k in range(max_num_of_customers)]

        elif kind == 'poisson':

            poisson_mean = customer_arrival_params
            queue_customer_arrival_probabilities = [poisson.pmf(
                k, mu=poisson_mean) for k in range(max_num_of_customers)]

        queue_customer_arrival_probabilities.append(
            1 - sum(queue_customer_arrival_probabilities))

        if not math.isclose(sum(queue_customer_arrival_probabilities), 1, rel_tol=1e-09):
            print('ERROR: The sum is not correct. It sums to',
                  sum(queue_customer_arrival_probabilities))

        return queue_customer_arrival_probabilities

    @staticmethod
    def get_next_state_probabilities_for_examples(state,
                                                  action,
                                                  max_num_of_customers,
                                                  customer_arrival_params,
                                                  service_completion_probabilities,
                                                  queue_customer_arrival_probabilities):

        def get_prob(num_of_new_customers_in_queue):

            return queue_customer_arrival_probabilities[num_of_new_customers_in_queue]

        if state == 0:

            pmf = [get_prob(num_of_new_customers_in_queue=i)
                   for i in range(max_num_of_customers + 1)]

        elif state >= 1 and state != max_num_of_customers:

            pmf = [0] * (max_num_of_customers + 1)

            # Case 1: next_state = 0, ..., state-2
            # pmf[next_state] = 0 since the max number of customers that can receive service during a time stage is one

            # Case 2: next_state = state - 1
            next_state = state - 1
            # One customer completes their service and no new customers arrive
            pmf[next_state] = service_completion_probabilities[action] * \
                get_prob(num_of_new_customers_in_queue=0)

            # Case 3: next_state = state, ..., max_num_of_customers - 2
            for next_state in range(state, max_num_of_customers - 1):

                # For if both Case 2 and Case 3 apply. This shouldn't happen.
                if pmf[next_state] != 0:
                    pmf[next_state] = 0

                # (next_state - state) customers arrived and no customer received complete service
                pmf[next_state] += get_prob(num_of_new_customers_in_queue=next_state - state) * (
                    1-service_completion_probabilities[action])

                # (next state - state + 1) customers arrived and one customer received complete service
                pmf[next_state] += get_prob(num_of_new_customers_in_queue=next_state -
                                            state + 1) * service_completion_probabilities[action]

            # Case 4: next_state = max_num_of_customers - 1
            next_state = max_num_of_customers - 1

            # For if both Case 2 and Case 4 apply. This happens when state = max_num_of_customers
            if pmf[next_state] != 0:
                pmf[next_state] = 0

            # (next_state - state) customers arrived and no customer received complete service
            pmf[next_state] += get_prob(num_of_new_customers_in_queue=next_state - state) * (
                1-service_completion_probabilities[action])

            # More than (next_state - state + 1) customers arrive and 1 customer receives complete service
            pmf[next_state] += service_completion_probabilities[action] * \
                sum([get_prob(num_of_new_customers_in_queue=i) for i in range(
                    max_num_of_customers - state, max_num_of_customers + 1)])

            # Case 5: next_state = max_num_of_new_customers_in_queue
            next_state = max_num_of_customers


            # No customers receive service and at least (max_num_of_customers - state) customers arrive
            pmf[next_state] = (1 - service_completion_probabilities[action]) * sum([get_prob(num_of_new_customers_in_queue=i)
                                                                                    for i in range(max_num_of_customers - state, max_num_of_customers + 1)])

        elif state == max_num_of_customers:

            pmf = [0] * (max_num_of_customers + 1)

            pmf[max_num_of_customers - 1] = service_completion_probabilities[action] * \
                get_prob(num_of_new_customers_in_queue=0)
            pmf[max_num_of_customers] = 1 - sum(pmf)

        if not math.isclose(sum(pmf), 1, rel_tol=1e-09):
            print(f'Situation: {state=}, {action=}, {
                  customer_arrival_params=}')
            print('ERROR: The sum is not correct. It sums to', sum(pmf))

        return pmf


def check_if_queue_control_parameter_values_are_valid(service_completion_probabilities, service_operating_costs):

    # The service probabilities are indeed probabilities.
    probabilities_check = (
        all([0 <= probability <= 1 for probability in service_completion_probabilities]))

    # The probability of completing service is smaller (or the same amount) when the service is 'slow'.
    service_completion_prob_check = (
        service_completion_probabilities[0] <= service_completion_probabilities[1])

    # The cost of operating 'slow' is less than (or equal to) the cost of operating 'fast' service.
    operating_costs_check = (
        service_operating_costs[0] <= service_operating_costs[1])

    if service_completion_prob_check and operating_costs_check and probabilities_check:
        params_are_valid = True
    else:
        params_are_valid = False

    return params_are_valid
