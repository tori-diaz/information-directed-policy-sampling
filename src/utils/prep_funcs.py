"""Functions used by data_creator.py """

from src.mdp.machine_repair_mdp import MachineRepairMDP
from src.mdp.dynamic_pricing_mdp import DynamicPricingMDP
from src.mdp.queue_control_mdp import QueueControlMDP


def create_mdps(mdp_name, mdp_kind, mdp_param_values):
    """
    Generate a list of MDPs.

    The following options are available for (mdp_name, mdp_kind): 
    - 'machine_repair', 'bernoulli' 
    - 'machine_repair', 'truncated_geometric' 
    - 'dynamic_pricing', 'binomial' 
    - 'dynamic_pricing', 'poisson' 
    - 'queue_control', 'binomial' 
    - 'queue_control', 'poisson' 
    - 'queue_control', 'finite_set'

    Parameters
    ----------
    mdp_name : str
        The name corresponding to the MDP. There are limited options.

    mdp_kind : str
        A specific example of the MDP.

    mdp_param_values 
        The data type and interpretation depends on the (mdp_name, mdp_kind) pair. 
        See the defintions of MachineRepairMDP, DynamicPricingMDP, or QueueControlMDP 
        for more information. 

    Returns
    -------
    list 
        List of either MachineRepairMDPs, DynamicPricingMDPs, or QueueControlMDPs 
    """

    if mdp_name == 'machine_repair':

        mdps = [MachineRepairMDP.example(p, mdp_kind)
                for p in mdp_param_values]

    elif mdp_name == 'dynamic_pricing':

        mdps = [DynamicPricingMDP.example(
            demand_params, mdp_kind) for demand_params in mdp_param_values]

    elif mdp_name == 'queue_control':

        if mdp_kind in ['binomial', 'poisson']:

            mdps = [QueueControlMDP.example(mdp_kind, customer_arrival_params=customer_arrival_params)
                    for customer_arrival_params in mdp_param_values]

        elif mdp_kind == 'finite_set':

            mdps = [QueueControlMDP.example(
                mdp_kind, service_probabilities=service_probabilities) for service_probabilities in mdp_param_values]

    return mdps


def create_optimal_policies(mdps):
    """
    Create a list of optimal policies.

    Parameters
    ----------
    mdps : list[MDPs]
        A list of either MachineRepairMDPs, DynamicPricingMDPs, or QueueControlMDPs     

    Returns
    -------
    list 
        List where the i-th element is an optimal policy for the i-th element in the input (mdps). 
    """

    optimal_policies = [mdp.get_an_optimal_policy() for mdp in mdps]

    return optimal_policies
