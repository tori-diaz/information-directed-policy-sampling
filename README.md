# About 
Code associated with "Information-Directed Policy Sampling for Episodic Bayesian Markov Decision Processes".

# How to run 
1. Create an environment based on `requirements.txt`. 

    For example, if using Windows, 
    ```bash 
    # Create a virtual environment
    py -m venv .venv

    # Activate the environment
    .venv/Scripts/activate

    # Install the requirements
    py -m pip install -r requirements.txt
    ```

2. Create `logs` and `data` folders in the root directory.

3. Create the data using: `python -m paper_results.data_creator`.

    The following directories and files will be created. 
    ```
    data
    ├── dynamic_pricing
    │   ├── binomial
    │   │     ├── incomplete_information_mdp.pkl
    │   │     ├── state_trajectory_probabilities_df.csv
    │   │     ├── scenario.csv
    │   │     ├── true_mdp_index_0
    │   │     │   ├── idps_regrets__iteration_0.csv
    │   │     │   ├── idps_rewards__iteration_0.csv
    │   │     │   ├── ps_regrets__iteration_0.csv
    │   │     │   ├── ps_rewards__iteration_0.csv
    │   │     │   └── ...
    │   │     ├── ...
    │   │     └── ...
    │   └── poisson
    │   │     └── ...
    ├── machine_repair
    │   └── ...
    └── queue_control
        └── ...
    ```

4. Analyze the data using: `python -m paper_results.data_analyzer`

    The following directory and files will be created. 
    ```
    data
    └── summarizing_tables
        ├── dynamic_pricing__binomial.csv
        ├── dynamic_pricing__poisson.csv
        ├── machine_repair__bernoulli.csv
        ├── machine_repair__truncated_geometric.csv
        ├── queue_control__binomial.csv
        ├── queue_control__finite_set.csv
        └── queue_control__poisson.csv
    ```

    The following describes the correspondence between the tables found in the paper and the data files.

    | Table | Data file | 
    | -- | -- | 
    | Table 2(a) | machine_repair__bernoulli.csv | 
    | Table 2(b) | machine_repair__truncated_geometric.csv | 
    | Table 4(a) | queue_control__poisson.csv | 
    | Table 4(b) | queue_control__binomial.csv | 
    | Table 4(c) | queue_control__finite_set.csv | 
    | Table 6(a) | dynamic_pricing__poisson.csv | 
    | Table 6(b) | dynamic_pricing__binomial.csv | 
