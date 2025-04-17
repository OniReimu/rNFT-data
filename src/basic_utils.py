"""
 @author suncj
 @email sun.cj@zhejianglab.com
 @create date 2023-09-20 16:34:33
 @modify date 2023-11-20 15:33:45
 @desc [description]
"""

import numpy as np

# Global variables
global_count = {}


def generate_unique_id(entity_type):
    """Returns the global ID for an entity, assigned sequentially starting from 1.

    Args:
        entity_type (_type_): _description_

    Returns:
        _type_: _description_
    """
    if entity_type not in global_count:
        global_count[entity_type] = -1

    global_count[entity_type] += 1

    return global_count[entity_type]


def generate_random_value(min_val, max_val, dist_model, include_min=False, include_max=False):
    """
    Returns a value between min_val and max_val based on the given probability distribution model dist_model.

    Args:
        min_val (_type_): Lower bound
        max_val (_type_): Upper bound
        dist_model (_type_): Probability distribution model
        include_min (bool, optional): Whether to include lower bound. Defaults to False.
        include_max (bool, optional): Whether to include upper bound. Defaults to False.

    Returns:
        _type_: _description_
    """

    num_samples = 10000  # Number of samples to generate

    # Generate values from uniform distribution
    if dist_model == "uniform":
        vals = np.random.uniform(min_val, max_val, num_samples)

    # Generate values from Gaussian distribution
    elif dist_model == "gaussian":
        mean = (max_val + min_val) / 2  # Mean
        std_dev = (max_val - min_val) / 6  # Standard deviation
        vals = np.random.normal(mean, std_dev, num_samples)
        vals = np.clip(vals, min_val, max_val)  # Clip generated values to specified range

    # Generate values from Pareto distribution
    elif dist_model == "pareto":
        shape = 3.0  # Shape parameter, must be greater than 0
        scale = 1.0  # Scale parameter, must be greater than 0

        # Generate Pareto distributed values between 0 and 1
        pareto_values = (np.random.pareto(shape, 1000) + 1) * scale

        # Map the generated Pareto values to the range of 0 to 1
        vals = 1 - (1 / pareto_values)

    # Generate values from Poisson distribution
    elif dist_model == "poisson":
        poisson_vals = np.random.poisson((max_val - min_val) / 2, num_samples) + min_val  # Mean is (max_val - min_val) / 2
        vals = poisson_vals / (poisson_vals + 1)

    for v in vals:
        if not include_min and v == min_val:
            continue

        if not include_max and v == max_val:
            continue

        return v

    return None
