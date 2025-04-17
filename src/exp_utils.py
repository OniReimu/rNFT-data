"""
 @author suncj
 @email sun.cj@zhejianglab.com
 @create date 2023-09-20 16:38:53
 @modify date 2023-11-20 10:51:46
 @desc [description]
"""

import numpy as np
import random
import math

from .env_objs import Transaction
from .constants import TRANS_TYPE_DATASET, TRANS_TYPE_MODEL


def compute_cost_price(quality, trans_type=None):
    """
    Compute the cost price

    Args:
        quality (_type_): _description_
        trans_type (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if trans_type == TRANS_TYPE_DATASET:
        return quality

    return quality


def compute_system_reward(base_quality, improved_quality, max_system_reward):
    max_quality = 1.0

    # Ensure that improved quality is never less than base quality
    if improved_quality < base_quality:
        raise ValueError("Improved quality must be greater than or equal to base quality.")

    # If there's no improvement, the reward is zero
    if improved_quality == base_quality:
        return 0

    # The gap to maximum quality from the base quality
    gap_to_max_from_base = max_quality - base_quality

    # The actual improvement achieved
    actual_improvement = improved_quality - base_quality

    # Calculate the effort ratio: a small improvement on a high base quality
    # should be worth as much as a larger improvement on a lower base quality.
    effort_ratio = (actual_improvement / gap_to_max_from_base) ** 0.5

    # Calculate the reward as proportional to the effort ratio
    system_reward = max_system_reward * effort_ratio

    # Ensure the reward is not above the maximum or below zero
    system_reward = min(max(system_reward, 0), max_system_reward)

    return system_reward        


def compute_transaction_quality(ref_dataset, ref_dratio, ref_model, ref_mratio,
                                my_dataset, my_model, out_trans_type):
    """
    Compute the transaction quality

    Args:
        ref_dataset (_type_): _description_
        ref_dratio (_type_): _description_
        ref_model (_type_): _description_
        ref_mratio (_type_): _description_
        my_dataset (_type_): _description_
        my_model (_type_): _description_
        out_trans_type (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    quality = 0

    if out_trans_type == TRANS_TYPE_DATASET:
        # If ref is larger than current, quality increases, maximum increase 0.1
        if ref_dataset is not None and ref_dataset.quality > my_dataset.quality:
            quality = min(ref_dataset.quality * (1 + 0.1 * ref_dratio), 1)      

        elif ref_dataset is not None and ref_dataset.quality <= my_dataset.quality:
            # NOTE: Only non-referencing will remain unchanged, if referenced, it will become "not worse than the worst", need to multiply a random decay
            quality = ref_dataset.quality * random.uniform(0.9, 1)
        else:
            quality = my_dataset.quality

    elif out_trans_type == TRANS_TYPE_MODEL:
        data_factor = 1
        # If ref dataset is better than current, it has a positive effect on model quality, maximum increase 0.1
        if ref_dataset is not None and ref_dataset.quality > my_dataset.quality:
            data_factor = 1 + 0.1 * ref_dratio
        elif ref_dataset is None or ref_dataset.quality == my_dataset.quality:
            data_factor = 1
        else:
            data_factor = 1 - 0.1 * ref_dratio

        # Similar to dataset processing, if ref's quality is better than current, maximum increase 0.1
        if ref_model is not None and ref_model.quality > my_model.quality:
            quality = min(ref_model.quality * (1 + 0.1 * ref_mratio), 1)   
        elif ref_model is not None and ref_model.quality <= my_model.quality:
            # NOTE: Same as dataset
            quality = ref_model.quality * random.uniform(0.9, 1)
        else:
            quality = my_model.quality

        # Multiply by the data set's improvement effect
        quality = min(quality * data_factor, 1)

    return quality


def convert_transactions_to_feature_vector(datasets, dataset_dim, models,
                                           model_dim):
    """
    Convert transactions to feature vectors

    Args:
        datasets (_type_): _description_
        dataset_dim (_type_): _description_
        models (_type_): _description_
        model_dim (_type_): _description_

    Returns:
        _type_: _description_
    """
    vec = []
    for i in range(dataset_dim):
        if i < len(datasets):
            vec.append([i, ] + datasets[i].to_feature_vector())
        else:
            vec.append([i, ] + Transaction.empty_vector())

    for i in range(model_dim):
        if i < len(models):
            vec.append([i, ] + models[i].to_feature_vector())
        else:
            vec.append([i, ] + Transaction.empty_vector())

    return np.array(vec).ravel()


def assign_rewards_to_publishers(publisher, ref_trans, budget, charge_fee):
    """_summary_

    Args:
        trans (_type_): _description_
        latest_block_height (_type_): _description_
    """
    # If trans is empty, return
    if ref_trans is None:
        return

    publisher.set_balance(publisher.get_balance() - budget) # budget includes charge_fee
    ref_trans.publisher.set_balance(ref_trans.publisher.get_balance() + budget - charge_fee) # charge_fee will be destroyed by the system and not transferred to the referenced publisher

    # # Calculate the discount factor    
    # # The newer the transaction, the larger the factor; the larger the charge_fee, the larger the factor
    # # discount_factor = math.pow(ref_trans.block_height / latest_block_height * ref_trans.charge_fee / 100, 2)

    # # Pay for ref dataset
    # if ref_trans.ref_dataset is not None:
    #     cost = ref_trans.ref_dataset.price * ref_trans.ref_dataset_ratio
    #     ref_trans.publisher.set_balance(ref_trans.publisher.get_balance() - cost)
    #     ref_trans.publisher.set_balance(publisher.get_balance() + cost)

    # # Pay for ref model
    # if hasattr(ref_trans, 'ref_model') and ref_trans.ref_model is not None:
    #     cost = ref_trans.ref_model.price * ref_trans.ref_model_ratio  # * discount_factor
    #     ref_trans.publisher.set_balance(publisher.get_balance() - cost)
    #     ref_trans.ref_model.publisher.set_balance(publisher.get_balance() + cost)

    # # Recursive money distribution
    # assign_rewards_to_publishers(ref_trans.publisher, ref_trans.ref_dataset, budget / 2 / 2, latest_block_height)

    # if hasattr(ref_trans, 'ref_model'):
    #     assign_rewards_to_publishers(ref_trans.publisher, ref_trans.ref_model, budget / 2 / 2, latest_block_height)

def system_rewards_to_publishers(publisher, reward):
    """
    The system directly gives rewards to publishers

    Args:
        publisher (_type_): _description_
        reward (_type_): _description_

    Returns:
        _type_: _description_
    """ 
    publisher.set_balance(publisher.get_balance() + reward)

def set_delayed_rewards(trans, current_block_height, reward, charge_fee):
    """
    Record the reward status of this trans

    Args:
        publisher (_type_): _description_
        trans (_type_): _description_
        current_block_height (_type_): _description_
        reward (_type_): _description_
        charge_fee (_type_): _description_

    Returns:
        extended_delay: _description_
    """ 
    return trans.publisher.delayed_rewards_processor.set_delayed_rewards(trans.id, current_block_height, reward, charge_fee)


def installment_payment(expense, down_payment_ratio, quality, charge_fee, extended_delay, raw_interest):
    """
    Calculate the down payment and subsequent payments based on the charge fee

    Args:
        expense (_type_): _description_
        down_payment_ratio (_type_): _description_
        quality (_type_): _description_
        charge_fee (_type_): _description_
        extended_delay (_type_): _description_

    Returns:
        final_expense (_type_): _description_
    """ 
    assert down_payment_ratio >= 0 and down_payment_ratio <= 1
    assert quality >= 0 and quality <= 1

    # k = 0.5
    # interest_rate_op = 1 + (raw_interest / (1 + k * charge_fee)) # Hyperbolic decay, i.e., g(\hat(q), \pi_r)
    interest_rate_op = 1 + (raw_interest - raw_interest * charge_fee) # Linear decay

    final_expense = expense * down_payment_ratio + \
                                        charge_fee + \
                                            sum(interest_rate_op**(j-quality) * (expense * (1 - down_payment_ratio) / extended_delay) for j in range(1, extended_delay+1))

    assert final_expense >= expense

    return final_expense

def compute_expense_decay_on_delayed_step(block_height_gap):
    """
    Compute the expense decay rate based on the difference between the current transaction block height and the referenced block height

    Args:
        block_height_gap (_type_): _description_

    Returns:
        decay (_type_): _description_
    """     

    assert block_height_gap >= 0

    # y = 2 - 1.2 * log(0.5+x)
    decay = max(2 - 1.2 * math.log(0.5 + block_height_gap), 0)

    return decay