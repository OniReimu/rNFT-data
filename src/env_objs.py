"""
 @author suncj
 @email sun.cj@zhejianglab.com
 @create date 2023-09-20 16:38:47
 @modify date 2023-11-20 14:38:00
 @desc [description]
"""

import numpy as np
import random

from collections import deque, namedtuple

from .constants import TRANS_TYPE_DATASET, TRANS_TYPE_MODEL, EXTENDED_DELAY_STEPS_FACTOR
from .basic_utils import generate_unique_id, generate_random_value
from .model import PPOAgent


class EnvConfig:
    """_summary_"""

    def __init__(self, config_data):
        if config_data is None:
            self.publisher_num = 10
            self.distribution_model = "uniform"
            self.raw_interest = 0.01
            self.candidate_num = 10
            self.back_d = 10
            self.max_system_reward = 2
            self.fixed_publish_cost = 0.1
        else:
            # Assign values to the corresponding instance variables
            self.publisher_num = config_data.get("publisher_num", 10)
            self.distribution_model = config_data.get("distribution_model", "uniform")
            self.raw_interest = config_data.get("raw_interest", 0.01)
            self.candidate_num = config_data.get("candidate_num", 10)
            self.back_d = config_data.get("back_d", 10)
            self.max_system_reward = config_data.get("max_system_reward", 2)
            self.fixed_publish_cost = config_data.get("fixed_publish_cost", 0.1)


DEFAULT_CONFIG_DATA = {
    "publisher_num": 10,
    "distribution_model": "uniform",
    "raw_interest": 0.01,
    "candidate_num": 15,
    "back_d": 8,
    "max_system_reward": 2,
    "fixed_publish_cost": 0.1,
}

# NOT USED
DEFAULT_CONFIG = EnvConfig(DEFAULT_CONFIG_DATA)


class Transaction:
    """_summary_

    Returns:
        _type_: _description_
    """

    publisher = None
    trans_type = None

    id = None
    charge_fee = 1
    quality = 1
    price = 1
    down_payment_ratio = 1

    def __init__(self, publisher, block_height, charge_fee, quality, price, down_payment_ratio, ttype):
        """_summary_

        Args:
            block_height (_type_): _description_
            ttype (_type_): _description_
            owner (_type_): _description_
            charge_fee (_type_): _description_
            quality (_type_): _description_
            price (_type_): _description_
        """
        self.publisher = publisher
        self.block_height = block_height
        self.charge_fee = charge_fee
        self.quality = quality
        self.price = price
        self.down_payment_ratio = down_payment_ratio
        self.trans_type = ttype
        # self.id = generate_unique_id(ttype)
        self.id = generate_unique_id("trans_id")

    def to_feature_vector(self):
        """转成训练所需的向量

        Returns:
            _type_: _description_
        """
        return [self.id, self.charge_fee, self.quality, self.price, self.down_payment_ratio]

    @classmethod
    def empty_vector(cls):
        """_summary_

        Returns:
            _type_: _description_
        """
        return [0, 0, 0, 0, 0]


class DatasetTrans(Transaction):
    """_summary_

    Args:
        Transaction (_type_): _description_
    """

    ref_dataset = None
    ref_dataset_ratio = 0

    def __init__(
        self, publisher, block_height, charge_fee, quality, price, down_payment_ratio, ref_dataset, ref_dratio
    ):
        super().__init__(publisher, block_height, charge_fee, quality, price, down_payment_ratio, TRANS_TYPE_DATASET)

        self.ref_dataset = ref_dataset
        self.ref_dataset_ratio = ref_dratio


class ModelTrans(Transaction):
    """_summary_

    Args:
        Transaction (_type_): _description_
    """

    ref_dataset = None
    ref_dataset_ratio = 0

    ref_model = None
    ref_model_ratio = 0

    def __init__(
        self,
        publisher,
        block_height,
        charge_fee,
        quality,
        price,
        down_payment_ratio,
        ref_dataset,
        ref_dratio,
        ref_model,
        ref_mratio,
    ):
        super().__init__(publisher, block_height, charge_fee, quality, price, down_payment_ratio, TRANS_TYPE_MODEL)

        self.ref_dataset = ref_dataset
        self.ref_dataset_ratio = ref_dratio

        self.ref_model = ref_model
        self.ref_model_ratio = ref_mratio


class Ledger:
    """_summary_

    Returns:
        _type_: _description_
    """

    config = None

    # All transactions are stored in transaction_ledger regardless of dataset/model type
    trans_ledger = None

    # Store transactions by block
    block_ledger = None
    block_height = -1

    # Track transactions that have not expired
    fresh_trans = None

    def __init__(self, config) -> None:
        """_summary_"""
        self.config = config
        self.reset()

    def reset(self) -> None:
        """_summary_"""
        self.block_height = 0
        self.block_ledger = {}

        self.trans_ledger = {}

        # self.init_trans = []
        self.fresh_trans = {}

    def add_transaction(self, trans, init=False):
        """_summary_

        Args:
            trans (_type_): _description_
        """

        if trans is None:
            return

        if trans.quality <= 0 or trans.quality > 1:
            print()
        assert 0 < trans.quality <= 1

        if trans.trans_type not in self.trans_ledger:
            self.trans_ledger[trans.trans_type] = {}
            self.fresh_trans[trans.trans_type] = {}

        # Store dataset/model in trans_ledger
        self.trans_ledger[trans.trans_type][trans.id] = trans

        # Store to block_ledger
        if self.block_height not in self.block_ledger:
            self.block_ledger[self.block_height] = []

        self.block_ledger[self.block_height].append(trans)

        # Add transaction to fresh_trans
        self.fresh_trans[trans.trans_type][trans.id] = trans

        # if init:
        #     self.init_trans.append(trans)

    def new_block(self):
        """_summary_"""
        self.block_height += 1

    def get_block_height(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.block_height

    def get_transaction(self, trans_id, ttype="dataset/model"):
        """_summary_

        Args:
            trans_id (_type_): _description_
            ttype (_type_): _description_

        Returns:
            _type_: _description_
        """
        if ttype not in self.trans_ledger:
            return None

        if trans_id not in self.trans_ledger[ttype]:
            return None

        return self.trans_ledger[ttype][trans_id]

    def get_latest_n_transaction(self, n, ttype="dataset/model"):
        """_summary_

        Args:
            n (_type_): _description_
            ttype (str, optional): _description_. Defaults to "dataset/model".

        Returns:
            _type_: _description_
        """
        if ttype not in self.trans_ledger:
            return []

        trans_dict = self.trans_ledger[ttype]

        trans_id_list = list(trans_dict.keys())
        sorted(trans_id_list)

        if len(trans_id_list) > n:
            trans_id_list = trans_id_list[-n:]

        return [self.trans_ledger[ttype][i] for i in trans_id_list]

    def get_freshest_n_transaction(self, n, ttype="dataset/model"):
        """_summary_

        Args:
            n (_type_): _description_
            ttype (str, optional): _description_. Defaults to "dataset/model".

        Returns:
            _type_: _description_
        """
        if ttype not in self.fresh_trans:
            return []

        # Make a copy of the dictionary to iterate over
        trans_dict = self.fresh_trans[ttype].copy()

        # Iterate over the copy and delete entries from the original dictionary
        for _, (trans_id, trans) in enumerate(trans_dict.items()):
            if self.block_height - trans.block_height >= EXTENDED_DELAY_STEPS_FACTOR * self.config.back_d:
                del self.fresh_trans[ttype][trans_id]

        # Reassign trans_dict to reflect the updates in self.fresh_trans[ttype]
        trans_dict = self.fresh_trans[ttype]

        # Selects n transactions randomly based on the distribution of quality, without replacement.
        if n >= len(trans_dict):
            return sorted(trans_dict.values(), key=lambda x: x.quality, reverse=True)

        trans_id_list = list(trans_dict.keys())

        qualities = np.array([trans_dict[trans_id].quality for trans_id in trans_id_list])
        probabilities = qualities / qualities.sum()

        selected_indices = np.random.choice(len(trans_id_list), size=n, replace=False, p=probabilities)
        selected_trans = [trans_dict[trans_id_list[i]] for i in selected_indices]

        # Sort the selected transactions based on quality
        selected_trans.sort(key=lambda x: x.quality, reverse=True)

        # select_height = [trans.block_height for trans in selected_trans]

        # for h in select_height:
        #     print(f"ttype: {ttype}, dataset height: {h}. current: {self.get_block_height()}")
        #     assert self.get_block_height() - h <= 20
        # for h in select_height:
        #     print(f"ttype: {ttype}, model height: {h}. current: {self.get_block_height()}")
        #     assert self.get_block_height() - h <= 20

        return selected_trans

    def get_quality_of_transactions(self, trans):
        """_summary_

        Args:
            trans (_type_): _description_

        Returns:
            _type_: _description_
        """
        tran_qualities = []
        for tran in trans:
            tran_qualities.append(tran.quality)

        return tran_qualities

    def get_ids_of_transactions(self, trans):
        """_summary_

        Args:
            trans (_type_): _description_

        Returns:
            _type_: _description_
        """
        tran_ids = []
        for tran in trans:
            tran_ids.append(tran.id)

        return tran_ids


# 1. Assign wallet, randomly select [60~100]
# 2. Assign dataset, randomly specify quality
# 3. Assign initial model, randomly specify quality
class Publisher:
    """_summary_"""

    config = None
    ledger = None

    name = None
    balance = []
    dataset = None
    model = None

    def __init__(self, config, ledger) -> None:
        """_summary_"""
        self.config = config
        self.ledger = ledger

        self.name = generate_unique_id("publisher")

        self.agent = PPOAgent()
        self.memory = ReplayMemory(capacity=1e6)
        self.delayed_rewards_processor = DelayedRewardProcessor(ledger, config)

        # Reset is not elegant here because it is immediately followed by a call to reset
        # self.reset()

    def reset(self):
        """_summary_"""

        # balance should be infinite
        self.balance = [
            random.randint(600, 1000),
        ]

        charge_fee = random.random()  # [0, 1)

        # Generate quality using different random distribution functions
        quality = generate_random_value(0, 1, self.config.distribution_model)
        price = quality
        down_payment_ratio = random.random()
        self.dataset = DatasetTrans(self, 0, charge_fee, quality, price, down_payment_ratio, None, 1)

        charge_fee = random.random()

        # Generate quality using different random distribution functions
        quality = generate_random_value(0, 1, self.config.distribution_model)
        price = quality
        self.model = ModelTrans(self, 0, charge_fee, quality, price, down_payment_ratio, None, 1, None, 1)

        self.delayed_rewards_processor.delayed_rewards.clear()
        self.delayed_rewards_processor.delayed_rewards_counter.clear()

        self.ledger.add_transaction(trans=self.dataset, init=True)
        self.ledger.add_transaction(trans=self.model, init=True)

    def get_balance(self, last_nth=1):
        """_summary_

        Args:
            back_n (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """
        if last_nth > len(self.balance):
            return self.balance[0]

        return self.balance[-last_nth]

    def set_balance(self, bal):
        """_summary_

        Args:
            bal (_type_): _description_
        """
        self.balance.append(bal)


Record = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ReplayMemory(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=int(capacity))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Record(*args))

    def sample(self, batch_size):
        # # Filter out the samples where the reward is None
        # available_samples = [record for record in self.memory if record.reward is not None]
        # return random.sample(available_samples, min(batch_size, len(available_samples)))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DelayedRewardProcessor:
    def __init__(self, ledger, config):
        self.ledger = ledger
        self.config = config

        self.delayed_rewards = {}
        self.delayed_rewards_counter = {}

    def extended_delay(self, charge_fee):
        """
        Calculate the extended delay based on the charge fee

        Args:
            charge_fee (_type_): _description_

        Returns:
            final_expense (_type_): _description_
        """
        if charge_fee <= 0:
            return self.config.back_d
        elif charge_fee >= 1:
            return EXTENDED_DELAY_STEPS_FACTOR * self.config.back_d
        else:
            return int(np.ceil(self.config.back_d + charge_fee * (self.config.back_d)))

    def set_delayed_rewards(self, trans_id, current_block_height, reward, charge_fee):
        """Set delayed rewards for a given transaction ID.

        Args:
            trans_id (str): Transaction identifier.
            current_block_height (int): Current height of the blockchain block.
            reward (float, optional): Reward to be added. Defaults to 0.
            charge_fee (float): _description_

        Returns:
            extended_delay (_type_): _description_
        """
        delayed_rewards_counter = self.delayed_rewards_counter
        entry = delayed_rewards_counter.get(trans_id)

        extended_delay = self.extended_delay(charge_fee)

        # Trigger cleanup with 10% chance
        if random.random() < 0.1:
            self.cleanup_expired_rewards(current_block_height)

        if entry is None:
            # Initialize if transaction ID doesn't exist.
            delayed_rewards_counter[trans_id] = (current_block_height, reward, extended_delay)
        else:
            # Update the reward value.
            delayed_rewards_counter[trans_id] = (entry[0], entry[1] + reward, entry[2])

        return extended_delay

    def cleanup_expired_rewards(self, current_block_height):
        # List of keys to delete based on the expiration condition
        keys_to_delete = [
            trans_id
            for trans_id, (block_height, _, extended_delay) in self.delayed_rewards_counter.items()
            if block_height + extended_delay < current_block_height
        ]

        # Delete the expired entries from the dictionary
        for trans_id in keys_to_delete:
            del self.delayed_rewards_counter[trans_id]

    def add_delayed_reward(self, trans_id, reward_info, delay_steps, total_expense):
        self.delayed_rewards[trans_id] = (reward_info, delay_steps, total_expense)

    def update_rewards(self, replay_memory, perf_dict, publisher_name, writer):
        for trans_id in list(self.delayed_rewards.keys()):
            reward_info, delay_steps, total_expense = self.delayed_rewards[trans_id]
            delay_steps -= 1

            if delay_steps <= 0:
                state, action, next_state_x, done = reward_info
                reward = self.calculate_reward(trans_id) - total_expense
                replay_memory.push(state, action, next_state_x, reward, done)

                # Update the perf_dict with the calculated reward
                perf_entry = perf_dict.get(trans_id)
                if perf_entry:
                    perf_entry["reward"] = reward
                    writer.add_scalar(f"publisher-{publisher_name}/eval", perf_entry["reward"])

                # Remove the trans_id entry from the delayed_rewards as it's processed
                del self.delayed_rewards[trans_id]

                # for ttype in list(self.ledger.fresh_trans.keys()):
                #     if trans_id in self.ledger.fresh_trans[ttype]:
                #         # trans = self.ledger.fresh_trans[ttype][trans_id]
                #         # print(f"height gap : {self.ledger.block_height - trans.block_height}, trans_height: {trans.block_height}, current: {self.ledger.block_height}")
                #         del self.ledger.fresh_trans[ttype][trans_id]

                # # Remove the initial transactions from the fresh_trans dict
                # if self.ledger.get_block_height() > self.config.back_d and len(self.ledger.init_trans) > 0:
                #     print(f"self.ledger.get_block_height(): {self.ledger.get_block_height()}, len(self.ledger.init_trans): {len(self.ledger.init_trans)}")
                #     for trans_id in self.ledger.init_trans:
                #         for ttype in list(self.ledger.fresh_trans.keys()):
                #             if trans_id in self.ledger.fresh_trans[ttype]: del self.ledger.fresh_trans[ttype][trans_id]
                #     self.ledger.init_trans.clear()

            else:
                # Update the delay_steps and total_expense in the dictionary
                self.delayed_rewards[trans_id] = (reward_info, delay_steps, total_expense)

    def calculate_reward(self, trans_id):
        return self.delayed_rewards_counter.get(trans_id, (0, 0))[1]
