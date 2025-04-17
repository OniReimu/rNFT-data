"""
 @author suncj
 @email sun.cj@zhejianglab.com
 @create date 2023-09-20 17:34:22
 @modify date 2023-11-20 11:22:06
 @desc [description]

"""

from typing import Any, Tuple
import numpy as np
import gymnasium as gym

from gymnasium.envs.registration import EnvSpec

from .constants import DATASET_DIM, MODEL_DIM, TRANS_TYPE_NONE, TRANS_TYPE_DATASET, TRANS_TYPE_MODEL
from .exp_utils import (
    assign_rewards_to_publishers,
    compute_transaction_quality,
    convert_transactions_to_feature_vector,
    system_rewards_to_publishers,
    set_delayed_rewards,
    installment_payment,
    compute_system_reward,
    compute_expense_decay_on_delayed_step,
)
from .env_objs import DatasetTrans, ModelTrans, Publisher


# pylint: disable=C0116
# pylint: disable=W0221
class DSPEnv(gym.Env):
    """
    DSP environment
    """

    config = None
    ledger = None

    publishers = []
    current_publisher = None

    def __init__(self, config, ledger, publishers):
        self.config = config
        self.ledger = ledger
        self.publishers = publishers

        self.spec = EnvSpec(
            id="dsp_env-v0",
            entry_point="src.dsp_env.DSPEnv:",
            reward_threshold=None,
            nondeterministic=False,
            max_episode_steps=30000,
            order_enforce=True,
            disable_env_checker=False,
            # apply_api_compatibility=False,
            kwargs={},
        )

        self.episode_time_steps = 0

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def step(self, publisher: Publisher, action: Tuple[Any]):
        """_summary_

        Args:
            agent (_type_): _description_
            action (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.current_publisher = publisher

        self.episode_time_steps += 1

        ttype, did, mid, dratio, mratio, charge_fee, down_payment_ratio = action
        # 1. ttype: transaction type # 0=no publish, 1=publish dataset, 2=publish model
        # 2. didx: dataset id # can be 0 (not selected)
        # 3. midx: model id # can be 0 (not selected) 
        # 4. dratio: dataset ratio
        # 5. mratio: model ratio
        # 6. charge_fee: fee in range [0~pi], burned directly (not transferred), this is a ratio
        # 7. down_payment_ratio: lambda in paper, [0~1], down payment ratio

        # Agent's wallet needs to deduct the expenses
        # Referenced agent nodes generate income

        # Q: Is there a relationship between the fixed publishing cost and reward?
        # A: No relationship, the fixed cost is to prevent too many junk transactions

        # NOTE: Each round allows no publishing. If publishing without referencing others, reward is 0
        if ttype == TRANS_TYPE_NONE:
            return (
                self.get_state(),
                (0, None, 0),
                self.get_done(),
                self.get_info(),
                None,
            )

        # Publish a transaction budget O, where O=w0+wd+wm, w0 is split evenly, wd(dataset), wm(model), w0+wd+wm=1

        # Q: Does CANDIDATE_NUM need to correspond to self.config.back_d?
        # A: No, CANDIDATE_NUM determines state shape, while back_d is for calculating balance difference over time (reward).

        # datasets = self.ledger.get_latest_n_transaction(self.config.candidate_num, ttype=TRANS_TYPE_DATASET)
        # models = self.ledger.get_latest_n_transaction(self.config.candidate_num, ttype=TRANS_TYPE_MODEL)
        datasets = self.ledger.get_freshest_n_transaction(self.config.candidate_num, ttype=TRANS_TYPE_DATASET)
        models = self.ledger.get_freshest_n_transaction(self.config.candidate_num, ttype=TRANS_TYPE_MODEL)

        ref_dataset = None
        if did != 0 and did < len(datasets):
            # It should be the index of the selected k transactions, not the transaction id
            # ref_dataset = self.ledger.get_transaction(did, 'dataset')
            ref_dataset = datasets[did]

        ref_model = None
        if mid != 0 and mid < len(models):
            # Same as above
            # ref_model = self.ledger.get_transaction(mid, 'model')
            ref_model = models[mid]

        # Start building the transaction
        trans = None

        # If both ref_dataset and ref_model are empty, publish the original model/dataset
        if ref_dataset is None and ref_model is None:
            if ttype == TRANS_TYPE_DATASET:
                publisher.dataset.block_height = self.ledger.get_block_height()
                trans = publisher.dataset

            elif ttype == TRANS_TYPE_MODEL:
                publisher.model.block_height = self.ledger.get_block_height()
                trans = publisher.model

            quality = trans.quality

        # If both ref_dataset and ref_model are valid values
        else:
            # Calculate quality in the env, not in the entity
            quality = compute_transaction_quality(
                ref_dataset, dratio, ref_model, mratio, publisher.dataset, publisher.model, ttype
            )

            # NOTE We set Data Quality == Price
            if ttype == TRANS_TYPE_DATASET:
                trans = DatasetTrans(
                    publisher,
                    self.ledger.get_block_height(),
                    charge_fee,
                    quality,
                    quality,  # price <-- quality
                    down_payment_ratio,
                    ref_dataset,
                    dratio,
                )
            elif ttype == TRANS_TYPE_MODEL:
                trans = ModelTrans(
                    publisher,
                    self.ledger.get_block_height(),
                    charge_fee,
                    quality,
                    quality,  # price <-- quality
                    down_payment_ratio,
                    ref_dataset,
                    dratio,
                    ref_model,
                    mratio,
                )

        if trans is not None:
            self.ledger.add_transaction(trans=trans, init=False)

            # NOTE: If the published dataset/model is better than all candidates, get extra rewards
            qualities_of_entities = []
            if ttype == TRANS_TYPE_DATASET:
                qualities_of_entities = self.ledger.get_quality_of_transactions(datasets)
                publisher.dataset = trans  # Update the latest dataset
            elif ttype == TRANS_TYPE_MODEL:
                qualities_of_entities = self.ledger.get_quality_of_transactions(models)
                publisher.model = trans  # Update the latest model

            # Calculate the range of your data
            max_quality = max(qualities_of_entities)

            # Find the greatest value in the list and calculate the difference
            improvement = max(quality - max_quality, 0)

            system_rewards = compute_system_reward(
                max_quality, improvement + max_quality, self.config.max_system_reward
            )
            system_rewards -= self.config.fixed_publish_cost  # The fixed cost in the paper

            system_rewards_to_publishers(publisher, system_rewards)
            # Register this transaction in delayed_rewards to track subsequent rewards, initial rewards default to 0
            # Must ensure publisher.name == trans.publisher.name
            extended_delay = set_delayed_rewards(trans, self.ledger.get_block_height(), system_rewards, charge_fee)

            # Distribute the rewards to each ref dataset and model
            # If no dataset/model is referenced, no money is distributed

            # Pay the referenced dataset
            total_expense = 0

            if trans.ref_dataset is not None:
                expense = (
                    trans.ref_dataset.price
                    * dratio
                    * compute_expense_decay_on_delayed_step(
                        (self.ledger.get_block_height() - trans.ref_dataset.block_height)
                    )
                )

                # Apply the installment payment mechanism
                expense = installment_payment(
                    expense, down_payment_ratio, quality, charge_fee, extended_delay, self.config.raw_interest
                )

                assign_rewards_to_publishers(publisher, trans.ref_dataset, expense, charge_fee)

                # Register this transaction in delayed_rewards to track subsequent rewards
                set_delayed_rewards(trans.ref_dataset, self.ledger.get_block_height(), expense, charge_fee)

                total_expense += expense

            # Pay the referenced model
            if hasattr(trans, "ref_model") and trans.ref_model is not None:
                expense = (
                    trans.ref_model.price
                    * mratio
                    * compute_expense_decay_on_delayed_step(
                        (self.ledger.get_block_height() - trans.ref_model.block_height)
                    )
                )

                # Apply the installment payment mechanism
                expense = installment_payment(
                    expense, down_payment_ratio, quality, charge_fee, extended_delay, self.config.raw_interest
                )

                assign_rewards_to_publishers(publisher, trans.ref_model, expense, charge_fee)

                # Register this transaction in delayed_rewards to track subsequent rewards
                set_delayed_rewards(trans.ref_model, self.ledger.get_block_height(), expense, charge_fee)

                total_expense += expense

        else:
            raise Exception('Not implemented')

        return (
            self.get_state(),
            # self.get_reward(),
            (extended_delay, trans.id, total_expense),
            self.get_done(),
            self.get_info(),
            None,
        )

    def reset(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        self.ledger.reset()

        self.current_publisher = None

        return self.get_state()

    def render(self):
        pass

    def next_round(self):
        """_summary_"""
        self.episode_time_steps += 1
        self.ledger.new_block()

    def get_state(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # datasets = self.ledger.get_latest_n_transaction(self.config.candidate_num, ttype=TRANS_TYPE_DATASET)
        # models = self.ledger.get_latest_n_transaction(self.config.candidate_num, ttype=TRANS_TYPE_MODEL)
        datasets = self.ledger.get_freshest_n_transaction(self.config.candidate_num, ttype=TRANS_TYPE_DATASET)
        models = self.ledger.get_freshest_n_transaction(self.config.candidate_num, ttype=TRANS_TYPE_MODEL)

        state = convert_transactions_to_feature_vector(datasets, DATASET_DIM, models, MODEL_DIM)

        # Balance is part of the state
        # NOTE The following is problematic, especially when state is used as next state, the associated object of balance has changed
        # if self.current_publisher is None:
        #     state = np.hstack((state, np.array([0, ])))
        # else:
        #     state = np.hstack((state, np.array([self.current_publisher.get_balance(), ])))

        return state

    # def get_reward(self):
    #     """_summary_

    #     Args:
    #         agent (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     return self.current_publisher.get_balance() - self.current_publisher.get_balance(self.config.back_d)

    def get_done(self):
        """_summary_

        Args:
            publisher (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.episode_time_steps > self.spec.max_episode_steps:
            self.episode_time_steps = 0

            return True

        if self.current_publisher.balance[-1] <= 0:
            return True

        return False

    def get_info(self):
        return {}

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_env_spec(self):
        return self.spec
