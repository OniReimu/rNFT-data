import torch
import numpy as np
import time

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from .dsp_env import DSPEnv
from .env_objs import Ledger, Publisher, Record, EnvConfig
from .model import ACTION_DIM
from .constants import EPISODES, EPOCHS, BATCH_SIZE
from .basic_utils import generate_unique_id
from .dp_baseline import DPPublisher  # Import the DP Publisher

writer = SummaryWriter('logs')


def run_exp(config: EnvConfig, agent_type="ddpg"):
    """
    Run experiment with specified agent type
    
    Args:
        config: Environment configuration
        agent_type: Type of agent to use ("ddpg" or "dp")
        
    Returns:
        Performance dictionary and action logs
    """
    ledger = Ledger(config)

    # Create publishers based on agent type
    if agent_type.lower() == "dp":
        publishers = [DPPublisher(config, ledger) for i in range(config.publisher_num)]
    else:  # Default to DDPG
        publishers = [Publisher(config, ledger) for i in range(config.publisher_num)]

    env = DSPEnv(config, ledger, publishers)

    perf_dict = {}
    action_logs = []
    timestamps = []  # Add timestamps list to track episode start times

    for epi in tqdm(range(EPISODES), desc='Episode'):
        timestamps.append(time.time())  # Record start time of each episode
        state = env.reset()

        # Reset participant node information, including balance, dataset and model quality
        for pub in publishers:
            pub.reset()

        for t in range(EPOCHS):
            env.next_round()

            for pub in publishers:
                # Q: If balance is less than 0, do you still participate in training?
                # A: balance is set to infinite

                # Add the current publisher's balance as part of the state
                state = np.hstack((state, np.array([pub.get_balance(), ])))

                action = pub.agent.action(state)
                action_logs.append({'publisher': pub.name, 'action': action[0], 'ref_mdl': action[1], 'ref_ds': action[2], 
                                    'ref_mrat': action[3], 'ref_drat': action[4], 'charge_fee': action[5], 'down_payment_ratio': action[6], 'balance': pub.get_balance()})

                assert len(action) == ACTION_DIM

                # NOTE The next_state returned by step should not include balance information, because this balance is the balance of the previous publisher
                # next_state, reward, done, _, _ = env.step(pub, action)

                # NOTE The reward is delayed, step should output the delay steps
                next_state, (delay_steps, trans_id, total_expense), done, _, _ = env.step(pub, action)

                next_state_x = np.hstack((next_state, np.array([pub.get_balance(), ])))

                # NOTE If no publishing, delay_steps is 0, then input to memory, otherwise input to delayed queue
                if delay_steps > 0 and trans_id is not None:
                    # Assuming 'state' is the current state before taking the action
                    # Push the action and its expected delay into the delayed reward queue

                    pub.delayed_rewards_processor.add_delayed_reward(trans_id, (state, action, next_state_x, done), delay_steps, total_expense)

                else:
                    # If no publishing, reward is 0, then store to memory
                    pub.memory.push(state, action, next_state_x, 0, done)

                # Update the queue and process rewards that are ready
                pub.delayed_rewards_processor.update_rewards(pub.memory, perf_dict, pub.name, writer)

                state = next_state

                # First output reward as None, until delay expires, update reward
                if trans_id is not None:
                    perf_dict[trans_id] = {
                        'episode': epi,
                        'epoch': t,
                        'publisher': pub.name,
                        'trans_id': trans_id,
                        'reward': None,
                        'delay_steps': delay_steps,
                        'total_expense': total_expense,
                        # 'timestamp': time.time()
                    }
                else:
                    perf_dict[f'None-{generate_unique_id("none_trans_id")}'] = {
                        'episode': epi,
                        'epoch': t,
                        'publisher': pub.name,
                        'trans_id': None,
                        'reward': 0,
                        'delay_steps': 0,
                        'total_expense': 0,
                        # 'timestamp': time.time()
                    }

                # Train the agent if enough samples are available
                if len(pub.memory) < BATCH_SIZE:
                    continue

                records = pub.memory.sample(BATCH_SIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Record(*zip(*records))
                pub.agent.train(batch)

    perf_dict['timestamps'] = timestamps  # Add timestamps to perf_dict
    return perf_dict, action_logs
