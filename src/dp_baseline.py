"""
Dynamic Programming baseline implementation for data sharing platform.
"""

import numpy as np
from .env_objs import Publisher, Transaction
from .constants import TRANS_TYPE_NONE, TRANS_TYPE_DATASET, TRANS_TYPE_MODEL

class DPAgent:
    """
    Dynamic Programming agent for the data sharing platform.
    This agent makes decisions based on expected future rewards.
    """
    
    def __init__(self, horizon=5, discount_factor=0.95):
        """
        Initialize the DP agent.
        
        Args:
            horizon (int): Planning horizon (how many steps ahead to consider)
            discount_factor (float): Discount factor for future rewards
        """
        self.horizon = horizon
        self.discount_factor = discount_factor
        self.value_table = {}  # State-value function
        self.policy = {}       # Optimal policy
        
    def action(self, state):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state observation
            
        Returns:
            tuple: Action to take (same format as DDPG agent)
        """
        # Convert state to hashable form
        state_key = self._get_state_key(state)
        
        # If we have a policy for this state, use it
        if state_key in self.policy:
            return self.policy[state_key]
        
        # Otherwise, use a heuristic approach
        # For simplicity, we'll use a rule-based strategy
        
        # Extract balance from state (last element)
        balance = state[-1]
        
        # Default action: don't publish
        if balance < 10:  # Low balance, don't risk publishing
            return (TRANS_TYPE_NONE, 0, 0, 0, 0, 0, 0)
        
        # With sufficient balance, decide whether to publish dataset or model
        if np.random.random() < 0.5:
            # Publish dataset
            return (TRANS_TYPE_DATASET, 
                   np.random.randint(0, 5),  # Reference dataset index
                   0,                        # No model reference
                   np.random.random(),       # Dataset ratio
                   0,                        # No model ratio
                   np.random.random() * 0.5, # Charge fee
                   np.random.random())       # Down payment ratio
        else:
            # Publish model
            return (TRANS_TYPE_MODEL,
                   np.random.randint(0, 5),  # Reference dataset index
                   np.random.randint(0, 5),  # Reference model index
                   np.random.random(),       # Dataset ratio
                   np.random.random(),       # Model ratio
                   np.random.random() * 0.5, # Charge fee
                   np.random.random())       # Down payment ratio
    
    def update_value_table(self, state, action, reward, next_state):
        """
        Update the value table based on observed transitions.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize value if not present
        if state_key not in self.value_table:
            self.value_table[state_key] = 0
        
        if next_state_key not in self.value_table:
            self.value_table[next_state_key] = 0
        
        # Update value using Bellman equation
        self.value_table[state_key] = reward + self.discount_factor * self.value_table[next_state_key]
        
        # Update policy
        self.policy[state_key] = action
    
    def _get_state_key(self, state):
        """
        Convert state array to hashable form.
        
        Args:
            state: State observation
            
        Returns:
            tuple: Hashable representation of state
        """
        # Round values to reduce state space
        # Convert to a more efficient representation to avoid memory issues
        # Only keep a subset of state features to reduce dimensionality
        state_array = np.array(state)
        if len(state_array) > 20:  # If state is large
            # Keep only every 5th element plus the last element (balance)
            indices = list(range(0, len(state_array), 5))
            if len(state_array)-1 not in indices:
                indices.append(len(state_array)-1)  # Always include balance
            reduced_state = state_array[indices]
            rounded_state = tuple(round(float(x), 1) for x in reduced_state)
        else:
            rounded_state = tuple(round(float(x), 1) for x in state_array)
        return rounded_state
    
    def train(self, batch):
        """
        Update the agent's knowledge based on a batch of experiences.
        
        Args:
            batch: Batch of experiences (state, action, next_state, reward, done)
        """
        # Extract batch components
        states = batch.state
        actions = batch.action
        next_states = batch.next_state
        rewards = batch.reward
        dones = batch.done
        
        # Update value table for each experience
        for i in range(len(states)):
            if not dones[i]:  # Only update if not terminal state
                self.update_value_table(states[i], actions[i], rewards[i], next_states[i])


class DPPublisher(Publisher):
    """
    Publisher using Dynamic Programming for decision making.
    """
    
    def __init__(self, config, ledger):
        """Initialize the DP Publisher."""
        super().__init__(config, ledger)
        # Replace with DP agent
        self.agent = DPAgent() 