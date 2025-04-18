"""
 @author suncj
 @email sun.cj@zhejianglab.com
 @create date 2023-09-20 16:40:00
 @modify date 2023-11-20 10:28:34
 @desc [description]
"""

import torch
import numpy as np

from torch import nn
from torch import optim

from .constants import DATASET_DIM, DEVICE, MODEL_DIM

# Define the dimensions of the state space and action space
ACTION_DIM = 7
STATE_DIM = (ACTION_DIM - 1) * (DATASET_DIM + MODEL_DIM) + 1  # add balance as state

# The dimensions of the discrete and continuous values in the state
DISCRETE_STATE_DIM = 2
CONTINUOUS_STATE_DIM = 4

# The dimensions of the discrete and continuous values in the action
DISCRETE_ACTION_DIM = 3
CONTINUOUS_ACTION_DIM = 4

# Define the hyperparameters
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft update coefficient


# pylint: disable=E1101


"""
Notice: 
    - This code demonstrates that the agent model architecture is modular and plug-and-play, allowing it to be replaced with alternative implementations.
    - The implementation below is barely a simple example of a PPO agent without any optimization. We would like to share the optimized version after the paper is accepted.
"""
class PPOPolicy(nn.Module):
    """Policy network for PPO

    Args:
        nn (Module): PyTorch Module
    """
    def __init__(self):
        super(PPOPolicy, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(STATE_DIM, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Discrete action output layer
        self.fc3_dis1 = nn.Linear(64, 3)  # no publish, model or dataset
        self.fc3_dis2 = nn.Linear(64, DATASET_DIM)  # select ref dataset
        self.fc3_dis3 = nn.Linear(64, MODEL_DIM)  # select ref model
        
        # Continuous action output layer
        self.fc3_cont1_mu = nn.Linear(64, 1)  # ref_dataset_ratio
        self.fc3_cont2_mu = nn.Linear(64, 1)  # ref_model_ratio
        self.fc3_cont3_mu = nn.Linear(64, 1)  # charge_fee
        self.fc3_cont4_mu = nn.Linear(64, 1)  # down_payment_ratio
        
        # Continuous action standard deviation
        self.fc3_cont1_sigma = nn.Linear(64, 1)
        self.fc3_cont2_sigma = nn.Linear(64, 1)
        self.fc3_cont3_sigma = nn.Linear(64, 1)
        self.fc3_cont4_sigma = nn.Linear(64, 1)
        
    def forward(self, state):
        """Forward pass through the network

        Args:
            state (tensor): State tensor

        Returns:
            tuple: All discrete distribution and continuous action parameters
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        # Discrete action distribution
        dis_out1 = torch.softmax(self.fc3_dis1(x), dim=-1)
        dis_out2 = torch.softmax(self.fc3_dis2(x), dim=-1)
        dis_out3 = torch.softmax(self.fc3_dis3(x), dim=-1)
        
        # Continuous action parameters (mean and standard deviation)
        cont1_mu = torch.sigmoid(self.fc3_cont1_mu(x))
        cont2_mu = torch.sigmoid(self.fc3_cont2_mu(x))
        cont3_mu = torch.sigmoid(self.fc3_cont3_mu(x))
        cont4_mu = torch.sigmoid(self.fc3_cont4_mu(x))
        
        cont1_sigma = torch.exp(self.fc3_cont1_sigma(x))
        cont2_sigma = torch.exp(self.fc3_cont2_sigma(x))
        cont3_sigma = torch.exp(self.fc3_cont3_sigma(x))
        cont4_sigma = torch.exp(self.fc3_cont4_sigma(x))
        
        return (
            dis_out1, dis_out2, dis_out3,
            cont1_mu, cont2_mu, cont3_mu, cont4_mu,
            cont1_sigma, cont2_sigma, cont3_sigma, cont4_sigma
        )
    
    def get_action(self, state, deterministic=False):
        """Sample an action from the policy

        Args:
            state (tensor): State tensor
            deterministic (bool, optional): If True, return deterministic action. Defaults to False.

        Returns:
            tuple: Sampled discrete and continuous actions
        """
        state = torch.FloatTensor(state).to(DEVICE)
        with torch.no_grad():
            (
                dis1, dis2, dis3,
                cont1_mu, cont2_mu, cont3_mu, cont4_mu,
                cont1_sigma, cont2_sigma, cont3_sigma, cont4_sigma
            ) = self(state)
            
            # Discrete action sampling
            if deterministic:
                dis_act1 = torch.argmax(dis1).item()
                dis_act2 = torch.argmax(dis2).item()
                dis_act3 = torch.argmax(dis3).item()
            else:
                dis_act1 = torch.multinomial(dis1, 1).item()
                dis_act2 = torch.multinomial(dis2, 1).item()
                dis_act3 = torch.multinomial(dis3, 1).item()
            
            # Continuous action sampling
            if deterministic:
                cont_act1 = cont1_mu.item()
                cont_act2 = cont2_mu.item()
                cont_act3 = cont3_mu.item()
                cont_act4 = cont4_mu.item()
            else:
                cont_act1 = torch.normal(cont1_mu, cont1_sigma).clamp(0, 1).item()
                cont_act2 = torch.normal(cont2_mu, cont2_sigma).clamp(0, 1).item()
                cont_act3 = torch.normal(cont3_mu, cont3_sigma).clamp(0, 1).item()
                cont_act4 = torch.normal(cont4_mu, cont4_sigma).clamp(0, 1).item()
            
            # Apply the action mask logic
            if cont_act4 == 1.0:
                cont_act3 = 0.0
                
        return dis_act1, dis_act2, dis_act3, cont_act1, cont_act2, cont_act3, cont_act4
    
    def evaluate_action(self, state, action):
        """Evaluate the log probability and entropy of an action

        Args:
            state (tensor): State tensor
            action (tensor): Action tensor

        Returns:
            tuple: Log probs and entropy
        """
        (
            dis1, dis2, dis3,
            cont1_mu, cont2_mu, cont3_mu, cont4_mu,
            cont1_sigma, cont2_sigma, cont3_sigma, cont4_sigma
        ) = self(state)
        
        # Discrete action log probability and entropy
        dis_act1, dis_act2, dis_act3 = action[:, 0].long(), action[:, 1].long(), action[:, 2].long()
        
        dis1_log_prob = torch.log(torch.gather(dis1, 1, dis_act1.unsqueeze(1))).squeeze()
        dis2_log_prob = torch.log(torch.gather(dis2, 1, dis_act2.unsqueeze(1))).squeeze()
        dis3_log_prob = torch.log(torch.gather(dis3, 1, dis_act3.unsqueeze(1))).squeeze()
        
        dis1_entropy = -torch.sum(dis1 * torch.log(dis1 + 1e-10), dim=1)
        dis2_entropy = -torch.sum(dis2 * torch.log(dis2 + 1e-10), dim=1)
        dis3_entropy = -torch.sum(dis3 * torch.log(dis3 + 1e-10), dim=1)
        
        # Continuous action log probability and entropy
        cont_act1, cont_act2, cont_act3, cont_act4 = action[:, 3], action[:, 4], action[:, 5], action[:, 6]
        
        cont1_distr = torch.distributions.Normal(cont1_mu.squeeze(), cont1_sigma.squeeze())
        cont2_distr = torch.distributions.Normal(cont2_mu.squeeze(), cont2_sigma.squeeze())
        cont3_distr = torch.distributions.Normal(cont3_mu.squeeze(), cont3_sigma.squeeze())
        cont4_distr = torch.distributions.Normal(cont4_mu.squeeze(), cont4_sigma.squeeze())
        
        cont1_log_prob = cont1_distr.log_prob(cont_act1)
        cont2_log_prob = cont2_distr.log_prob(cont_act2)
        cont3_log_prob = cont3_distr.log_prob(cont_act3)
        cont4_log_prob = cont4_distr.log_prob(cont_act4)
        
        cont1_entropy = cont1_distr.entropy()
        cont2_entropy = cont2_distr.entropy()
        cont3_entropy = cont3_distr.entropy()
        cont4_entropy = cont4_distr.entropy()
        
        # Total log probability and entropy
        log_prob = dis1_log_prob + dis2_log_prob + dis3_log_prob + cont1_log_prob + cont2_log_prob + cont3_log_prob + cont4_log_prob
        entropy = dis1_entropy + dis2_entropy + dis3_entropy + cont1_entropy + cont2_entropy + cont3_entropy + cont4_entropy
        
        return log_prob, entropy


class PPOValueNetwork(nn.Module):
    """Value network for PPO

    Args:
        nn (Module): PyTorch Module
    """
    def __init__(self):
        super(PPOValueNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state):
        """Forward pass to get state value

        Args:
            state (tensor): State tensor

        Returns:
            tensor: State value
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, clip_param=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        """Initialize PPO agent

        Args:
            clip_param (float, optional): PPO clip parameter. Defaults to 0.2.
            value_coef (float, optional): Value loss coefficient. Defaults to 0.5.
            entropy_coef (float, optional): Entropy coefficient. Defaults to 0.01.
            max_grad_norm (float, optional): Max gradient norm for clipping. Defaults to 0.5.
        """
        self.policy = PPOPolicy().to(DEVICE)
        self.value_network = PPOValueNetwork().to(DEVICE)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.0003)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)
        
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
    def action(self, state):
        """Get action from policy

        Args:
            state (numpy.ndarray): State array

        Returns:
            tuple: Discrete and continuous actions
        """
        return self.policy.get_action(state)
    
    def train(self, rollouts):
        """Train the PPO agent

        Args:
            rollouts (object): Object containing collected experience
                with state, action, reward, next_state, done, old_log_probs
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array(rollouts.state)).to(DEVICE)
        actions = torch.FloatTensor(np.array(rollouts.action)).to(DEVICE)
        returns = torch.FloatTensor(np.array(rollouts.returns)).unsqueeze(1).to(DEVICE)
        advantages = torch.FloatTensor(np.array(rollouts.advantages)).unsqueeze(1).to(DEVICE)
        old_log_probs = torch.FloatTensor(np.array(rollouts.log_probs)).to(DEVICE)
        
        # Multiple updates of the policy and value network
        for _ in range(10):  # Usually PPO does multiple updates
            # Evaluate the action under the current policy
            new_log_probs, entropy = self.policy.evaluate_action(states, actions)
            
            # Calculate the ratio of the policy
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy loss, used to encourage exploration
            entropy_loss = -entropy.mean()
            
            # Value network loss
            values = self.value_network(states)
            value_loss = nn.MSELoss()(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize the policy network
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
    
    def compute_returns_and_advantages(self, batch, gamma=0.99, lam=0.95):
        """Compute GAE returns and advantages

        Args:
            batch (object): Object containing collected experience
            gamma (float, optional): Discount factor. Defaults to 0.99.
            lam (float, optional): GAE parameter. Defaults to 0.95.
            
        Returns:
            tuple: Returns and advantages arrays
        """
        # Convert data to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(DEVICE)
        rewards = torch.FloatTensor(np.array(batch.reward)).to(DEVICE)
        dones = torch.FloatTensor(np.array(batch.done)).to(DEVICE)
        
        # Calculate the value of the current state and the next state
        with torch.no_grad():
            values = self.value_network(states).squeeze()
            next_values = self.value_network(next_states).squeeze()
        
        # Calculate TD error
        deltas = rewards + gamma * next_values * (1 - dones) - values
        
        # Calculate GAE
        advantages = torch.zeros_like(rewards).to(DEVICE)
        returns = torch.zeros_like(rewards).to(DEVICE)
        
        last_gae = 0
        last_return = 0
        
        # Backwards computation
        for t in reversed(range(len(rewards))):
            last_gae = deltas[t] + gamma * lam * (1 - dones[t]) * last_gae
            last_return = rewards[t] + gamma * (1 - dones[t]) * last_return
            
            advantages[t] = last_gae
            returns[t] = last_return
        
        return returns.cpu().numpy(), advantages.cpu().numpy()


def soft_update(target, source):
    """
    Helper function for soft update of target network

    Args:
        target (_type_): _description_
        source (_type_): _description_
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)
