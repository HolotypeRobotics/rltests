from her_environment import Environment
from her import HER
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


"""
Inputs to the agent should be the same form as the outputs from the bottom layer.
(It should predict its own inputs.)
The input/predicted features should include 1` hot encoding of 
 - the action taken (turn left/right, move forward)
 - the current position(x/y cell)
 - direction(up/down/left/right)
 - the object distances
 - Timescale features 

 Later on, we will add the action to inhibit external input, so that the inputs to the agent could be purely predictive.
 this allows the agent to formulate a plan
 We will use this to experiment with how planning loops work, 
 e.g. what determines when to stop predicting? is it determined by local desicion points?
 ultimately we want to produce foraging type behivior, where the agent tracks value of current patch vs other local patches.
"""



class HER_Agent():
    def __init__(self, grid_size, n_layers, n_hidden, beta, epsilon, bias, gate_alpha, layer_alpha, gamma, lambda_):

        self.epsilon = epsilon
        self.beta = beta
        self.bias = bias
        self.gate_alpha = gate_alpha
        self.layer_alpha = layer_alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        # Initialize environment with random parameters
        n_rewards = np.random.randint(0, 10)
        n_efforts = np.random.randint(1, 5)
        n_objects = np.random.randint(0, 3)
        n_distances = np.random.randint(1, 5)
        self.env = Environment( grid_size, grid_size, n_rewards, n_efforts, n_objects)

        # Initialize HER model
        n_layers = n_layers
        n_hidden = n_hidden  # Example hidden size
        n_responses = 3 # turn left, right, move forward
        n_directions = 4 # up, down, left, right
        # All 1 hot width + height, + orientation + previous action + object distances
        n_stimuli = (grid_size * 2) + n_directions + n_responses + (n_objects * n_distances)
        n_outcomes = n_stimuli # output should be the same as input
        self.model = HER(n_layers=n_layers,
                         n_hidden=n_hidden,
                         n_stimuli=n_stimuli,
                         n_outcomes=n_outcomes,
                         n_responses=n_responses,
                         beta=beta, 
                         bias=bias,
                         gate_alpha=gate_alpha, 
                         layer_alpha=layer_alpha,
                         gamma=gamma,
                         lambda_=lambda_)
        
        self.env = Environment() 

        self.optimizer = optim.Adam(self.model.parameters(), lr=layer_alpha)

    def run_episode(self, steps):
        state, reward, done = self.env.reset()
        total_reward = 0
        log_probs = []
        rewards = []
        rs = [] # Store working memory representations
        ps = [] # Store predictions

        for t in range(steps):
            if done:
                break

            action_logits = self.model(state)  # Remove torch.no_grad() for training
            action_probs = F.softmax(action_logits, dim=-1)

            # Sample action (not argmax) for policy gradient
            action_dist = torch.distributions.Categorical(probs=action_probs)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action) # Store log probability for REINFORCE
            log_probs.append(log_prob)
            # Todo: the previous action is always 0 in the environment. must fix.
            next_state, reward, done = self.env.step(action.item())

            # Store r and p for backpropagation
            rs.append(self.model.WM.clone()) # Store WM representations
            ps.append(action_logits.clone()) # Store predictions (logits)

            rewards.append(reward)
            total_reward += reward
            state = next_state

        # REINFORCE update (after the episode)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) # Normalize returns

        policy_loss = 0
        for log_prob, R in zip(log_probs, returns):
            policy_loss -= log_prob * R

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # HER backpropagation
        self.model.backward(state, action.item(), reward, rs, ps)

        return total_reward

n_episodes = 100
n_steps_per_episode = 100
grid_size = 10
n_layers = 3
n_hidden = 10
beta = 0.001
bias = 0.0
gate_alpha = 0.001
layer_alpha = 0.001
gamma = 0.95
lambda_ = 0.95
epsilon = 0.1

"""
acc predictions
future value, effort, error, control


"""


agent = HER_Agent(grid_size, 
                n_layers=n_layers, 
                n_hidden=n_hidden, 
                beta=beta, 
                bias=bias,
                gate_alpha=gate_alpha, 
                layer_alpha=layer_alpha, 
                gamma=gamma,
                epsilon=epsilon,
                lambda_=lambda_)

if __name__ == "__main__":
    for i in range(n_episodes):
        print(f"----------Episode: {i}----------")
        agent.run_episode(n_steps_per_episode)