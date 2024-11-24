import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BoostModule(nn.Module):
    def __init__(self, num_boost_levels, state_size):
        super(BoostModule, self).__init__()
        self.fc1 = nn.Linear(state_size, num_boost_levels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        boost_values = self.fc1(state)
        boost_probs = self.softmax(boost_values)
        return boost_probs

class ActionSelectionModule(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActionSelectionModule, self).__init__()
        self.fc1 = nn.Linear(state_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, boost_level):
        action_values = self.fc1(state) - boost_level  # Effort cost influences action values
        action_probs = self.softmax(action_values)
        return action_probs

class ControlModule(nn.Module):
    def __init__(self, action_size):
        super(ControlModule, self).__init__()
        self.fc1 = nn.Linear(action_size, 1)  # Output a reward prediction

    def forward(self, action_taken):
        reward_pred = self.fc1(action_taken)
        return reward_pred


class RML(nn.Module):
    def __init__(self, num_states, num_actions, const):
        super(RML, self).__init__()
        # Constants from MATLAB code
        self.alpha = const['alpha']
        self.beta = const['beta']
        self.eta = const['eta']
        self.Temp = const['temp']
        self.gamma = const['gamma']
        self.omega = const['omega']
        self.mu = const['mu']
        self.NElesion = const['NElesion']
        self.DAlesion = const['DAlesion']
        
        # Value matrices
        self.V = torch.zeros(num_states + 1, num_actions)  # State-action value
        self.costs = const['costs']
        
        # Variance matrices
        self.varK = torch.full((num_states, num_actions), 0.3)
        self.varD = torch.full((num_states, num_actions), 0.5)
        self.varV = torch.full((num_states, num_actions), 0.1)
        self.varV2 = torch.full((num_states, num_actions), 0.5)
        
        # Prediction error
        self.D = torch.zeros(num_states, num_actions)
        
        # Kalman Gain
        self.k = const['k']
        
        # State-action transition trace
        self.sat = torch.zeros(num_states, 1)

    def action(self, state, available_actions, b):
        available_actions = available_actions.reshape(-1)
        res = self.V[state, available_actions] - self.costs[state, available_actions] / (b * self.NElesion)

        # Softmax-based action selection
        p = F.softmax(res / self.Temp, dim=0)
        
        # Sample an action based on softmax probabilities
        action = torch.multinomial(p, 1).item()
        
        # Update state-action trace
        self.sat[state] = 1
        
        return action

    def action_eval(self, state, available_actions, b, action_taken):
        available_actions = available_actions.reshape(-1)
        res = self.V[state, available_actions] - self.costs[state, available_actions] / (b * self.NElesion)
        
        # Softmax probability for the chosen action
        p = F.softmax(res / self.Temp, dim=0)
        return p[action_taken]

    def KG_update(self, state, action):
        # Kalman Gain update logic
        self.varD[state, action] += self.alpha * (torch.abs(self.D[state, action]) - self.varD[state, action])
        self.varV2[state, action] += self.alpha * (self.V[state, action] - self.varV2[state, action])
        self.varV[state, action] = ((self.V[state, action] - self.varV2[state, action]) ** 2) / (self.varD[state, action] ** 2)

        # Ensure Kalman gain is between 0 and 1
        self.varV[state, action] = torch.clamp(self.varV[state, action], 0, 1)
        self.varK[state, action] = self.varV[state, action]

        # Mean Kalman gain across actions
        self.k = torch.mean(self.varK[state, :])

    def learn(self, rw, state, next_state, action, RW, b):
        # Learning update (prediction error and value update)
        if self.omega == 0:  # ACC Action case
            R = rw * self.DAlesion * (RW + self.mu * b)
            VTA = R + (1 - self.mu) * b * self.gamma * self.DAlesion * torch.max(self.V[next_state, :])
        else:  # ACC Boost case
            R = rw * RW * self.DAlesion - self.omega * b
            VTA = R + self.DAlesion * torch.max(self.V[next_state, :])

        self.D[state, action] = VTA - self.V[state, action]  # TD learning

        # Kalman gain update
        self.KG_update(state, action)
        
        # Kalman gain-based learning rate adjustment
        learning_rate = torch.clamp(self.k, self.eta, 1)
        
        # Value update
        self.V[state, action] += learning_rate * self.D[state, action]

# Define constants from MATLAB
const = {
    'alpha': 0.3,
    'beta': 0.5,
    'eta': 0.1,
    'temp': 1.0,
    'gamma': 0.9,
    'omega': 0.5,
    'mu': 0.1,
    'NElesion': 1.0,
    'DAlesion': 1.0,
    'costs': torch.randn(10, 5),  # Replace with actual cost matrix
    'k': 0.5
}

# Instantiate model
num_states = 10
num_actions = 5
model = RML(num_states, num_actions, const)

# Sample state and available actions
state = 0
available_actions = torch.tensor([0, 1, 2, 3, 4])
b = 0.8  # Boost level
rw = 1.0
RW = 1.0
next_state = 1

# Action selection
action_taken = model.action(state, available_actions, b)
print(f"Selected Action: {action_taken}")

# Learning update
model.learn(rw, state, next_state, action_taken, RW, b)
print(f"Updated Value: {model.V[state, action_taken]}")
