import torch
import torch.nn as nn
import torch.optim as optim

class BoostModule(nn.Module):
    def __init__(self, state_dim, boost_levels):
        super(BoostModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, boost_levels)
        self.softmax = nn.Softmax(dim=1)
        self.Tboost = 0.3  # Temperature for boost softmax

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        boost_probs = self.softmax(x / self.Tboost)
        return boost_probs

class ActionSelectionModule(nn.Module):
    def __init__(self, state_dim, action_dim, boost_levels):
        super(ActionSelectionModule, self).__init__()
        self.fc1 = nn.Linear(state_dim + boost_levels, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=1)
        self.Tact = 0.5  # Temperature for action softmax
        self.cost = nn.Parameter(torch.randn(action_dim)) # Cost of each action

    def forward(self, state, boost_probs):
        # Combine state and boost intensity
        x = torch.cat((state, boost_probs), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        action_probs = self.softmax(x / self.Tact - (self.cost / boost_probs)) # Factor in effort cost 
        return action_probs

class RML(nn.Module):
    def __init__(self, state_dim, action_dim, boost_levels):
        super(RML, self).__init__()
        self.boost_module = BoostModule(state_dim, boost_levels)
        self.action_selection_module = ActionSelectionModule(state_dim, action_dim, boost_levels)

        # Kalman Filter parameters
        self.t_smoothing = 0.3  # Smoothing parameter for Kalman filter
        self.min_lr = 0.15 # Minimum learning rate
        self.td_discount = 0.99  # Discount factor for TD error
        self.variance = torch.zeros(action_dim)
        self.PE = torch.zeros(action_dim)
        self.boost_cost_unit = 0.001  # Boost cost unit

    def forward(self, state):
        boost_probs = self.boost_module(state)
        action_probs = self.action_selection_module(state, boost_probs)
        return action_probs

    def update_learning_rate(self, Vact, exper_val_of_action, learning_rate):
        # Calculate prediction error
        self.PE = (1 - self.t_smoothing) * self.PE + self.t_smoothing * torch.abs(exper_val_of_action - Vact)

        # Calculate estimated variance
        self.variance = (1 - self.t_smoothing) * self.variance + self.t_smoothing * torch.square(Vact - Vact.mean())

        # Calculate learning rate using Kalman filter
        new_learning_rate = torch.max(torch.div(self.variance, torch.square(self.PE), out=torch.zeros_like(self.variance)), self.min_lr)
        return new_learning_rate

    def train_step(self, state, action, reward, next_state, learning_rate):
            optimizer.zero_grad()

            # Forward pass
            # Calculate the boost probs from the state
            boost_probs = self.boost_module(state)

            # Calculate the action probs from the state and boost probs
            # x = nn(cat(state, boost_probs))
            # action probs = softmax(x / self.Tact - (action costs/ boost_probs))
            action_probs = self.action_selection_module(state, boost_probs)

            # Select the boost intensity based on boost_probs
            boost_intensity = torch.argmax(boost_probs, dim=1)
            
            # Calculate the discounted future reward ---------------

            # 1. Calculate the action probs and boost probs from the next state
            next_action = torch.max(self.action_selection_module(next_state, boost_probs), dim=1)[0]
            next_boost = torch.max(self.boost_module(next_state), dim=1)[0]
            
            rt = 0
            if reward != 0:
                rt = 1
            # 
            # VTA = (R + (mu * b)) + (1 - mu) * b * gamma * max(state_action_value);
            # D(s,resp) = VTA - V(s,resp) %TD learning for Action

            exper_val_of_action = (rt * (reward + (self.DA_dynamics * boost_intensity))) + (boost_intensity * (1 - self.DA_dynamics) * self.td_discount * next_action)
            da = (reward + (y * boost)) + (boost * (1 - y) * g * next_action)
            # experienced value of action = reward + boost +
            exper_val_of_boost = (rt * reward) - (self.boost_cost_unit * boost_intensity) + next_boost 

            # Calculate loss as the sum of negative log likelihood of chosen action and RPE 
            loss = -torch.log(action_probs[0, action]) + torch.square(exper_val_of_action - action_probs[0, action]) + torch.square(exper_val_of_boost - boost_probs[0, boost_intensity]) # Add the RPE to the loss

            # Update learning rate
            new_learning_rate = self.update_learning_rate(action_probs, exper_val_of_action, learning_rate)

            # Backward pass
            loss.backward()
            optimizer.step()

            return loss.item(), new_learning_rate


# Initialize environment, state, and action dimensions
state_dim = 10 
action_dim = 5
boost_levels = 4

# Instantiate the RML model
model = RML(state_dim, action_dim, boost_levels)

# Initialize optimizer
optimizer = optim.Adam(model.parameters())

# Training loop (example)
for epoch in range(1000):
    # Sample state, action, reward, next_state from environment
    state = torch.randn(1, state_dim)
    action = torch.randint(0, action_dim, (1,))
    reward = torch.randn(1)
    next_state = torch.randn(1, state_dim)

    # Train the model 
    loss, new_learning_rate = model.train_step(state, action, reward, next_state, learning_rate)

    # Print loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Learning Rate: {new_learning_rate.mean():.4f}")