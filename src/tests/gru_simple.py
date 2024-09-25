import numpy as np
from tests.environment import Environment
import torch
import torch.nn as nn
import torch.nn.functional as F


rewards = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 10]
]


grid_size = len(rewards)
env = Environment(rewards)
env.set_exit(grid_size - 1, grid_size - 1)

# PL is a GRU that integrate goal information G(t), 
# context from the hippocampus C(t), 
# and feedback from the ACC (E(t), F(t)) to dynamically update 
# its task set representations T_i.

# Interaction with Hippocampus:
#       - Bidirectional connections with ventral CA1 and subiculum create a loop for continuous goal-context integration
#       - PL-to-CA1 projections modulate sharp-wave ripple events, influencing replay content


# ACC modeled as a separate GRU,
# taking in P(t), O(t), and computing Error signal: E(t),  conflict signal: F(t).
# E(t) = ||P(t) - O(t)||^2 + λ * ||∇h_t||^2  # Error signal computation
# F(t) = g(h_t)                          # Conflict signal computation

# Interaction with Hippocampus:
#       - Receives place cell input from dorsal CA1, allowing alignment of effort coding with specific spatial locations
#       - ACC-to-CA1 feedback influences the rate of theta sequences, potentially scaling mental exploration speed based on effort


class ACC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(ACC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Parameters for various computations
        self.effort_discount = nn.Parameter(torch.tensor(0.1))
        self.volatility_lr = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.9))  # Discount factor for EVC
        self.theta_0 = nn.Parameter(torch.tensor(0.9))  # For learning rate update
        self.theta_j = nn.Parameter(torch.tensor(0.1))  # For learning rate update
        
    def forward(self, predicted_outcomes, observed_outcomes, effort_level, previous_control_signals, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.init_hidden(predicted_outcomes.size(0))
        
        # 1. Reward Prediction Error (RPE)
        rpe = observed_outcomes - predicted_outcomes
        
        # 2. Expected Value of Control (EVC)
        immediate_reward = observed_outcomes
        future_evc = self.gamma * torch.max(previous_control_signals, dim=-1)[0]
        evc = immediate_reward + future_evc - self.effort_discount * effort_level
        
        # 3. Conflict Monitoring
        conflict = -torch.sum(predicted_outcomes * predicted_outcomes, dim=-1, keepdim=True)
        
        # 4. Surprise
        epsilon = 1e-10  # To avoid log(0)
        surprise = -torch.log2(predicted_outcomes + epsilon)
        
        # 5. Volatility Estimation
        volatility = self.volatility_lr * rpe
        
        # 6. Effort Discounting (already used in EVC calculation)
        subjective_value = observed_outcomes * (1 - self.effort_discount * effort_level)
        
        # 7. Cost-Benefit Ratio
        cbr = observed_outcomes / (effort_level + epsilon)
        
        # 8. Error-Related Negativity (ERN)
        ern = torch.mean(torch.abs(rpe), dim=-1, keepdim=True)
        
        # 9. Learning Rate Update
        prev_learning_rate = self.volatility_lr
        new_learning_rate = self.theta_0 * prev_learning_rate + self.theta_j * torch.abs(rpe.mean())
        self.volatility_lr.data = torch.clamp(new_learning_rate, min=0.01, max=1.0)
        
        # Combine all signals for GRU processing
        combined_signals = torch.cat([
            rpe, evc, conflict, surprise, volatility, subjective_value, cbr, ern
        ], dim=-1)
        
        # Process through GRU to maintain state
        out, hidden = self.gru(combined_signals, hidden_state)
        
        return {
            'rpe': rpe,
            'evc': evc,
            'conflict': conflict,
            'surprise': surprise,
            'volatility': volatility,
            'subjective_value': subjective_value,
            'cbr': cbr,
            'ern': ern,
            'learning_rate': self.volatility_lr.item(),
            'hidden': hidden
        }
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=next(self.parameters()).device)

class PrelimbicCortex(nn.Module):
    def __init__(self, goal_input_size, action_output_size, hidden_size, num_layers=1):
        super(PrelimbicCortex, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.goal_gru = nn.GRU(goal_input_size, hidden_size, num_layers, batch_first=True)
        self.action_fc = nn.Linear(hidden_size, action_output_size)
        
        self.update_gate = nn.Linear(hidden_size + 5, hidden_size)  # +5 for error and conflict signals
        self.reset_gate = nn.Linear(hidden_size + 5, hidden_size)
        self.candidate_state = nn.Linear(hidden_size + 5, hidden_size)
        
    def forward(self, goal_inputs, acc_error, acc_conflict, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.init_hidden(goal_inputs.size(0))
        
        # Process the goal inputs using the GRU
        goal_outputs, hidden = self.goal_gru(goal_inputs, hidden_state)
        goal_representation = goal_outputs[:, -1, :]
        
        # Combine ACC inputs
        acc_inputs = torch.cat((acc_error, acc_conflict), dim=-1)
        
        # Use the ACC inputs to control the update of the hidden state
        combined = torch.cat((hidden[-1], acc_inputs), dim=-1)
        update_gate = torch.sigmoid(self.update_gate(combined))
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        candidate_state = torch.tanh(self.candidate_state(torch.cat((reset_gate * hidden[-1], acc_inputs), dim=-1)))
        new_hidden = update_gate * hidden[-1] + (1 - update_gate) * candidate_state
        
        # Compute the action hierarchy prediction
        action_hierarchy = self.action_fc(new_hidden)
        
        return action_hierarchy, new_hidden.unsqueeze(0)
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=next(self.parameters()).device)




class RLGRUModel:
    def __init__(self, input_size, hidden_size, action_size, learning_rate=0.001):
            self.gru = GRUCell(input_size, hidden_size)
            self.W_action = np.random.randn(action_size, hidden_size) / np.sqrt(hidden_size)
            self.b_action = np.zeros((action_size, 1))
            self.W_reward = np.random.randn(1, hidden_size) / np.sqrt(hidden_size)
            self.b_reward = np.zeros((1, 1))
            self.learning_rate = learning_rate

    def predict(self, state, is_prediction=False):
            h, effort = self.gru.forward(state, is_prediction)
            action_probs = self.softmax(np.dot(self.W_action, h) + self.b_action)
            predicted_reward = np.dot(self.W_reward, h) + self.b_reward
            return action_probs, predicted_reward, self.gru.effort


    def calculate_re_value(self, reward, effort):
            return  reward / (1 + effort)

    def softmax(self, x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)

    def step(self, state, action, reward, next_state, done):
           
        #    Modulation of PL by ACC:
        # IPLACC(t)=g(∑m=1MU(am)⋅WmACC-PL)
        # IPLACC​(t)=g(m=1∑M​U(am​)⋅WmACC-PL​)

        #     IPLACC(t)IPLACC​(t): Input from the ACC to PL at time tt.
        #     U(am)U(am​): Utility of action amam​.
        #     WmACC-PLWmACC-PL​: Synaptic weight between ACC outputs and PL neurons.
        #     g(⋅)g(⋅): Modulation function that increases or decreases PL activity based on the computed utility.


input_size = grid_size * 2
# number of possible trajectories
hidden_size = grid_size * (grid_size - 1) * 2
action_size = 4

model = RLGRUModel(input_size, hidden_size, action_size, n_step_ahead=5)

# Training loop
num_episodes = 1000
action = 0
for episode in range(num_episodes):
    state, reward = env.reset()
    state = model.preprocess_state(state['image'])
    done = False
    total_reward = 0

    while not done:


            next_state, reward, done, coordinates, = env.step(action)

            action_probs = model.step(state, action, reward, next_state, done)

            action = np.random.choice(action_size, p=action_probs.flatten())

            state = next_state
            total_reward += reward

    if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}")

print("Training completed!")