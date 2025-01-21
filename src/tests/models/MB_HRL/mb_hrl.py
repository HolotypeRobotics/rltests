import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import torchviz

torch.autograd.set_detect_anomaly(True)

# Constants (Hyperparameters)
GAMMA = 0.99  # Discount factor
N_OPTIONS = 4  # Number of options
N_LAYERS = 2  # Number of layers in the hierarchy
CONTROL_LR = 0.01
HABITUAL_LR = 0.001
ERROR_ALPHA = 0.1
N_MAX_LAYER_ITERATIONS = 10
ATTENTION_DIVERSITY_WEIGHT = 1.0  # Weight for attention diversity loss
ATTENTION_SPARSITY_WEIGHT = 0.1  # Weight for attention sparsity loss
DELIBERATION_COST = 0.01  # Cost for switching options

class AttentionMechanism(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.attention_layer = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # x: [batch_size, input_size]
        attn_weights = torch.tanh(self.attention_layer(x))
        # attn_weights: [batch_size, hidden_size]
        return attn_weights

class HierarchicalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_options, n_layers,
                 control_lr=0.01, habitual_lr=0.001, error_alpha=0.1, n_max_layer_iterations=10):
        super(HierarchicalGRU, self).__init__()
        self.input_size = input_size
        self.n_options = n_options
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.error_alpha = error_alpha
        self.average_error = defaultdict(float)
        self.control_gains = defaultdict(lambda: 0.5)
        self.n_max_layer_iterations = n_max_layer_iterations
        self.current_option = None

        self.attention = AttentionMechanism(input_size, hidden_size)

        # Use GRU instead of GRUCell
        self.grus = nn.ModuleList()
        self.grus.append(nn.GRU(hidden_size, hidden_size, batch_first=True))
        for i in range(1, n_layers):
            self.grus.append(nn.GRU(hidden_size * 2, hidden_size, batch_first=True))

        self.value_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_options) for _ in range(n_layers)
        ])

        self.termination_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_options) for _ in range(n_layers - 1)
        ])

        self.control_optimizer = optim.Adam(
            list(self.value_heads.parameters()) +
            list(self.termination_heads.parameters()) +
            list(self.attention.parameters()),
            lr=control_lr
        )
        self.habitual_optimizer = optim.Adam(self.grus.parameters(), lr=habitual_lr)

    def apply_gain_contrast(self, tensor, gain):
        centered_tensor = tensor - 0.5
        scaled_tensor = centered_tensor * gain
        contrasted = scaled_tensor + 0.5
        return contrasted

    def select_option(self, option_probabilities, chosen_option):
        dist = torch.distributions.Categorical(logits=option_probabilities)
        new_option_index = dist.sample()

        # Clamp the sampled option index to ensure it's within the valid range
        new_option_index = torch.clamp(new_option_index, 0, self.n_options - 1)

        if self.current_option is not None and new_option_index != self.current_option:
            deliberation_cost = DELIBERATION_COST
        else:
            deliberation_cost = 0
        self.current_option = new_option_index.item()  # Convert to a scalar (Python number)

        # Create a new tensor for chosen_option instead of modifying it in-place
        new_chosen_option = torch.tensor(self.current_option, dtype=torch.long, device=option_probabilities.device)

        return new_chosen_option, deliberation_cost

    def update_control_gain(self, layer_idx, reward, value):
        with torch.no_grad():
            prediction_error = reward - value.item()
            self.average_error[layer_idx] = (1 - self.error_alpha) * self.average_error[layer_idx] + self.error_alpha * prediction_error
            self.control_gains[layer_idx] = max(0, 1.0 - abs(self.average_error[layer_idx]))

    def terminate_option(self, termination_prob):
        return termination_prob > random.random()

    def forward_layer(self, layer_idx, x, hidden_state, chosen_option, obs):

        # Apply attention to the input observation (obs)
        attn_weights = self.attention(obs)  # attn_weights: [batch_size, hidden_size]

        # Apply attention weights
        x_weighted = attn_weights  # x_weighted: [batch_size, hidden_size]

        # Get input for the current layer
        if layer_idx == 0:
            input_tensor = x_weighted.unsqueeze(1)  # Input for the first layer is the weighted observation
        else:
            input_tensor = torch.cat((x_weighted, hidden_state.squeeze(0)), dim=-1).unsqueeze(1)

        # Use GRU
        _, new_hidden_state = self.grus[layer_idx](input_tensor, hidden_state)

        # Apply gain contrast to a copy of the hidden state to avoid in-place modification
        hidden_state_copy = new_hidden_state.clone()
        hidden_state_copy = self.apply_gain_contrast(hidden_state_copy, self.control_gains[layer_idx])

        # Get value, effort, and termination outputs
        values = self.value_heads[layer_idx](hidden_state_copy.squeeze(0))

        if layer_idx < self.n_layers - 1:
            termination_prob = torch.sigmoid(self.termination_heads[layer_idx](hidden_state_copy.squeeze(0))).gather(1, chosen_option.unsqueeze(1))
        else:
            termination_prob = None

        return values, termination_prob, new_hidden_state, new_hidden_state # returning new_hidden_state twice since we removed the output variable
    
    def process_event(self, layer_idx, x, hidden_states, chosen_option, obs, reward=None, effort=None, timestep=0, accumulated_deliberation_cost=0):
        # Ensure chosen_option is a tensor
        if not isinstance(chosen_option, torch.Tensor):
            chosen_option = torch.tensor(chosen_option, dtype=torch.long, device=x.device)
        elif chosen_option.device != x.device:
            chosen_option = chosen_option.to(x.device)

        # Forward pass for the current layer
        values, termination_output, hidden_states[layer_idx], output = self.forward_layer(
            layer_idx, x, hidden_states[layer_idx], chosen_option, obs
        )

        # Option selection logic for higher layers (before recursion)
        if layer_idx < self.n_layers - 1:
            if self.terminate_option(termination_output):
                option_probabilities = values  # Use value as a proxy for option selection
                new_chosen_option, deliberation_cost = self.select_option(option_probabilities, chosen_option)
                accumulated_deliberation_cost += deliberation_cost

                # Update hidden state for the new option
                new_hidden_states = []
                for i in range(self.n_layers):
                    if i == layer_idx:
                        new_hidden_states.append(self.init_hidden(x.size(0))[i])
                    else:
                        new_hidden_states.append(hidden_states[i])
                hidden_states = new_hidden_states
                chosen_option = new_chosen_option  # Update chosen_option

        # Recursive call for lower layers
        if layer_idx < self.n_layers - 1:
            _, lower_values, hidden_states = self.process_event(
                layer_idx + 1, x, hidden_states, chosen_option, obs, reward, effort, timestep, accumulated_deliberation_cost
            )
            timestep += 1

        # Update control gain based on reward prediction error (both layers)
        if reward is not None:
            self.update_control_gain(layer_idx, reward, values.gather(1, chosen_option.unsqueeze(1)))

        # Ensure chosen_option is always valid
        chosen_option = torch.clamp(chosen_option, 0, self.n_options - 1)

        return output, values, hidden_states

    def init_hidden(self, batch_size):
        # Adjust the initialization for GRU, which has a different hidden state shape
        return [torch.zeros(1, batch_size, self.hidden_size, device=self.grus[0].weight_hh_l0.device) for _ in range(self.n_layers)]

    def update_layer_weights(self, layer_idx, obs, chosen_option, reward, effort, next_obs, done, hidden_state):
        # habitual network update
        self.habitual_optimizer.zero_grad()
        if layer_idx < self.n_layers - 1:
            self.control_optimizer.zero_grad()

        # Clone hidden_state to avoid in-place modifications
        hidden_state_clone = hidden_state.detach().clone()


        # Forward pass for the current layer
        values, termination_output, new_hidden_state, _ = self.forward_layer(
            layer_idx, obs.unsqueeze(1), hidden_state_clone, chosen_option, obs
        )

        # Get the target option from the model's output
        target_option = values.argmax(dim=-1)

        # Compute loss for value function using TD error
        with torch.no_grad():
            next_values, _, _, _ = self.forward_layer(
                layer_idx,
                next_obs.unsqueeze(1),
                new_hidden_state.detach().clone(),
                # new_hidden_state,
                target_option,
                next_obs
            )

            # Compute target Q-value with proper handling of termination
            if done:
                target_q = torch.tensor([reward], dtype=torch.float, device=values.device)
            else:
                target_option = torch.clamp(target_option, 0, self.n_options - 1)
                next_q = next_values.max(dim=1, keepdim=True)[0]

                if layer_idx == 0:
                    current_return = torch.tensor([reward], dtype=torch.float, device=values.device) - DELIBERATION_COST

                    # Ensure chosen_option has the correct shape for gather
                    chosen_option = chosen_option.view(-1, 1)

                    continue_value = (1 - termination_output) * next_values.gather(1, chosen_option)
                    switch_value = next_values.max(dim=1, keepdim=True)[0] - DELIBERATION_COST
                    target_q = current_return + GAMMA * torch.max(continue_value, switch_value)
                else:
                    target_q = torch.tensor([reward], dtype=torch.float, device=values.device) - effort + GAMMA * next_q

        # Ensure chosen_option has the correct shape for gathering
        chosen_option = chosen_option.view(-1, 1)
        q_values = values.gather(1, chosen_option).squeeze(1)

        # Make sure target_q has the same shape as q_values
        target_q = target_q.view_as(q_values)  # Reshape target_q

        value_loss = F.mse_loss(q_values, target_q)

        # Combine losses if it's a higher layer
        total_loss = value_loss
        if layer_idx < self.n_layers - 1:
            # Compute advantage
            advantage = values.gather(1, chosen_option) - values.max(1, keepdim=True)[0]

            # Compute loss for policy over options
            policy_loss = -torch.log(F.softmax(values, dim=-1).gather(1, chosen_option)).mean()

            # Compute loss for termination function
            termination_loss = -torch.log(1 - termination_output.clamp(min=1e-8)).mean() if advantage.mean() < 0 else torch.tensor(0.0, requires_grad=True, device=values.device)

            # Compute loss for attention diversity
            attn_weights = self.attention(obs)
            attention_diversity_loss = 0
            for i in range(self.n_options):
                for j in range(i + 1, self.n_options):
                    attention_diversity_loss += F.cosine_similarity(attn_weights[:, i], attn_weights[:, j], dim=0)

            # Compute loss for attention sparsity
            attention_sparsity_loss = torch.norm(attn_weights, p=1, dim=-1).mean()

            # Combine control losses
            total_control_loss = (
                policy_loss
                + termination_loss
                + ATTENTION_DIVERSITY_WEIGHT * attention_diversity_loss
                + ATTENTION_SPARSITY_WEIGHT * attention_sparsity_loss
            )
            total_loss += total_control_loss

        # Update weights
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.habitual_optimizer.step()
        if layer_idx < self.n_layers - 1:
            self.control_optimizer.step()



    def forward(self, x, chosen_option, hidden_states=None):
        obs = x
        if hidden_states is None:
            hidden_states = self.init_hidden(x.size(0))

        # Process the event through the hierarchy
        final_output, values, _ = self.process_event(0, x, hidden_states, chosen_option, obs)

        # Return both final_output and values from the lowest layer
        return final_output, values

# Test parameters
N_EPISODES = 10
MAX_STEPS = 50
EPSILON = 0.3
TEST_EVERY_N_EPISODES = 10

# Define a simple grid world environment
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)  # Start at bottom left
        self.goal = (size - 1, size - 1)  # Goal at top right

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            next_state = (x, min(y + 1, self.size - 1))
        elif action == 1:  # Down
            next_state = (x, max(y - 1, 0))
        elif action == 2:  # Left
            next_state = (max(x - 1, 0), y)
        elif action == 3:  # Right
            next_state = (min(x + 1, self.size - 1), y)
        else:
            next_state = self.state  # Invalid action

        self.state = next_state
        reward = 1 if next_state == self.goal else 0
        done = next_state == self.goal
        effort = 0.1 if action < 4 else 0  # Assume some effort for movement actions
        return next_state, reward, done, effort

    def get_observation(self):
        # One-hot encoding of the state
        obs = torch.zeros(1, self.size * self.size)
        obs[0, self.state[0] * self.size + self.state[1]] = 1
        return obs

# Initialize environment and agent
env = GridWorld()
INPUT_SIZE = env.size * env.size  # One-hot encoding of states
HIDDEN_SIZE = 6
model = HierarchicalGRU(INPUT_SIZE, HIDDEN_SIZE, N_OPTIONS, N_LAYERS)

# Function to run an episode and collect data for training
def run_episode(model, env, epsilon=0.3):
    obs = env.reset()
    obs = env.get_observation()
    done = False
    hidden_states = model.init_hidden(1)
    episode_data = []
    total_reward = 0
    total_effort = 0
    steps = 0
    accumulated_deliberation_cost = 0

    chosen_option = torch.tensor([random.randint(0, N_OPTIONS - 1)], dtype=torch.long, device=obs.device)

    while not done and steps < MAX_STEPS:
        # Get model output for the current layer
        output, values = model.forward(obs, chosen_option, hidden_states)

        # Select action based on the chosen option's policy
        if random.random() < epsilon:
            action = torch.tensor([random.randint(0, 3)], device=obs.device)
        else:
            action = output[0].argmax(dim=-1).unsqueeze(-1)

        # Take the chosen action in the environment
        next_obs, reward, done, effort = env.step(action.item())
        next_obs = env.get_observation()

        # Get the target option from the model's output (use values from the lowest layer)
        target_option = values.argmax(dim=-1)

        # Apply accumulated deliberation cost
        reward = reward - accumulated_deliberation_cost
        accumulated_deliberation_cost = 0  # Reset for the next step

        # Store data for training (including target_option)
        episode_data.append(
            (obs, chosen_option.clone(), reward, effort, next_obs, done, target_option, hidden_states[0].clone())
        )

        # Update current state, chosen option, and hidden states
        obs = next_obs
        chosen_option = target_option.clone()
        hidden_states = [h.detach().clone() for h in hidden_states]

        total_reward += reward
        total_effort += effort
        steps += 1

    return episode_data, total_reward, total_effort

# Function to perform model updates
def update_model(model, episode_data):
    for layer_idx in reversed(range(model.n_layers)):
        for obs, chosen_option, reward, effort, next_obs, done, target_option, hidden_state in episode_data:
            model.update_layer_weights(
                layer_idx,
                obs,
                chosen_option,
                reward,
                effort,
                next_obs,
                done,
                hidden_state,
            )

# Function to visualize option usage and terminations
def visualize_option_usage(model, env):
    option_usage = defaultdict(list)
    termination_counts = defaultdict(int)

    for x in range(env.size):
        for y in range(env.size):
            env.state = (x, y)
            obs = env.get_observation()
            hidden_states = model.init_hidden(1)
            chosen_option = torch.tensor([random.randint(0, N_OPTIONS - 1)], dtype=torch.long, device=obs.device)

            for _ in range(5):
                output, values = model.forward(obs, chosen_option, hidden_states)
                if values is not None:
                    chosen_option_idx, _ = model.select_option(values, chosen_option)
                    chosen_option = torch.tensor([chosen_option_idx], dtype=torch.long, device=obs.device)

                if len(output) > 2 and output[2] is not None:
                    termination_prob = output[2]
                    if model.terminate_option(termination_prob):
                        termination_counts[(x, y)] += 1

            option_usage[(x, y)].append(chosen_option.item())

    return option_usage, termination_counts

# Main training loop
rewards_over_time = []
efforts_over_time = []
for episode in range(N_EPISODES):
    episode_data, total_reward, total_effort = run_episode(model, env, EPSILON)
    update_model(model, episode_data)
    rewards_over_time.append(total_reward)
    efforts_over_time.append(total_effort)

    if episode % TEST_EVERY_N_EPISODES == 0:
        option_usage, termination_counts = visualize_option_usage(model, env)
        print(f"Episode {episode}:")
        for state, options in option_usage.items():
            print(f"  State {state}: Option usage = {options}")
        print(f"  Termination counts: {termination_counts}")

# Plotting the results
plt.figure(figsize=(12, 5))

plt.style.use("rose-pine")
plt.subplot(1, 2, 1)
plt.plot(rewards_over_time)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(efforts_over_time)
plt.title("Total Effort per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Effort")

plt.tight_layout()
plt.show()

# Visualize the model using torchviz
batch_size = 1
input_size = env.size * env.size
input_tensor = torch.randn(batch_size, input_size)
chosen_option = torch.tensor([0], dtype=torch.long)
hidden_states = model.init_hidden(batch_size)

# Create a dummy input with the correct shape
dummy_input = (input_tensor, chosen_option, hidden_states)

# Pass the dummy input through the model
output, values = model(*dummy_input)

# Visualize the model graph
dot = torchviz.make_dot(values, params=dict(model.named_parameters()))
# dot.render("model", format="png")