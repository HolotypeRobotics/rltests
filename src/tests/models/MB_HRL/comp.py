import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class OnlineGRU(nn.Module):
    def __init__(self, input_size, hidden_size, has_output_layer=False, output_size=None, learning_rate=0.01, gamma=0.99, switch_cost=0.1):
        super(OnlineGRU, self).__init__()
        print(f"Creating gru with input_size: {input_size}, hidden_size: {hidden_size}, output_size: {output_size}")

        self.hidden_size = hidden_size
        self.has_output_layer = has_output_layer
        self.gamma = gamma
        self.switch_cost = switch_cost
        self.learning_rate = learning_rate

        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.value_head = nn.Linear(hidden_size, 1)  # For value estimation
        self.termination_head = nn.Linear(hidden_size, 1)  # For termination probability

        # Output layer only if specified
        if has_output_layer:
            self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Loss function
        self.value_criterion = nn.MSELoss()  # Loss for value estimate
        self.termination_criterion = nn.BCEWithLogitsLoss()  # Loss for termination probability

        # Initialize hidden state
        self.hidden = None

        # Accumulated reward and effort
        self.accumulated_reward = 0
        self.accumulated_effort = 0

        # Termination probability
        self.termination_prob = 0.5

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward(self, x, layer_idx):
        print(f"{'   ' * (layer_idx)}Layer {layer_idx} forward")

        # Ensure hidden state is initialized correctly
        if self.hidden is None:
            self.hidden = self.init_hidden()

        # Reshape input to (sequence_length, batch_size, input_size)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        gru_out, self.hidden = self.gru(x, self.hidden)
        self.hidden = self.hidden.detach()

        # Use hidden state for value and termination heads
        value_estimate = self.value_head(self.hidden).squeeze()

        # Estimate termination probability using the termination head
        termination_logit = self.termination_head(self.hidden).squeeze()
        self.termination_prob = torch.sigmoid(termination_logit).item()  # Update termination probability

        if self.has_output_layer:
            return self.output_layer(gru_out[-1, 0, :]), value_estimate, self.termination_prob, termination_logit
        else:
            return gru_out[-1, 0, :], value_estimate, self.termination_prob, termination_logit

    def update_weights(self, value_estimate, termination_logit, value_target, termination_target, layer_idx):
        print(f"{'   ' * (layer_idx)}Layer {layer_idx} update")

        # Calculate value loss
        value_loss = self.value_criterion(value_estimate, value_target)

        # Calculate termination loss
        termination_loss = self.termination_criterion(termination_logit, termination_target)

        # Total loss
        total_loss = value_loss + termination_loss

        # Update weights
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        return total_loss.item()

    def calculate_termination_target(self, continue_value, switch_value):
        # Calculate termination target based on value difference
        # If continue_value > switch_value, termination target is 0 (continue)
        # Otherwise, termination target is 1 (switch)
        value_diff = continue_value - switch_value
        return torch.tensor(1 / (1 + np.exp(value_diff)), dtype=torch.float)

    def reset_accumulated_values(self):
        self.accumulated_reward = 0
        self.accumulated_effort = 0

    def should_terminate(self):
        # Decide whether to terminate based on termination probability
        return torch.rand(1).item() < self.termination_prob

class MetaRL:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rates, gamma=0.99, switch_cost=0.1, termination_threshold=0.5):
        self.n_layers = len(hidden_sizes)
        self.grus = nn.ModuleList()
        self.gamma = gamma
        self.switch_cost = switch_cost
        self.termination_threshold = termination_threshold

        for i in range(self.n_layers):
            print(f"Creating GRU {i + 1}/{self.n_layers}")

            if i < self.n_layers - 1:
                layer_input_size = input_size + hidden_sizes[i + 1]
            else:
                layer_input_size = input_size

            layer_output_size = hidden_sizes[i]
            has_output_layer = (i == 0)  # Only the last layer has an output layer
            self.grus.append(OnlineGRU(
                input_size=layer_input_size,
                hidden_size=layer_output_size,
                has_output_layer=has_output_layer,
                output_size=output_size if has_output_layer else None,
                learning_rate=learning_rates[i],
                gamma=self.gamma,
                switch_cost=self.switch_cost
            ))

    def meta_step(self, x, env, layer_idx, ext, top=False):
        """
        Perform the meta-learning step for the specified layer and recursively process lower layers.
        """
        # Get the current GRU layer
        current_gru = self.grus[layer_idx]

        # Check for termination before running the forward pass
        if not top and current_gru.should_terminate():
            current_gru.reset_accumulated_values()
            return 0, 0, 0  # Indicate termination

        # Initialize loss, reward, effort
        loss = 0
        total_reward = 0
        total_effort = 0

        # Ensure x and ext are 2D before concatenation
        if top == False:
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Reshape to (1, hidden_size)
            if ext.dim() == 1:
                ext = ext.unsqueeze(0)  # Reshape ext if needed
            x = torch.cat([x, ext], dim=1)  # Concatenate x and ext

        # Forward pass for the current layer
        output, value_estimate, termination_prob, termination_logit = current_gru(x, layer_idx)

        if layer_idx > 0:
            # Recursively call meta_step for the lower layer
            lower_loss, reward, effort = self.meta_step(x=output, env=env, layer_idx=layer_idx - 1, ext=ext)
            loss += lower_loss
            total_reward += reward
            total_effort += effort

            # Get the value estimate from the lower layer (if it exists)
            _, lower_value_estimate, _, _ = self.grus[layer_idx - 1](output, layer_idx - 1)

            # Calculate switch value (value of switching to the best other option)
            switch_value = lower_value_estimate.detach() - self.switch_cost if layer_idx > 0 else torch.tensor(0.0)

        else:
            switch_value = torch.tensor(0.0)  # No lower layer to switch to
            # If at the lowest level, perform the action in the environment
            reward, effort = env.step(output.squeeze(0))  # Remove extra dimension from output
            total_reward += reward
            total_effort += effort

        # Accumulate reward and effort for the current layer
        current_gru.accumulated_reward += total_reward
        current_gru.accumulated_effort += total_effort

        # Calculate continue value (value of continuing with the current option)
        continue_value = current_gru.accumulated_reward - current_gru.accumulated_effort + (
                self.gamma * value_estimate.detach() * (1 - termination_prob))

        # Calculate termination target based on whether to continue or switch
        termination_target = current_gru.calculate_termination_target(continue_value, switch_value)

        # Calculate value target (TD target)
        value_target = current_gru.accumulated_reward - current_gru.accumulated_effort + (
                self.gamma * value_estimate.detach() * (1 - termination_target))

        # Update weights for the current layer
        layer_loss = current_gru.update_weights(value_estimate, termination_logit, value_target, termination_target, layer_idx)
        loss += layer_loss

        return loss, total_reward, total_effort

    def get_reward(self, layer_idx, step):
        # Placeholder function to simulate getting a reward from the environment
        # You'll need to implement the actual logic based on your task
        # For now, let's just return a random value
        return np.random.rand()

    def get_effort(self, layer_idx, step):
        # Placeholder function to simulate effort associated with an action
        # You'll need to implement the actual logic based on your task
        # For now, let's just return a constant value
        return 0.1

# Dummy environment for testing
class DummyEnv:
    def step(self, action):
        # Replace this with your actual environment logic
        reward = np.random.rand()
        effort = 0.1
        return reward, effort

if __name__ == "__main__":
    # Define parameters
    input_size = 10
    hidden_sizes = [20, 15, 10]  # Three layers with different hidden sizes
    output_size = 5
    learning_rates = [0.01, 0.005, 0.001]  # Different learning rates for each GRU
    gamma = 0.99
    switch_cost = 0.1
    termination_threshold = 0.5

    # Create the MetaRL model
    meta_model = MetaRL(input_size, hidden_sizes, output_size, learning_rates, gamma, switch_cost, termination_threshold)

    # Create a dummy environment
    env = DummyEnv()

    # Example training loop
    for epoch in range(5):
        # Generate dummy data
        x = torch.randn(1, input_size)

        # Perform meta-learning step
        loss, reward, effort = meta_model.meta_step(x=x, env=env, layer_idx=meta_model.n_layers - 1, ext=x, top=True)

        print(f"\nEpoch {epoch + 1}")
        print(f"Total Loss: {loss}")
        print(f"Total Reward: {reward}")
        print(f"Total Effort: {effort}")