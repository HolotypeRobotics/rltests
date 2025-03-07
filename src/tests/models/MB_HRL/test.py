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

        # Estimate value using the value head
        value_estimate = self.value_head(gru_out[-1, 0, :]).squeeze()

        # Estimate termination probability using the termination head
        termination_logit = self.termination_head(gru_out[-1, 0, :]).squeeze()
        self.termination_prob = torch.sigmoid(termination_logit).item()  # Update termination probability

        if self.has_output_layer:
            return self.output_layer(gru_out[-1, 0, :]), value_estimate, self.termination_prob
        else:
            return gru_out[-1, 0, :], value_estimate, self.termination_prob

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
        # return torch.tensor(0.0 if continue_value > switch_value else 1.0, dtype=torch.float)
        value_diff = continue_value - switch_value
        return 1 / (1 + np.exp(value_diff))

    def reset_accumulated_values(self):
        self.accumulated_reward = 0
        self.accumulated_effort = 0

class MetaRL:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rates, steps_per_layer, gamma=0.99, switch_cost=0.1):
        self.n_layers = len(hidden_sizes)
        self.grus = nn.ModuleList()
        self.steps_per_layer = steps_per_layer
        self.gamma = gamma
        self.switch_cost = switch_cost

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

    def meta_step(self, x, targets, layer_idx, ext, reward, effort, top=False):
        """
        Perform the meta-learning step for the specified layer and recursively process lower layers.
        """
        loss = 0
        # Ensure x and ext are 2D before concatenation
        if top == False:
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Reshape to (1, hidden_size)
            if ext.dim() == 1:
                ext = ext.unsqueeze(0)  # Reshape ext if needed
            x = torch.cat([x, ext], dim=1)  # Concatenate x and ext

        # Get the current GRU layer
        current_gru = self.grus[layer_idx]

        # Accumulate reward and effort for the current layer
        current_gru.accumulated_reward += reward
        current_gru.accumulated_effort += effort

        for step in range(self.steps_per_layer[layer_idx]):

            # Forward pass for the current layer
            output, value_estimate, termination_prob = current_gru(x, layer_idx)
            termination_logit = current_gru.termination_head(output).squeeze()

            if layer_idx > 0:
                # Recursively call meta_step for the lower layer
                lower_layer_loss = self.meta_step(x=output, targets=targets, layer_idx=layer_idx - 1, ext=ext, reward=reward, effort=effort)
                loss += lower_layer_loss

                # Get the value estimate from the lower layer (if it exists)
                _, lower_value_estimate, _ = self.grus[layer_idx - 1](output, layer_idx - 1)

                # Calculate switch value (value of switching to the best other option)
                # For simplicity, let's assume the best other option is the one with the highest value
                # In a more complex scenario, you might consider other factors or explore other options
                switch_value = lower_value_estimate.detach() - self.switch_cost if layer_idx > 0 else torch.tensor(0.0)

            else:
                switch_value = torch.tensor(0.0)  # No lower layer to switch to

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

            # Decide whether to terminate the current layer based on termination probability
            if torch.rand(1).item() < termination_prob:
                current_gru.reset_accumulated_values()
                break

        return loss

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

if __name__ == "__main__":
    # Define parameters
    input_size = 10
    hidden_sizes = [20, 15, 10]  # Three layers with different hidden sizes
    output_size = 5
    learning_rates = [0.01, 0.005, 0.001]  # Different learning rates for each GRU
    steps_per_layer = [5,4,3]

    # Create the MetaRL model
    meta_model = MetaRL(input_size, hidden_sizes, output_size, learning_rates, steps_per_layer)

    # Example training loop
    for epoch in range(5):
        # Generate dummy data
        x = torch.randn(1, input_size)
        targets = []
        for i in range(meta_model.n_layers):
            out_size = hidden_sizes[i] if i > 0 else output_size
            targets.append(torch.randn(out_size))

        # Get reward and effort (you'll need to define how these are obtained)
        reward = meta_model.get_reward(meta_model.n_layers - 1, epoch)  # Placeholder function
        effort = meta_model.get_effort(meta_model.n_layers - 1, epoch)  # Placeholder function

        # Perform meta-learning step
        loss = meta_model.meta_step(x=x, ext=x, targets=targets, layer_idx=meta_model.n_layers - 1, reward=reward, effort=effort, top=True)

        print(f"\nEpoch {epoch + 1}")
        print(f"Total Loss: {loss}")