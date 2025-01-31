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
        self.prediction_layer = nn.Linear(hidden_size, input_size) # Predictis the next input state

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
        self.recent_predicted_value = 0

        # Termination probability
        self.termination_prob = 0.5

        # Used as threshold for choosing options
        self.option_value_threshold = 0.5

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

        state_prediction = self.prediction_layer(self.hidden).squeeze()

        value_estimate = self.value_head(self.hidden).squeeze()

        # Estimate termination probability using the termination head
        termination_logit = self.termination_head(self.hidden).squeeze()
        self.termination_prob = torch.sigmoid(termination_logit).item()  # Update termination probability

        if self.has_output_layer:
            return self.output_layer(gru_out[-1, 0, :]), value_estimate, state_prediction, self.termination_prob, termination_logit
        else:
            return gru_out[-1, 0, :], value_estimate, self.termination_prob, state_prediction, termination_logit

    def update_weights(self, value_estimate, state_prediction, termination_logit, value_target, actual_state, termination_target, layer_idx):
        print(f"{'   ' * (layer_idx)}Layer {layer_idx} update")

        # Calculate value loss
        value_loss = self.value_criterion(value_estimate, value_target)

        # Calculate state prediction loss
        state_prediction_loss = self.value_criterion(state_prediction, actual_state)

        # Calculate termination loss
        termination_loss = self.termination_criterion(termination_logit, termination_target)

        # Total loss
        total_loss = value_loss + termination_loss + state_prediction_loss

        # Update weights
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        return total_loss.item()
    
    def compare_option_values(self, next_value, previous_value, alpha=0.8, beta=0.2):
        return (alpha * next_value / (1 + (beta * previous_value))) > self.option_value_threshold

    def calculate_termination_target(self, continue_value, switch_value):
        # Calculate termination target based on value difference
        value_diff = continue_value - switch_value
        return torch.tensor(1 / (1 + np.exp(value_diff)), dtype=torch.float)

    def reset_accumulated_values(self):
        self.accumulated_reward = 0
        self.accumulated_effort = 0
        self.best_option_value = -float('inf')  # Reset best option value


class MetaRL:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rates, gamma=0.99, switch_cost=0.1, termination_threshold=0.5, min_activation=0.1, num_max_steps=100):
        self.n_layers = len(hidden_sizes)
        self.grus = nn.ModuleList()
        self.gamma = gamma
        self.switch_cost = switch_cost
        self.termination_threshold = termination_threshold
        self.min_activation = min_activation  # Minimum activation threshold for options
        self.num_max_steps = num_max_steps  # Maximum number of steps to take in the environment


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

    # Gets the options based on the confidence of each option
    def filter_options(self, x):
        """
        Filter the options to the range [min_activation, 1]
        """
        options = torch.clamp_min(x, self.min_activation)
        options[options == self.min_activation] = 0
        return options

    def rollout_option(self, ext, layer_idx, option_idx):
        """
        Perform a rollout for a single option in the specified layer.
        """
        current_gru = self.grus[layer_idx]
        # Set the option as the 1 hot input
        x = torch.zeros(self.grus[layer_idx].hidden_size)
        x[option_idx] = 1
        accumulated_reward = 0
        with torch.no_grad():
            for t in range(self.num_max_steps):
                inputs = torch.cat([x, ext], dim=1)  # Concatenate x and ext
                output, reward, termination_prob, ext, termination_logit = current_gru(inputs, layer_idx)
                accumulated_reward += ((self.gamma **t) * reward) # Accumulate reward discounted over time
                if termination_prob > self.termination_threshold:
                    break

        return accumulated_reward


    def meta_step(self, x, env, layer_idx, ext, top=False):
        """
        Perform the meta-learning step for the specified layer and recursively process lower layers.
        """
        # Get the current GRU layer
        current_gru = self.grus[layer_idx]
        # Initialize loss, reward, effort
        loss = 0
        total_reward = 0

        # Ensure x and ext are 2D before concatenation
        if top == False:
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Reshape to (1, hidden_size)
            if ext.dim() == 1:
                ext = ext.unsqueeze(0)  # Reshape ext if needed
            x = torch.cat([x, ext], dim=1)  # Concatenate x and ext


        for step in range(self.steps_per_layer[layer_idx]):

            # Forward pass for the current layer
            output, option_value_estimate, termination_prob, state_prediction, termination_logit = current_gru(x, layer_idx)

            # Check if we should terminate or take another step in the environment/layer
            if termination_prob > self.termination_threshold:
                break

            # Get the options based on the confidence of each option
            affordances = self.filter_options(output)

            # Look at each option, holding the last one in working memory to compare it to the current one
            last_best_sub_option = None
            chosen_option_value  = option_value_estimate.clone().detach()
            switch_value = 0

            # iterate through all the options not equal to 0, going from highest to lowest
            for option_idx in torch.argsort(affordances, descending=True):
                sub_option_probability = affordances[option_idx]
                if sub_option_probability == 0:
                    break
                # Get the option
                sub_option_value = self.rollout_option(x, ext, layer_idx, option_idx)

                # If the value is greater than the last best value, then set the option as the best option
                if self.compare_option_values(next_value=sub_option_value, previous_value=chosen_option_value):
                    last_best_sub_option = option_idx
                    switch_value = chosen_option_value # setting the value for the alternative option, since we switched, and will have to switch back to regain the value
                    chosen_option_value = sub_option_value # the value for the chosen option

                # Clear unused option
                else:
                    affordances[layer_idx] = 0

            # Choose the best option
            # set the output to the best option (1-hot)
            output = torch.zeros_like(output)
            output[last_best_sub_option] = 1

            # Then observe the reward and effort
            if layer_idx > 0:
                # Recursively call meta_step for the lower layer
                l,r,e = self.meta_step(x=output, env=env, layer_idx=layer_idx - 1, ext=ext)
                loss += l
                reward += r
                effort += e
            else:
                r, e = env.step(output) # Placeholder function
                reward += r
                effort += e

            # Update weights for the current layer based on what we observed

            # Calculate continue value (value of continuing with the current option)
            continue_value = reward - effort + (self.gamma * chosen_option_value * (1 - termination_prob))

            # Calculate termination target based on whether to continue or switch
            termination_target = current_gru.calculate_termination_target(continue_value, switch_value)

            # Calculate value target (TD target)
            value_target = reward - effort + (self.gamma * chosen_option_value * (1 - termination_target))

            # Update weights for the current layer
            layer_loss = current_gru.update_weights(option_value_estimate, termination_logit, state_prediction, value_target, termination_target, layer_idx)
            loss += layer_loss

        return loss, reward, effort

if __name__ == "__main__":
    # Define parameters
    input_size = 10
    hidden_sizes = [20, 15, 10]  # Three layers with different hidden sizes
    output_size = 5
    learning_rates = [0.01, 0.005, 0.001]  # Different learning rates for each GRU
    steps_per_layer = [5,4,3]

    # Create the MetaRL model
    meta_model = MetaRL(input_size, hidden_sizes, output_size, learning_rates, steps_per_layer)
...