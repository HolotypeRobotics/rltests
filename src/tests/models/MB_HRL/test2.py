import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# TODO:
# - Implement attention network functionality to external GRU inputs
# - Imlement control functionality
# - Simulate head movements or gaze shifts that correspond to the current focus of mental simulation.
# - Prediction error should trigger option switching trhough diminishing confidence in the current option
#   - It should also effect the learning rate, and control by increasing both

class OnlineGRU(nn.Module):
    def __init__(self, input_size, external_input_size, hidden_size, has_output_layer=False, output_size=None, learning_rate=0.01, gamma=0.99, switch_cost=0.1, termination_threshold=0.9):
        super(OnlineGRU, self).__init__()
        print(f"Creating gru with input_size: {input_size}, hidden_size: {hidden_size}, output_size: {output_size}")

        self.input_size = input_size
        self.external_input_size = external_input_size
        self.hidden_size = hidden_size
        self.has_output_layer = has_output_layer
        self.gamma = gamma
        self.switch_cost = switch_cost
        self.learning_rate = learning_rate
        self.output_size = self.hidden_size if output_size is None else output_size
        self.prediction_error_history = []
        self.base_learning_rate = learning_rate
        self.adaptive_learning_rate = learning_rate

        # GRU layer
        self.gru = nn.GRU(input_size+external_input_size, hidden_size, batch_first=False)
        self.value_head = nn.Linear(hidden_size, 1)  # For value estimation
        self.termination_head = nn.Linear(hidden_size, 1)  # For termination probability
        self.prediction_layer = nn.Linear(hidden_size, external_input_size) # Predictis the next input state

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
        self.termination_threshold = termination_threshold


    def update_learning_rate(self, prediction_error):
        """Adjust learning rate based on prediction error"""
        self.prediction_error_history.append(prediction_error)
        if len(self.prediction_error_history) > 10:  # Keep last 10 errors
            self.prediction_error_history.pop(0)
        
        # Increase learning rate if recent errors are high
        mean_error = sum(self.prediction_error_history) / len(self.prediction_error_history)
        self.adaptive_learning_rate = self.base_learning_rate * (1 + mean_error)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.adaptive_learning_rate

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward(self, x, execute=True):

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
        termination_prob = torch.sigmoid(termination_logit).item()  # Update termination probability

        # Output action only if bottom layer, and we are not planning
        if self.has_output_layer and execute == True:
            return self.output_layer(gru_out[-1, 0, :]), value_estimate, state_prediction, termination_prob, termination_logit
        else:
            return gru_out[-1, 0, :], value_estimate, state_prediction,  termination_prob, termination_logit

    def update_weights(self, value_estimate, state_prediction, termination_logit, value_target, actual_state, termination_target, layer_idx):
        print(f"{'   ' * (layer_idx)}Layer {layer_idx} update")

        # Calculate prediction error
        prediction_error = torch.mean(torch.abs(state_prediction - actual_state))
        
        # Update learning rate based on prediction error
        self.update_learning_rate(prediction_error.item())

        # Calculate value loss
        value_loss = self.value_criterion(value_estimate, value_target)

        # Calculate state prediction loss
        state_prediction_loss = self.value_criterion(state_prediction, actual_state)
        termination_loss = self.termination_criterion(termination_logit, termination_target)

        # Total loss
        prediction_weight = 1.0 + prediction_error.item()
        total_loss = (value_loss + termination_loss + state_prediction_loss * prediction_weight)


        # Update weights
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        return total_loss.item(), prediction_error.item()


    def calculate_termination_target(self, continue_value, switch_value):
        # Ensure we're working with detached tensors for this calculation
        if isinstance(continue_value, torch.Tensor):
            continue_value = continue_value.detach()
        if isinstance(switch_value, torch.Tensor):
            switch_value = switch_value.detach()
            
        # Calculate value difference
        value_diff = continue_value - switch_value
        
        # Convert to tensor if it's not already
        if not isinstance(value_diff, torch.Tensor):
            value_diff = torch.tensor(value_diff, dtype=torch.float32)
            
        # Use torch operations instead of numpy
        return torch.sigmoid(torch.exp(value_diff))


class MetaRL:
    def __init__(self, external_input_size, hidden_sizes, output_size, learning_rates, gamma=0.99, switch_cost=0.1, termination_threshold=0.5, min_activation=0.1, num_max_steps=10, option_value_threshold=0.5):
        self.n_layers = len(hidden_sizes)
        self.grus = nn.ModuleList()
        self.gamma = gamma
        self.switch_cost = switch_cost
        self.termination_threshold = termination_threshold
        self.min_activation = min_activation  # Minimum activation threshold for options
        self.num_max_steps = num_max_steps  # Maximum number of steps to take in the environment
        self.option_value_threshold = option_value_threshold  # Threshold for option value comparison

        for i in range(self.n_layers):
            print(f"Creating GRU {i + 1}/{self.n_layers}")

            if i < self.n_layers - 1:
                # If not the top layer, the input size is the hidden size of the above layer
                layer_input_size = hidden_sizes[i + 1]
            else:
                # If the top layer, the input size is just the external input size
                layer_input_size = 0

            layer_output_size = hidden_sizes[i]
            has_output_layer = (i == 0)  # Only the last layer has an output layer
            self.grus.append(OnlineGRU(
                input_size=layer_input_size,
                external_input_size=external_input_size,
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
        x = torch.zeros(current_gru.input_size if layer_idx == self.n_layers - 1 else self.grus[layer_idx + 1].hidden_size)
        x[option_idx] = 1
        accumulated_reward = 0
        with torch.no_grad():
            for t in range(self.num_max_steps):
                _, predicted_reward, ext, termination_prob, _ = self.feed(x, ext, layer_idx, execute=False)
                accumulated_reward += ((self.gamma **t) * predicted_reward) # Accumulate reward discounted over time
                if termination_prob > self.termination_threshold:
                    break

        return accumulated_reward

    def apply_control_over_options(self, x, option_idx):
        """
        Apply control over the options by contrasting the values based on
        the chosen option, and the frequency of negative prediction errors.
        """
        # Use sigmoid, with a gain based on error frequency
        x = torch.sigmoid(x - option_idx)

        return x
    
    def feed(self, x, ext, layer_idx, execute=True):
        # Ensure x and ext are 2D before concatenation


        # If not the top layer, concatenate the external input. Otherwise, just use x, which is the external input
        if layer_idx < self.n_layers - 1: 
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Reshape to (1, hidden_size)
    
            if ext.dim() == 1:
                ext = ext.unsqueeze(0)  # Reshape ext if needed

            # Take only the first input_size elements from ext
            ext = ext[:, :self.grus[layer_idx].external_input_size]
            x = torch.cat([x, ext], dim=1)  # Concatenate x and ext

        out = self.grus[layer_idx](x, execute)

        return out


    def meta_step(self, x, env, layer_idx, ext):
        """
        Perform the meta-learning step for the specified layer and recursively process lower layers.
        """
        # Get the current GRU layer
        current_gru = self.grus[layer_idx]
        # Initialize loss, reward, effort
        loss = 0
        total_reward = 0
        prediction_errors = []

        reward = 0
        effort = 0

        for step in range(self.num_max_steps):

            # Forward pass for the current layer
            output, option_value_estimate, state_prediction, termination_prob, termination_logit = self.feed(x=x, ext=ext, layer_idx=layer_idx)

            # Detach output for option selection to prevent gradient issues
            output_detached = output.detach()

            # Check if we should terminate or take another step in the environment/layer
            if termination_prob > self.termination_threshold:
                print(f"Terminating at layer {layer_idx}, step {step}, termination_prob: {termination_prob} > {self.termination_threshold}")
                break

            # Get the options based on the confidence of each option
            affordances = self.filter_options(output_detached)
            mean_error = 0
            if len(prediction_errors) > 0:
                mean_error = sum(prediction_errors) / len(prediction_errors)
                affordances = affordances * (1 - mean_error)  # Reduce confidence based on errors

            # Look at each option, holding the last one in working memory to compare it to the current one
            last_best_sub_option = torch.argmax(affordances)
            chosen_option_value = option_value_estimate.clone().detach()
            chosen_option_confidence = 0.0
            alternative_option_value = 0.0
            alternative_option_confidence = 0.0

            # Sort options by confidence and iterate
            sorted_indices = torch.argsort(affordances, descending=True)
            for option_idx in sorted_indices:
                sub_option_confidence = affordances[option_idx].item()  # Convert to scalar
                if sub_option_confidence == 0:
                    break

                # Get the sub-option value for the current option by rolling out the option in the lower layer
                if layer_idx > 0:
                    sub_option_value = self.rollout_option(ext=ext, layer_idx=layer_idx - 1, option_idx=option_idx)
                else:
                    sub_option_value = chosen_option_value

                # Compare values
                if isinstance(sub_option_value, torch.Tensor):
                    sub_option_value = sub_option_value.detach()
                if isinstance(chosen_option_value, torch.Tensor):
                    chosen_option_value = chosen_option_value.detach()

                # If the value is greater than the last best value, then set the option as the best option
                if self.compare_option_values(next_value=sub_option_value, previous_value=chosen_option_value):
                    last_best_sub_option = option_idx

                    alternative_option_value = float(chosen_option_value) # setting the value for the alternative option, since we switched, and will have to switch back to regain the value
                    alternative_option_confidence = chosen_option_confidence

                    chosen_option_value = float(sub_option_value) # The value for the chosen option
                    chosen_option_confidence = sub_option_confidence # The confidence for the chosen option

                # Eliminate unused option
                else:
                    affordances[option_idx] = 0

            # Choose the best option
            # set the output to the best option (1-hot)
            output = self.apply_control_over_options(output, last_best_sub_option)

            # Then observe the reward and effort
            if layer_idx > 0:
                # Recursively call meta_step for the lower layer
                l,r,e, p_errors = self.meta_step(x=output, env=env, layer_idx=layer_idx - 1, ext=ext)
                loss += l
                reward += r
                effort += e
                prediction_errors.extend(p_errors)
            else:
                r, e = env.step(output)
                reward += r
                effort += e

            # Update weights for the current layer based on what we observed

            # Calculate continue value (value of continuing with the current option)
            # continue_value = reward - effort + (self.gamma * chosen_option_value * (1 - termination_prob))
            # continue_value = chosen_option_value * chosen_option_confidence
            continue_value = (chosen_option_value * chosen_option_confidence) * (1 - mean_error if len(prediction_errors) > 0 else 0)
            switch_value =  alternative_option_value * alternative_option_confidence

            # Calculate termination target based on whether to continue or switch
            termination_target = current_gru.calculate_termination_target(continue_value, switch_value)

            # Calculate value target (TD target)
            value_target = torch.tensor(reward - effort + (self.gamma * chosen_option_value * (1 - termination_target)))

            # Update weights for the current layer
            
            layer_loss, pred_error = current_gru.update_weights(
                option_value_estimate, state_prediction, termination_logit,
                value_target, env.state, termination_target, layer_idx
            )
            prediction_errors.append(pred_error)
            loss += layer_loss

        return loss, reward, effort, prediction_errors

    def compare_option_values(self, next_value, previous_value, alpha=0.8, beta=0.2):
        # Ensure we're working with scalar values
        if isinstance(next_value, torch.Tensor):
            next_value = next_value.item()
        if isinstance(previous_value, torch.Tensor):
            previous_value = previous_value.item()
            
        return (alpha * next_value / (1 + (beta * previous_value))) > self.option_value_threshold

# Dummy environment for testing
class DummyEnv:
    def __init__(self, state_size):
        self.state_size = state_size
        self.state = torch.zeros(self.state_size, dtype=torch.float32)
    def step(self, action):
        # Replace this with your actual environment logic
        reward = np.random.rand()
        self.state = torch.rand(self.state_size, dtype=torch.float32)
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
    termination_threshold = 0.79

    # Create the MetaRL model
    meta_model = MetaRL(input_size, hidden_sizes, output_size, learning_rates, gamma, switch_cost, termination_threshold)
    # Create a dummy environment
    env = DummyEnv(input_size)
    # Example training loop
    for epoch in range(1):
        # Generate dummy data
        x = torch.randn(1, input_size)
        # Perform meta-learning step
        loss, reward, effort, prediction_errors = meta_model.meta_step(x=x, env=env, layer_idx=meta_model.n_layers - 1, ext=x)
        print(f"\nEpoch {epoch + 1}")
        print(f"Total Loss: {loss}")
        print(f"Total Reward: {reward}")
        print(f"Total Effort: {effort}")
