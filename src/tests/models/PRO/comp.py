import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

# Constants (Hyperparameters)
GAMMA = 0.99  # Discount factor
N_OPTIONS = 4  # Number of options
N_GOALS = 8  # Number of goals
N_LAYERS = 2  # Number of layers in the hierarchy
CONTROL_LR = 0.01
HABITUAL_LR = 0.001
ERROR_ALPHA = 0.1
N_MAX_LAYER_ITERATIONS = 10

class AttentionMechanism(nn.Module):
    def __init__(self, input_size, n_options):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(input_size, n_options)

    def forward(self, x):
        # x: input observation (or features)
        # Apply a linear layer followed by a softmax to get attention weights for each option
        attn_weights = F.softmax(self.attention_weights(x), dim=-1)  # Softmax over options
        return attn_weights

class HierarchicalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_options, n_goals, n_layers,
                 control_lr=0.01, habitual_lr=0.001, error_alpha=0.1, error_window=10, n_max_layer_iterations=10):
        super(HierarchicalGRU, self).__init__()
        self.input_size = input_size
        self.n_options = n_options
        self.hidden_size = hidden_size
        self.option_efforts = torch.zeros(n_options)
        self.n_goals = n_goals
        self.n_layers = n_layers
        self.error_alpha = error_alpha
        self.average_error = defaultdict(float)  # Initialize average error for each layer
        self.control_gains = defaultdict(lambda: 0.5)  # Initialize control gains to 0.5 (or a value <1)
        self.n_max_layer_iterations = n_max_layer_iterations

        # Add the attention mechanism as a module
        self.attention = AttentionMechanism(input_size, n_options)

        # GRU cells for each layer
        # Input to each GRU layer: environment observation + hidden state from the layer above
        self.grus = nn.ModuleList([
            nn.GRUCell(
                input_size + (hidden_size if i < n_layers - 1 else 0),  # Include context from the higher layer
                hidden_size
            ) for i in range(n_layers)
        ])

        # Predicted efforts for individual options/actions
        self.effort_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_options) for _ in range(n_layers)
        ])

        # Predicted value for having taken a specific action in a given state
        self.value_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_options) for _ in range(n_layers) # Output Q-values for each option
        ])

        # Separate heads for permanent and transient value functions
        self.permanent_value_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_options) for _ in range(n_layers)
        ])
        self.transient_value_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_options) for _ in range(n_layers)
        ])
        
        # Goal prediction heads for each layer
        self.goal_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_goals) for _ in range(n_layers)
        ])

        # Termination functions for each layer (except the lowest)
        self.termination_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_options) for _ in range(n_layers - 1) # Output termination probability for each option
        ])

        # Policy over options (for each layer except the lowest)
        self.policy_over_options_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_options) for _ in range(n_layers - 1)
        ])

        # Optimizers
        self.control_optimizer = optim.Adam(
            list(self.effort_heads.parameters()) +
            list(self.value_heads.parameters()) +
            list(self.goal_heads.parameters()) +
            list(self.termination_heads.parameters()) +
            list(self.policy_over_options_heads.parameters()),
            lr=control_lr
        )
        self.habitual_optimizer = optim.Adam(self.grus.parameters(), lr=habitual_lr)

    # Apply gain contrast using softmax temperature
    def apply_gain_contrast(self, tensor, gain):
        # gain determines the strength of contrast
        # centered around 0.5, the midpoint of the sigmoid.
        # gain = 0 means no contrast, gain = 1 means maximum contrast
        # gain is learned, and stored in self.control_gains
        
        # center around 0.5
        centered_tensor = tensor - 0.5
        # scale by gain
        scaled_tensor = centered_tensor * gain
        # return to original range
        contrasted = scaled_tensor + 0.5
        
        return contrasted

    def get_cost(self, efforts, values):
        """Calculate cost ratio. Add a small epsilon to avoid division by zero."""
        # Cost is effort divided by value. The higher the value, the lower the cost.
        # The higher the effort, the higher the cost.
        # Adding a small epsilon ensures numerical stability.
        return efforts / (values + 1e-6)

    def select_option(self, policy_over_options_output):
        """Sample an option based on the policy over options output."""
        # Use torch.distributions to create a categorical distribution
        dist = torch.distributions.Categorical(logits=policy_over_options_output)
        # Sample from the distribution
        option_index = dist.sample()
        return option_index.item()

    def update_control_gain(self, layer_idx, reward, value):
        """Update control gain based on reward prediction error."""
        # Calculate the prediction error (RPE)
        prediction_error = reward - value.item()  # .item() to get a scalar value

        # Update running average of error using exponential moving average
        self.average_error[layer_idx] = (1 - self.error_alpha) * self.average_error[layer_idx] + self.error_alpha * prediction_error

        # Update gain based on average error
        # Example: Increase gain if error is large (positive or negative), decrease if error is small
        # The specific function can be adjusted
        self.control_gains[layer_idx] = max(0, 1.0 - abs(self.average_error[layer_idx]))

    def terminate_option(self, layer_idx, termination_output, chosen_option_index):
        """Decide whether to terminate the current option based on the termination head output."""
        termination_prob = torch.sigmoid(termination_output[0, chosen_option_index]).item()
        # returns true if termination probability is greater than a random number between 0 and 1
        return termination_prob > torch.rand(1).item()

    def forward_layer(self, layer_idx, x, hidden_state, chosen_option, context=None):
        """
        Perform a forward pass for a single layer.
        Returns output and updated hidden state.
        """
        # Concatenate input with context from the higher layer
        if context is not None:
            # context is the hidden state of the layer above
            input_with_context = torch.cat((x, context), dim=-1)
        else:
            input_with_context = x
        

        hidden_state = self.grus[layer_idx](input_with_context, hidden_state)

        # Apply gain contrast to hidden state
        hidden_state = self.apply_gain_contrast(hidden_state, self.control_gains[layer_idx])

        goals = self.goal_heads[layer_idx](hidden_state)
        efforts = self.effort_heads[layer_idx](hidden_state)
        # values = self.value_heads[layer_idx](hidden_state)

        # Get policy over options for this layer (except for the lowest layer)
        if layer_idx < self.n_layers - 1:
            policy_over_options_output = self.policy_over_options_heads[layer_idx](hidden_state)
            termination = self.termination_heads[layer_idx](hidden_state)
        else:
            policy_over_options_output = None
            termination = None

        # Option Selection (for the master policy)
        if layer_idx == 0: # Master policy is at the lowest level
            # One-hot encode the chosen option
            chosen_option_one_hot = F.one_hot(chosen_option, num_classes=self.n_options).float()
            # Concatenate the chosen option with the observation for the next layer
            x = torch.cat((x, chosen_option_one_hot), dim=-1)  # Pass chosen option as input to next layer
            if layer_idx < self.n_layers - 1:
                return (goals, efforts, values, termination, policy_over_options_output, chosen_option), hidden_state
            else:
                return (goals, efforts, values, chosen_option), hidden_state
        else:
            if layer_idx < self.n_layers - 1:
                return (goals, efforts, values, termination, policy_over_options_output), hidden_state
            else:
                return (goals, efforts, values), hidden_state

    def process_event(self, layer_idx, x, hidden_states, chosen_option, reward=None, timestep=0):
        """Process a single timestep within an option."""
        # Apply attention to the input observation
        attn_weights = self.attention(x) # Get attention weights
        # based on the chosen option, select the corresponding attention weights
        attn_weights_for_option = attn_weights.gather(1, chosen_option.unsqueeze(1))
        # apply attention to x
        x = x * attn_weights_for_option

        # Get initial output for this layer
        # Pass the hidden state of the layer above as context to the current layer
        output, hidden_states[layer_idx] = self.forward_layer(
            layer_idx,
            x,
            hidden_states[layer_idx],
            chosen_option,
            hidden_states[layer_idx + 1] if layer_idx < self.n_layers - 1 else None
        )

        goals = output[0]
        efforts = output[1]
        # values = output[2]
        permanent_values = self.permanent_value_heads[layer_idx](hidden_state)
        transient_values = self.transient_value_heads[layer_idx](hidden_state)
        values = permanent_values + transient_values

        # If not at the lowest layer, recursively process lower layer events
        if layer_idx < self.n_layers - 1:
            # Execute the chosen option for a fixed number of steps or until termination
            termination_output = output[3]
            termination = self.terminate_option(layer_idx, termination_output, chosen_option)

            if not termination:
                # Process lower layer event, passing chosen_option as context
                lower_layer_output, hidden_states = self.process_event(
                    layer_idx + 1, x, hidden_states, chosen_option, reward, timestep)

                # Update this layer's state based on lower layer output
                # Concatenate lower layer output with current input
                # Use value as a proxy for lower layer output
                x_combined = torch.cat((x, lower_layer_output[2]), dim=-1)
                output, hidden_states[layer_idx] = self.forward_layer(
                    layer_idx,
                    x_combined,
                    hidden_states[layer_idx],
                    chosen_option,
                    hidden_states[layer_idx + 1] if layer_idx < self.n_layers - 1 else None
                )

                # Update control gain based on reward prediction error for this layer
                if reward is not None:
                    self.update_control_gain(layer_idx, reward, output[2])

                # Update goals, efforts, and values with the new output
                goals = output[0]
                efforts = output[1]
                # values = output[2]
                permanent_values = self.permanent_value_heads[layer_idx](hidden_state)
                transient_values = self.transient_value_heads[layer_idx](hidden_state)
                values = permanent_values + transient_values

                timestep += 1 # Increment timestep within the option
            
            if termination:
                # If the option terminated, choose a new option for the next step
                policy_over_options_output = output[4]
                chosen_option = self.select_option(policy_over_options_output)
        else:
            # Update control gain based on reward prediction error for the lowest layer
            if reward is not None:
                self.update_control_gain(layer_idx, reward, values)

        return output, hidden_states

    def init_hidden(self, batch_size):
        """Initialize hidden states for all layers."""
        return [torch.zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]

    def update_layer_weights(self, layer_idx, target_value):
        """Update weights for a specific layer using accumulated values as targets."""
        # habitual network update
        self.habitual_optimizer.zero_grad()

        # Get current predictions
        output, hidden_state = self.forward_layer(
            layer_idx,
            torch.zeros_like(target_value),  # placeholder input
            self.grus[layer_idx].weight_ih_l0.new_zeros(self.hidden_size), # use weights from layer to initialize hidden state
            None if layer_idx == 0 else target_value # use target value for previous layer output
        )

        # Compute loss for value function
        value_loss = F.mse_loss(output[2], target_value)

        # Update weights for value function
        value_loss.backward()
        self.habitual_optimizer.step()

        if layer_idx < self.n_layers - 1:
            # control network update
            self.control_optimizer.zero_grad()
            # Compute loss for goal prediction
            goal_loss = F.mse_loss(output[0], target_value) # Assuming target_value can also represent target goals

            # Update weights for goal prediction
            goal_loss.backward()
            self.control_optimizer.step()

    def forward(self, x, chosen_option, hidden_states=None):
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [torch.zeros(1, self.hidden_size) for _ in range(self.n_layers)]

        # Process the event through the hierarchy
        final_output, _ = self.process_event(0, x, hidden_states, chosen_option)
        return final_output