import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class OnlineGRU(nn.Module):
    def __init__(self, input_size, external_input_size, hidden_size, has_output_layer=False, output_size=None, learning_rate=0.01, gamma=0.99, switch_cost=0.1, termination_threshold=0.9):
        super(OnlineGRU, self).__init__()
        print(f"Creating gru with input_size: {input_size}, hidden_size: {hidden_size}, output_size: {output_size}")

        self.input_size = input_size
        self.external_input_size = external_input_size
        self.hidden_size = hidden_size
        self.has_output_layer = has_output_layer
        self.gamma = gamma
        self.energy = 0.0
        self.switch_cost = switch_cost
        self.output_size = self.hidden_size if output_size is None else output_size
        self.prediction_error_history = []
        self.base_learning_rate = learning_rate

        self.previous_reward = 0

        # GRU layer
        self.gru = nn.GRU(input_size+external_input_size, hidden_size, batch_first=False)
        self.value_head = nn.Linear(hidden_size, 1)  # For value estimation
        self.termination_head = nn.Linear(hidden_size, 1)  # For termination probability
        with torch.no_grad():
            self.termination_head.bias.fill_(-5.0)  # This will make initial termination probabilities very low
        # Predictis the next input state
        self.prediction_layer = nn.Sequential(
            nn.Linear(hidden_size, external_input_size),
            nn.Sigmoid()
        )

        # Output layer only if specified
        if has_output_layer:
            self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Loss function
        self.value_criterion = nn.MSELoss()  # Loss for value estimate
        self.control_output_criterion = nn.MSELoss() # Or nn.CrossEntropyLoss() if appropriate
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

        # Increase learning rate more aggressively if recent errors are consistently high
        mean_error = sum(self.prediction_error_history) / len(self.prediction_error_history)
        error_std = np.std(self.prediction_error_history) if len(self.prediction_error_history) > 1 else 0
        # Adaptive LR scaling: base + mean + std of error.  Higher std means more volatility, increase LR more
        self.adaptive_learning_rate = self.base_learning_rate * (1 + mean_error + error_std)

        print(f"New Learning rate: {self.adaptive_learning_rate:.4f} using prediction error: {mean_error:.4f} std: {error_std:.4f}")

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

    def update_weights(self,
                       value_estimate,
                       state_prediction,
                       termination_logit,
                       value_target,
                       actual_state,
                       termination_target,
                       raw_output_affordances,
                       controlled_output_affordances,
                       layer_idx):
        print(f"{'   ' * (layer_idx)}Layer {layer_idx} update")

        # Calculate prediction error
        prediction_error = torch.mean(torch.abs(state_prediction - actual_state))

        # Update learning rate based on prediction error
        self.update_learning_rate(prediction_error.item())

        # Calculate value loss
        value_loss = self.value_criterion(value_estimate, value_target)

        # Calculate Control Output Loss
        # Detach the controlled output so gradients don't flow back through the control mechanism itself.
        controlled_output_target = controlled_output_affordances.detach()
        control_output_loss = self.control_output_criterion(raw_output_affordances, controlled_output_target)

        # Calculate state prediction loss
        state_prediction_loss = self.value_criterion(state_prediction, actual_state)
        termination_loss = self.termination_criterion(termination_logit, termination_target)

        # Total loss (add control_output_loss)
        prediction_weight = 1.0 + prediction_error.item()
        total_loss = (value_loss + termination_loss + state_prediction_loss * prediction_weight + control_output_loss)

        # Update weights
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        return total_loss.item(), prediction_error.item()

    def calculate_termination_target(self, continue_value, switch_value, current_reward=0.0):
        """
        Calculate whether the previous step should have terminated based on current knowledge.
        High termination target means the previous step was a peak we should have stopped at.
        
        Args:
            continue_value (float/tensor): Current predicted future value
            switch_value (float/tensor): Current predicted value of switching
            current_reward (float): Current step's reward (after the decision)
            prev_reward (float): Previous step's reward (at decision point)
            
        Returns:
            tensor: Termination probability between 0 and 1
        """
        # Ensure we're working with tensors
        if isinstance(continue_value, torch.Tensor):
            continue_value = continue_value.detach()
        if isinstance(switch_value, torch.Tensor):
            switch_value = switch_value.detach()
        
        # Convert everything to tensors if they aren't already
        current_reward = torch.tensor(current_reward, dtype=torch.float32)
        self.previous_reward = torch.tensor(self.previous_reward, dtype=torch.float32)
        
        # Calculate reward drop (how much worse did we get?)
        reward_drop = self.previous_reward - current_reward
        
        # Should have terminated if:
        # 1. Previous reward was high AND
        # 2. Current reward or predicted future value is low
        
        # Normalize previous reward to 0-1 range
        prev_reward_score = torch.sigmoid(self.previous_reward * 2)
        
        # Compute how "bad" the result of continuing was
        # High if current reward AND future value are both low
        future_loss = 1 - torch.sigmoid(current_reward + continue_value)
        
        # Combine: High termination if self.previous_reward was good AND continuing was bad
        termination_score = prev_reward_score * future_loss

        self.previous_reward = current_reward
        
        return termination_score

class MetaRL:
    def __init__(self, external_input_size, hidden_sizes, output_size, learning_rates, gamma=0.99, switch_cost=0.1, termination_threshold=0.9, min_activation=0.1, num_max_steps=100, option_value_threshold=0.5, curiosity_weight=0.5, epsilon=0.2, epsilon_decay = 0.99):
        self.n_layers = len(hidden_sizes)
        self.grus = nn.ModuleList()
        self.gamma = gamma
        self.switch_cost = switch_cost
        self.termination_threshold = termination_threshold
        self.min_activation = min_activation  # Minimum activation threshold for options
        self.num_max_steps = num_max_steps  # Maximum number of steps to take in the environment
        self.option_value_threshold = option_value_threshold  # Threshold for option value comparison

        self.error_threshold_multiplier = 2.0  # Multiplier for error threshold
        self.error_history_length = 5  # Number of recent errors to consider for control
        self.termination_factor = 0.5  # Factor for adaptive termination threshold


        self.curiosity_weight = curiosity_weight
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        for i in range(self.n_layers):
            print(f"Creating GRU {i + 1}/{self.n_layers}")

            if i < self.n_layers - 1:
                # If not the top layer, the input size is the hidden size of the above layer plus the output size of the current layer so that the chosen option is fed back as input
                layer_input_size = hidden_sizes[i + 1] + output_size
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


    def high_confidence_positions_kl(self, affordances, delta=0.1, eps=1e-8):

        # Convert to probabilities
        p = F.softmax(affordances, dim=0)
        n = affordances.numel()

        # For each option, compute d_i = log(p_i * n)
        divergence_contrib = torch.log(p + eps) + torch.log(torch.tensor(float(n)))

        # Options where divergence contribution is above delta are flagged
        indices = torch.nonzero(divergence_contrib > delta, as_tuple=False).view(-1)
        
        # Sort the selected indices by their corresponding affordance values in descending order
        if indices.numel() > 0:  # Check if any indices were selected
            sorted_vals, sorted_order = torch.sort(affordances[indices], descending=True)
            sorted_indices = indices[sorted_order]
        else:
            sorted_indices = indices

        return sorted_indices

    def rollout_option(self, ext, layer_idx, option_idx):

        current_gru = self.grus[layer_idx]
        # Set the option as the 1 hot input, basically temporarilly choosing and holding the option, during the rollout
        x = torch.zeros(current_gru.input_size if layer_idx == self.n_layers - 1 else self.grus[layer_idx + 1].hidden_size)
        x[option_idx] = 1
        accumulated_value = 0
        num_steps = 0
        output = torch.zeros(current_gru.output_size)
        with torch.no_grad():
            for t in range(1):
                num_steps += 1
                # The predicted next state is fed back in as the external input, producing imagined states
                output, predicted_value, ext, termination_prob, _ = self.feed(x=x, ext=ext, opt=output, layer_idx=layer_idx, execute=False)# Modified value accumulation: Emphasize termination probability
                accumulated_value += predicted_value * (self.gamma**t)  # Discounted accumulated value
                print(f"Rollout step {t}, predicted reward: {predicted_value}, accumulated reward: {accumulated_value}")
                print(f"Termination probability: {termination_prob}")
                print(f"State prediction: {ext}")
                # input()
                if termination_prob > self.termination_threshold:
                    print(f"Rollout for option {option_idx} in layer {layer_idx} completed in {num_steps}/{self.num_max_steps} steps with reward {accumulated_value}")

                    break

        return accumulated_value, ext

    def apply_control_over_options(self, x, option_idx, prediction_errors):

        print(f"Controlled x before: {x}")
        
        # Calculate error-based metrics
        if not prediction_errors:
            return x
        
        recent_errors = prediction_errors[-5:]  # Look at last 5 errors
        mean_error = np.mean(recent_errors)
        error_trend = np.mean(np.diff(recent_errors)) if len(recent_errors) > 1 else 0 #how are we differentiating between different options errors? it is just the overall. TODO: separate this.
        
        # Calculate adaptive gain
        # Base gain increases with error magnitude
        base_gain = 1 + (mean_error * 2)

        # Additional gain if errors are increasing
        trend_factor = max(0, error_trend * 5)
        total_gain = base_gain + trend_factor
        
        print(f"Mean error: {mean_error:.4f}, Error trend: {error_trend:.4f}, Total gain: {total_gain:.4f}")
        
        # --- Error Thresholding ---
        error_threshold = 0.3  # TODO: make this a parameter
        if mean_error < error_threshold:
            # print("Error below threshold. Minimal control.")
            return F.softmax(x, dim=0)  # Return original softmax (or slightly modified)

        # Create a copy of x
        controlled_x = x.clone()
        
        # Apply exponential boost to chosen option based on gain
        boost_factor = torch.exp(torch.tensor(total_gain))
        controlled_x[option_idx] *= boost_factor
        print(f"Controlled x after boost: {controlled_x}")
        
        # Calculate suppression factor based on distance from chosen option
        for i in range(len(controlled_x)):
            if i != option_idx:
                # Stronger suppression for options with higher initial values
                base_suppression = 1 - (mean_error * controlled_x[i])
                distance_factor = 1 / (1 + abs(i - option_idx))
                controlled_x[i] *= (base_suppression * distance_factor)
        
        print(f"Controlled x after suppression: {controlled_x}")
        
        # Apply softmax with temperature scaling
        temperature = max(0.1, 1 - mean_error)  # Lower temperature = sharper distribution
        controlled_x = F.softmax(controlled_x / temperature, dim=0)
        
        print(f"Controlled x after temperature-scaled normalization: {controlled_x}")
        
        return controlled_x
    
    def feed(self, x, ext, opt, layer_idx, execute=True):
        # Ensure x and ext are 2D before concatenation


        # If not the top layer, concatenate the external input. Otherwise, just use x, which is the external input
        if layer_idx < self.n_layers - 1: 
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Reshape to (1, hidden_size)
    
            if ext.dim() == 1:
                ext = ext.unsqueeze(0)  # Reshape ext if needed

            if opt.dim() == 1:
                opt = opt.unsqueeze(0)

            # Take only the first input_size elements from ext
            ext = ext[:, :self.grus[layer_idx].external_input_size]
            opt = opt[:, :self.grus[layer_idx].output_size]
            x = torch.cat([x, ext, opt], dim=1)  # Concatenate x and ext

        out = self.grus[layer_idx](x, execute)

        return out


    # Cant consider different ooptions if we are executing
    # Termination probability should remain high during planning, and go low to trigger execution

    def meta_step(self, x, env, layer_idx, ext):

        # Get the current GRU layer
        current_gru = self.grus[layer_idx]
        # Initialize loss, reward, effort
        loss = 0
        total_reward = 0
        prediction_errors = []
        output = torch.zeros(current_gru.output_size)

        done = False

        for step in range(self.num_max_steps):

            state, _, done  = env.get_outputs()

            # Forward pass for the current layer
            output, option_value_estimate, state_prediction, termination_prob, termination_logit = self.feed(x=x, ext=state, opt=output, layer_idx=layer_idx)

            # print(f"\nLayer {layer_idx}, Step {step}")
            # print(f"Option values: {option_value_estimate}")
            # print(f"State prediction: {state_prediction}")  # First 5 values
            # print(f"Actual state: {state}")
            # print(f"Prediction error: {torch.mean(torch.abs(state_prediction - state))}")
            # print(f"Termination probability: {termination_prob}")
            # print(f"Termination logit: {termination_logit}")
            # print(f"Output: {output}")
            # print(f"Epsilon: {self.epsilon}")

            # Detach output for option selection to prevent gradient issues
            affordances = output.clone().detach()

            # --- Error-Modulated Termination ---
            mean_error = 0 # set default
            if len(prediction_errors) > 0:
                mean_error = np.mean(prediction_errors[-self.error_history_length:]) # Get recent errors

            # adaptive_threshold = self.termination_threshold - (mean_error * self.termination_factor) # Decrease termination probability as the mean error increases
            # if termination_prob > adaptive_threshold:

            # Check if we should terminate or take another step in the environment/layer
            if termination_prob > self.termination_threshold:
                # print(f"Terminating at layer {layer_idx}, step {step}, adaptive termination prob: {termination_prob} > {self.termination_threshold}")

                break

            # Get the options based on the confidence of each option
            affordances_idx = self.high_confidence_positions_kl(affordances)
            # print(f"Top options after filtering: {affordances} pos: {affordances_idx}")

            if len(affordances_idx) == 0:
                # ...   No known options to choose from
                # Rollout?
                # print("No options to choose from")
                affordances_idx = torch.argmax(affordances).unsqueeze(0)
                print(affordances_idx)


            # Look at each option, holding the last one in working memory to compare it to the current one
            last_best_sub_option = affordances_idx[0]
            chosen_option_value = option_value_estimate.clone().detach()
            alternative_option_value = 0.0

            # Sort options by confidence and iterate

            for option_idx in affordances_idx:
                sub_option_confidence = affordances[option_idx].item()  # Convert to scalar
                if sub_option_confidence == 0:
                    break

                # Get the sub-option value for the current option by rolling out the option in the lower layer
                if layer_idx > 0:
                    sub_option_value, sub_predicted_ext = self.rollout_option(ext=ext, layer_idx=layer_idx - 1, option_idx=option_idx)

                    # TODO: Check the followeing (termination probabiltiy)
                    # Get the termination probability of the current layers goal from the terminal state of the lower layer.
                    with torch.no_grad():
                        _, _, _, termination_prob, _ = self.feed(x=x, ext=sub_predicted_ext, opt=output, layer_idx=layer_idx)
                        sub_option_value = sub_option_value * termination_prob 

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

                    chosen_option_value = float(sub_option_value) # The value for the chosen option

                # Eliminate unused option
                else:
                    affordances[option_idx] = 0

            last_best_sub_option = torch.argmax(affordances)

            # Choose the best option
            # set the output to the best option (1-hot)
            output = self.apply_control_over_options(output, last_best_sub_option, prediction_errors)
            controlled_output_affordances = output.clone().detach()
            # print(f"Chosen option: {last_best_sub_option}")
            # print(f"Chosen value: {chosen_option_value}")
            # Then observe the reward and effort

            # Take a step in the environment if we are at the bottom layer
            if layer_idx == 0:
                # Get action from the output of the bottom layer GRU
                action_probs = F.softmax(output, dim=0)
                # Epsilon-greedy action selection
                if np.random.rand() < self.epsilon:
                    # print("Random action")
                    action = np.random.choice(len(action_probs))
                else:
                    action = torch.argmax(action_probs).item()

                self.epsilon *= self.epsilon_decay  # Decay epsilon

                # Step in the environment
                _, reward, done = env.step(action)
                # Update effort based on environment's effort
                total_reward += reward

                # print(f"Action: {action}, Reward: {reward}, total reward: {total_reward}, Done: {done}")
                # If at the bottom layer and the episode is done, break the inner loop
                if done:
                    break
            else:
                # Recursively call meta_step for the lower layer
                loss_lower, reward, pred_errors_lower = self.meta_step(
                    x=output, env=env, layer_idx=layer_idx - 1, ext=state
                )
                loss += loss_lower
                total_reward += reward  # Accumulate reward from lower layers
                prediction_errors.extend(pred_errors_lower)

                # If a lower layer indicates termination, respect that decision
                if done:
                    break

            # Update weights for the current layer based on what we observed

            # Calculate continue value (value of continuing with the current option)
            net_reward = reward
            continue_value = chosen_option_value
            switch_value = (alternative_option_value) - self.switch_cost

            # Calculate termination target based on whether to continue or switch
            termination_target = current_gru.calculate_termination_target(continue_value, switch_value, reward)
            print(f"Continue value: {continue_value}, Switch value: {switch_value}")
            print(f"Termination target: {termination_target}")
            input()

            # Calculate value target (TD target)
            value_target = (net_reward + (continue_value * (1 - termination_target)))

            # Update weights for the current layer

            layer_loss, pred_error = current_gru.update_weights(
                value_estimate=option_value_estimate,
                state_prediction=state_prediction,
                termination_logit=termination_logit,
                value_target=value_target,
                actual_state=state,
                termination_target=termination_target,
                raw_output_affordances=affordances,
                controlled_output_affordances=controlled_output_affordances,
                layer_idx=layer_idx
            )

            prediction_errors.append(pred_error)
            loss += layer_loss

        return loss, total_reward, prediction_errors

    def compare_option_values(self, next_value, previous_value, alpha=0.8, beta=0.2):
        # Ensure we're working with scalar values
        if isinstance(next_value, torch.Tensor):
            next_value = next_value.item()
        if isinstance(previous_value, torch.Tensor):
            previous_value = previous_value.item()
            
        return (alpha * next_value / (1 + (beta * previous_value))) > self.option_value_threshold


class Environment:
    def __init__(self, seq):
        self.sequence = seq
        self.position = 0
        self.done = False
        self.action = 0

    def step(self, action):
        self.action = action
        return self.get_outputs()

    def get_outputs(self):
        if self.action == 1:
            if self.position < self.sequence.shape[0] - 1:
                self.position +=1

            else:
                self.done = True

        one_hot_position = torch.zeros_like(self.sequence)
        one_hot_position[self.position] = 1
        self.action = 0

        return one_hot_position, self.sequence[self.position - 1],  self.done # state, reward, done

# Define MetaRL parameters
external_input_size = 5
hidden_sizes = [2, 2]  # Example hidden sizes for two layers
output_size = 2  # 2 possible actions 0 and 1
learning_rates = [0.02, 0.05]  # Example learning rates

# Create MetaRL model
meta_rl = MetaRL(external_input_size, hidden_sizes, output_size, learning_rates,
                gamma=0.99, switch_cost=0.1,
                termination_threshold=0.5,
                min_activation=0.1,
                num_max_steps=5,
                option_value_threshold=0.5,
                curiosity_weight=0.5,
                epsilon=0.7,
                epsilon_decay=0.999)


states = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
env = Environment(seq=states)
state, reward, done = env.get_outputs()
meta_rl.meta_step(x=state, env=env, layer_idx=1, ext=state)
