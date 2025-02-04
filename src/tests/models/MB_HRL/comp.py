import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from environment import Environment

# ... (OnlineGRU class remains the same)

class MetaRL:
    # ... (other methods remain the same)

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

            print(f"\nLayer {layer_idx}, Step {step}")
            print(f"Option values: {option_value_estimate}")
            print(f"State prediction: {state_prediction[:5]}...")  # First 5 values
            print(f"Actual state: {env.state[:5]}...")
            print(f"Prediction error: {torch.mean(torch.abs(state_prediction - env.state))}")

            # Detach output for option selection to prevent gradient issues
            output_detached = output.detach()

            # Check if we should terminate or take another step in the environment/layer
            if termination_prob > self.termination_threshold:
                print(f"Terminating at layer {layer_idx}, step {step}, termination_prob: {termination_prob} > {self.termination_threshold}")
                break

            # Get the options based on the confidence of each option
            affordances = self.filter_options(output_detached)
            print(f"Available options after filtering: {affordances}")

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

                    alternative_option_value = float(chosen_option_value)
                    alternative_option_confidence = chosen_option_confidence

                    chosen_option_value = float(sub_option_value)
                    chosen_option_confidence = sub_option_confidence

                # Eliminate unused option
                else:
                    affordances[option_idx] = 0

            # Choose the best option
            output = self.apply_control_over_options(output, last_best_sub_option)
            print(f"Chosen option: {last_best_sub_option}")
            print(f"Chosen value: {chosen_option_value}")

            # Take a step in the environment if we are at the bottom layer
            if layer_idx == 0:
                # Get action from the output of the bottom layer GRU
                action_probs = F.softmax(output, dim=0)
                action = torch.argmax(action_probs).item()

                # Step in the environment
                next_state, reward, done = env.step(action)
                next_ext = torch.tensor(next_state, dtype=torch.float32)

                # Update effort based on environment's effort
                effort = env.effort  # Get effort from the environment after the step
                total_reward += reward

                print(f"Action: {env.action_to_string(action)}, Reward: {reward}, Effort: {effort}, Done: {done}")

                # If at the bottom layer and the episode is done, break the inner loop
                if done:
                    break
            else:
                # Recursively call meta_step for the lower layer
                loss_lower, reward_lower, effort_lower, pred_errors_lower = self.meta_step(
                    x=output, env=env, layer_idx=layer_idx - 1, ext=ext
                )
                loss += loss_lower
                total_reward += reward_lower  # Accumulate reward from lower layers
                effort += effort_lower  # Accumulate effort from lower layers
                prediction_errors.extend(pred_errors_lower)

                # If a lower layer indicates termination, respect that decision
                if done:
                    break

            # Prepare for the next step: update state and external input
            if layer_idx == 0:
                ext = next_ext  # Update external input for the next step in the bottom layer
            else:
                # For higher layers, use the output of the current layer as input for the next step
                x = output

            # Update weights for the current layer
            state, _, _ = env.get_outputs()
            state = torch.from_numpy(state).float()
            continue_value = (chosen_option_value * chosen_option_confidence) * (
                        1 - mean_error if len(prediction_errors) > 0 else 0)
            switch_value = alternative_option_value * alternative_option_confidence
            termination_target = current_gru.calculate_termination_target(continue_value, switch_value)
            value_target = (reward - effort + (self.gamma * chosen_option_value * (1 - termination_target)))

            layer_loss, pred_error = current_gru.update_weights(
                option_value_estimate, state_prediction, termination_logit,
                value_target, state, termination_target, layer_idx
            )
            prediction_errors.append(pred_error)
            loss += layer_loss

        return loss, total_reward, effort, prediction_errors

def train_meta_rl(env, meta_rl, num_episodes, batch_size):
    for episode in range(num_episodes):
        # Reset the environment and get initial state
        s, _, done = env.reset()
        ext = torch.tensor(s, dtype=torch.float32)  # Convert to tensor

        # Initialize hidden states for all GRUs
        for gru in meta_rl.grus:
            gru.hidden = gru.init_hidden()

        total_reward = 0
        episode_steps = 0

        # Meta-learning step for the top layer (recursively processes lower layers)
        loss, reward, effort, prediction_errors = meta_rl.meta_step(
            x=ext,  # Pass 'ext' as input to the top-level GRU
            env=env,
            layer_idx=meta_rl.n_layers - 1,
            ext=ext
        )
        total_reward += reward
        episode_steps += effort

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Steps: {episode_steps}")
        env.plot_path(meta_rl)  # Assuming you want to visualize the path after each episode

# Example Usage (remains the same)