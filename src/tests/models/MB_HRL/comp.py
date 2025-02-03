class OnlineGRU(nn.Module):
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
    def meta_step(self, x, env, layer_idx, ext):
        current_gru = self.grus[layer_idx]
        loss = 0
        total_reward = 0
        reward = 0
        effort = 0

        for step in range(self.num_max_steps):
            # Ensure input tensors are properly shaped
            if isinstance(x, torch.Tensor) and x.dim() == 1:
                x = x.unsqueeze(0)
            
            output, option_value_estimate, state_prediction, termination_prob, termination_logit = self.feed(x=x, ext=ext, layer_idx=layer_idx)
            
            # Detach output for option selection to prevent gradient issues
            output_detached = output.detach()
            
            if termination_prob > self.termination_threshold:
                break

            # Get the options based on the confidence of each option
            affordances = self.filter_options(output_detached)

            # Look at each option
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

                # Get sub-option value
                if layer_idx > 0:
                    sub_option_value = self.rollout_option(ext=ext, layer_idx=layer_idx - 1, option_idx=option_idx)
                else:
                    sub_option_value = chosen_option_value

                # Compare values
                if isinstance(sub_option_value, torch.Tensor):
                    sub_option_value = sub_option_value.detach()
                if isinstance(chosen_option_value, torch.Tensor):
                    chosen_option_value = chosen_option_value.detach()

                if self.compare_option_values(
                    next_value=sub_option_value,
                    previous_value=chosen_option_value
                ):
                    last_best_sub_option = option_idx
                    alternative_option_value = float(chosen_option_value)  # Convert to scalar
                    alternative_option_confidence = chosen_option_confidence
                    chosen_option_value = float(sub_option_value)  # Convert to scalar
                    chosen_option_confidence = sub_option_confidence
                else:
                    affordances[option_idx] = 0

            # Apply control and get new output
            output = self.apply_control_over_options(output, last_best_sub_option)

            # Process reward and effort
            if layer_idx > 0:
                l, r, e = self.meta_step(x=output, env=env, layer_idx=layer_idx - 1, ext=ext)
                loss += l
                reward += r
                effort += e
            else:
                r, e = env.step(output)
                reward += r
                effort += e

            # Calculate values for termination
            continue_value = chosen_option_value * chosen_option_confidence
            switch_value = alternative_option_value * alternative_option_confidence

            # Calculate targets
            termination_target = current_gru.calculate_termination_target(continue_value, switch_value)
            value_target = torch.tensor(reward - effort + (self.gamma * chosen_option_value * (1 - termination_target)))

            # Update weights
            layer_loss = current_gru.update_weights(
                option_value_estimate,
                state_prediction,
                termination_logit,
                value_target,
                env.state,
                termination_target,
                layer_idx
            )
            loss += layer_loss

        return loss, reward, effort

    def compare_option_values(self, next_value, previous_value, alpha=0.8, beta=0.2):
        # Ensure we're working with scalar values
        if isinstance(next_value, torch.Tensor):
            next_value = next_value.item()
        if isinstance(previous_value, torch.Tensor):
            previous_value = previous_value.item()
            
        return (alpha * next_value / (1 + (beta * previous_value))) > self.option_value_threshold