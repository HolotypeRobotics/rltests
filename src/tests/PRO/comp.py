import torch
# ... (other imports)

# ... (OutcomeRepresentation class - no changes)

class PROControl(nn.Module):
    # ... (Existing __init__ code)
        self.outcome_rep = OutcomeRepresentation(n_stimuli, self.n_ro_conjunctions).to(device)
        self.optimizer_or = None # Add optimizer for outcome representation
    # ... (rest of the code)


class GoNoGoTrainer:
    def __init__(self, model, task, learning_rate=0.001, or_learning_rate=0.001):
        self.model = model
        self.task = task
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model.optimizer_or = optim.Adam(model.outcome_rep.parameters(), lr=or_learning_rate) # Initialize OR optimizer here
        # ... (rest of existing code)

    def run_trial(self, stimulus, correct_response, is_go_trial=None):
        """Run a single trial"""
        self.optimizer.zero_grad()
        self.model.optimizer_or.zero_grad()  # Zero grad for outcome rep optimizer

        # Forward pass
        response, pred_outcomes, pred_temporal, response_discrete = self.model(stimulus)

        # Evaluate response
        eval_results = self.evaluate_response(response, correct_response)

        # Create outcome vector
        actual_outcomes = torch.zeros(self.task.n_responses * self.task.n_outcomes, device=device)
        response_idx = int(torch.argmax(eval_results['response_made']).item())
        outcome_idx = 0 if eval_results['correct'] else 1
        actual_outcomes[response_idx * self.task.n_outcomes + outcome_idx] = 1.0

        # Reward (task-specific - needs to be implemented for each task)
        reward = 1.0 if eval_results['correct'] else -0.5  # Placeholder - replace with task-specific reward
        reward = torch.tensor([reward], device=device).float()

        # Outcome valence (task-specific)
        outcome_valence = actual_outcomes.clone().detach()  # Placeholder - replace with task-specific valence
        # Important: Detach outcome_valence to avoid double gradients through OutcomeRepresentation

        # Gating signal
        gating_signal = 1.0  # Placeholder - can be task-specific

        # Update weights (get losses from here)
        update_results = self.model.update_weights(stimulus, response, actual_outcomes, reward, outcome_valence, gating_signal)

        # Calculate actual_values (task-specific - depends on timing of outcomes)
        actual_values = reward + self.model.gamma * self.model.compute_temporal_prediction() # Placeholder - replace with task-specific target
        actual_values = actual_values.detach() # Detach to avoid gradients through temporal prediction pathway

        # Compute losses (using update_results)
        losses = {
            'total_loss': update_results['ro_loss'] + update_results['td_loss'] + update_results['valence_loss'],
            'ro_loss': update_results['ro_loss'].item(),
            'td_loss': update_results['td_loss'].item(),
            'valence_loss': update_results['valence_loss'].item()
        }

        # Backward pass (on total_loss only)
        update_results['ro_loss'].backward(retain_graph=True) # Backpropagate ro_loss
        update_results['td_loss'].backward(retain_graph=True) # Backpropagate td_loss
        update_results['valence_loss'].backward() # Backpropagate valence loss

        self.optimizer.step()
        self.model.optimizer_or.step() # Update outcome representation weights

        return losses, eval_results


    def train_epoch(self, n_trials):
        """Train for one epoch"""
        epoch_metrics = defaultdict(list)

        # ... (Task-specific training logic - examples below)

        if isinstance(self.task, GoNoGoTask):
            for _ in range(n_trials):
                stimulus, correct_response, is_go_trial = self.task.generate_trial()
                stimulus, correct_response = stimulus.to(device), correct_response.to(device)

                losses, eval_results = self.run_trial(stimulus, correct_response, is_go_trial) # Call run_trial ONLY ONCE

                # ... (Record metrics as before)

        # ... (Add similar blocks for other tasks - ChangeSignalTask, ForagingTask, etc.)

        return {k: np.mean(v) for k, v in epoch_metrics.items()}

    # ... (Rest of GoNoGoTrainer - train, plot_learning_curves)

# Example usage (no changes)
# ...