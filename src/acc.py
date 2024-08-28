import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ACCModel:
    def __init__(self, num_actions, num_outcomes, state_dim, alpha=0.1, gamma=0.9, k_effort=0.1, volatility_lr=0.1, switch_threshold=0.5, temperature=1.0):
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.state_dim = state_dim
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.k_effort = k_effort  # Effort discount parameter
        self.volatility_lr = volatility_lr  # Volatility learning rate
        self.switch_threshold = switch_threshold  # Threshold for task switching
        self.temperature = temperature  # Temperature for softmax action selection

        # Weights (GRU)
        
        
        # Initialize outcome predictions for each state-action pair
        self.outcome_predictions = np.zeros((state_dim, num_actions, num_outcomes))
        
        # Initialize state-action values
        self.state_action_values = np.zeros((state_dim, num_actions))

    def predict_outcomes(self, state, action):
        """Predict multiple outcomes for a given state-action pair"""
        state_index = np.argmax(state)  # Assuming one-hot encoded state
        return self.outcome_predictions[state_index, action]

    def update_predictions(self, state, action, observed_outcomes):
        """Update outcome predictions using temporal difference learning"""
        state_index = np.argmax(state)
        predictions = self.predict_outcomes(state, action)
        prediction_errors = observed_outcomes - predictions
        
        # TD learning update
        self.outcome_predictions[state_index, action] += self.alpha * prediction_errors
        
        return prediction_errors

    def compute_surprise(self, prediction_errors):
        """Compute surprise based on unexpected non-occurrences"""
        negative_pes = np.clip(prediction_errors, None, 0)
        return -np.sum(negative_pes)  # Higher value indicates more surprise

    def update_state_action_values(self, state, action, reward):
        """Update state-action values using Q-learning"""
        state_index = np.argmax(state)
        current_q = self.state_action_values[state_index, action]
        max_next_q = np.max(self.state_action_values[state_index])
        td_error = reward + self.gamma * max_next_q - current_q
        self.state_action_values[state_index, action] += self.alpha * td_error

    def action_selection(self, state):
        """Select action based on predicted outcomes, surprise, and state-action values"""
        state_index = np.argmax(state)
        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            predictions = self.predict_outcomes(state, action)
            expected_value = np.mean(predictions)
            potential_surprise = self.compute_surprise(-predictions)
            q_value = self.state_action_values[state_index, action]
            
            # Combine value, surprise, and Q-value
            action_values[action] = 0.4 * expected_value + 0.3 * potential_surprise + 0.3 * q_value

        # Softmax decision rule
        probabilities = np.exp(action_values / self.temperature) / np.sum(np.exp(action_values / self.temperature))
        return np.random.choice(self.num_actions, p=probabilities)

    def detect_task_switch_need(self, surprise, volatility, conflict):
        """Detect if a task switch is needed based on surprise, volatility, and conflict"""
        switch_probability = surprise * volatility * conflict
        return switch_probability > self.switch_threshold

    def update_volatility(self, volatility, prediction_error):
        """Update volatility estimate"""
        return np.clip(volatility + self.volatility_lr * np.abs(prediction_error), 0, 1)

    def compute_conflict(self, action_values):
        """Compute conflict as the similarity between top action values"""
        sorted_values = np.sort(action_values)[::-1]
        return 1 - (sorted_values[0] - sorted_values[1]) / sorted_values[0] if sorted_values[0] != 0 else 1

    def simulate_decision_making(self, num_trials, initial_state, transition_prob=0.1):
        """Simulate ACC-based decision-making process with PL state"""
        choices = []
        surprises = []
        switches = []
        states = [initial_state]
        volatility = 0.1
        conflict = 0.5

        for _ in range(num_trials):
            current_state = states[-1]
            
            # Select action
            action = self.action_selection(current_state)
            choices.append(action)

            # Generate outcomes and reward (simplified)
            observed_outcomes = np.random.rand(self.num_outcomes)
            reward = np.mean(observed_outcomes)  # Simplified reward

            # Update predictions and compute surprise
            prediction_errors = self.update_predictions(current_state, action, observed_outcomes)
            surprise = self.compute_surprise(prediction_errors)
            surprises.append(surprise)

            # Update state-action values
            self.update_state_action_values(current_state, action, reward)

            # Compute conflict
            action_values = [self.state_action_values[np.argmax(current_state), a] for a in range(self.num_actions)]
            conflict = self.compute_conflict(action_values)

            # Update volatility
            volatility = self.update_volatility(volatility, np.mean(prediction_errors))

            # Check for task switch
            if self.detect_task_switch_need(surprise, volatility, conflict):
                switches.append(_)

            # Transition to next state (simplified)
            if np.random.rand() < transition_prob:
                next_state = np.zeros(self.state_dim)
                next_state[np.random.choice(self.state_dim)] = 1
            else:
                next_state = current_state
            states.append(next_state)

        return choices, surprises, switches, states

# Example usage
state_dim = 5  # Number of possible PL states
num_actions = 3
num_outcomes = 4
acc_model = ACCModel(num_actions=num_actions, num_outcomes=num_outcomes, state_dim=state_dim)

# Create initial state (one-hot encoded)
initial_state = np.zeros(state_dim)
initial_state[0] = 1

choices, surprises, switches, states = acc_model.simulate_decision_making(num_trials=100, initial_state=initial_state)

print(f"Number of actions: {len(choices)}")
print(f"Number of switches: {len(switches)}")
print(f"Average surprise: {np.mean(surprises)}")
print(f"Number of unique states visited: {len(set([tuple(state) for state in states]))}")