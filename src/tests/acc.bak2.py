import numpy as np

class ACCModel:
    def __init__(self, num_actions, num_outcomes, state_dim, alpha=0.1, gamma=0.9, k_effort=0.1, volatility_lr=0.1, switch_threshold=0.5):
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.state_dim = state_dim
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.k_effort = k_effort  # Effort discount parameter
        self.volatility_lr = volatility_lr  # Volatility learning rate
        self.switch_threshold = switch_threshold  # Threshold for task switching
        
        # State-action value (Q-values)
        self.state_action_values = np.zeros((state_dim, num_actions))
        
        # Outcome predictions for each state-action pair
        self.outcome_predictions = np.zeros((state_dim, num_actions, num_outcomes))
        
        # Initialize volatility, conflict, and performance history
        self.volatility = 0.1
        self.conflict = 0.5
        self.performance_history = []
        
    # Update functions for learning
    def update_state_action_values(self, state, action, reward):
        state_index = np.argmax(state)
        current_q = self.state_action_values[state_index, action]
        max_next_q = np.max(self.state_action_values[state_index])
        td_error = reward + self.gamma * max_next_q - current_q
        self.state_action_values[state_index, action] += self.alpha * td_error
    
    def update_outcome_predictions(self, state, action, observed_outcomes):
        state_index = np.argmax(state)
        prediction_errors = observed_outcomes - self.outcome_predictions[state_index, action]
        self.outcome_predictions[state_index, action] += self.alpha * prediction_errors
        return prediction_errors
    
    def update_volatility(self, prediction_errors, window_size=10):
        recent_errors = prediction_errors[-window_size:] if len(prediction_errors) >= window_size else prediction_errors
        volatility_change = np.var(recent_errors)
        self.volatility += self.volatility_lr * volatility_change
    
    # Core functions for decision-making
    def compute_subjective_value(self, reward, effort):
        return reward * (1 - self.k_effort * effort)
    
    def action_selection(self, state):
        state_index = np.argmax(state)
        action_values = np.zeros(self.num_actions)
        
        for action in range(self.num_actions):
            expected_value = np.mean(self.outcome_predictions[state_index, action])
            effort_cost = np.random.random()  # Placeholder for actual effort computation
            subjective_value = self.compute_subjective_value(expected_value, effort_cost)
            
            q_value = self.state_action_values[state_index, action]
            action_values[action] = 0.5 * subjective_value + 0.5 * q_value
        
        probabilities = np.exp(action_values) / np.sum(np.exp(action_values))
        return np.random.choice(self.num_actions, p=probabilities)
    
    # Task switching logic
    def detect_task_switch_need(self, performance, conflict):
        avg_performance = np.mean([p for _, p in self.performance_history])
        return avg_performance < 0.7 and conflict > self.switch_threshold
    
    def simulate_decision_making(self, num_trials, initial_state):
        choices, surprises, switches, states = [], [], [], [initial_state]
        
        for t in range(num_trials):
            current_state = states[-1]
            
            # Action selection
            action = self.action_selection(current_state)
            choices.append(action)
            
            # Simulate outcomes and reward (simplified)
            observed_outcomes = np.random.rand(self.num_outcomes)
            reward = np.mean(observed_outcomes)
            
            # Update models and compute surprise
            prediction_errors = self.update_outcome_predictions(current_state, action, observed_outcomes)
            surprise = -np.sum(np.clip(prediction_errors, None, 0))  # Higher means more surprise
            surprises.append(surprise)
            
            # Update Q-values and performance history
            self.update_state_action_values(current_state, action, reward)
            self.performance_history.append((current_state, reward))
            
            # Task switching
            if self.detect_task_switch_need(reward, self.conflict):
                switches.append(t)
            
            # Transition to next state (simplified)
            next_state = np.random.choice(self.state_dim, p=np.ones(self.state_dim)/self.state_dim)
            states.append(np.eye(self.state_dim)[next_state])
        
        return choices, surprises, switches, states

# Example usage
state_dim = 5  # Number of possible states
num_actions = 3
num_outcomes = 4

acc_model = ACCModel(num_actions=num_actions, num_outcomes=num_outcomes, state_dim=state_dim)
initial_state = np.eye(state_dim)[0]  # One-hot encoded initial state

choices, surprises, switches, states = acc_model.simulate_decision_making(num_trials=100, initial_state=initial_state)

print(f"Actions taken: {choices}")
print(f"Task switches: {switches}")
print(f"Average surprise: {np.mean(surprises)}")
print(f"Unique states visited: {len(set(map(tuple, states)))}")
