# ACC Model for Action Selection and Task Switching

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
        
        # Initialize outcome predictions for each state-action pair
        self.outcome_predictions = np.zeros((state_dim, num_actions, num_outcomes))
        
        # Initialize state-action values
        self.state_action_values = np.zeros((state_dim, num_actions))


    def value_representation_update(self, v_old, reward):
        """Update value representation based on feedback"""
        prediction_error = reward - v_old
        v_new = v_old + self.alpha * prediction_error
        return v_new, prediction_error

    def expected_value_of_control(self, p_states, rewards, costs):
        """Compute the Expected Value of Control"""
        return np.sum([p * (r - c) for p, r, c in zip(p_states, rewards, costs)])

    def temporal_discounting(self, reward, time, k=0.1):
        """Apply temporal discounting to a reward"""
        return reward / (1 + k * time)

    def conflict_monitoring(self, p_correct, p_alternatives, prediction_error):
        """Compute the conflict signal based on action probabilities and prediction error"""
        conflict = np.sum(p_correct * (1 - p_correct)) + np.sum([p_alt * (1 - p_alt) for p_alt in p_alternatives])
        # Incorporating prediction error into conflict
        conflict += np.abs(prediction_error) * 0.5  # Adjust weight as needed
        return conflict


    def cost_benefit_ratio(self, reward, effort):
        """Compute the cost-benefit ratio"""
        return reward / effort

    def error_related_negativity(self, erp_error, erp_correct):
        """Compute the Error-Related Negativity"""
        return erp_error - erp_correct

    def surprise(self, probability):
        """Compute surprise based on information theory"""
        return -np.log2(probability)

    # def update_volatility(self, volatility, prediction_error):
    #     """Update volatility based on prediction error and its variance"""
    #     volatility_change = np.var(prediction_error)  # Larger variance increases volatility
    #     return volatility + self.volatility_lr * volatility_change
    
    def update_volatility(self, volatility, prediction_errors, window_size=10):
        """Update volatility based on the variance of prediction errors over time"""
        # Track a moving window of recent prediction errors
        recent_errors = prediction_errors[-window_size:] if len(prediction_errors) >= window_size else prediction_errors
        volatility_change = np.var(recent_errors)
        return volatility + self.volatility_lr * volatility_change


    def subjective_value(self, reward, effort):
        """Compute subjective value with effort discounting"""
        return reward * (1 - self.k_effort * effort)

    def dynamic_subjective_value(self, reward, effort, recent_performance, fatigue=0.05):
        """Compute subjective value with dynamically adjusted effort cost"""
        # Adjust effort discounting based on recent performance and accumulated fatigue
        dynamic_effort = effort * (1 + fatigue * (1 - recent_performance))
        return self.subjective_value(reward, dynamic_effort)


    def action_outcome_contingency(self, p_outcome_given_action, p_outcome_given_no_action):
        """Compute action-outcome contingency"""
        return p_outcome_given_action - p_outcome_given_no_action

    def hierarchical_error(self, reward, value_next, value_current):
        """Compute hierarchical prediction error"""
        return reward + self.gamma * value_next - value_current

    def hierarchical_error_update(self, reward, value_next, value_current, uncertainty):
        """Compute hierarchical prediction error with uncertainty weighting"""
        error = self.heirarchical_error(reward, value_next, value_current)
        # Adjust the weight of the error based on uncertainty (could represent task volatility)
        weighted_error = error / (1 + uncertainty)
        return weighted_error

    # Proper goal-directed behavior requires that an organism internalizes
    # behaviorally relevant stimuli, brings appropriate stimulus–response
    # rules online, and monitors outcomes to adjust behavior accordingly.
    # Preparatory attention or the selection of task-relevant stimuli is
    # critical for the execution of proper goal-directed behavior.
    def preparatory_attention(self, x, x_bar, alpha=0.1):
        """Compute preparatory attention gain"""
        return 1 + alpha * (x - x_bar)
    
    def adjust_preparatory_attention(self, conflict_signal, task_relevance):
        """Adjust preparatory attention allocation based on conflict and task relevance"""
        attention_gain = 1 + self.alpha * (conflict_signal + task_relevance)
        return np.clip(attention_gain, 0.5, 2.0)  # Limit gain to prevent extreme focus

    def adjust_learning_rate(self, volatility, base_lr=0.1):
        """Adjust the learning rate based on volatility"""
        return base_lr * (1 + volatility)

    def task_switching_cost(self, current_task, previous_task, switch_cost=0.2):
        """Compute task switching cost"""
        return switch_cost if current_task != previous_task else 0

    def select_task_set(self, task_sets, context, performance_history):
        """Select the appropriate task set based on hierarchical relevance"""
        top_level_task = self.select_top_level_goal(task_sets)
        sub_tasks = top_level_task.get_sub_tasks()
        
        # Evaluate subtasks based on performance and context
        task_set_scores = {}
        for sub_task in sub_tasks:
            context_score = self.compute_context_relevance(sub_task, context)
            performance_score = self.compute_historical_performance(sub_task, performance_history)
            task_set_scores[sub_task] = 0.5 * context_score + 0.5 * performance_score

        return max(task_set_scores, key=task_set_scores.get)

    def compute_context_relevance(self, task_set, context):
        """Compute the relevance of a task set to the current context based on feature overlap"""
        feature_overlap = np.dot(task_set, context)  # Assuming task_set and context are feature vectors
        return feature_overlap / (np.linalg.norm(task_set) * np.linalg.norm(context))


    def compute_historical_performance(self, task_set, performance_history):
        """Compute the historical performance of a task set"""
        relevant_performances = [perf for ts, perf in performance_history if ts == task_set]
        return np.mean(relevant_performances) if relevant_performances else 0.5

    def simulate_task_switching(self, num_trials, task_sets, initial_task):
        """Simulate ACC-based task switching process"""
        current_task = initial_task
        performance_history = []
        task_switches = []
        volatility = 0.1
        conflict = 0.5

        for _ in range(num_trials):
            # Simulate task performance (simplified)
            performance = np.random.normal(0.7, 0.1)  # Mean performance of 0.7 with some variance
            performance_history.append((current_task, performance))

            # Update volatility (simplified)
            volatility = self.update_volatility(volatility, 0.7 - performance)

            # Detect if a task switch is needed
            if self.detect_task_switch_need(performance, volatility, conflict):
                # Select new task set
                new_task = self.select_task_set(task_sets, context="current_context", performance_history=performance_history)
                
                if new_task != current_task:
                    task_switches.append((_+1, current_task, new_task))
                    current_task = new_task

            # Update conflict (simplified)
            conflict = max(0, min(1, conflict + np.random.normal(0, 0.1)))

        return performance_history, task_switches


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

    def reward_prediction_error(self, reward, expected_reward):
        """Compute reward prediction error similar to dopamine signals"""
        return reward - expected_reward


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

    def compute_effort_cost(self, action):
        """Placeholder for computing effort cost based on action properties"""
        # In real models, effort cost might depend on action difficulty, distance, etc.
        return np.random.random()

    def action_selection(self, state):
        """Select action based on predicted outcomes, effort cost, and state-action values"""
        state_index = np.argmax(state)
        action_values = np.zeros(self.num_actions)
        
        for action in range(self.num_actions):
            predictions = self.predict_outcomes(state, action)
            expected_value = np.mean(predictions)
            effort_cost = self.compute_effort_cost(action)
            subjective_value = self.subjective_value(expected_value, effort_cost)
            
            q_value = self.state_action_values[state_index, action]
            
            action_values[action] = 0.5 * subjective_value + 0.5 * q_value
        
        # Softmax decision rule
        probabilities = np.exp(action_values) / np.sum(np.exp(action_values))
        return np.random.choice(self.num_actions, p=probabilities)

    def detect_task_switch_need(self, prediction_error, surprise, volatility, conflict):
        """Detect task switch based on prediction error, surprise, volatility, and conflict"""
        switch_signal = prediction_error * (surprise + volatility + conflict)
        return switch_signal > self.switch_threshold
    
    def detect_task_switch_need(self, performance, volatility, conflict, historical_performance, performance_threshold=0.7):
        """Detect task switch based on performance, conflict, and historical performance"""
        avg_performance = np.mean(historical_performance)
        if avg_performance < performance_threshold and conflict > self.switch_threshold:
            return True
        return False

    def uncertainty_modulated_action_selection(self, state, uncertainty):
        """Action selection modulated by uncertainty in state-action values"""
        state_index = np.argmax(state)
        action_values = self.state_action_values[state_index]
        
        # Modulate the action values by uncertainty (e.g., higher uncertainty could encourage exploration)
        action_values += np.random.randn(len(action_values)) * uncertainty
        
        probabilities = np.exp(action_values) / np.sum(np.exp(action_values))
        return np.random.choice(self.num_actions, p=probabilities)


    def simulate_decision_making(self, num_trials, initial_state, transition_prob=0.1, volatility=0.1, conflict=0.5):
        """Simulate ACC-based decision-making process with PL state"""
        choices = []
        surprises = []
        switches = []
        states = [initial_state]

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

            # Update volatility and conflict (simplified)
            volatility = max(0, min(1, volatility + np.random.normal(0, 0.1)))
            conflict = max(0, min(1, conflict + np.random.normal(0, 0.1)))

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