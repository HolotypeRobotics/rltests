import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class TDTerminationLearner:
    def __init__(self, max_steps, alpha=0.9, gamma=0.75):
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.values = np.zeros(max_steps)  # State values
        self.termination_probs = np.ones(max_steps) * 0.5
        
    def update(self, rewards, efforts):
        accumulated_reward = 0
        accumulated_effort = 0
        value_history = []
        
        # Forward pass to accumulate rewards and efforts
        for t in range(self.max_steps):
            if t < len(rewards):
                accumulated_reward += rewards[t]
                accumulated_effort += efforts[t]
                
                # Current return is accumulated_reward minus accumulated_effort
                current_return = accumulated_reward - accumulated_effort
                
                # TD update
                if t < self.max_steps - 1:
                    # Calculate TD error
                    td_target = current_return + (1-self.termination_probs[t+1]) * self.values[t + 1]
                    td_error = td_target - self.values[t]
                    self.values[t] += self.alpha * td_error
                else:
                    # Terminal state
                    td_error = current_return - self.values[t]
                    self.values[t] += self.alpha * td_error
                
                # Update termination probability based on value comparison
                if t > 0:
                    # Compare current value with previous value
                    value_diff = self.values[t] - self.values[t-1]
                    # If value is increasing, decrease termination probability
                    # If value is decreasing, increase termination probability
                    self.termination_probs[t] += self.alpha * (-np.tanh(value_diff))
                    self.termination_probs[t] = np.clip(self.termination_probs[t], 0.01, 0.99)
                
                value_history.append(self.values[t])
        
        return value_history

def visualize_learning(rewards, efforts, n_passes=3):
    max_steps = len(rewards)
    learner = TDTerminationLearner(max_steps)
    
    # Track history for visualization
    prob_history = []
    value_history = []
    
    # Training loop
    for i in range(n_passes):
        values = learner.update(rewards, efforts)
        value_history.append(values)
        prob_history.append(learner.termination_probs.copy())
    
    # Plotting
    timesteps = range(max_steps)
    plt.style.use('rose-pine')
    plt.figure(figsize=(15, 10))
    
    # Plot rewards and efforts
    plt.subplot(2, 2, 1)
    plt.plot(timesteps, rewards, marker='o', label='Reward', color='green')
    plt.plot(timesteps, efforts, marker='x', label='Effort', color='red')
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("Reward and Effort over Time")
    plt.legend()
    plt.grid(True)
    
    # Plot termination probabilities over training
    plt.subplot(2, 2, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, n_passes))
    for i in range(n_passes):
        plt.plot(timesteps, prob_history[i], marker='o', 
                label=f'Pass {i+1}', color=colors[i])
    plt.xlabel("Timestep")
    plt.ylabel("Termination Probability")
    plt.title("Termination Probabilities Over Training")
    plt.legend()
    plt.grid(True)
    
    # Plot value estimates over training
    plt.subplot(2, 2, 3)
    for i in range(n_passes):
        plt.plot(timesteps, value_history[i], marker='o', 
                label=f'Pass {i+1}', color=colors[i])
    plt.xlabel("Timestep")
    plt.ylabel("Value Estimate")
    plt.title("Value Estimates Over Training")
    plt.legend()
    plt.grid(True)
    
    # Plot final accumulated values
    plt.subplot(2, 2, 4)
    accumulated_rewards = np.cumsum(rewards)
    accumulated_efforts = np.cumsum(efforts)
    net_value = accumulated_rewards - accumulated_efforts
    plt.plot(timesteps, net_value, marker='o', 
            label='Net Value', color='purple')
    plt.xlabel("Timestep")
    plt.ylabel("Accumulated Value")
    plt.title("Net Accumulated Value (Reward - Effort)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    rewards = [0, 2, 0, 5, 0, 0, 0, 0, 0, 0]  # Example reward sequence
    efforts = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Example effort sequence
    visualize_learning(rewards, efforts, n_passes=3)