import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


SWITCH_COST = 0.1
EXPLORATION_PROB = 0.2  # Epsilon for epsilon-greedy exploration

class TDTwoBanditsLearner:
    def __init__(self, max_steps, alpha=0.1, gamma=1.0):  # Set gamma to 1.0
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        # Values for both machines (initialized to small random values)
        self.values_machine1 = np.random.uniform(0, 0.1, size=max_steps)
        self.values_machine2 = np.random.uniform(0, 0.1, size=max_steps)
        self.termination_probs = np.ones(max_steps) * 0.5
        self.current_machine = 0  # Keep track of the current machine

    def update(self, rewards1, efforts1, rewards2, efforts2):
        value_history1 = []
        value_history2 = []
        switch_probs = []

        # Update loop
        for t in range(self.max_steps):
            # Exploration vs. Exploitation
            if np.random.rand() < EXPLORATION_PROB:
                # Explore: choose a random machine
                current_machine = np.random.choice([0, 1])
            else:
                # Exploit: choose the machine with the highest value
                current_machine = 0 if self.values_machine1[t] > self.values_machine2[t] else 1
            
            self.current_machine = current_machine # Update the current machine

            if current_machine == 0:
                # Update for Machine 1
                if t < len(rewards1):
                    accumulated_reward = sum(rewards1[:t+1])
                    accumulated_effort = sum(efforts1[:t+1])
                    current_return = accumulated_reward - accumulated_effort

                    if t < self.max_steps - 1:
                        continue_value = (1 - self.termination_probs[t]) * self.values_machine1[t + 1]
                        switch_value = self.values_machine2[0] - SWITCH_COST

                        td_target = current_return + self.gamma * max(continue_value, switch_value)
                        td_error = td_target - self.values_machine1[t]
                        self.values_machine1[t] += self.alpha * td_error

                        # Update termination probability only for the current machine
                        # value_diff = continue_value - switch_value
                        value_diff = switch_value - continue_value
                        self.termination_probs[t] = 1 / (1 + np.exp(value_diff))
                    else:
                        td_error = current_return - self.values_machine1[t]
                        self.values_machine1[t] += self.alpha * td_error

                    value_history1.append(self.values_machine1[t])
                    switch_probs.append(self.termination_probs[t])
            else:
                # Update for Machine 2 (similar process)
                if t < len(rewards2):
                    accumulated_reward = sum(rewards2[:t+1])
                    accumulated_effort = sum(efforts2[:t+1])
                    current_return = accumulated_reward - accumulated_effort

                    if t < self.max_steps - 1:
                        continue_value = (1 - self.termination_probs[t]) * self.values_machine2[t + 1]
                        switch_value = self.values_machine1[0] - SWITCH_COST

                        td_target = current_return + self.gamma * max(continue_value, switch_value)
                        td_error = td_target - self.values_machine2[t]
                        self.values_machine2[t] += self.alpha * td_error
                    else:
                        td_error = current_return - self.values_machine2[t]
                        self.values_machine2[t] += self.alpha * td_error

                    value_history2.append(self.values_machine2[t])
                    switch_probs.append(self.termination_probs[t])

        return value_history1, value_history2, switch_probs

def visualize_learning(rewards1, efforts1, rewards2, efforts2, n_passes=3):
    max_steps = max(len(rewards1), len(rewards2))
    learner = TDTwoBanditsLearner(max_steps)
    
    # Track history for visualization
    value_history1_all = []
    value_history2_all = []
    switch_probs_all = []
    
    # Training loop
    for i in range(n_passes):
        vh1, vh2, sp = learner.update(rewards1, efforts1, rewards2, efforts2)
        value_history1_all.append(vh1)
        value_history2_all.append(vh2)
        switch_probs_all.append(sp)
    
    # Plotting
    timesteps = range(max_steps)
    plt.style.use('rose-pine')
    plt.figure(figsize=(15, 12))
    
    # Plot rewards and efforts for both machines
    plt.subplot(3, 2, 1)
    plt.plot(timesteps[:len(rewards1)], rewards1, marker='o', label='Reward M1', color='green')
    plt.plot(timesteps[:len(efforts1)], efforts1, marker='x', label='Effort M1', color='red')
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("Machine 1: Reward and Effort")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(timesteps[:len(rewards2)], rewards2, marker='o', label='Reward M2', color='blue')
    plt.plot(timesteps[:len(efforts2)], efforts2, marker='x', label='Effort M2', color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("Machine 2: Reward and Effort")
    plt.legend()
    plt.grid(True)
    
    # Plot value estimates over training
    plt.subplot(3, 2, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, n_passes))
    for i in range(n_passes):
        plt.plot(timesteps[:len(value_history1_all[i])], value_history1_all[i], 
                label=f'Pass {i+1}', color=colors[i])
    plt.xlabel("Timestep")
    plt.ylabel("Value Estimate")
    plt.title("Machine 1: Value Estimates Over Training")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    for i in range(n_passes):
        plt.plot(timesteps[:len(value_history2_all[i])], value_history2_all[i], 
                label=f'Pass {i+1}', color=colors[i])
    plt.xlabel("Timestep")
    plt.ylabel("Value Estimate")
    plt.title("Machine 2: Value Estimates Over Training")
    plt.legend()
    plt.grid(True)
    
    # Plot switching probabilities
    plt.subplot(3, 2, 5)
    for i in range(n_passes):
        plt.plot(timesteps[:len(switch_probs_all[i])], switch_probs_all[i], 
                label=f'Pass {i+1}', color=colors[i])
    plt.xlabel("Timestep")
    plt.ylabel("Switching Probability")
    plt.title("Probability of Switching Machines")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Machine 1: High early rewards
    rewards1 = [0, 2, 0, 5, 0, 0, 0, 0, 0, 0]
    efforts1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    # Machine 2: Steady smaller rewards
    rewards2 = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    efforts2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    visualize_learning(rewards1, efforts1, rewards2, efforts2, n_passes=3)