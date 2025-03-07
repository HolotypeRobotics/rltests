import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


# Ensure results directory exists
os.makedirs('results', exist_ok=True)

class SequenceEnv:
    """Environment with a sequence of states, rewards, and effort costs"""
    def __init__(self, seq, topology):
        self.seq = np.array(seq, dtype=np.float32)
        self.topology = np.array(topology, dtype=np.float32)
        self.seq_len = len(seq)
        self.max_index = self.seq_len - 1
        self.index = 0
        self.reset()

    def reset(self):
        self.index = 0
        self.done = False
        self.prev_action = np.zeros(2, dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        """Create one-hot encoding of position (place cell representation)"""
        pos_onehot = np.zeros(self.seq_len, dtype=np.float32)
        pos_onehot[self.index] = 1.0
        obs = np.concatenate([pos_onehot, self.prev_action])
        return obs

    def step(self, action, terminate=False): # Renamed terminate to action_terminate, but using action directly now
        """Take a step in the environment"""
        if self.done:
            raise RuntimeError("Episode already terminated.")
        effort = 0.0001
        reward = 0.0
        action_terminate = False # Initialize action_terminate

        if action == 1 and self.index < self.max_index: # Action 1 is still "move"
            prev_topology = self.topology[self.index]
            prev_reward = self.seq[self.index]
            self.index += 1
            effort = self.topology[self.index] - prev_topology
            reward = self.seq[self.index] - prev_reward
        elif action == 0: # Action 0 is now "terminate"
            action_terminate = True # Set terminate flag if action is 0

        self.prev_action = np.eye(2, dtype=np.float32)[action]

        if action_terminate or self.index == self.max_index: # Check action_terminate for termination
            self.done = True

        next_obs = self._get_obs() if not self.done else None
        return next_obs, reward, effort, self.done, self.index


class SimpleNeuralAgent(nn.Module):
    """
    Simplified Neural Agent with action 0 as terminate.
    """
    def __init__(self, input_dim, hidden_dim=64, num_actions=2): # num_actions still 2
        super(SimpleNeuralAgent, self).__init__()

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy and value - No term_head anymore
        self.policy_head = nn.Linear(hidden_dim, num_actions)  # Action 0: terminate, Action 1: move
        self.value_head = nn.Linear(hidden_dim, 1)   # Expected value

        # Input Prediction Head
        self.prediction_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, state):
        """Forward pass: features, policy logits, value, input prediction"""
        features = self.feature_net(state)
        logits = self.policy_head(features) # Policy logits for action 0 (terminate) and 1 (move)
        value = self.value_head(features)
        input_prediction = self.prediction_head(features)
        return logits, value, input_prediction # Removed term_prob from output


def train_agent(env, agent, optimizer, num_episodes=1000):
    """Train agent with action 0 as terminate."""
    rewards_history = []
    positions_history = []
    surprise_history = []

    # Track surprise at each position
    position_surprise = np.zeros(env.seq_len)
    surprise_count = np.zeros(env.seq_len)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False

        episode_reward = 0
        episode_effort = 0
        episode_steps = 0

        # Storage for previous prediction
        prev_input_pred = None
        position_specific_errors = np.zeros(env.seq_len)

        while not done:
            # Get policy, value, and input prediction - No term_prob anymore
            logits, value, input_prediction = agent.forward(state)

            # Calculate input prediction error
            input_prediction_error = 0.0
            if prev_input_pred is not None:
                input_prediction_error = F.mse_loss(prev_input_pred, state).item()

            # Action selection - sample action directly from policy logits
            probs = F.softmax(logits, dim=1) # Probabilities for action 0 (terminate) and 1 (move)
            action = torch.multinomial(probs, 1).item() # Sample action (0 or 1)

            # Execute action in environment - action 0 is terminate now
            next_state, reward, effort, done, position = env.step(action, terminate=False) # Pass action directly, terminate handled in env.step
            episode_reward += reward
            episode_effort += effort
            episode_steps += 1

            # Calculate surprise based on INPUT PREDICTION ERROR
            if prev_input_pred is not None:
                surprise = F.mse_loss(prev_input_pred, state).item()
                position_surprise[position] = 0.9 * position_surprise[position] + 0.1 * surprise
                surprise_count[position] += 1

            # Learn from experience
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                _, next_value, _ = agent.forward(next_state_tensor) # Only need next_value, no term_prob
                next_value = next_value.detach()
                target = reward - effort + 0.95 * next_value.item()
            else:
                target = reward - effort

            # Calculate losses
            value_loss = F.mse_loss(value, torch.tensor([[target]]))

            # Policy gradient loss
            advantage = target - value.item()
            log_prob = torch.log(probs.gather(1, torch.tensor([[action]])))
            policy_loss = -log_prob * advantage

            # Input Prediction Loss
            input_loss = F.mse_loss(input_prediction, state)

            # Entropy regularization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            entropy_loss = -0.01 * entropy

            # Uncertainty loss (placeholder - can refine later)
            uncertainty_target = torch.tensor([[min(1.0, input_prediction_error * 5)]])
            uncertainty_loss = F.mse_loss(probs[:,0].unsqueeze(1), uncertainty_target) # Using prob of terminate action as uncertainty

            # Total loss - no term_loss anymore
            loss = policy_loss + value_loss + input_loss + entropy_loss + uncertainty_loss

            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            # Save prediction for next step
            if not done:
                prev_input_pred = input_prediction.detach()
                state = next_state_tensor

        # End of episode - record metrics
        rewards_history.append(episode_reward)
        positions_history.append(position)
        surprise_history.append(np.mean(position_surprise))

        # Log progress
        if episode % 50 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Effort = {episode_effort:.2f}, "
                  f"Position = {position}")

        # Step the scheduler
        scheduler.step()

    return {
        'rewards': rewards_history,
        'positions': positions_history,
        'surprise': surprise_history,
        'position_surprise': position_surprise
    }


def plot_training_results(results):
    """Plot training results for the simplified agent."""

    plt.figure(figsize=(12, 8))

    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(results['rewards'], label='Episode Reward')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Plot positions
    plt.subplot(2, 2, 2)
    plt.plot(results['positions'], label='Final Position')
    plt.title('Final Positions')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.legend()

    # Plot surprise
    plt.subplot(2, 2, 3)
    plt.plot(results['surprise'], label='Average Surprise')
    plt.title('Average Surprise per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Surprise')
    plt.legend()

    # Plot final surprise by position
    plt.subplot(2, 2, 4)
    positions = np.arange(len(results['position_surprise']))
    plt.bar(positions, results['position_surprise'])
    plt.title('Surprise by Position')
    plt.xlabel('Position')
    plt.ylabel('Surprise')

    plt.tight_layout()
    plt.savefig('results/training_results_simple_action0_terminate.png')
    plt.close()


def run_experiment(seq, topology, num_episodes=1000):
    """Run experiment with the simplified agent."""
    # Initialize environment
    env = SequenceEnv(seq, topology)
    input_dim = env.seq_len + 2  # State + prev action

    # Create simplified agent
    agent = SimpleNeuralAgent(input_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.01, weight_decay=1e-5)

    # Train agent
    print("Training simplified agent with action 0 as terminate...")
    results = train_agent(env, agent, optimizer, num_episodes=num_episodes)

    # Plot training results
    plot_training_results(results)

    return agent, results


if __name__ == "__main__":
    # Define sequence with reward structure
    seq = [1, 3, 10, 11, 9, 20, -100, 0]  # Rewards
    topology = [0, 0, 4, 1, 1, 5, 100, 2]  # Topology affects effort

    print("Reward sequence:", seq)
    print("Topology sequence:", topology)

    # Run experiment with simplified agent
    agent, results = run_experiment(seq, topology, num_episodes=1000)