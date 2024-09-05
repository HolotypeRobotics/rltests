import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from minigrid.wrappers import FullyObsWrapper

# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define Q-learning Agent
class QLearningAgent:
    def __init__(self, env, lr=0.01, gamma=0.99, epsilon=0.1):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Extract the image shape from the observation space dictionary
        obs_space = env.observation_space['image']
        input_dim = np.prod(obs_space.shape)

        # Initialize Q-Network
        self.q_network = QNetwork(input_dim, env.action_space.n)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.q_network(state_tensor)

        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            max_next_q_value = torch.max(next_q_values)

        target_q_value = reward + (1 - done) * self.gamma * max_next_q_value
        current_q_value = q_values[0, action]

        loss = self.criterion(current_q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Main function to train the agent
def main():
    env = FullyObsWrapper(gym.make('MiniGrid-Empty-5x5-v0'))  # Fully observable environment
    agent = QLearningAgent(env)

    episodes = 500
    for episode in range(episodes):
        state, _ = env.reset()  # Updated to match the new Gymnasium API
        state = state['image'].flatten()  # Flatten the state image for simplicity
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)  # Updated to match the new Gymnasium API
            next_state = next_state['image'].flatten()
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
