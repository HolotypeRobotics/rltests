import numpy as np
from minigrid.core.constants import COLORS, IDX_TO_COLOR, OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
import gymnasium as gym

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        self.W_h = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.W_z = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.W_r = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)

        self.U_h = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.U_z = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.U_r = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)

        self.b_h = np.zeros((hidden_size, 1))
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))

        self.h = np.zeros((hidden_size, 1))

        self.effort = 0

    def forward(self, x):
        x = x.reshape(-1, 1)

        z = self.sigmoid(np.dot(self.W_z, x) + np.dot(self.U_z, self.h) + self.b_z)
        r = self.sigmoid(np.dot(self.W_r, x) + np.dot(self.U_r, self.h) + self.b_r)
        h_hat = np.tanh(np.dot(self.W_h, x) + np.dot(self.U_h, r * self.h) + self.b_h)
        
        
        self.h = (1 - z) * self.h + z * h_hat

        return self.h

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def reset_hidden_state(self):
        self.h = np.zeros((self.hidden_size, 1))

class RLGRUModel:
    def __init__(self, input_size, hidden_size, action_size, learning_rate=0.001):
        self.gru = GRUCell(input_size, hidden_size)
        self.W_action = np.random.randn(action_size, hidden_size) / np.sqrt(hidden_size)
        self.b_action = np.zeros((action_size, 1))
        self.W_value = np.random.randn(1, hidden_size) / np.sqrt(hidden_size)
        self.b_value = np.zeros((1, 1))
        self.learning_rate = learning_rate

    def predict(self, state):
        h = self.gru.forward(state)
        action_probs = self.softmax(np.dot(self.W_action, h) + self.b_action)
        value = np.dot(self.W_value, h) + self.b_value
        return action_probs, value

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def train_step(self, state, action, reward, next_state, done):
        # Forward pass
        action_probs, value = self.predict(state)
        _, next_value = self.predict(next_state)

        # Compute target and advantage
        target = reward + (1 - done) * 0.99 * next_value  # 0.99 is the discount factor
        advantage = target - value

        # Compute gradients
        dL_dW_action = np.zeros_like(self.W_action)
        dL_db_action = np.zeros_like(self.b_action)
        dL_dW_value = advantage * self.gru.h.T
        dL_db_value = advantage

        # For the action probabilities, we'll use the REINFORCE algorithm
        dL_dW_action[action] = -advantage * (1 - action_probs[action]) * self.gru.h.T
        dL_db_action[action] = -advantage * (1 - action_probs[action])

        # Update weights
        self.W_action -= self.learning_rate * dL_dW_action
        self.b_action -= self.learning_rate * dL_db_action
        self.W_value -= self.learning_rate * dL_dW_value
        self.b_value -= self.learning_rate * dL_db_value

        return np.sum(advantage**2)  # Return loss

# Utility functions
def preprocess_state(state):
    # This function should be adapted based on the specific MiniGrid environment
    # For simplicity, let's flatten the state and normalize it
    return state.flatten() / 255.0

# Example usage
env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')
input_size = env.observation_space['image'].shape[0] * env.observation_space['image'].shape[1] * env.observation_space['image'].shape[2]
hidden_size = 64
action_size = env.action_space.n

model = RLGRUModel(input_size, hidden_size, action_size)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state, info = env.reset()
    state = preprocess_state(state['image'])
    done = False
    total_reward = 0
    
    while not done:
        action_probs, _ = model.predict(state)
        action = np.random.choice(action_size, p=action_probs.flatten())
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_state(next_state['image'])
        
        loss = model.train_step(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}")

print("Training completed!")