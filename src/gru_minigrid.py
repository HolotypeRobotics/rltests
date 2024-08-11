import numpy as np
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

    def forward(self, x):
        x = x.reshape(-1, 1)

        z = self.sigmoid(np.dot(self.W_z, x) + np.dot(self.U_z, self.h) + self.b_z)
        r = self.sigmoid(np.dot(self.W_r, x) + np.dot(self.U_r, self.h) + self.b_r)
        h_hat = np.tanh(np.dot(self.W_h, x) + np.dot(self.U_h, r * self.h) + self.b_h)
        
        self.h = (1 - z) * self.h + z * h_hat

        return self.h, z

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def reset_hidden_state(self):
        self.h = np.zeros((self.hidden_size, 1))

class RLGRUModel:
    def __init__(self, input_size, hidden_size, action_size, n_step_ahead=5, learning_rate=0.001):
        self.gru = GRUCell(input_size, hidden_size)
        self.W_action = np.random.randn(action_size, hidden_size) / np.sqrt(hidden_size)
        self.b_action = np.zeros((action_size, 1))
        self.W_reward = np.random.randn(1, hidden_size) / np.sqrt(hidden_size)
        self.b_reward = np.zeros((1, 1))
        self.learning_rate = learning_rate
        self.n_step_ahead = n_step_ahead

    def predict(self, state):
        h, z = self.gru.forward(state)
        action_probs = self.softmax(np.dot(self.W_action, h) + self.b_action)
        predicted_reward = np.dot(self.W_reward, h) + self.b_reward
        return action_probs, predicted_reward, z

    def predict_n_step(self, initial_state):
        state = initial_state
        predictions = []
        for _ in range(self.n_step_ahead):
            action_probs, predicted_reward, z = self.predict(state)
            predictions.append((action_probs, predicted_reward, z))
            # For simplicity, we'll use the most probable action as the next state
            # In a real scenario, you might want to use the environment model here
            next_state = np.argmax(action_probs)
            state = self.preprocess_state(next_state)
        return predictions

    def calculate_re_value(self, predictions):
        re_value = 0
        for t, (_, reward, z) in enumerate(predictions):
            effort = np.mean(z)  # Average effort across all cells
            re_value += reward / (1 + effort * t)
        return re_value

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def train_step(self, state, action, reward, next_state, done):
        # Forward pass
        action_probs, predicted_reward, z = self.predict(state)
        
        # Compute n-step ahead predictions
        future_predictions = self.predict_n_step(next_state)
        
        # Calculate RE_value
        re_value = self.calculate_re_value([(action_probs, predicted_reward, z)] + future_predictions)

        # Compute loss (you may want to adjust this based on your specific requirements)
        action_loss = -np.log(action_probs[action]) * (reward - predicted_reward)
        reward_loss = (reward - predicted_reward) ** 2
        re_value_loss = -re_value  # We want to maximize RE_value
        total_loss = action_loss + reward_loss + re_value_loss

        # Compute gradients (this is a simplified version, you might need more complex backpropagation)
        dL_dW_action = np.zeros_like(self.W_action)
        dL_db_action = np.zeros_like(self.b_action)
        dL_dW_reward = (predicted_reward - reward) * self.gru.h.T
        dL_db_reward = predicted_reward - reward

        dL_dW_action[action] = -1 / action_probs[action] * (reward - predicted_reward) * self.gru.h.T
        dL_db_action[action] = -1 / action_probs[action] * (reward - predicted_reward)

        # Update weights
        self.W_action -= self.learning_rate * dL_dW_action
        self.b_action -= self.learning_rate * dL_db_action
        self.W_reward -= self.learning_rate * dL_dW_reward
        self.b_reward -= self.learning_rate * dL_db_reward

        return total_loss

    def preprocess_state(self, state):
        # This function should be adapted based on the specific MiniGrid environment
        # For simplicity, let's assume the state is already in the correct format
        return state

# Example usage
env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')
input_size = env.observation_space['image'].shape[0] * env.observation_space['image'].shape[1] * env.observation_space['image'].shape[2]
hidden_size = 64
action_size = env.action_space.n

model = RLGRUModel(input_size, hidden_size, action_size, n_step_ahead=5)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state, info = env.reset()
    state = model.preprocess_state(state['image'])
    done = False
    total_reward = 0
    
    while not done:
        action_probs, _, _ = model.predict(state)
        action = np.random.choice(action_size, p=action_probs.flatten())
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = model.preprocess_state(next_state['image'])
        
        loss = model.train_step(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}")

print("Training completed!")