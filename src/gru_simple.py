import numpy as np
from environment import Environment


rewards = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 10]
]



grid_size = len(rewards)
env = Environment(rewards)
env.set_exit(grid_size - 1, grid_size - 1)

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        self.W_h = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)  # PL transition probablilities (Rules)
        self.W_z = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)  # ACC Effort estimation
        self.W_r = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)  # IL Behavioral extinction
        


        self.U_h = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.U_z = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.U_r = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)

        self.b_h = np.zeros((hidden_size, 1))
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))


        self.h = np.zeros((hidden_size, 1))
        self.predicted_h = np.zeros((hidden_size, 1))

        self.predicted_state = np.zeros((hidden_size, 1))

        # OFC Reward prediction weights based on predicted state from chosen goal state
        self.W_R = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.U_R = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size) # recurrent weights
        self.b_R = np.zeros((hidden_size, 1)) # bias weights

        # efforts based on transition from current state to goal state
        self.effort = np.zeros((hidden_size, 1))

        self.positive_valence = 0
        self.negative_valence = 0


    # Todo: rework so that the ACC predict the outcome associated with the choice from the PL
    # Calcualte the effort and compare it to the predicted reward to tell if 

    def forward(self, x, execute=True):

        # Calculate the state for the PL. this is the predicted hidden state
        # The ACC Takes this as the action, and predicts the resulting states and gives it to the OFC to get the reward
        # The predicted reward is  used with the calcualted effort to determine if the action/sub-actions need to be switched
        # If predicted state does not match actual state, then the ACC will trigger a switch

            x = x.reshape(-1, 1)

            z = self.sigmoid(np.dot(self.W_z, x) + np.dot(self.U_z, self.h) + self.b_z)
            r = self.sigmoid(np.dot(self.W_r, x) + np.dot(self.U_r, self.h) + self.b_r)
            h_hat = np.tanh(np.dot(self.W_h, x) + np.dot(self.U_h, r * self.h) + self.b_h)

            self.predicted_h = (1 - z) * self.h + z * h_hat # PL state

            trajectory = self.sigmoid(np.dot(self.W_))

            # Predict the reward for the predicted options
            reward = self.sigmoid(np.dot(self.W_R, x) + np.dot(self.U_R, self.predicted_h) + self.b_R)

            # Estimate the mental effort for next step for each possibility
            # (mental effort is the effort to maintain the current state)
            self.effort += z

            # calculate the value of the next state based on the reward and effort
            re_value = reward / (1 + self.effort)

            # apply the value to the hidden state
            self.predicted_h *= re_value

            if execute:
                    self.h = self.predicted_h
                    # Recalculate the effort for the current new self.h
                    z = self.sigmoid(np.dot(self.W_z, x) + np.dot(self.U_z, self.h) + self.b_z)
                    self.effort -= z

            return self.h

    def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

    def reset_hidden_state(self):
            self.h = np.zeros((self.hidden_size, 1))
            self.effort = 0

class RLGRUModel:
    def __init__(self, input_size, hidden_size, action_size, learning_rate=0.001):
            self.gru = GRUCell(input_size, hidden_size)
            self.W_action = np.random.randn(action_size, hidden_size) / np.sqrt(hidden_size)
            self.b_action = np.zeros((action_size, 1))
            self.W_reward = np.random.randn(1, hidden_size) / np.sqrt(hidden_size)
            self.b_reward = np.zeros((1, 1))
            self.learning_rate = learning_rate

    def predict(self, state, is_prediction=False):
            h, effort = self.gru.forward(state, is_prediction)
            action_probs = self.softmax(np.dot(self.W_action, h) + self.b_action)
            predicted_reward = np.dot(self.W_reward, h) + self.b_reward
            return action_probs, predicted_reward, self.gru.effort


    def calculate_re_value(self, reward, effort):
            return  reward / (1 + effort)

    def softmax(self, x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)

    def step(self, state, action, reward, next_state, done):


input_size = grid_size * 2
# number of possible trajectories
hidden_size = grid_size * (grid_size - 1) * 2
action_size = 4

model = RLGRUModel(input_size, hidden_size, action_size, n_step_ahead=5)

# Training loop
num_episodes = 1000
action = 0
for episode in range(num_episodes):
    state, reward = env.reset()
    state = model.preprocess_state(state['image'])
    done = False
    total_reward = 0

    while not done:


            next_state, reward, done, coordinates, = env.step(action)

            action_probs = model.step(state, action, reward, next_state, done)

            action = np.random.choice(action_size, p=action_probs.flatten())

            state = next_state
            total_reward += reward

    if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}")

print("Training completed!")