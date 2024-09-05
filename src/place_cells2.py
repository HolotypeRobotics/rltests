import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

GRID_SIZE = 10
MAX_STEPS = 200
HIDDEN_SIZE = 64
TRAJECTORY_LENGTH = 5
NUM_SHAPES = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridEnvironment:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.agent_pos = [0, 0]
        self.place_shapes()

    def place_shapes(self):
        for _ in range(NUM_SHAPES):
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            shape = np.random.randint(1, NUM_SHAPES + 1)
            self.grid[x, y] = shape

    def step(self, action):
        # 0: up, 1: right, 2: down, 3: left
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        new_x, new_y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            self.agent_pos = [new_x, new_y]
        return self.agent_pos, self.grid[self.agent_pos[0], self.agent_pos[1]]

class ShapePredictionModel(nn.Module):
    def __init__(self, trajectory_length, hidden_size, num_shapes):
        super(ShapePredictionModel, self).__init__()
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_shapes + 1)  # +1 for "no shape"

    def forward(self, trajectory):
        lstm_out, _ = self.lstm(trajectory)
        return self.fc(lstm_out[:, -1, :])

class Agent:
    def __init__(self, model, trajectory_length):
        self.model = model
        self.optimizer = optim.Adam(model.parameters())
        self.trajectory = deque(maxlen=trajectory_length)
        self.criterion = nn.CrossEntropyLoss()

    def add_to_trajectory(self, position):
        self.trajectory.append(position)

    def predict(self):
        if len(self.trajectory) < self.trajectory.maxlen:
            return None
        with torch.no_grad():
            trajectory_tensor = torch.tensor(list(self.trajectory), dtype=torch.float32).unsqueeze(0).to(device)
            prediction = self.model(trajectory_tensor)
            return torch.softmax(prediction, dim=1).squeeze().cpu().numpy()

    def learn(self, true_shape):
        if len(self.trajectory) < self.trajectory.maxlen:
            return
        trajectory_tensor = torch.tensor(list(self.trajectory), dtype=torch.float32).unsqueeze(0).to(device)
        true_shape_tensor = torch.tensor([true_shape], dtype=torch.long).to(device)

        self.optimizer.zero_grad()
        prediction = self.model(trajectory_tensor)
        loss = self.criterion(prediction, true_shape_tensor)
        loss.backward()
        self.optimizer.step()

def train():
    env = GridEnvironment()
    model = ShapePredictionModel(TRAJECTORY_LENGTH, HIDDEN_SIZE, NUM_SHAPES).to(device)
    agent = Agent(model, TRAJECTORY_LENGTH)

    for step in range(MAX_STEPS):
        action = np.random.randint(0, 4)
        position, shape = env.step(action)
        agent.add_to_trajectory(position)
        agent.learn(shape)

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}")

    return agent

def evaluate(agent, env):
    predictions = np.zeros((GRID_SIZE, GRID_SIZE, NUM_SHAPES + 1))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            agent.trajectory.clear()
            for _ in range(TRAJECTORY_LENGTH):
                agent.add_to_trajectory([x, y])
            prediction = agent.predict()
            if prediction is not None:
                predictions[x, y] = prediction

    correct_predictions = 0
    total_shapes = 0
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            true_shape = env.grid[x, y]
            predicted_shape = np.argmax(predictions[x, y])
            if true_shape > 0:
                total_shapes += 1
                if predicted_shape == true_shape:
                    correct_predictions += 1

    accuracy = correct_predictions / total_shapes if total_shapes > 0 else 0
    print(f"Prediction Accuracy: {accuracy:.2f}")

    return predictions

if __name__ == "__main__":
    trained_agent = train()
    env = GridEnvironment()  # Create a new environment for evaluation
    predictions = evaluate(trained_agent, env)

    print("True Environment:")
    print(env.grid)
    print("\nPredicted Environment (most likely shape at each position):")
    print(np.argmax(predictions, axis=2))