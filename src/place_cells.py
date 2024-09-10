import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import torch.nn.init as init
import torch.nn.functional as F

def grid_cell(x, frequency, grid_size):
    return (1 - torch.cos(frequency * math.pi * x / grid_size)) / 2

# --- Fourier Feature Encoding ---
def generate_fourier_features(coords, grid_size):
    features = [] 
    x = (coords[:, 0:1]) / grid_size # x position
    y = (coords[:, 1:2]) / grid_size # y position
    features.append(x)
    features.append(y)
    features.append(grid_cell(x, 5, grid_size)) # x place grid activation
    features.append(grid_cell(y, 5, grid_size)) # y place grid activation
    features.append(grid_cell(x, 1, grid_size)) # x env grid activation
    features.append(grid_cell(y, 1, grid_size)) # y env grid activation
    return torch.cat(features, dim=1)

def shape_to_int(shape):
    return np.argmax(shape)

def create_unique_environments(num_environments, grid_size, shapes_per_env, n_sequences, path_length):
    environments = []
    num_shapes = shapes_per_env * num_environments
    # Create a list of shapes to pull from
    shapes = list(range(num_shapes))
    # Shuffle the shapes
    np.random.shuffle(shapes)
    # Divide the shapes into groups for each environment
    shape_groups = [shapes[i:(i + num_shapes // num_environments)] for i in range(0, num_shapes, num_shapes // num_environments)]

    for _ in range(num_environments):
        env = Environment(grid_size, num_shapes)
        env.generate_paths(n_sequences, path_length)
        env.place_shapes(shape_groups.pop(0))
        environments.append(env)
    return environments

class Environment:
    def __init__(self, grid_size, n_diff_shapes):
        self.grid_size = grid_size
        # populate the grid with empty shapes
        self.grid = np.zeros((grid_size, grid_size, n_diff_shapes))
        self.paths = []

    def generate_paths(self, n_paths, path_length):
        """Generates paths within the grid."""
        for _ in range(n_paths):
            path = []
            x, y = np.random.randint(0, self.grid_size, size=2)
            for _ in range(path_length):
                move = np.random.choice(['up', 'down', 'left', 'right'])
                if move == 'up' and y > 0: y -= 1
                elif move == 'down' and y < self.grid_size - 1: y += 1
                elif move == 'left' and x > 0: x -= 1
                elif move == 'right' and x < self.grid_size - 1: x += 1
                path.append((x, y))
            self.paths.append(path)

    def place_shapes(self, shapes):
        """Places one of each shape on the grid at locations along the paths."""
        all_positions = [pos for path in self.paths for pos in path]
        np.random.shuffle(all_positions)

        for pos in all_positions:
            if not shapes:
                break
            x, y = pos
            if np.sum(self.grid[y, x]) == 0:  # If no shape at this position
                self.grid[y, x, shapes.pop(0)] = 1 # Creates 1 hot encoding of shape

    def get_shape_at(self, x, y):
        """Returns the shape at the given (x, y) coordinates."""
        return self.grid[y, x]


# --- Shape Predictor RNN (Modified) ---
class ShapePredictorRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, grid_size):
        super(ShapePredictorRNN, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, output_size)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.grid_size = grid_size
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')


    def forward(self, position, h=None):
        fourier_features = self.positional_encoding(position)
        # combined_input = torch.cat((fourier_features), dim=1)
        lin_out = self.fc1(fourier_features)
        rnn_out, h = self.rnn(fourier_features, h)
        combined_hidden = torch.cat((rnn_out.squeeze(1), lin_out), dim=1)
        out = self.leaky_relu(combined_hidden)

        out = self.fc2(out)
        out = self.leaky_relu(out)
        return out, h

    def positional_encoding(self, position):
        return generate_fourier_features(position, self.grid_size)

def visualize_shape_probabilities(model, environments, num_shapes, resolution=100):
    model.eval()
    for env_idx, env in enumerate(environments):
        grid_size = env.grid_size
        probabilities = np.zeros((resolution, resolution, num_shapes))

        with torch.no_grad():
            for y in range(resolution):
                for x in range(resolution):
                    grid_x = (x / (resolution - 1)) * (grid_size - 1)
                    grid_y = (y / (resolution - 1)) * (grid_size - 1)
                    input_pos = torch.tensor([grid_x, grid_y], dtype=torch.float32).unsqueeze(0)
                    output, _ = model(input_pos)
                    probabilities[y, x, :] = torch.softmax(output, dim=1).numpy().squeeze()

        fig, axes = plt.subplots(2, (num_shapes + 1) // 2, figsize=(20, 10))
        axes = axes.flatten()

        for shape in range(num_shapes):
            im = axes[shape].imshow(probabilities[:, :, shape], cmap='viridis', interpolation='nearest', alpha=0.7,
                                    extent=[-0.5, grid_size - 0.5, grid_size - 0.5, -0.5])
            axes[shape].set_title(f'Env: {env_idx}, Shape {shape}')

            # Overlay actual shape positions
            for y in range(grid_size):
                for x in range(grid_size):
                    if env.grid[y, x, shape] == 1:
                        axes[shape].plot(x, y, 'ro', markersize=10, markeredgecolor='white')

            axes[shape].set_xlim(-0.5, grid_size - 0.5)
            axes[shape].set_ylim(grid_size - 0.5, -0.5)
            axes[shape].axis('off')
            fig.colorbar(im, ax=axes[shape])

    plt.tight_layout()
    plt.show()

# --- Model and Training Parameters ---
num_environments = 1
num_sequences = 10
shapes_per_environment = 15
path_length = 20
grid_size = 10
hidden_size = 10
num_epochs = 200
learning_rate = 0.01
input_size = 6  # normalized x, normalized y, Fourier x, Fourier y, Fourier x_env, Fourier y_env
num_shapes = shapes_per_environment * num_environments

# --- Create Multiple Environments ---
environments = create_unique_environments(num_environments, grid_size, shapes_per_environment, num_sequences, path_length)

model = ShapePredictorRNN(input_size, hidden_size, num_shapes, grid_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(num_epochs):
    total_loss = 0
    for env_idx, env in enumerate(environments):
        for path in env.paths:
            hidden = None
            sequence_loss = 0
            for (x, y) in path:
                input_pos = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)
                output, hidden = model(input_pos, hidden)
                target = torch.tensor([shape_to_int(env.get_shape_at(x, y))])
                loss = criterion(output, target)
                sequence_loss += loss

            # Backward pass and optimize (once per sequence)
            optimizer.zero_grad()
            sequence_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += sequence_loss.item()

    avg_loss = total_loss / (num_sequences * num_environments)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Testing the RNN
model.eval()
with torch.no_grad():
    for env_idx, env in enumerate(environments):
        one_hot_env = np.zeros(num_environments)
        one_hot_env[env_idx] = 1
        one_hot_env = torch.tensor(one_hot_env, dtype=torch.float32).unsqueeze(0)
        print(f"Testing Environment {env_idx + 1}")
        errors = []
        for path in env.paths:
            sequence_errors = []
            hidden = None
            for (x, y) in path:
                input_pos = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)

                output, hidden = model(input_pos, hidden)

                predicted = torch.argmax(output).item()
                actual = shape_to_int(env.get_shape_at(x, y))
                print(f"position x {x}, y {y}: Predicted: {predicted}, Actual: {actual}")

                error = abs(predicted - actual)
                sequence_errors.append(error)
            errors.append(np.mean(sequence_errors))
            print()

        print(f"Average prediction error for Environment {env_idx + 1}: {np.mean(errors):.4f}")

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot 1: Training Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot 2: Prediction Errors
plt.subplot(1, 2, 2)
plt.bar(range(1, len(errors) + 1), errors)
plt.title('Prediction Errors by Sequence')
plt.xlabel('Sequence')
plt.ylabel('Average Error')

plt.tight_layout()
plt.show()

visualize_shape_probabilities(model, environments, num_shapes)