import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import torch
import torch.nn as nn
import torch.optim as optim


# Define the RNN model
class PlaceCellRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PlaceCellRNN, self).__init__()
        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          nonlinearity='relu',
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, h):
        # Ensure x is 3D: (batch_size, seq_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Ensure h is 3D: (num_layers * num_directions, batch_size, hidden_size)
        if h.dim() == 2:
            h = h.unsqueeze(0)  # Adjust to (1, batch_size, hidden_size)

        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h


# Function to calculate boundary proximity feature
def calculate_boundary_proximity(x, y, env_corners):
    # Convert environment corners to a Path object
    env_boundary = Path(env_corners)

    # Calculate the minimum distance to the boundary from the point (x, y)
    distances = [
        np.linalg.norm(np.array([x, y]) - np.array(corner))
        for corner in env_corners
    ]
    min_distance = min(distances)
    return min_distance


# Generate structured input features including boundary proximity
def generate_inputs(seq_length, env_corners):
    # Generate a structured path (e.g., a circular trajectory)
    t = np.linspace(0, 2 * np.pi, seq_length)
    positions = 0.5 + 0.4 * np.vstack(
        (np.cos(t), np.sin(t))).T  # Circular path

    # Calculate velocities based on sequential positions
    velocities = np.diff(positions, axis=0, prepend=positions[:1])

    # Calculate boundary proximity for each position
    boundary_proximity = np.array([
        calculate_boundary_proximity(x, y, env_corners) for x, y in positions
    ])

    # Combine velocity, context, and boundary proximity into inputs
    context = np.random.randint(0, 2, (seq_length, 2))
    inputs = np.concatenate(
        (velocities, context, boundary_proximity[:, np.newaxis]), axis=1)
    return torch.tensor(inputs, dtype=torch.float32), positions


# Estimate centers of place fields (mu_i) based on firing rates and true positions
def estimate_center(firing_rates, true_positions, epsilon=1e-6):
    centers = []
    for i in range(firing_rates.shape[1]):
        weights = firing_rates[:, i]
        center = np.sum(weights[:, np.newaxis] * true_positions,
                        axis=0) / (np.sum(weights) + epsilon)
        centers.append(center)
    return np.array(centers)


# Estimate positions using firing rates and estimated centers
def estimate_position(firing_rates, centers, epsilon=1e-6):
    positions = []
    for t in range(firing_rates.shape[0]):
        weights = firing_rates[t, :]
        estimated_position = np.sum(weights[:, np.newaxis] * centers,
                                    axis=0) / (np.sum(weights) + epsilon)
        positions.append(estimated_position)
    return np.array(positions)


# Parameters
input_size = 5
hidden_size = 500
output_size = 100
seq_length = 100
batch_size = 1

model = PlaceCellRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_env_corners = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
inputs, true_positions = generate_inputs(seq_length, train_env_corners)

# Initial hidden state
h0 = torch.zeros(1, batch_size, hidden_size)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    inputs = inputs.unsqueeze(0) if inputs.dim(
    ) == 2 else inputs  # Shape: (batch_size, seq_length, input_size)
    outputs, hn = model(inputs, h0)
    outputs_np = outputs.detach().numpy().reshape(-1, output_size)

    estimated_centers = estimate_center(outputs_np, true_positions)
    estimated_positions = estimate_position(outputs_np, estimated_centers)

    estimated_positions_tensor = torch.tensor(estimated_positions,
                                              dtype=torch.float32,
                                              requires_grad=True)
    true_positions_tensor = torch.tensor(true_positions, dtype=torch.float32)

    loss = criterion(estimated_positions_tensor, true_positions_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')

print("Model training completed.")

# Define testing environment corners (e.g., L-shape)
test_env_corners = [(0.1, 0.1), (0.8, 0.1), (0.8, 0.4), (0.5, 0.4), (0.5, 0.8),
                    (0.1, 0.8)]

# Generate testing data with a different environment geometry
test_inputs, test_positions = generate_inputs(seq_length, test_env_corners)

# Test the model on the new environment
with torch.no_grad():
    test_inputs = test_inputs.unsqueeze(0)  # Add batch dimension
    test_outputs, _ = model(test_inputs, h0)

# Visualize learned place fields for new geometry
grid_size = 20
xx, yy = np.meshgrid(np.linspace(0, 1, grid_size),
                     np.linspace(0, 1, grid_size))
test_outputs_np = test_outputs.squeeze(0).numpy().reshape(
    grid_size, grid_size, -1)

plt.figure(figsize=(15, 10))
for i in range(min(16, output_size)):  # Show up to 16 place fields
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_outputs_np[:, :, i],
               cmap='viridis',
               extent=[0, 1, 0, 1],
               origin='lower')
    plt.colorbar(label='Firing Rate')
    plt.title(f'Place Cell {i+1}')
    plt.plot(*zip(*test_env_corners), 'wo-',
             linewidth=2)  # Show environment boundary
    plt.axis('equal')

plt.tight_layout()
plt.suptitle('Learned Place Fields in New Environment', fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()
