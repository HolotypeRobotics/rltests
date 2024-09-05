import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.sequences = []

    def generate_sequences(self, n_sequences, trajectory_length, n_diff_shapes):
        for _ in range(n_sequences):
            sequence = []
            # Fill the sequence with no shapes
            for _ in range(trajectory_length):
                shape = np.zeros(n_diff_shapes)
                sequence.append(shape)
            # Add a random number of shapes, selecting from n_diff_shapes
            for _ in range(np.random.randint(1, 3)):
                # Select a random position and shape
                pos = np.random.randint(0, trajectory_length)
                shape = np.random.randint(0, n_diff_shapes)
                # Add the shape to the sequence as a 1-hot encoded vector
                v = np.zeros(n_diff_shapes)
                v[shape] = 1
                sequence[pos] = v
            self.sequences.append(sequence)

    def get_shape(self, sequence, position):
        return self.sequences[sequence][position]

class ShapePredictorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShapePredictorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# Define the RNN parameters
num_sequences = 10
num_shapes = 3
sequence_length = 10
hidden_size = 1
input_size = num_sequences + sequence_length
output_size = num_shapes
num_epochs = 100

env = Environment()
env.generate_sequences(num_sequences, sequence_length, num_shapes)

# Instantiate the RNN
model = ShapePredictorRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=.01)

# Training loop
losses = []
for epoch in range(num_epochs):
    total_loss = 0
    # Shuffle the order of sequences, but not the positions within sequences
    sequence_order = np.random.permutation(num_sequences)
    
    for seq_idx in sequence_order:
        hidden = model.init_hidden(1)  # Initialize hidden state for each sequence
        for pos in range(sequence_length):
            # Prepare input: one-hot encoded sequence index + position
            sequence_onehot = np.zeros(num_sequences)
            sequence_onehot[seq_idx] = 1
            position_onehot = np.zeros(sequence_length)
            position_onehot[pos] = 1
            
            input_tensor = torch.tensor(np.concatenate([sequence_onehot, position_onehot])[np.newaxis, np.newaxis, :], dtype=torch.float32)
            target_tensor = torch.tensor(env.get_shape(seq_idx, pos), dtype=torch.float32)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output, hidden = model(input_tensor, hidden)

            # Compute the loss
            loss = criterion(output.squeeze(), target_tensor)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Detach hidden state
            hidden = hidden.detach()

    avg_loss = total_loss / (num_sequences * sequence_length)
    losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Function to calculate prediction error
def prediction_error(predicted, actual):
    return np.linalg.norm(predicted - actual)

# Testing the RNN
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    errors = []
    for i in range(num_sequences):
        sequence_errors = []
        hidden = model.init_hidden(1)
        for j in range(sequence_length):
            sequence_onehot = np.zeros(num_sequences)
            sequence_onehot[i] = 1
            position_onehot = np.zeros(sequence_length)
            position_onehot[j] = 1
            input_tensor = torch.tensor(np.concatenate([sequence_onehot, position_onehot])[np.newaxis, np.newaxis, :], dtype=torch.float32)
            
            output, hidden = model(input_tensor, hidden)
            predicted = output.numpy().squeeze()
            actual = env.get_shape(i, j)
            
            error = prediction_error(predicted, actual)
            sequence_errors.append(error)
        errors.append(np.mean(sequence_errors))

    print(f"Average prediction error: {np.mean(errors):.4f}")

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