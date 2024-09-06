import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

def shape_to_int(shape):
    return np.argmax(shape)

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
    def __init__(self, num_sequences, embedding_dim, hidden_size, output_size, num_layers=1):
        super(ShapePredictorRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_sequences, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, sequence_idx, position, h=None):
        embedded_seq = self.embedding(sequence_idx)
        pos_enc = self.positional_encoding(position, self.embedding_dim)
        rnn_input = torch.cat((embedded_seq, pos_enc), dim=1).unsqueeze(1)  # Add sequence dimension
        out, h = self.rnn(rnn_input, h)
        out = self.fc(out.squeeze(1))  # Remove sequence dimension
        return out, h

    def positional_encoding(self, position, d_model, max_len=1000):
       """
        Computes the sinusoidal positional encoding 
        """
        pe = torch.zeros(position.size(0), d_model)
        position = position.unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

# Define the RNN parameters
num_sequences = 10
num_shapes = 3
sequence_length = 10
embedding_dim = 16  # Dimension for sequence embeddings
hidden_size = 64
num_layers = 2  # Use 2 GRU layers
num_epochs = 100
learning_rate = 0.01
teacher_forcing_ratio = 0.5

env = Environment()
env.generate_sequences(num_sequences, sequence_length, num_shapes)

model = ShapePredictorRNN(num_sequences, embedding_dim, hidden_size, num_shapes, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(num_epochs):
    total_loss = 0
    for seq_idx in range(num_sequences):
        hidden = None  # Initialize hidden state for each sequence
        sequence_loss = 0
        for pos in range(sequence_length):
            # Prepare input
            input_seq_idx = torch.tensor([seq_idx]).long()
            input_pos = torch.tensor([pos], dtype=torch.float32)

            # Forward pass
            output, hidden = model(input_seq_idx, input_pos, hidden)

            # Get target
            target = torch.tensor([shape_to_int(env.get_shape(seq_idx, pos))])

            # Compute the loss
            loss = criterion(output, target)
            sequence_loss += loss

            # Teacher forcing 
            if np.random.random() < teacher_forcing_ratio:
                input_shape = target
            else:
                input_shape = torch.argmax(output, dim=1)

        # Backward pass and optimize (once per sequence)
        optimizer.zero_grad()
        sequence_loss.backward()
        optimizer.step()

        total_loss += sequence_loss.item()

    avg_loss = total_loss / num_sequences
    losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Testing the RNN
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    errors = []
    for i in range(num_sequences):
        sequence_errors = []
        hidden = None
        for j in range(sequence_length):
            input_seq_idx = torch.tensor([i]).long()
            input_pos = torch.tensor([j], dtype=torch.float32)

            output, hidden = model(input_seq_idx, input_pos, hidden)
            predicted = output.numpy().squeeze()
            actual = env.get_shape(i, j)

            _predicted = np.argmax(predicted)
            _actual = shape_to_int(actual)
            print(f"Sequence {i}, Position {j}: Predicted: {_predicted}, Actual: {_actual}")

            error = np.sum(np.abs(predicted - actual))
            sequence_errors.append(error)
        errors.append(np.mean(sequence_errors))
        print()

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