import numpy as np




class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # W: Weight matrices for input connections
        # U: Weight matrices for recurrent connections
        # b: Bias vectors

        # The subscripts denote which gate or operation each weight or bias is associated with:

        # r: Reset gate
        # z: Update gate
        # h: Candidate hidden state

        # input weights for canidate hidden state
        self.W_h = np.random.randn(hidden_size, input_size)
        # input weights for update gate
        self.W_z = np.random.randn(hidden_size, input_size)
        # input weights for reset gate
        self.W_r = np.random.randn(hidden_size, input_size)

        # recurrent weights for canidate hidden state
        self.U_h = np.random.randn(hidden_size, hidden_size)
        # recurrent weights for update gate
        self.U_z = np.random.randn(hidden_size, hidden_size)
        # recurrent weights for reset gate
        self.U_r = np.random.randn(hidden_size, hidden_size)

        # biases for canidate hidden state
        self.b_h = np.zeros((hidden_size, 1))
        # biases for update gate
        self.b_z = np.zeros((hidden_size, 1))
        # biases for reset gate
        self.b_r = np.zeros((hidden_size, 1))

        # Initialize hidden state
        self.h = np.zeros((hidden_size, 1))

    def forward(self, x):
        # Reshape input
        x = x.reshape(-1, 1)


       # input weights for canidate hidden state
        self.W_h.fill(0)
        # input weights for update gate
        self.W_z.fill(0)
        # input weights for reset gate
        self.W_r.fill(0)

        # recurrent weights for canidate hidden state
        self.U_h.fill(0)
        # recurrent weights for update gate
        self.U_z.fill(0)
        # recurrent weights for reset gate
        self.U_r.fill(0)


        # Update gate
        z = self.sigmoid(np.dot(self.W_z, x) + np.dot(self.U_z, self.h) + self.b_z)

        # Reset gate
        r = self.sigmoid(np.dot(self.W_r, x) + np.dot(self.U_r, self.h) + self.b_r)

        # Candidate hidden state
        h_hat = np.tanh(np.dot(self.W_h, x) + np.dot(self.U_h, r * self.h) + self.b_h)

        # New hidden state
        self.h = (1 - z) * self.h + z * h_hat
        print(f"z: {z}, r: {r}, h_hat: {h_hat}, h: {self.h}")

        return self.h

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def reset_hidden_state(self):
        self.h = np.zeros((self.hidden_size, 1))

class GRUModel:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.2):
        self.gru = GRUCell(input_size, hidden_size)
        self.W_out = np.random.randn(output_size, hidden_size)
        self.b_out = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

    def predict(self, x):
        h = self.gru.forward(x)
        out = np.dot(self.W_out, h) + self.b_out
        print(f"out weights: {self.W_out}")
        return self.softmax(out)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def train_step(self, x, y_true):
        # Forward pass
        y_pred = self.predict(x)

        # Calculate loss (cross-entropy)
        loss = -np.sum(y_true * np.log(y_pred))

        # Backward pass (simplified gradient descent for demonstration)
        dL_dy = y_pred - y_true
        dL_dW_out = np.dot(dL_dy, self.gru.h.T)
        dL_db_out = dL_dy

        # Update weights
        self.W_out -= self.learning_rate * dL_dW_out
        self.b_out -= self.learning_rate * dL_db_out

        return loss

# Utility functions
def one_hot_encode(color, color_to_index, num_classes):
    encoded = np.zeros((num_classes, 1))
    encoded[color_to_index[color]] = 1
    return encoded

# Example usage
colors = ["red", "green", "blue", "yellow"]
color_to_index = {color: i for i, color in enumerate(colors)}
index_to_color = {i: color for i, color in enumerate(colors)}

input_size = len(colors)
hidden_size = 1
output_size = len(colors)

model = GRUModel(input_size, hidden_size, output_size)

# Online learning loop
print("Start playing Simon! Input the colors one by one.")

try:
    while True:
        # Get user input
        user_input = input("Enter a color (red, green, blue, yellow): ").strip().lower()
        if user_input not in colors:
            print("Invalid color. Try again.")
            continue

        # Encode input
        x = one_hot_encode(user_input, color_to_index, input_size)

        # Make a prediction
        y_pred = model.predict(x)
        print(f"Prediction: {y_pred}")
        predicted_color = index_to_color[np.argmax(y_pred)]

        print(f"Model predicts the next color will be: {predicted_color}")

        # Get true next color from user
        next_color = input("Enter the actual next color: ").strip().lower()
        if next_color not in colors:
            print("Invalid color. Try again.")
            continue

        # Encode true next color
        y_true = one_hot_encode(next_color, color_to_index, output_size)

        # Train model with the true next color
        loss = model.train_step(x, y_true)
        print(f"Training step completed. Loss: {loss}")

except KeyboardInterrupt:
    print("Game over. Goodbye!")
