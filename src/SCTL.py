import numpy as np
from itertools import product

class EnhancedSCTL:
    def __init__(self, input_size, num_columns, cells_per_column, active_columns, sparsity=0.02, boost_strength=1.0, max_sequence_length=10):
        self.input_size = input_size
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.active_columns = active_columns
        self.sparsity = sparsity
        self.boost_strength = boost_strength
        self.max_sequence_length = max_sequence_length
        
        # Initialize columns (each column connects to a subset of input bits)
        self.columns = np.random.rand(num_columns, input_size) < sparsity
        
        # Initialize cells
        self.cells = np.zeros((num_columns, cells_per_column, input_size), dtype=bool)
        
        # Initialize predictions
        self.predictions = np.zeros(input_size, dtype=bool)
        
        # Keep track of active cells from the previous time step
        self.previous_active_cells = []
        
        # Boosting factors to ensure all columns get activated over time
        self.boost_factors = np.ones(num_columns)
        
        # Sequence memory for temporal sequences
        self.sequence_memory = []

        # Initialize sequence history
        self.sequence_history = []

    def activate_columns(self, input_data):
        # Compute overlap scores
        overlap = np.dot(self.columns.astype(int), input_data).astype(float)
        
        # Apply boosting factors
        overlap *= self.boost_factors
        
        # Add small random values to break ties
        overlap += np.random.rand(self.num_columns) * 0.01
        
        # Select top active columns
        active_columns = np.argsort(overlap)[-self.active_columns:]
        
        # Update boosting factors
        self.boost_factors *= (1 - self.boost_strength)
        self.boost_factors[active_columns] += self.boost_strength
        
        return active_columns
    
    def activate_cells(self, active_columns, input_data):
        active_cells = []
        for col in active_columns:
            # Find best matching cell in the column
            cell_scores = np.dot(self.cells[col].astype(int), input_data)
            
            # If no cell matches well, activate a new cell
            if np.max(cell_scores) == 0:
                best_cell = np.argmin(np.sum(self.cells[col], axis=1))
            else:
                best_cell = np.argmax(cell_scores)
            
            active_cells.append((col, best_cell))
        
        return active_cells
    
    def learn(self, active_cells, input_data):
        # Strengthen connections between previously active cells and currently active input
        for col, cell in self.previous_active_cells:
            self.cells[col][cell] |= input_data
        
        # Learn new patterns
        for col, cell in active_cells:
            self.cells[col][cell] |= input_data
        
        # Update previous active cells
        self.previous_active_cells = active_cells
        
        # Update sequence history
        self.sequence_history.append(active_cells)
        if len(self.sequence_history) > self.max_sequence_length:
            self.sequence_history.pop(0)
    
    def predict_next_state(self, active_cells):
        self.predictions = np.zeros(self.input_size, dtype=bool)
        for col, cell in active_cells:
            self.predictions |= self.cells[col][cell]
        
        # Ensure prediction sparsity
        if np.sum(self.predictions) > self.active_columns:
            top_indices = np.argsort(self.predictions.astype(int))[-self.active_columns:]
            self.predictions = np.zeros(self.input_size, dtype=bool)
            self.predictions[top_indices] = True
    
    def train(self, input_sequence, iterations=1):
        for _ in range(iterations):
            for t in range(len(input_sequence) - 1):
                current_input = input_sequence[t]
                next_input = input_sequence[t+1]
                
                # Process current input
                active_columns = self.activate_columns(current_input)
                active_cells = self.activate_cells(active_columns, current_input)
                
                # Learn
                self.learn(active_cells, next_input)
                
                # Make prediction for the next state
                self.predict_next_state(active_cells)
                
                # Store sequence memory
                self.sequence_memory.append((active_columns, active_cells))
    
    def predict(self, input_data):
        active_columns = self.activate_columns(input_data)
        active_cells = self.activate_cells(active_columns, input_data)
        self.predict_next_state(active_cells)
        return self.predictions
    
    def online_learning(self, new_input, true_next):
        # Make a prediction
        prediction = self.predict(new_input)
        
        # Learn from this new example (online learning)
        active_columns = self.activate_columns(new_input)
        active_cells = self.activate_cells(active_columns, new_input)
        self.learn(active_cells, true_next)
        
        return prediction

# Generate a sample input sequence (binary SDRs)
input_size = 100
num_columns = 500  # Further increased number of columns
cells_per_column = 64
active_columns = 70  # Increased number of active columns
sparsity = 0.01
boost_strength = 0.005
max_sequence_length = 20

# Generate a more complex input sequence (binary SDRs)
sequence_length = 1000
pattern_length = 5  # Length of repeating patterns
input_sequence = np.zeros((sequence_length, input_size), dtype=bool)

patterns = [np.random.choice(input_size, 5, replace=False) for _ in range(pattern_length)]
for i in range(sequence_length):
    pattern = patterns[i % pattern_length]
    input_sequence[i, pattern] = True

# Train the model with the best hyperparameters
sctl = EnhancedSCTL(input_size, num_columns, cells_per_column, active_columns, sparsity, boost_strength, max_sequence_length)
sctl.train(input_sequence)

# Test the model with online learning
correct_predictions = 0
total_predictions = 100

for _ in range(total_predictions):
    # Generate a new input
    new_input = np.zeros(input_size, dtype=bool)
    new_input[np.random.choice(input_size, 5, replace=False)] = True
    
    # Get the true next state (for testing purposes)
    true_next = np.zeros(input_size, dtype=bool)
    true_next[np.random.choice(input_size, 5, replace=False)] = True
    
    # Perform online learning and get prediction
    prediction = sctl.online_learning(new_input, true_next)
    
    # Check if the prediction is correct (allowing for partial matches)
    if np.sum(prediction & true_next) > 0:
        correct_predictions += 1

print(f"Prediction accuracy on complex sequences: {correct_predictions / total_predictions:.2f}")

# Detailed output of the internal state
print("\nInternal state details:")
print("Number of active columns:", np.sum(np.any(sctl.columns, axis=1)))
print("Number of active cells:", np.sum(np.any(sctl.cells, axis=2)))
print
