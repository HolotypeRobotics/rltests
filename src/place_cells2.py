import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''

    This is a simple implementation of a hippocampal circuit model using PyTorch.
It takes in grid cell-like inputs from the entorhinal cortex (EC) and object location
inputs from the local entorhinal cortex (LEC). The CA3 module predicts the next input
based on the current input and hidden state. The CA1 module combines the CA3 and EC
inputs to predict the place fields. The model is trained to predict the center of the
place fields with the locations of previous salient stimuli at the center.

    The model can be used more in a prediction-like manner with the CA3 module
predicting the next input. Or it can be used in a more based on current raw inputs
from the MEC and LEC. This is controlled by percentages of reality and prediction
in the CA1 module.

    The idea is that the prediction of place cells in CA3 is used for tracking and recalling
salient stimuli in the environment for later use. e.g. for remembering the location of landmarks 
on the way to a reward.

This should show if places are experienced/learned in a sequence, and then if the end of the 
sequence is moved, then learned, other places should move too because of RNN/CA3 prediction.

''' 

# Configuration
CONFIG = {
    'mec': {
        'scales': [0.1, 0.2, 0.4, 0.8],
        'offsets': [(0, 0), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
    },
    'environments': {
        'environment1': {
            'size': (1.0, 1.0),
            'objects': [(0.5, 0.5)],
            'corners': [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]  # Square
        },
        'environment2': {
            'size': (1.0, 1.0),
            'objects': [(0.5, 0.5)],
            'corners': [(0.1, 0.1), (0.8, 0.1), (0.8, 0.4), (0.5, 0.4), (0.5, 0.8), (0.1, 0.8)]  # L-shape
        },
        'environment3': {
            'size': (1.0, 1.0),
            'objects': [(0.5, 0.5)],
            'corners': [(0.1, 0.1), (0.9, 0.1), (0.9, 0.3), (0.3, 0.3), (0.3, 0.7), (0.9, 0.7), (0.9, 0.9), (0.1, 0.9)]  # U-shape
        },
        'environment4': {
            'size': (1.0, 1.0),
            'objects': [(0.5, 0.5)],
            'corners': [(0.1, 0.1), (0.9, 0.1), (0.9, 0.3), (0.6, 0.3), (0.6, 0.9), (0.4, 0.9), (0.4, 0.3), (0.1, 0.3)]  # T-shape
        },
        'environment5': {
            'size': (1.0, 1.0),
            'objects': [(0.5, 0.5)],
            'corners': [(0.1, 0.1), (0.3, 0.1), (0.3, 0.4), (0.7, 0.4), (0.7, 0.1), (0.9, 0.1), (0.9, 0.9), (0.7, 0.9), (0.7, 0.6), (0.3, 0.6), (0.3, 0.9), (0.1, 0.9)]  # H-shape
        },
        'environment6': {
            'size': (1.0, 1.0),
            'objects': [(0.5, 0.5)],
            'corners': [(0.1, 0.1), (0.5, 0.1), (0.5, 0.3), (0.3, 0.3), (0.3, 0.7), (0.5, 0.7), (0.5, 0.9), (0.1, 0.9)]  # S-shape
        }
    }
}


# Define MEC activation function using periodic activity like grid cells with overlapping increasing sizes and offsets
def mec_activation(x, y, config):
    activations = []
    for scale in config['mec']['scales']:
        for offset in config['mec']['offsets']:
            activation = np.sin(2 * np.pi * (x + offset[0]) / scale) * np.sin(2 * np.pi * (y + offset[1]) / scale)
            activations.append(activation)
    return np.array(activations)

# Define LEC activation function using distance to specific objects in the environment
def lec_activation(x, y, environment):
    distances = []
    for obstacle in environment['objects']:
        distance = np.sqrt((x - obstacle[0])**2 + (y - obstacle[1])**2)
        distances.append(distance)
    return np.array(distances)

class CA3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CA3, self).__init__()
        # Projections from EC to CA3 (skipping DG)
        self.input_projection = nn.Linear(input_size, hidden_size)
        # Recurrent connections
        self.recurrent = nn.RNN(hidden_size, hidden_size, batch_first=True)
        # Output projections to CA1
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_state=None):
        # Project input to hidden state
        projected_input = self.input_projection(input).unsqueeze(1)
        # Recurrent activation
        output, hidden_state = self.recurrent(projected_input, hidden_state)
        # Output
        predicted_ec = self.output(output.squeeze(1))
        return predicted_ec, hidden_state

class CA1(nn.Module):
    def __init__(self, ec_input_size, output_size):
        super(CA1, self).__init__()
        # Projections from EC to CA1
        self.process_ec = nn.Linear(ec_input_size, output_size)

    def forward(self, ca3_input, ec_input, reality, prediction):
        # Project EC input to the same size as CA3 input
        processed_ec_input = self.process_ec(ec_input)
        # inputs weighted more heavily based on prediction vs reality
        # prediction comes from CA3, reality comes from EC
        combined = prediction * ca3_input + reality * processed_ec_input  
        # Activation function
        return F.softmax(combined, dim=-1)

class HippocampalCircuit(nn.Module):
    def __init__(self, ec_input_size, num_places):
        super(HippocampalCircuit, self).__init__()
        # Number of place fields we want to predict
        self.num_place_fields = num_places
        # CA3 and CA1 modules
        self.ca3 = CA3(ec_input_size, num_places, num_places)
        self.ca1 = CA1(ec_input_size, num_places)

    def forward(self, ec_input, hidden_state=None):
        # have ca3 predict the next input from its hidden state
        ca3_prediction, hidden_state = self.ca3(ec_input, hidden_state)
        # Predict the place fields based on the EC input and the CA3 prediction
        places_field_activations = self.ca1(ca3_prediction, ec_input, .5, .5)
        return places_field_activations, ca3_prediction, hidden_state

def get_place_field_center(place_field_activations, num_place_fields):
    # Assume place fields are arranged in a grid
    grid_size = int(np.sqrt(num_place_fields))
    x_coords = torch.linspace(0, 1, grid_size).repeat(grid_size)
    y_coords = torch.linspace(0, 1, grid_size).repeat_interleave(grid_size)
    coords = torch.stack((x_coords, y_coords), dim=1)
    
    # Weight the coordinates by the place field activations
    weighted_coords = coords * place_field_activations.unsqueeze(1)
    center = weighted_coords.sum(dim=0) / place_field_activations.sum()
    return center


# Training function
def train_model(model, ec_inputs, object_locations, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for ec_input, object_location in zip(ec_inputs, object_locations):
            optimizer.zero_grad()
            
            place_field_activations, _, _ = model(ec_input.unsqueeze(0))
            predicted_location = get_place_field_center(place_field_activations.squeeze(), model.num_place_fields)
            
            loss = criterion(predicted_location, object_location)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(ec_inputs):.4f}")


# Test function
def test_model(model, ec_input, true_location):
    model.eval()
    with torch.no_grad():
        place_field_activations, _, _ = model(ec_input.unsqueeze(0))
        predicted_location = get_place_field_center(place_field_activations.squeeze(), model.num_place_fields)
        error = torch.sqrt(torch.sum((predicted_location - true_location) ** 2))
        print(f"Predicted location: ({predicted_location[0]:.4f}, {predicted_location[1]:.4f})")
        print(f"Actual location: ({true_location[0]:.4f}, {true_location[1]:.4f})")
        print(f"Error: {error:.4f}")
    return predicted_location, error

# Main execution
if __name__ == "__main__":
    # Setup
    ec_input_size = 10  # Adjust based on your EC input size
    num_place_fields = 100  # 10x10 grid of place fields
    model = HippocampalCircuit(ec_input_size, num_place_fields)

    # Generate training data
    num_samples = 100
    ec_inputs = torch.rand(num_samples, ec_input_size)
    object_locations = torch.rand(num_samples, 2)  # Random object locations

    # Train the model
    train_model(model, ec_inputs, object_locations, num_epochs=200, learning_rate=0.001)

    # Test the model
    test_ec_input = torch.rand(ec_input_size)
    test_object_location = torch.rand(2)
    predicted_location, error = test_model(model, test_ec_input, test_object_location)

    # Move the object and test again
    moved_object_location = test_object_location + torch.tensor([0.2, 0.2])  # Move object
    print("\nAfter moving the object:")
    predicted_location_after_move, error_after_move = test_model(model, test_ec_input, moved_object_location)

    # Retrain with new object location
    print("\nRetraining with new object location:")
    train_model(model, ec_inputs, object_locations, num_epochs=50, learning_rate=0.0001)

    # Test again after retraining
    print("\nAfter retraining:")
    final_predicted_location, final_error = test_model(model, test_ec_input, moved_object_location)