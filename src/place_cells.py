import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd

'''

    This is a simple implementation of a hippocampal circuit model using PyTorch.
It takes in grid cell-like inputs from the entorhinal cortex (EC) and object location
inputs from the lateral entorhinal cortex (LEC). The CA3 module predicts the next input
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

# Define MEC activation function using periodic activity like grid cells with overlapping increasing sizes and offsets
def mec_activation(x, y, config):
    activations = []
    for scale in config['mec']['scales']:
        for offset in config['mec']['offsets']:
            activation = np.sin(2 * np.pi * (x + offset[0]) / scale) * np.sin(2 * np.pi * (y + offset[1]) / scale)
            activations.append(activation)
    return np.array(activations, dtype=np.float32)

# Define LEC activation function using distance to specific objects in the environment
def lec_activation(x, y, environment):
    activations = []
    for target in environment['place_field_targets']:
        distance = np.sqrt((x - target['position'][0])**2 + (y - target['position'][1])**2)
        activation = np.exp(-distance / 0.2)  # Gaussian activation
        activations.append(activation)
    return np.array(activations, dtype=np.float32)

class CA3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CA3, self).__init__()
        self.input_size = input_size
        # Projections from EC to CA3 (skipping DG)
        self.input_projection = nn.Linear(input_size, hidden_size)
        # Recurrent connections
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

        # Initialize weights and biases to zero
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.constant_(param.data, 0.0)
                
        # Output projections to CA1
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_state=None):
        # Project input to hidden state
        projected_input = F.relu(self.input_projection(input)).unsqueeze(1)
        # Recurrent activation
        output, hidden_state = self.rnn(projected_input, hidden_state)
        # Output
        predicted_ec = self.output(output.squeeze(1))
        return predicted_ec, hidden_state

class CA1(nn.Module):
    def __init__(self, ec_input_size, output_size):
        super(CA1, self).__init__()
        # Projections from EC to CA1
        self.ec_input_size = ec_input_size
        self.process_ec = nn.Linear(ec_input_size, output_size)

    def forward(self, ec_input):
        out = self.process_ec(ec_input)
        return F.relu(out)

class HippocampalCircuit(nn.Module):
    def __init__(self, ec_input_size, hidden_size, num_place_fields):
        super(HippocampalCircuit, self).__init__()
        # Number of place fields we want to predict
        self.num_place_fields = num_place_fields
        # CA3 and CA1 modules
        self.ca3 = CA3(ec_input_size, hidden_size, num_place_fields)
        # self.ca1 = CA1(ec_input_size, num_place_fields)

    def forward(self, ec_input, hidden_state=None, reality=0.1, prediction=0.9):
        # have ca3 predict the next input from its hidden state
        ca3_prediction, hidden_state = (self.ca3(ec_input, hidden_state))
        # have ec-ca1 net predict place from raw inputs
        # ca1_output = self.ca1(ec_input)
        # Combine the predictions based on reality and prediction percentages
        # combined_output = prediction * ca3_prediction + reality * ca1_output
        # place_field_activations = torch.tanh(combined_output)
        place_field_activations = torch.tanh(ca3_prediction)
        
        return place_field_activations, ca3_prediction, hidden_state

# Generate a path by taking steps in random directions
def generate_path(num_steps, step_size=0.05):
    path = torch.zeros(num_steps, 2, dtype=torch.float32)
    current_position = torch.rand(2, dtype=torch.float32)
    path[0] = current_position
    for i in range(1, num_steps):
        direction = torch.randn(2, dtype=torch.float32)
        direction /= direction.norm()
        new_position = current_position + step_size * direction
        new_position = torch.clamp(new_position, 0, 1)
        path[i] = new_position
        current_position = new_position
    return path

def create_place_field_target(position, strength=1.0, spread=0.2):
    def activation(x, y):
        distance = np.sqrt((x - position[0])**2 + (y - position[1])**2)
        return strength * np.exp(-distance / spread)
    return activation

def train_model(model, config, num_epochs, learning_rate, bptt_len=30):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        path = generate_path(config['num_steps'])
        hidden_state = None
        total_loss = 0

        for step in range(0, config['num_steps'] - bptt_len, bptt_len):
            optimizer.zero_grad()
            losses = []
            hidden_state = hidden_state.detach() if hidden_state is not None else None

            for sub_step in range(bptt_len):
                position = path[step + sub_step]
                ec_input = torch.cat([
                    torch.tensor(mec_activation(position[0].item(), position[1].item(), config), dtype=torch.float32),
                    torch.tensor(lec_activation(position[0].item(), position[1].item(), config['environment']), dtype=torch.float32)
                ])
                
                place_field_activations, _, hidden_state = model(ec_input.unsqueeze(0), hidden_state)
                
                # Calculate target place field activations
                target_activations = torch.zeros_like(place_field_activations)
                for i, target in enumerate(config['environment']['place_field_targets']):
                    target_activation = target['activation'](position[0].item(), position[1].item())
                    target_activations[0, i] = target_activation
                
                loss = criterion(place_field_activations, target_activations)
                losses.append(loss)

            # Accumulate losses and perform a single backward pass
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/(config['num_steps']/bptt_len):.4f}")

    return model


def train_model_on_moved_target(model, config, num_epochs, learning_rate, bptt_len=30):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    moved_target_index = len(config['environment']['place_field_targets']) - 1

    for epoch in range(num_epochs):
        path = generate_path(config['num_steps'])
        hidden_state = None
        total_loss = 0

        for step in range(0, config['num_steps'] - bptt_len, bptt_len):
            optimizer.zero_grad()
            losses = []
            hidden_state = hidden_state.detach() if hidden_state is not None else None

            for sub_step in range(bptt_len):
                position = path[step + sub_step]
                ec_input = torch.cat([
                    torch.tensor(mec_activation(position[0].item(), position[1].item(), config), dtype=torch.float32),
                    torch.tensor(lec_activation(position[0].item(), position[1].item(), config['environment']), dtype=torch.float32)
                ])
                
                place_field_activations, _, hidden_state = model(ec_input.unsqueeze(0), hidden_state)
                
                # Calculate target place field activation only for the moved target
                target_activations = torch.zeros_like(place_field_activations)
                target_activation = config['environment']['place_field_targets'][moved_target_index]['activation'](position[0].item(), position[1].item())
                target_activations[0, moved_target_index] = target_activation  # Only update the last place field target
                
                # Calculate the loss only for the moved target
                loss = criterion(place_field_activations[0, moved_target_index], target_activations[0, moved_target_index])
                losses.append(loss)

            # Accumulate losses and perform a single backward pass
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/(config['num_steps']/bptt_len):.4f}")

    return model

def predict_sequence(model, start_position, num_steps, config):
    model.eval()
    predictions = []
    hidden_state = None
    current_position = start_position

    with torch.no_grad():
        for _ in range(num_steps):
            ec_input = torch.cat([
                torch.tensor(mec_activation(current_position[0].item(), current_position[1].item(), config), dtype=torch.float32),
                torch.tensor(lec_activation(current_position[0].item(), current_position[1].item(), config['environment']), dtype=torch.float32)
            ])
            
            place_field_activations, _, hidden_state = model(ec_input.unsqueeze(0), hidden_state)
            predictions.append(place_field_activations.squeeze().numpy())
            
            # Update current_position based on the predicted place field activations
            max_activation_index = torch.argmax(place_field_activations)
            current_position = torch.tensor(config['environment']['place_field_targets'][max_activation_index]['position'], dtype=torch.float32)

    return np.array(predictions)

def visualize_place_fields(model, config):
    model.eval()
    resolution = 100
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    
    place_field_maps = []
    
    with torch.no_grad():
        for i in range(len(config['environment']['place_field_targets'])):
            activations = np.zeros((resolution, resolution))
            for xi in range(resolution):
                for yi in range(resolution):
                    ec_input = torch.cat([
                        torch.tensor(mec_activation(xx[xi, yi], yy[xi, yi], config), dtype=torch.float32),
                        torch.tensor(lec_activation(xx[xi, yi], yy[xi, yi], config['environment']), dtype=torch.float32)
                    ])
                    place_field_activations, _, _ = model(ec_input.unsqueeze(0))
                    activations[xi, yi] = place_field_activations[0, i].item()
            place_field_maps.append(activations)
    
    fig, axes = plt.subplots(1, len(place_field_maps), figsize=(5*len(place_field_maps), 5))
    if len(place_field_maps) == 1:
        axes = [axes]
    
    for i, (ax, field_map) in enumerate(zip(axes, place_field_maps)):
        im = ax.imshow(field_map, extent=[0, 1, 0, 1], origin='lower')
        ax.set_title(f"Place Field {i+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        target_pos = config['environment']['place_field_targets'][i]['position']
        ax.plot(target_pos[0], target_pos[1], 'r*', markersize=10)
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = {
        'num_steps': 100,
        'mec': {
            'scales': [ 1, 0.5],
            'offsets': [(0, 1), (1,0)]
        },
        'environment': {
            'place_field_targets': [
                {'position': (0.2, 0.3), 'activation': create_place_field_target((0.2, 0.3))},
                {'position': (0.7, 0.8), 'activation': create_place_field_target((0.5, 0.1))},
                {'position': (0.5, 0.5), 'activation': create_place_field_target((0.3, 0.5))},
                {'position': (0.1, 0.9), 'activation': create_place_field_target((0.7, 0.7))},
                {'position': (0.9, 0.1), 'activation': create_place_field_target((0.9, 0.9))},
            ]
        }
    }

    ec_input_size = len(config['mec']['scales']) * len(config['mec']['offsets']) + len(config['environment']['place_field_targets'])
    hidden_size = 100
    num_place_fields = len(config['environment']['place_field_targets'])
    
    model = HippocampalCircuit(ec_input_size, hidden_size, num_place_fields)
    model.float()

    # Initial training on all place field targets
    trained_model = train_model(model, config, num_epochs=100, learning_rate=0.001)
    print("Place field activations after initial training:")
    visualize_place_fields(trained_model, config)

    # Move the last place field target
    new_position = (0.9, 0.9)
    config['environment']['place_field_targets'][-1] = {
        'position': new_position,
        'activation': create_place_field_target(new_position)
    }

    # Retrain the model only on the moved target
    trained_model_on_moved_target = train_model_on_moved_target(trained_model, config, num_epochs=100, learning_rate=0.001)
    print("Place field activations after retraining on moved target:")
    visualize_place_fields(trained_model_on_moved_target, config)

    # Predict a sequence starting from a random position
    start_position = torch.rand(2)
    predicted_sequence = predict_sequence(trained_model_on_moved_target, start_position, num_steps=200, config=config)

    # Visualize the predicted sequence
    plt.figure(figsize=(10, 5))
    plt.imshow(predicted_sequence.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Place Field Activation')
    plt.xlabel('Time Step')
    plt.ylabel('Place Field')
    plt.title('Predicted Place Field Activations Over Time')
    plt.show()
