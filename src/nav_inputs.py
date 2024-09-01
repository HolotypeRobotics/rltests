import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch

# Configuration parameters
CONFIG = {
    'environment': {
        'size': (100, 100),
        'n_obstacles': 3,
        'n_targets': 3,
        'n_places': 10,
        'target_rewards': {
            'food': 1.0,
            'water': 0.5,
            'shelter': 0.3,
        },
        'target_stresses': {
            'food': 0.1,
            'water': 0.2,
            'shelter': 0.4,
        }
    },
    'agent': {
        'hunger_range': (0, 1),
        'target_sensitivities': {
            'food': 0.9,
            'water': 0.7,
            'shelter': 0.5
        },
    },
    'dca1': {
        'scales': [10, 20, 40, 80],
        'offsets': [(0, 0), (5, 5), (10, 10)],
    },
    'dsub': {
        'n_directions': 8,
        'direction_distance': 5,
    },
    'ica1': {
        'directions': [(0, 1), (1, 0), (0, -1), (-1, 0), (-.5, .5)],
    },
    'visualization': {
        'grid_resolution': 100,
        'agent_arrow_length': 5,
        'agent_arrow_width': 2,
        'subplot_layout': (2, 3),
    }
}


class Environment:
    def __init__(self, size, obstacles, targets):
        self.size = size
        self.obstacles = obstacles
        self.targets = targets

# Define PyTorch layer to predict place based on LEC activation
class PlacePredictionModel(nn.Module):
    def __init__(self, num_task_sets, num_objects, num_places, hidden_size=64):
        # super(PlacePredictionModel, self).__init__()
        # self.fc = nn.Linear(num_task_sets * num_objects, num_places)  # Predicting a single value (e.g., place index or score)
        # self.sigmoid = nn.Sigmoid()  # Activation function

        super(PlacePredictionModel, self).__init__()
        self.fc1 = nn.Linear(num_task_sets*num_objects, hidden_size)  # Input layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, num_places)  # Output layer


    # def forward(self, x):
    #     x = self.fc(x)
    #     x = self.sigmoid(x)
    #     return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output

class Agent:
    def __init__(self, x, y, heading, config):
        self.x = x
        self.y = y
        self.heading = heading
        self.hunger = np.random.uniform(*config['agent']['hunger_range'])
        self.target_sensitivities = config['agent']['target_sensitivities']

def dca1_activation(x, y, environment, config):
    """Fine allocentric place using overlapping grids of increasing resolution."""
    activations = []
    for scale in config['dca1']['scales']:
        for offset in config['dca1']['offsets']:
            activation = np.sin(2 * np.pi * (x + offset[0]) / scale) * np.sin(2 * np.pi * (y + offset[1]) / scale)
            activations.append(activation)
    return activations

def lec_activation(x, y, environment):
    """Calculate the distance to all objects (obstacles and targets) in the environment."""
    distances = []
    # Calculate distances to obstacles
    for obstacle in environment.obstacles:
        distance = np.sqrt((x - obstacle[0])**2 + (y - obstacle[1])**2)
        distances.append(distance)
    
    # Calculate distances to targets
    for target in environment.targets:
        distance = np.sqrt((x - target['position'][0])**2 + (y - target['position'][1])**2)
        distances.append(distance)
    
    return distances


def vca1_activation(x, y, environment, config, task_set_index):
    """
    Broad allocentric place prediction using lec_activation as input to a PyTorch model.
    
    Parameters:
    - task_set_index: Current task set index to decide where to input data
    """
    # Calculate distances to all objects
    distances = lec_activation(x, y, environment)
    
    # Collect rewards and stresses for all objects (obstacles have 0 reward/stress)
    rewards = [target['reward'] for target in environment.targets] + [0] * len(environment.obstacles)
    stresses = [target['stress'] for target in environment.targets] + [0] * len(environment.obstacles)
    
    # Calculate input as distances weighted by (reward - stress)
    weighted_distances = [dist * (reward - stress) for dist, reward, stress in zip(distances, rewards, stresses)]
    
    # Prepare input tensor for the model
    num_objects = len(weighted_distances)
    input_tensor = torch.zeros(config['environment']['n_targets'] * num_objects)  # assuming a 1D input
    start_index = task_set_index * num_objects
    input_tensor[start_index:start_index + num_objects] = torch.tensor(weighted_distances, dtype=torch.float32)
    
    # Create model and predict
    num_task_sets = config['environment']['n_targets']  # Assuming number of task sets is the length of targets list
    num_places = config['environment']['n_places']
    model = PlacePredictionModel(num_task_sets, num_objects, num_places)
    output = model(input_tensor)
    
    return output.detach().numpy()

def dsub_heading_activation(heading):
    """Broad allocentric environment heading."""
    return (np.sin(heading) + 1) / 2

def dsub_direction_options(x, y, environment, config):
    """Possible allocentric directional options respecting obstacles."""
    directions = []
    for angle in np.linspace(0, 2*np.pi, config['dsub']['n_directions'], endpoint=False):
        dx, dy = config['dsub']['direction_distance'] * np.cos(angle), config['dsub']['direction_distance'] * np.sin(angle)
        new_x, new_y = x + dx, y + dy
        if not any((abs(new_x - obs[0]) < config['dsub']['direction_distance'] and 
                    abs(new_y - obs[1]) < config['dsub']['direction_distance']) for obs in environment.obstacles):
            directions.append(angle)
    return directions

def vsub_activation(agent, environment):
    """Egocentric distances of motivationally relevant targets."""
    distances = {}
    for target in environment.targets:
        distance = np.sqrt((agent.x - target['position'][0])**2 + (agent.y - target['position'][1])**2)
        sensitivity = agent.target_sensitivities.get(target['type'], 0.5)  # Default sensitivity 0.5
        distances[target['type']] = distance * sensitivity
    return distances

def ica1_activation(agent, environment, config):
    """Egocentric transition points respecting immediate boundaries."""
    transitions = []
    for dx, dy in config['ica1']['directions']:
        new_x, new_y = agent.x + dx, agent.y + dy
        if not any((new_x, new_y) == obstacle for obstacle in environment.obstacles):
            transition = (new_x - agent.x, new_y - agent.y)
            transitions.append(transition)
    return transitions

def simulate_hippocampal_activity(agent, environment, config, task_set_index):
    dca1 = dca1_activation(agent.x, agent.y, environment, config)
    vca1 = vca1_activation(agent.x, agent.y, environment, CONFIG, task_set_index)
    dsub_heading = dsub_heading_activation(agent.heading)
    dsub_directions = dsub_direction_options(agent.x, agent.y, environment, config)
    vsub = vsub_activation(agent, environment)
    ica1 = ica1_activation(agent, environment, config)
    
    return {
        'dCA1': dca1,
        'vCA1': vca1[0],
        'dSub_heading': dsub_heading,
        'dSub_directions': dsub_directions,
        'vSub': vsub,
        'iCA1': ica1
    }

def visualize_environment(agent, environment, hippocampal_activity, config):
    plt.figure(figsize=(12, 8))
    
    # Plot environment
    plt.scatter(*zip(*environment.obstacles), color='red', s=100, label='Obstacles')
    for target in environment.targets:
        plt.scatter(*target['position'], color='green', s=100, label=f"{target['type']}")
    
    # Plot agent
    plt.scatter(agent.x, agent.y, color='blue', s=200, label='Agent')
    arrow_length = config['visualization']['agent_arrow_length']
    arrow_width = config['visualization']['agent_arrow_width']
    plt.arrow(agent.x, agent.y, arrow_length*np.cos(agent.heading), arrow_length*np.sin(agent.heading), 
              head_width=arrow_width, head_length=arrow_width, fc='blue', ec='blue')
    
    # Add text for hippocampal activations
    activation_text = "\n".join([
        f"dCA1: {len(hippocampal_activity['dCA1'])} activations",
        f"vCA1: {hippocampal_activity['vCA1']:.2f}",
        f"dSub heading: {hippocampal_activity['dSub_heading']:.2f}",
        f"dSub directions: {len(hippocampal_activity['dSub_directions'])} options",
        f"vSub: {', '.join(f'{k}: {v:.2f}' for k, v in hippocampal_activity['vSub'].items())}",
        f"iCA1: {len(hippocampal_activity['iCA1'])} transitions"
    ])
    plt.text(0.02, 0.98, activation_text, transform=plt.gca().transAxes, verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlim(0, environment.size[0])
    plt.ylim(0, environment.size[1])
    plt.legend()
    plt.title("Simulated Environment and Hippocampal Activations")
    plt.show()

def visualize_region_activations(environment, agent, config, task_set_index):
    fig, axs = plt.subplots(*config['visualization']['subplot_layout'], figsize=(18, 12))
    axs = axs.flatten()
    
    resolution = config['visualization']['grid_resolution']
    x = np.linspace(0, environment.size[0], resolution)
    y = np.linspace(0, environment.size[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # dCA1 (average of activations)
    Z_dca1 = np.vectorize(lambda x, y: np.mean(dca1_activation(x, y, environment, config)))(X, Y)
    axs[0].imshow(Z_dca1, extent=[0, environment.size[0], 0, environment.size[1]], origin='lower', cmap='viridis')
    axs[0].set_title('dCA1 Activation (Average)')
    
    # vCA1
    Z_vca1 = np.vectorize(lambda x, y: vca1_activation(x, y, environment, config, task_set_index)[0])(X, Y)
    axs[1].imshow(Z_vca1, extent=[0, environment.size[0], 0, environment.size[1]], origin='lower', cmap='viridis')
    axs[1].set_title('vCA1 Activation')
    
    # dSub (heading)
    Z_dsub_heading = np.full_like(X, dsub_heading_activation(agent.heading))
    axs[2].imshow(Z_dsub_heading, extent=[0, environment.size[0], 0, environment.size[1]], origin='lower', cmap='viridis')
    axs[2].set_title('dSub Heading Activation')
    
    # dSub (direction options)
    Z_dsub_directions = np.vectorize(lambda x, y: len(dsub_direction_options(x, y, environment, config)))(X, Y)
    axs[3].imshow(Z_dsub_directions, extent=[0, environment.size[0], 0, environment.size[1]], origin='lower', cmap='viridis')
    axs[3].set_title('dSub Direction Options')
    
    # vSub (average of distances)
    Z_vsub = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temp_agent = Agent(X[i,j], Y[i,j], agent.heading, config)
            Z_vsub[i,j] = np.mean(list(vsub_activation(temp_agent, environment).values()))
    axs[4].imshow(Z_vsub, extent=[0, environment.size[0], 0, environment.size[1]], origin='lower', cmap='viridis')
    axs[4].set_title('vSub Activation (Average Distance)')
    
    # iCA1 (transition points)
    Z_ica1 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temp_agent = Agent(X[i,j], Y[i,j], agent.heading, config)
            transitions = ica1_activation(temp_agent, environment, config)
            Z_ica1[i,j] = len(transitions)
    axs[5].imshow(Z_ica1, extent=[0, environment.size[0], 0, environment.size[1]], origin='lower', cmap='viridis')
    axs[5].set_title('iCA1 Activation (Transition Points)')
    
    # Plot environment on all subplots
    for ax in axs:
        ax.scatter(*zip(*environment.obstacles), color='red', s=20)
        for target in environment.targets:
            ax.scatter(*target['position'], color='green', s=20)
        ax.scatter(agent.x, agent.y, color='blue', s=50)
        ax.set_xlim(0, environment.size[0])
        ax.set_ylim(0, environment.size[1])
    
    plt.tight_layout()
    plt.show()

def visualize_vsub(agent, environment, config):
    """Visualize the vSub distances weighted by sensitivities."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    distances = vsub_activation(agent, environment)
    target_types = list(distances.keys())
    distance_values = list(distances.values())
    
    ax.bar(target_types, distance_values, color=['green', 'blue', 'purple'])
    ax.set_ylabel('Weighted Distance')
    ax.set_title('vSub Target Distances (Weighted by Sensitivity)')
    
    plt.show()

def visualize_dca1_grids(environment, config):
    """Visualize the dCA1 activation patterns for different scales and offsets."""
    scales = config['dca1']['scales']
    offsets = config['dca1']['offsets']
    
    fig, axs = plt.subplots(len(scales), len(offsets), figsize=(15, 10))
    resolution = config['visualization']['grid_resolution']
    x = np.linspace(0, environment.size[0], resolution)
    y = np.linspace(0, environment.size[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    for i, scale in enumerate(scales):
        for j, offset in enumerate(offsets):
            Z = np.sin(2 * np.pi * (X + offset[0]) / scale) * np.sin(2 * np.pi * (Y + offset[1]) / scale)
            ax = axs[i, j]
            ax.imshow(Z, extent=[0, environment.size[0], 0, environment.size[1]], origin='lower', cmap='viridis')
            ax.set_title(f'Scale: {scale}, Offset: {offset}')
            ax.scatter(*zip(*environment.obstacles), color='red', s=20)
            for target in environment.targets:
                ax.scatter(*target['position'], color='green', s=20)
    
    plt.tight_layout()
    plt.show()

def visualize_ica1_transitions(environment, config):
    """Visualize iCA1 transition points across the grid."""
    resolution = config['visualization']['grid_resolution']
    x = np.linspace(0, environment.size[0], resolution)
    y = np.linspace(0, environment.size[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temp_agent = Agent(X[i,j], Y[i,j], 0, config)
            transitions = ica1_activation(temp_agent, environment, config)
            Z[i,j] = len(transitions)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=[0, environment.size[0], 0, environment.size[1]], origin='lower', cmap='viridis')
    plt.scatter(*zip(*environment.obstacles), color='red', s=20, label='Obstacles')
    for target in environment.targets:
        plt.scatter(*target['position'], color='green', s=20, label=f"{target['type']}")
    plt.title('iCA1 Transition Points')
    plt.colorbar(label='Number of Transitions')
    plt.show()

def create_random_environment(config):
    """Creates a random environment with obstacles and targets using the configuration."""
    size = config['environment']['size']
    obstacles = [(np.random.randint(0, size[0]), np.random.randint(0, size[1])) 
                 for _ in range(config['environment']['n_obstacles'])]
    targets = [{'position': (np.random.randint(0, size[0]), np.random.randint(0, size[1])),
                'type': np.random.choice(['food', 'water', 'shelter']),
                'value': np.random.random()} 
               for _ in range(config['environment']['n_targets'])]
    
    # Add rewards and stresses to targets based on their type
    for target in targets:
        target_type = target['type']
        target['reward'] = config['environment']['target_rewards'][target_type]
        target['stress'] = config['environment']['target_stresses'][target_type]
    
    return Environment(size, obstacles, targets)

environment = create_random_environment(CONFIG)
agent = Agent(x=np.random.randint(0, environment.size[0]),
              y=np.random.randint(0, environment.size[1]),
              heading=np.random.uniform(0, 2*np.pi),
              config=CONFIG)

task_set_index = 0 

hippocampal_activity = simulate_hippocampal_activity(agent, environment, CONFIG, task_set_index)

visualize_vsub(agent, environment, CONFIG)
visualize_dca1_grids(environment, CONFIG)
visualize_environment(agent, environment, hippocampal_activity, CONFIG)
visualize_region_activations(environment, agent, CONFIG, task_set_index)
visualize_ica1_transitions(environment, CONFIG)