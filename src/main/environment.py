import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn.functional as F


class Action:
    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2

class Direction:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

# ANSI color codes
COLORS = {
    1: '\033[31m',  # Red
    2: '\033[32m',  # Green
    3: '\033[34m',  # Blue
    4: '\033[33m',  # Yellow
    5: '\033[35m',  # Magenta
    6: '\033[36m',  # Cyan
    7: '\033[37m',  # White
}
RESET = '\033[0m'  # Reset color

class Environment:
    def action_to_string(self, action):
        if action == Action.TURN_LEFT:
            return "TURN LEFT"
        elif action == Action.TURN_RIGHT:
            return "TURN RIGHT"
        elif action == Action.MOVE_FORWARD:
            return "MOVE FORWARD"
        else:
            raise ValueError("Invalid action")

    def __init__(self, rewards=None, efforts=None, n_objects=None): # Added default dimensions and objects
        self.width = len(rewards[0])
        self.height = len(rewards)
        self.n_objects = n_objects
        if rewards is not None:
            self.rewards = np.array(rewards)
            self.available_rewards = self.rewards
        if efforts is not None:
            self.efforts = np.array(efforts)
        self.objects = np.zeros((self.height, self.width), dtype=int) # Initialize with zeros
        self.state = (0, 0)
        self.previous_action = Action.MOVE_FORWARD
        self.direction = Direction.EAST
        self.start = (0, 0)
        self.exits = []
        self.place_objects() # Place objects randomly
        self.object_positions = self.find_object_positions()
        self.distances = self.calculate_distances()
        self.reward = 0
        self.effort = 0
        self.done = False
        self.imagined_scenes = []
        self.path = []

        # Precompute SDRs for efficiency
        self.state_sdrs = {}
        for y in range(self.height):
            for x in range(self.width):
                self.state_sdrs[(y, x)] = self._coord_to_sdr(x, y)
        self.direction_sdrs = {
            Direction.NORTH: np.array([1, 0, 0, 0]),
            Direction.EAST: np.array([0, 1, 0, 0]),
            Direction.SOUTH: np.array([0, 0, 1, 0]),
            Direction.WEST: np.array([0, 0, 0, 1])
        }

    def place_objects(self):
        # Place objects randomly (excluding start and exit)
        available_positions = [(y, x) for y in range(self.height) for x in range(self.width) if (y, x) != (0, 0)]
        object_positions = np.random.choice(len(available_positions), size=self.n_objects, replace=False)
        for i, idx in enumerate(object_positions):
            self.objects[available_positions[idx]] = i + 1 # Assign unique object IDs

    def _coord_to_sdr(self, x, y):
        # Helper function (no change in logic, just renaming)
        sdr = np.zeros(self.width + self.height)
        sdr[y] = 1
        sdr[self.height + x] = 1
        return sdr 

    def state_to_sdr(self):
        # One-hot encode location
        y = F.one_hot(torch.tensor(self.state[0]), num_classes=self.rewards.shape[0]).float()
        x = F.one_hot(torch.tensor(self.state[1]), num_classes=self.rewards.shape[1]).float()

        # One-hot encode direction
        direction_sdr = F.one_hot(torch.tensor(self.direction), num_classes=4).float()

        distances_sdr = self.distances / np.linalg.norm(self.distances) # Normalize distances

        s = torch.cat([x, y, distances_sdr, direction_sdr], dim=0)
        return s

    def state_to_matrix(self):
        mat = np.zeros(self.rewards.shape)
        mat[self.state] = 1
        return mat

    def coord_to_matrix(self, x, y):
        mat = np.zeros(self.rewards.shape)
        mat[y, x] = 1
        return mat

    def state_to_one_hot(self, state):
        size = self.rewards.shape[0] * self.rewards.shape[1]
        one_hot = np.zeros(size)
        index = state[0] * self.rewards.shape[1] + state[1]
        one_hot[index] = 1
        return one_hot
    
    def one_hot_to_state(self, one_hot):
        index = np.argmax(one_hot)
        y = index // self.width
        x = index % self.width
        print(f"Index: {index}, Y: {y}, X: {x}")
        return (y, x)

    def direction_to_one_hot(self, direction):
        one_hot = np.zeros(4)
        one_hot[direction] = 1
        return one_hot

    def load_from_file(self, filepath):
        """
        Load environment layout from a text file.
        
        File format:
        # - Wall
        . - Empty space
        S - Starting position
        E - Exit
        1-9 - Objects (numbered)
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Strip whitespace and filter out empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        # Determine dimensions
        height = len(lines)
        width = max(len(line) for line in lines)
        
        # Resize environment if needed
        if height > self.height or width > self.width:
            self.height = height
            self.width = width
            self.rewards = np.zeros((height, width))
            self.efforts = np.zeros((height, width))
            self.walls = np.zeros((height, width))
            self.objects = np.zeros((height, width), dtype=int)
        
        # Parse the file
        start_found = False
        exit_positions = []
        object_positions = {}
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == '#':
                    # Wall
                    self.walls[y, x] = 1
                    self.rewards[y, x] = -0.1  # Slight negative reward for walls
                elif char == 'S':
                    # Starting position
                    self.set_start(x, y)
                    start_found = True
                elif char == 'E':
                    # Exit
                    exit_positions.append((x, y))
                    self.rewards[y, x] = 10.0  # High reward for exit
                elif char.isdigit() and char != '0':
                    # Object
                    obj_id = int(char)
                    self.objects[y, x] = obj_id
                    object_positions[obj_id] = (y, x)
                else:
                    # Empty space
                    pass
        
        # Set exits
        self.exits = [(y, x) for x, y in exit_positions]
        
        # If no start was specified, use default
        if not start_found:
            self.set_start(0, 0)
        
        # Update object positions
        self.object_positions = self.find_object_positions()
        self.distances = self.calculate_distances()
        
        # Initialize state
        self.state = self.start
        
        return True

    def action_to_one_hot(self, action):
        one_hot = np.zeros(3)
        one_hot[action] = 1
        return one_hot

    def reset(self):
        self.state = self.start
        self.direction = Direction.EAST
        self.reward = 0  # No reward on reset
        self.effort = 0
        self.done = False
        self.path = []
        return self.get_outputs()

    def set_effort(self, x, y, effort):
        self.efforts[y, x] = effort

    def set_reward(self, x, y, reward):
        self.rewards[y, x] = reward

    def set_start(self, x, y):
        self.start = (y, x)

    def add_exit(self, x, y):
        self.exits.append((y, x))

    def find_object_positions(self):
        object_positions = {}
        for obj in range(1, np.max(self.objects) + 1):  # Assuming 0 means no object
            positions = np.argwhere(self.objects == obj)
            if len(positions) > 0:
                object_positions[obj] = positions[0]
        return object_positions

    def calculate_distances(self):
        distances = np.zeros(self.n_objects)
        for i in range(self.n_objects):
            object_positions = np.argwhere(self.objects == i + 1)
            if len(object_positions) > 0:
                distances[i] = np.linalg.norm(np.array(self.state) - object_positions[0])
            else: # Handle cases where an object might not be present
                distances[i] = -1 # Or some other sentinel value

        return distances

    def render(self):
        # Create a matrix for just the visual characters
        mat = np.full(self.rewards.shape, ' ', dtype=str)
        
        # Track object types in a separate matrix for coloring later
        obj_types = np.full(self.rewards.shape, 0, dtype=int)
        
        # Fill in objects
        object_positions = self.find_object_positions()
        for obj_id, pos in object_positions.items():
            mat[pos[0], pos[1]] = '●'
            obj_types[pos[0], pos[1]] = obj_id
        
        # Place agent on top
        if self.direction == Direction.NORTH:
            sprite = '▲'
        elif self.direction == Direction.SOUTH:
            sprite = '▼'
        elif self.direction == Direction.WEST:
            sprite = '◀'
        elif self.direction == Direction.EAST:
            sprite = '▶'

        for direction, state, color in self.imagined_scenes:
            if direction == Direction.NORTH:
                ghost = '▲'
            elif direction == Direction.SOUTH:
                ghost = '▼'
            elif direction == Direction.WEST:
                ghost = '◀'
            elif direction == Direction.EAST:
                ghost = '▶'
            else:
                ghost = ' '
            imagined_row, imagined_col = state
            mat[imagined_row, imagined_col] = ghost
            obj_types[imagined_row, imagined_col] = color

        agent_row, agent_col = self.state
        mat[agent_row, agent_col] = sprite
        obj_types[agent_row, agent_col] = 6  # Special code for agentobj_types
        
        
        # Calculate width based on visual characters only
        width = 2 * mat.shape[1] + 1
        
        # Render with fixed-width borders
        print('┌' + '─' * width + '┐')
        for i in range(mat.shape[0]):
            line = '│ '
            for j in range(mat.shape[1]):
                # Apply color based on object type
                obj_id = obj_types[i, j]
                char = mat[i, j]
                
                if obj_id == -1:  # Agent
                    line += f'\033[36m{char}\033[0m '  # Cyan for agent
                elif obj_id > 0:  # Object with color
                    line += f'{COLORS[obj_id]}{char}{RESET} '
                else:  # Empty space
                    line += f'{char} '
            
            line += '│'
            print(line)
        
        print('└' + '─' * width + '┘')

        self.remove_projections()

    def project(self, state, direction, color):
        state = self.one_hot_to_state(state)
        self.imagined_scenes.append((direction, state, color))

    def remove_projections(self):
        self.imagined_scenes = []

    def create_grid_and_path(self):
        grid = np.zeros_like(self.rewards, dtype=int)
        grid[self.exit] = 2  # Mark the exit cell

        path = []
        for x, y in self.path:
            path.append([y, x])

        return grid, path

    def plot_path(self, agent):
        fig, ax = plt.subplots(figsize=(5, 5))

        grid, path = self.create_grid_and_path()

        # Get the critic values for each state
        critic_values = np.zeros_like(self.rewards)
        for i in range(self.rewards.shape[0]):
            for j in range(self.rewards.shape[1]):
                # state = self._coord_to_sdr(j, i)
                # critic_values[i, j] = agent.critic.value(state) # Gets the critic value for the state
                critic_values[i, j] = self.rewards[i, j]

        # Plot the grid
        cmap = colors.ListedColormap(['white', 'black', 'red', 'blue', 'green', 'yellow', 'purple', 'cyan'])
        ax.imshow(self.objects, cmap=cmap)

        # Add cell values to the grid
        for i in range(self.rewards.shape[0]):
            for j in range(self.rewards.shape[1]):
                cell_value = critic_values[i, j]
                ax.text(j, i, f"{cell_value:.2f}", ha='center', va='center', color='black')

        # Plot the path
        path = np.array(path)
        print(path)
        if path.size > 0:
            ax.plot(path[:, 1], path[:, 0], '-o', markersize=10, color='orange')
            ax.plot(path[0, 1], path[0, 0], 'oy', markersize=12)  # Start position
            ax.plot(path[-1, 1], path[-1, 0], 'ob', markersize=12)  # End position

        # Set the axis limits and remove ticks
        ax.set_xticks(range(self.rewards.shape[1]))
        ax.set_yticks(range(self.rewards.shape[0]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, self.rewards.shape[1] - 0.5)
        ax.set_ylim(self.rewards.shape[0] - 0.5, -0.5)

        num_steps = len(path)
        ax.text(0.05, 0.95, f"Steps: {num_steps}", transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', pad=3.0))

        plt.show()

    def reset_path(self):
        self.path = []
        self.effort = 0  # Reset effort
        self.reward = 0  # Reset reward

    def get_outputs(self):
        self.distances = self.calculate_distances()

        position = self.state_to_one_hot(self.state)
        direction = self.direction_to_one_hot(self.direction)
        previous_action = self.action_to_one_hot(self.previous_action)
        previous_action = torch.from_numpy(previous_action).float().unsqueeze(0)
        position = torch.from_numpy(position).unsqueeze(0).float()
        direction = torch.from_numpy(direction).unsqueeze(0).float()
        obj_distances = F.softmax(torch.from_numpy(self.distances).float(), dim=-1).unsqueeze(0)
        # previous action doesnt need to be converted
        return (position, direction, obj_distances, previous_action), self.reward, self.effort, self.done

    def step(self, action):
        
        self.previous_action = action
        new_state = self.state

        if len(self.path) == 0:
            self.path.append(self.state)

        if action == Action.TURN_LEFT:
            self.direction = (self.direction - 1) % 4
            self.effort += 0.0012
        elif action == Action.TURN_RIGHT:
            self.direction = (self.direction + 1) % 4
            self.effort += 0.0012
        elif action == Action.MOVE_FORWARD:
            self.effort += 0.001

            if self.direction == Direction.NORTH:
                new_state = (max(self.state[0] - 1, 0), self.state[1])
            elif self.direction == Direction.SOUTH:
                new_state = (min(self.state[0] + 1, self.rewards.shape[0] - 1), self.state[1])
            elif self.direction == Direction.WEST:
                new_state = (self.state[0], max(self.state[1] - 1, 0))
            elif self.direction == Direction.EAST:
                new_state = (self.state[0], min(self.state[1] + 1, self.rewards.shape[1] - 1))
        else:
            raise ValueError(f"Invalid action: {action}")
        
        self.reward = self.available_rewards[new_state]
        self.available_rewards[new_state] = 0 # Mark as visited
        self.effort += self.efforts[new_state]

        # Figure out if the agent moved
        if new_state != self.state:
            self.path.append(self.state)

        self.state = new_state

        for exit in self.exits:
            if self.state == exit:
                self.done = True

        print()
        self.render()
        print(f"Reward: {self.reward}")
        
        return self.get_outputs()
        
    def close(self):
        pass
