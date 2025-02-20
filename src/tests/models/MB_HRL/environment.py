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

    def __init__(self, width, height, n_efforts, n_objects): # Added default dimensions and objects
        self.width = width
        self.height = height
        self.n_objects = n_objects
        self.rewards = np.zeros((height, width)) # Initialize rewards to zero
        self.objects = np.zeros((height, width), dtype=int) # Initialize objects to zero
        self.efforts = np.array(n_efforts)
        self.state = (0, 0)
        self.direction = Direction.EAST
        self.exit = (height - 1, width - 1) # Bottom-right corner
        self.place_objects() # Place objects randomly
        self.object_positions = self.find_object_positions()
        self.distances = self.calculate_distances()
        self.reward = 0
        self.effort = 0
        self.done = False
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
        available_positions = [(y, x) for y in range(self.height) for x in range(self.width) if (y, x) != (0, 0) and (y, x) != self.exit]
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

    def direction_to_one_hot(self, direction):
        one_hot = np.zeros(4)
        one_hot[direction] = 1
        return one_hot

    def proximity_to_one_hot(self, distance, n_bins=3, max_distance=None):
        if max_distance is None:
            max_distance = np.max(self.rewards.shape) * np.sqrt(2) # Diagonal
        bins = np.linspace(0, max_distance, n_bins + 1)
        bin_index = np.digitize(distance, bins) - 1 # Subtract 1 to make indices 0-based
        one_hot = np.zeros(n_bins)
        one_hot[min(bin_index, n_bins - 1)] = 1 # Ensure index is within bounds
        return one_hot

    def action_to_one_hot(self, action):
        one_hot = np.zeros(3)
        one_hot[action] = 1
        return one_hot

    def reset(self):
        self.state = (0, 0)
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

    def set_exit(self, x, y):
        self.exit = (y, x)

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
        print(self.state_to_matrix())

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

        state_one_hot = self.state_to_one_hot(self.state)
        direction_one_hot = self.direction_to_one_hot(self.direction)
        proximity_one_hots = np.array([self.proximity_to_one_hot(d) for d in self.distances])
        previous_action_one_hot = self.action_to_one_hot(0) # Initialize with no previous action
        s = np.concatenate([state_one_hot, direction_one_hot, proximity_one_hots.flatten(), previous_action_one_hot])

        return s, self.reward, self.effort, self.done

    def step(self, action):
        moved = False
        new_state = self.state

        if len(self.path) == 0:
            self.path.append(self.state)

        if self.state == self.exit:
            self.done = True

        if action == Action.TURN_LEFT:
            self.direction = (self.direction - 1) % 4
            self.effort = 0  # Minimum effort for turning
        elif action == Action.TURN_RIGHT:
            self.direction = (self.direction + 1) % 4
            self.effort = 0  # Minimum effort for turning
        elif action == Action.MOVE_FORWARD:
            self.effort = 0  # Minimum effort for moving forward
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
        
        # Figure out if the agent moved
        if new_state != self.state:
            print(f"Moved to {new_state} from {self.state}")
            print(f"got reward: {self.rewards[new_state]}")

            self.state = new_state
            moved = True
            self.effort = max(1, self.efforts[self.state])  # Minimum effort of 1 for movement
            self.reward = self.rewards[self.state]  # Only get reward if actually moved
        else:
            print(f"No movement. reward: {self.reward}")
            self.effort = 0
            self.reward = 0

        if moved:
            self.path.append(self.state)
        
        return self.get_outputs()
        
    def close(self):
        pass
