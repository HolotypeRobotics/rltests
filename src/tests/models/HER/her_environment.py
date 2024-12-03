import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

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

    def __init__(self, n_rewards, n_efforts, n_objects):
        self.rewards = np.array(n_rewards)
        self.efforts = np.array(n_efforts)
        self.objects = np.array(n_objects)
        self.state = (0, 0)
        self.direction = Direction.EAST
        self.object_positions = self.find_object_positions()
        self.distances = self.calculate_distances()
        self.reward = 0
        self.effort = 0
        self.done = False
        self.exit = (len(n_rewards) - 1, len(n_rewards[0]) - 1)
        self.path = []

    def state_to_sdr(self):
        y = np.zeros(self.rewards.shape[0])
        y[self.state[0]] = 1
        x = np.zeros(self.rewards.shape[1])
        x[self.state[1]] = 1
        return np.concatenate((x, y))

    def coord_to_sdr(self, x, y):
        _y = np.zeros(self.rewards.shape[0])
        _y[y] = 1
        _x = np.zeros(self.rewards.shape[1])
        _x[x] = 1
        return np.concatenate((_x, _y))

    def state_to_matrix(self):
        mat = np.zeros(self.rewards.shape)
        mat[self.state] = 1
        return mat

    def coord_to_matrix(self, x, y):
        mat = np.zeros(self.rewards.shape)
        mat[y, x] = 1
        return mat

    def reset(self):
        self.state = (0, 0)
        self.direction = Direction.EAST
        self.reward = 0  # No reward on reset
        self.effort = 0
        self.done = False
        distances = self.calculate_distances()
        return np.array(self.state_to_sdr(), dtype=int), self.reward, self.effort, self.direction, distances

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
        distances = np.zeros(len(self.object_positions))
        for i, (obj, pos) in enumerate(self.object_positions.items()):
            distances[i] = np.linalg.norm(np.array(self.state) - pos)
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
                state = self.coord_to_sdr(j, i)
                critic_values[i, j] = agent.critic.value(state)

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

    def step(self, action):
        if len(self.path) == 0:
            self.path.append(self.state)

        if self.state == self.exit:
            if np.random.rand() < 0.5:
                self.done = True

        old_state = self.state
        moved = False
        self.effort = 0  # Reset effort
        self.reward = 0  # Reset reward

        if action == Action.TURN_LEFT:
            self.direction = (self.direction - 1) % 4
            self.effort = 1  # Minimum effort for turning
        elif action == Action.TURN_RIGHT:
            self.direction = (self.direction + 1) % 4
            self.effort = 1  # Minimum effort for turning
        elif action == Action.MOVE_FORWARD:
            new_state = self.state
            if self.direction == Direction.NORTH:
                new_state = (max(self.state[0] - 1, 0), self.state[1])
            elif self.direction == Direction.SOUTH:
                new_state = (min(self.state[0] + 1, self.rewards.shape[0] - 1), self.state[1])
            elif self.direction == Direction.WEST:
                new_state = (self.state[0], max(self.state[1] - 1, 0))
            elif self.direction == Direction.EAST:
                new_state = (self.state[0], min(self.state[1] + 1, self.rewards.shape[1] - 1))
            
            if new_state != self.state:
                self.state = new_state
                moved = True
                self.effort = max(1, self.efforts[self.state])  # Minimum effort of 1 for movement
                self.reward = self.rewards[self.state]  # Only get reward if actually moved
        else:
            raise ValueError(f"Invalid action: {action}")

        if moved:
            self.path.append(self.state)

        self.distances = self.calculate_distances()

        return np.array(self.state_to_sdr(), dtype=int), self.reward, self.done, self.state, self.effort, self.direction, self.distances

    def close(self):
        pass
