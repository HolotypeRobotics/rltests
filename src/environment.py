import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Environment:

    def action_to_string(self, action):
        if action == Action.UP:
            return "UP"
        elif action == Action.DOWN:
            return "DOWN"
        elif action == Action.LEFT:
            return "LEFT"
        elif action == Action.RIGHT:
            return "RIGHT"
        else:
            raise ValueError("Invalid action")

    def __init__(self, rewards):
        self.rewards = np.array(rewards)
        self.state = (0, 0)
        self.reward = 0
        self.done = False
        self.exit = (len(rewards) - 1, len(rewards[0]) - 1)
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
        self.reward = self.rewards[self.state]
        self.done = False
        return np.array(self.state_to_sdr(), dtype=int), self.reward

    def set_reward(self, x, y, reward):
        self.rewards[y, x] = reward

    def set_exit(self, x, y):
        self.exit = (y, x)

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
        cmap = colors.ListedColormap(['white', 'black', 'red'])
        ax.imshow(grid, cmap=cmap)

        # Add cell values to the grid
        for i in range(self.rewards.shape[0]):
            for j in range(self.rewards.shape[1]):
                cell_value = critic_values[i, j]
                ax.text(j,
                                i,
                                f"{cell_value:.4f}",
                                ha='center',
                                va='baseline',
                                color='black')

        # Plot the path
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], '-o', markersize=10, color='g')
        ax.plot(path[0, 1], path[0, 0], 'oy', markersize=12)  # Start position
        ax.plot(path[-1, 1], path[-1, 0], 'ob', markersize=12)  # End position

        # Set the axis limits and remove ticks
        ax.set_xticks(range(self.rewards.shape[1]))
        ax.set_yticks(range(self.rewards.shape[0]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, self.rewards.shape[1] - 0.5)
        ax.set_ylim(self.rewards.shape[0] - 0.5, -0.5)

        # # Invert the y-axis to match the grid coordinates
        # ax.invert_axis()

        num_steps = len(path)
        ax.text(0.05,
                        0.95,
                        f"Steps: {num_steps}",
                        transform=ax.transAxes,
                        fontsize=12,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', edgecolor='black', pad=3.0))

        plt.show()

    def reset_path(self):
        self.path = []

    def step(self, action):
        if len(self.path) == 0:
            self.path.append(self.state)

        if self.state == self.exit:
            if np.random.rand() < 0.5:
                self.done = True
            # self.done = True

        elif action == Action.UP:
            self.state = (max(self.state[0] - 1, 0), self.state[1])

        elif action == Action.DOWN:
            self.state = (min(self.state[0] + 1,
                                                self.rewards.shape[0] - 1), self.state[1])

        elif action == Action.LEFT:
            self.state = (self.state[0], max(self.state[1] - 1, 0))

        elif action == Action.RIGHT:
            self.state = (self.state[0],
                                        min(self.state[1] + 1, self.rewards.shape[1] - 1))

        elif action == None:
            print("No action taken")
            pass

        else:
            raise ValueError(f"Invalid action: {action}")

        self.path.append(self.state)

        self.reward = self.rewards[self.state]

        return np.array(self.state_to_sdr(),
                                        dtype=int), self.reward, self.done, self.state

    def close(self):
        pass
