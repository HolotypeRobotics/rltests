import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from environment import Environment
from agent import Agent

# Hyperparameters
grid_size = 5
num_actions = 3
num_envs = 1
num_directions = 4
coords_size = grid_size * 2
num_objects = 3
# state, env_input, direction, distances
input_size = coords_size + num_envs + num_directions + num_objects
hidden_size = 32
motor_efferent_size = num_actions
learning_rate = 0.01

agent = Agent(coords_size=coords_size,
            env_enc_size=num_envs,
            direction_size=num_directions,
            distance_size=num_objects,
            hidden_size=hidden_size,
            output_size_1=num_actions) # reward, effort, and predicted motor efferents for action

# Initialize environment, ACC, and PL
env = Environment(rewards=[[0, 0, 0, 1, 10],
                           [0, 0, 1, 1, 0],
                           [0, 1, 1, 0, 0],
                           [0, 1, 0, 1, 0],
                           [0, 0, 0, 0, 10]],
                  efforts=[[2, 1, 2, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 2, 1, 1],
                           [1, 1, 1, 1, 1],
                           [2, 1, 2, 1, 3]],
                  objects=[[0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0],
                           [0, 2, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 3, 0]])
env.set_exit(4, 4)

torch.autograd.set_detect_anomaly(True)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    done = False
    total_reward = 0
    total_effort = 0
    total_loss = 0

    # Reset environment and agent's hidden states
    env.reset()
    agent.reset()

    while not done:

        # Step
        loss, done = agent.step(env=env)
        total_loss += loss
