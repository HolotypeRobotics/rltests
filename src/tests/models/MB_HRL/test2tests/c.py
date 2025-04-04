import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MultiSeqEnv:
    def __init__(self, reward_seq, effort_seq, max_steps=6):
        # reward_seq and effort_seq: a single sequence each.
        assert len(reward_seq) == len(effort_seq), "Mismatch in sequence lengths"
        self.reward_seq = reward_seq
        self.effort_seq = effort_seq
        self.seq_len = len(reward_seq)
        self.max_steps = max_steps
        self.num_steps = 0
    
    def reset(self, episode=None):
        # Start in the middle of the sequence.
        self.pos = self.seq_len // 2
        self.recent_pos = self.pos  # store the starting position
        self.done = False
        self.num_steps = 0
        # Previous action: one-hot vector for 3 actions.
        self.prev_action = torch.zeros(3, dtype=torch.float32)
        # Head direction: one-hot for left/right. Default to "right" ([0,1]).
        self.head_direction = torch.tensor([0, 1], dtype=torch.float32)
        return self._get_obs()
    
    def _get_obs(self):
        # Position: one-hot vector (length = seq_len).
        pos_onehot = torch.zeros(self.seq_len, dtype=torch.float32)
        pos_onehot[self.pos] = 1.0
        # External state: we return the position one-hot and head direction.
        return torch.cat([pos_onehot, self.head_direction], dim=-1).unsqueeze(0), self.prev_action.unsqueeze(0)
    
    def step(self, action):
        """
        Action space:
         0: move forward  
         1: move reverse  
         2: terminate  
        """
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            self.done = True
        # Compute reward as difference relative to the recent position.
        reward = self.reward_seq[self.pos] - self.reward_seq[self.recent_pos]
        effort = self.effort_seq[self.pos] - self.effort_seq[self.recent_pos]
        reward -= effort
        if action == 2:  # terminate: no movement.
            pass
        else:
            if action == 0 and self.pos < self.seq_len - 1:
                self.pos += 1
            elif action == 1 and self.pos > 0:
                self.pos -= 1
        
        # Update previous action.
        prev_action_onehot = torch.zeros(3, dtype=torch.float32)
        prev_action_onehot[action] = 1.0
        self.prev_action = prev_action_onehot
        # Update head direction based on action.
        if action == 1:  # reverse
            self.head_direction = torch.tensor([1, 0], dtype=torch.float32)
        elif action == 0:  # forward
            self.head_direction = torch.tensor([0, 1], dtype=torch.float32)
        
        return self._get_obs(),
