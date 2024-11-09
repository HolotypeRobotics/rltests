import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUGRU(nn.Module):
    def __init__(self, coords_size, env_enc_size, direction_size, distance_size, hidden_size, output_size_1):
        super(GRUGRU, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.hidden_size = hidden_size

        # The total input size is the sum of the state, environment encoding, direction, and distance inputs
        input_size = coords_size + env_enc_size + direction_size + distance_size

        # GRU1 (PL analog)
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=False)
        self.reward_predictor_1 = nn.Linear(hidden_size, 1)  # OFC Analog
        self.effort_predictor_1 = nn.Linear(hidden_size, 1)  # BLA Analog
        self.output_layer = nn.Linear(hidden_size, output_size_1)
        self.output_layer.weight.data.fill_(1)
        # Disable gradients for the output layer
        for param in self.output_layer.parameters():
            param.requires_grad = False
        
        # GRU2 (ACC analog) predicts next state, reward, effort, and loss
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.coord_predictor = nn.Linear(hidden_size, coords_size)  # Predicts next state
        self.reward_predictor_2 = nn.Linear(hidden_size, 1)  # Predicts reward
        self.effort_predictor_2 = nn.Linear(hidden_size, 1)
        self.loss_predictor = nn.Linear(hidden_size, 1)  # Predicts loss

    def forward(self, x, h1=None, h2=None):        
        # TODO: rethink efforrts values advantage, and policy
        # GRU1 (PL) forward pass
        out1, h1 = self.gru1(x, h1)
        reward_current_state = self.reward_predictor_1(out1)
        effort_current_state = self.effort_predictor_1(out1)
        action_probs = self.output_layer(out1)
        action_probs = F.softmax(action_probs, dim=-1)
        
        # GRU2 (ACC) forward pass
        out2, h2    = self.gru2(h1, h2)
        coords_pred = self.coord_predictor(out2)
        loss_pred   = self.loss_predictor(out2)
        reward_pred = self.reward_predictor_2(out2)
        effort_pred = self.effort_predictor_2(out2)

        coords_pred = torch.relu(coords_pred)
        loss_pred   = torch.relu(loss_pred)
        reward_pred = torch.relu(reward_pred)
        effort_pred = torch.relu(effort_pred)

        
        return action_probs, h1, h2, coords_pred, reward_pred, effort_pred, loss_pred
