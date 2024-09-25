import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from grugru import GRUGRU

class Agent:
    def __init__(self, coords_size, env_enc_size, direction_size, distance_size, hidden_size, output_size_1, learning_rate=0.01):
        self.coords_size = coords_size
        self.env_enc_size = env_enc_size
        self.direction_size = direction_size
        self.distance_size = distance_size
        self.model = GRUGRU(coords_size, env_enc_size, direction_size, distance_size, hidden_size, output_size_1)
        self.criterion = nn.MSELoss()
        self.hidden_size = hidden_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.h1 = torch.zeros(1, 1, self.hidden_size)
        self.h2 = torch.zeros(1, 1, self.hidden_size)

    def reset(self):
        self.h1 = torch.zeros(1, self.hidden_size)
        self.h2 = torch.zeros(1, self.hidden_size)
    
    def step(self, env):

            self.optimizer.zero_grad()  # Reset gradients

            # Detach hidden states from previous computations
            if self.h1 is not None:
                self.h1 = self.h1.detach()
            if self.h2 is not None:
                self.h2 = self.h2.detach()

            # Format the inputs
            direction = F.one_hot(torch.tensor([env.direction]), num_classes=self.direction_size).float().squeeze(0)
            env_enc = F.one_hot(torch.tensor([0]), num_classes=self.env_enc_size).float().squeeze(0)
            distances = torch.from_numpy(env.distances/self.coords_size).float()
            coords = torch.tensor(env.state_to_sdr()).float()

            # Forward pass with updated inputs
            x = torch.cat((coords, env_enc, direction, distances), dim=-1).unsqueeze(0)
            action_probs, self.h1, self.h2, coords_pred, reward_pred, effort_pred, loss_pred = self.model(x, self.h1, self.h2)

            # Take Action in Environment:
            # action = action[0].argmax().item() # Convert tensor action to integer
            action = torch.distributions.Categorical(action_probs[0]).sample().item()

            coords, reward, done, _, effort, direction, distances = env.step(action) # Take action in environment
            reward = torch.tensor([reward]).float()
            effort = torch.tensor([effort]).float()

            # Update actual resulting state
            coords = torch.tensor(coords).float()

            # Compute GRU2 loss (prediction error)
            gru2_loss = self.criterion(coords_pred, coords)
            gru2_loss += self.criterion(reward_pred, reward)
            gru2_loss += self.criterion(effort_pred, effort)

            # print(f"Acutal state: {actual_resulting_state}")
            # Compute influence on GRU1 hidden state
            h1_loss = torch.sigmoid(loss_pred * gru2_loss)
            h1_gain = (reward_pred) / (effort_pred + 1e-6)

            # Compute target for GRU1 hidden state
            # loss decreases the weights for the current hidden state activation
            h1_target = self.h1.clone() - (self.h1.clone() * h1_loss) + (self.h1.clone() * h1_gain)
            
            # Compute loss for GRU1 hidden state
            gru1_h_loss = self.criterion(self.h1, h1_target)
            
            # Total loss (combined GRU1 and GRU2 losses)
            loss = gru1_h_loss + gru2_loss
            
            # Backpropagation
            loss.backward(retain_graph=True)
            
            # Update model parameters
            self.optimizer.step()

            input()

            # print(f"Gru1 loss: {gru1_h_loss}, Gru2 loss: {gru2_loss}")
            return loss.item(), done
