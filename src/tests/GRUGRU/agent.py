import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tests.grugru.grugru import GRUGRU

class Agent:
    def __init__(self, coords_size, env_enc_size, direction_size, distance_size, hidden_size, output_size_1, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.9999):
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
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay


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
            action_probs, self.h1, self.h2, coords_pred, value_pred, effort_pred, loss_pred = self.model(x, self.h1, self.h2)

            # print(f"\nInput Tensor (x):\n{x}")
            # print(f"\nPredicted Coordinates: {coords_pred}")
            # print(f"\nActual Coordinates:    {coords}")
            # print(f"Predicted Reward: {reward_pred}")
            # print(f"Actual Reward:    {env.reward}")
            # print(f"Predicted Effort: {effort_pred}")
            # print(f"Actual Effort:     {env.effort}")
            # print(f"Predicted Loss: {loss_pred}")
            # print(f"\nAction Probabilities: {action_probs}")
            # input()

            # Take Action in Environment:
            if torch.rand(1) < self.epsilon:
                action = torch.distributions.Categorical(action_probs[0]).sample().item()
                # action = torch.randint(0, action_probs.size(1), (1,)).item()
            else:
                action = action_probs[0].argmax().item() # Convert tensor action to integer
            self.epsilon *= self.epsilon_decay

            coords, reward, done, _, effort, direction, distances = env.step(action) # Take action in environment
            
            # TODO: predicted effort should control ability to hold working memory, but we want to minimize effort in the objective function
            # We want to maximize the action that leads to the highest reward but least effort.
            # WM feeds reward prediction to RL

            # Format the outputs
            coords = torch.tensor(coords).float().unsqueeze(0)
            reward = torch.tensor([reward]).float().unsqueeze(0)
            effort = torch.tensor([effort]).float().unsqueeze(0)


            # Compute GRU2 loss (prediction error)
            gru2_loss = self.criterion(coords_pred, coords)
            print(f"GRU2 Loss coords: {gru2_loss}")
            gru2_loss += self.criterion(value_pred, reward)
            print(f"GRU2 Loss + reward: {gru2_loss}")
            gru2_loss += self.criterion(effort_pred, effort)
            print(f"GRU2 Loss + effort: {gru2_loss}")
            gru2_loss = (gru2_loss/3)
            loss_pred = loss_pred.squeeze(0).squeeze(0)
            loss_loss = self.criterion(loss_pred, gru2_loss/3)
            print(f"Loss Loss: {loss_loss}")
            input()

            # Compute influence on GRU1 hidden state
            h1_loss = torch.sigmoid(gru2_loss) * loss_pred
            advantage = (value_pred) / (effort_pred + 1e-6)

            # Compute target for GRU1 hidden state
            # loss decreases the weights for the current hidden state activation
            h1_target = self.h1.clone() - (self.h1.clone() * h1_loss) + (self.h1.clone() * advantage)
            
            # Compute loss for GRU1 hidden state
            gru1_h_loss = self.criterion(self.h1, h1_target)
            
            # Total loss (combined GRU1 and GRU2 losses)
            loss = gru1_h_loss + gru2_loss + loss_loss
            
            # Backpropagation
            loss.backward(retain_graph=True)
            
            # Update model parameters
            self.optimizer.step()

            # input()
            return loss.item(), reward, effort, done
