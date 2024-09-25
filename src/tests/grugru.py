import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GRUGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_1, output_size_2):
        super(GRUGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # GRU1 (PL analog)
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size_1)
        
        # GRU2 (ACC analog)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.state_predictor = nn.Linear(hidden_size, output_size_2)  # Predicts next state
        self.loss_predictor = nn.Linear(hidden_size, 1)  # Predicts loss

    def forward(self, x, h1=None, h2=None):
        # GRU1 (PL) forward pass
        out1, h1 = self.gru1(x, h1)
        action = self.output_layer(out1)
        
        # GRU2 (ACC) forward pass
        out2, h2 = self.gru2(h1.transpose(0,1), h2)
        predicted_next_state = self.state_predictor(out2)
        predicted_loss = self.loss_predictor(out2)
        
        return action, h1, h2, predicted_next_state, predicted_loss

# Training loop
def train_model(model, input_data, target_data, next_state_data, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.zero_grad()
        h1, h2 = None, None
        total_loss = 0
        
        for i in range(len(input_data) - 1):  # -1 because we need next state
            x = input_data[i].unsqueeze(0).unsqueeze(0)
            actual_next_state = next_state_data[i+1].unsqueeze(0).unsqueeze(0)
            
            action, h1, h2, predicted_next_state, predicted_loss = model(x, h1, h2)
            
            # Compute GRU2 loss (prediction error)
            gru2_loss = criterion(predicted_next_state, actual_next_state)
            
            # Compute influence on GRU1 hidden state
            h1_influence = torch.sigmoid(predicted_loss * gru2_loss)
            
            # Compute target for GRU1 hidden state
            h1_target = h1 * (1 - h1_influence) + h1_influence * h1.detach().clone()
            
            # Compute loss for GRU1 hidden state
            gru1_h_loss = criterion(h1, h1_target)
            
            # Total loss
            loss = gru1_h_loss + gru2_loss
            
            total_loss += loss.item()
            
            loss.backward(retain_graph=True)
        
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

# Example usage
input_size = 5
hidden_size = 20
output_size = 1

model = GRUGRU(input_size, hidden_size, output_size)

# Generate some dummy data
input_data = torch.randn(100, input_size)
target_data = torch.randn(100, output_size)
next_state_data = torch.randn(100, input_size)  # Represents the actual next states

# Train the model
train_model(model, input_data, target_data, next_state_data, num_epochs=100, learning_rate=0.01)