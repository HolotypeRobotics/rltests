import torch
import torch.nn as nn
import torch.nn.functional as F


class HMBRL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HMBRL, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=False)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.local_valence = nn.Linear(hidden_size, 1)
        self.gru1_hidden = None
        self.gru2_hidden = None

    def forward(self, x):
        x, self.gru1_hidden = self.gru1(x, self.gru1_hidden)
        x, self.gru2_hidden = self.gru2(x, self.gru2_hidden)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return
    
