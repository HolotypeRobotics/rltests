import torch
import torch.nn as nn


class GruChain(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(GruChain, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # GRU layers
        self.grus = nn.ModuleList([
            nn.GRU(input_size if (i == n_layers - 1 ) else hidden_size, 
                  hidden_size, 
                  batch_first=False)
            for i in range(n_layers)
        ])

    def forward(self, x):
        pass

    def forward_layer(self, x, hidden, layer_idx, context=None):
        pass

    def process_event(self, event):
        
