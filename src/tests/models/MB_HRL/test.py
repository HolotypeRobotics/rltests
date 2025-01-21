import torch
import torch.nn as nn
import torch.optim as optim

class OnlineGRU(nn.Module):
    def __init__(self, input_size, hidden_size, has_output_layer=False, output_size=None, learning_rate=0.01):
        super(OnlineGRU, self).__init__()
        print(f"Creating gru with input_size: {input_size}, hidden_size: {hidden_size}, output_size: {output_size}")
        
        self.hidden_size = hidden_size
        self.has_output_layer = has_output_layer
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Output layer only if specified
        if has_output_layer:
            self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function (only used if has_output_layer is True)
        self.criterion = nn.MSELoss()
        
        # Initialize hidden state
        self.hidden = None
        
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward(self, x, layer_idx):
        print(f"{'   ' * (layer_idx)}Layer {layer_idx} forward")
        
        # Ensure hidden state is initialized correctly
        if self.hidden is None:
            self.hidden = self.init_hidden()
        
        # Reshape input to (sequence_length, batch_size, input_size)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        gru_out, self.hidden = self.gru(x, self.hidden)

        self.hidden = self.hidden.detach()

        if self.has_output_layer:
            return self.output_layer(gru_out[-1, 0, :])  # Last time step
        else:
            return gru_out[-1, 0, :]  # Hidden state from last time step

    def update_weights(self, output, target, layer_idx):
        print(f"{'   ' * (layer_idx)}Layer {layer_idx} update")
        self.optimizer.zero_grad()
        loss = self.criterion(output, target)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

class MetaRL:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rates, steps_per_layer):
        """
        Initialize a MetaRL model with multiple layers of GRU.
        
        Args:
            input_size: Input size for the first GRU.
            hidden_sizes: List of hidden sizes for each GRU layer.
            output_size: Output size for the final GRU.
            learning_rates: List of learning rates for each GRU layer.
        """
        self.n_layers = len(hidden_sizes)
        self.grus = nn.ModuleList()
        self.steps_per_layer = steps_per_layer
        
        # Dynamically create GRU layers
        '''
        information goes top to bottom
        inputs at the top (n_layers -1) and
        outputs at the bottom (0)
        '''
        for i in range(self.n_layers):
            print(f"Creating GRU {i + 1}/{self.n_layers}")

            if i < self.n_layers - 1:
                layer_input_size =  input_size + hidden_sizes[i + 1 ]
            else:
                layer_input_size = input_size

            layer_output_size = hidden_sizes[i]
            has_output_layer = (i == 0)  # Only the last layer has an output layer
            self.grus.append(OnlineGRU(
                input_size=layer_input_size,
                hidden_size=layer_output_size,
                has_output_layer=has_output_layer,
                output_size=output_size if has_output_layer else None,
                learning_rate=learning_rates[i]
            ))
            
    def meta_step(self, x, targets, layer_idx, ext, top=False):
        """
        Perform the meta-learning step for the specified layer and recursively process lower layers.
        """
        loss = 0
        # Ensure x and ext are 2D before concatenation
        if top == False:
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Reshape to (1, hidden_size)
            if ext.dim() == 1:
                ext = ext.unsqueeze(0)  # Reshape ext if needed
            x = torch.cat([x, ext], dim=1)  # Concatenate x and ext

        for _ in range(self.steps_per_layer):

            # Forward pass for the current layer
            output = self.grus[layer_idx](x, layer_idx)

            if layer_idx > 0:
                # Pass concatenated x as ext to the next layer
                loss = self.meta_step(x=output, targets=targets, layer_idx=layer_idx - 1, ext=ext, )

            # Update weights for the current layer
            loss += self.grus[layer_idx].update_weights(output, targets[layer_idx], layer_idx)

        return loss


if __name__ == "__main__":
    # Define parameters
    input_size = 10
    hidden_sizes = [20, 15, 10]  # Three layers with different hidden sizes
    output_size = 5
    learning_rates = [0.01, 0.005, 0.001]  # Different learning rates for each GRU
    steps_per_layer = 3
    
    # Create the MetaRL model
    meta_model = MetaRL(input_size, hidden_sizes, output_size, learning_rates, steps_per_layer)
    
    # Example training loop
    for epoch in range(5):
        # Generate dummy data
        x = torch.randn(1, input_size)
        targets = []
        for i in range(meta_model.n_layers):
            out_size = hidden_sizes[i] if i > 0 else output_size
            targets.append(torch.randn(out_size))
        
        # Perform meta-learning stepclone
        loss = meta_model.meta_step( x=x, ext=x, targets=targets, layer_idx=meta_model.n_layers - 1, top=True)
        
        print(f"\nEpoch {epoch + 1}")
        print(f"Total Loss: {loss}")
