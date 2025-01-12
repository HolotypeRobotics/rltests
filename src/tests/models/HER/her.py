import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # For optimization

class WM_Gate(nn.Module):
    def __init__(self, n_stimuli, beta, bias, alpha, lambda_):
        super(WM_Gate, self).__init__()
        self.X = nn.Parameter(torch.zeros(n_stimuli))  # (Eq. 2.4, 2.6)
        self.beta = beta
        self.bias = bias
        self.alpha = alpha
        self.lambda_ = lambda_ # Eligibility trace decay rate

    def forward(self, s, r_prev, layer_idx):
        v = torch.matmul(self.X.T, s)  # Value of storing current stimulus s (Eq. 2.4)
        v_prev = torch.matmul(self.X.T, r_prev)  # Value of maintaining previous item (Eq. 2.4)
        # Missing decay for eligibiltiy trace?

        # Probabilistic gating (softmax) (Eq. 2.5)
        prob = (torch.exp(self.beta * v) + self.bias) / (torch.exp(self.beta * v) + self.bias + torch.exp(self.beta * v_prev))
        prob_dist = torch.distributions.Bernoulli(probs=prob)  # Create a Bernoulli distribution
        r = prob_dist.sample() * s + (1 - prob_dist.sample()) * r_prev  # Sample and update WM

        return r
    
    # Update weights (Eq. 2.6)
    def update_weights(self, e, W, r, d):
            self.X.data += torch.outer(torch.matmul(e, W.T), r) * d.unsqueeze(1)

class HERLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, alpha=0.075): # n_stimuli not needed here
        super(HERLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(n_inputs, n_outputs))  # (Eq. 2.1, 2.3) Initialize with random values
        self.alpha = alpha # (Eq. 2.3)

    def forward(self, r, wm_mask, modulation=None):  # Added modulation
        p = torch.matmul(self.W.T, r)  # Predictions (Eq. 2.1)

        if modulation is not None:
            # Implement additive modulation (Eq. 2.9, 2.10)
            modulation = modulation.view(self.W.shape) # Reshape modulation to match W
            masked_modulation = modulation * wm_mask.unsqueeze(1) # Unsqueeze for broadcasting
            p = p + torch.matmul(masked_modulation.T, r) # Additive modulation
        
        return p

    def update_weights(self, e, r, ):
        self.W.data += self.alpha * torch.outer(e, r) # Update weights (Eq. 2.3)


class HER(nn.Module):
    def __init__(self, n_layers, n_hidden, n_stimuli, n_outcomes, n_responses, beta, bias, gate_alpha, layer_alpha, gamma, lambda_):
        super(HER, self).__init__()
        self.n_layers = n_layers
        self.n_stimuli = n_stimuli # Now takes state dimension directly
        self.gamma = gamma # Softmax temperature
        self.lambda_ = lambda_ # Eligibility trace decay rate
        self.layers = nn.ModuleList()

        # Working Memory and Gating Mechanisms
        self.WM_gates = nn.ModuleList([WM_Gate(n_stimuli, beta, bias, gate_alpha, lambda_) for i in range(n_layers)])
        self.register_buffer('WM', torch.randint(0, n_stimuli, (n_layers,))) # Corrected shape 
        self.register_buffer('eligibility_traces', torch.zeros(n_layers, n_stimuli))

        # Layer Modules
        self.layers.append(HERLayer(n_responses, n_outcomes)) # First layer is output layer
        
        for i in range(1, n_layers):
            self.layers.append(HERLayer(n_responses, n_hidden))

        # Response Selection
        self.W_response = nn.Parameter(torch.randn(n_responses))
        self.response_optimizer = optim.Adam([self.W_response], lr=layer_alpha) # Optimizer for response selection weights

        # Optimizers (one for each WM_Gate)
        self.optimizers = nn.ModuleList([optim.Adam(self.WM_gates[i].parameters(), lr=gate_alpha) for i in range(self.n_layers)])
    
    # Layer 0 is bottom (output) layer
    # Highest layer is input
    def forward(self, s):
        rs = []
        ps = []
        modulation = None

        for i in range(self.n_layers - 1, -1, -1):
            r_prev_index = self.WM[i]  # Get WM index
            r_prev = F.one_hot(r_prev_index.long().unsqueeze(0), num_classes=self.n_stimuli).float() # One-hot encode

            r_index = self.WM_gates[i](s, r_prev.squeeze(0), i)  # Get next WM index
            self.WM[i] = r_index # Update WM with index

            r = F.one_hot(r_index.long().unsqueeze(0), num_classes=self.n_stimuli).float()  # One-hot encode for layer input
            rs.append(r.squeeze(0))

            wm_mask = r.clone() # WM mask is now the one-hot r

            if modulation is not None:
                p = self.layers[i](r.squeeze(0), wm_mask.squeeze(0), modulation)
            else:
                p = self.layers[i](r.squeeze(0), wm_mask.squeeze(0))

            ps.append(p)

            if i > 0:
                modulation = p  # No need to flatten, keep as vector

        return ps[0]  # Return prediction (logits)

    def backward(self, s, a_index, outcome, rs, ps):
        """
        Backpropagation through the HER model.

        Args:
            s: Input stimulus vector.
            a_index: Index of the selected action.
            o: Outcome vector.
            rs: List of working memory representations (r) for each layer.
            ps: List of predictions (p) for each layer.
        """

        es = [] # Store error signals for each layer
        # Initialize outcome for lowest layer
        # One-hot encode the outcome
        o_prime = torch.tensor(outcome, dtype=torch.float32) # Outcome is now a tensor

        for i in range(self.n_layers):
            # Calculate modulated prediction (m)
            modulation = None
            if i < self.n_layers - 1:
                modulation = ps[i+1].flatten()
            m = self.layers[i](rs[i], modulation)

            wm_mask = F.one_hot(self.WM[i].long(), num_classes=self.n_representations).float()
            m = self.layers[i](rs[i], wm_mask, modulation) # Pass wm_mask for modulation


            # Filter errors (Eq. 2.2)
            filter_ = torch.zeros_like(o_prime)
            filter_[a_index] = 1
            e = filter_ * (o_prime - m) # Error calculation (Eq. 2.11)
            es.append(e) # Store error

            # Update layer weights
            self.layers[i].update_weights(e, rs[i])

            # Update WM gating weights (Eq. 2.6)
            error_backpropagated = torch.matmul(e, self.layers[i].W.T) # Backpropagate error
            self.WM_gates[i].update_weights(error_backpropagated, self.eligibility_traces[i])


            if i < self.n_layers - 1: # Calculate outcome for next higher layer (Eq. 2.7)
                o_prime = torch.outer(rs[i], e) # Outer product
                o_prime = o_prime.flatten() # Flatten for next layer

        # Recalculate response_probs within backward (or pass it from forward)
        response_logits = torch.matmul(ps[-1], self.W_response) # Use ps[-1] from the arguments
        response_probs = F.softmax(response_logits * self.gamma, dim=-1)

        # Update response selection weights (using error from lowest layer)
        self.response_optimizer.zero_grad()
        response_loss = -torch.log(response_probs[a_index])  # Negative log-likelihood loss
        response_loss.backward() # retain_graph to avoid issues with multiple backward passes
        self.response_optimizer.step()
