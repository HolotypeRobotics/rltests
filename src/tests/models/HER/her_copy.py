import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # For optimization

class WM_Gate(nn.Module):
    def __init__(self, n_stimuli, n_representations, beta=15, bias=0.1, alpha=0.075, lambda_=0.95):
        super(WM_Gate, self).__init__()
        self.X = nn.Parameter(torch.zeros(n_stimuli, n_representations))  # (Eq. 2.4, 2.6)
        self.beta = beta
        self.bias = bias
        self.alpha = alpha
        self.lambda_ = lambda_ # Eligibility trace decay rate

    def forward(self, s, r_prev, layer_idx):
        v = torch.matmul(self.X.T, s)  # Value of storing current stimulus s (Eq. 2.4)
        v_prev = torch.matmul(self.X.T, r_prev)  # Value of maintaining previous item (Eq. 2.4)

        # Probabilistic gating (softmax) (Eq. 2.5)
        prob = (torch.exp(self.beta * v) + self.bias) / (torch.exp(self.beta * v) + self.bias + torch.exp(self.beta * v_prev))
        prob_dist = torch.distributions.Bernoulli(probs=prob)  # Create a Bernoulli distribution
        r = prob_dist.sample() * s + (1 - prob_dist.sample()) * r_prev  # Sample and update WM

        return r
    
    # Update weights (Eq. 2.6)
    def update_weights(self, e, W, r, d):
            self.X.data += torch.outer(torch.matmul(e, W.T), r) * d.unsqueeze(1)

class HERLayer(nn.Module):
    def __init__(self, n_representations, n_outcomes, alpha=0.075): # n_stimuli not needed here
        super(HERLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(n_representations, n_outcomes))  # (Eq. 2.1, 2.3) Initialize with random values
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
    def __init__(self, n_layers, n_stimuli, n_outcomes, n_representations, n_responses, betas, biases, alphas_gate, alphas_layer, gamma=10, lambdas=0.95):
        super(HER, self).__init__()
        self.n_layers = n_layers
        self.n_stimuli = n_stimuli
        self.n_representations = n_representations
        self.n_outcomes = n_outcomes
        self.n_responses = n_responses
        self.gamma = gamma
        self.lambdas = lambdas # List of lambdas for each layer

        # Working Memory and Gating Mechanisms
        self.WM_gates = nn.ModuleList([WM_Gate(n_stimuli, n_representations, betas[i], biases[i], alphas_gate[i],  lambdas[i]) for i in range(n_layers)])
        self.register_buffer('WM', torch.randint(0, n_representations, (n_layers,)))  # Initialize WM with random indices
        self.register_buffer('eligibility_traces', torch.zeros(n_layers, n_stimuli))

        # Layer Modules
        self.layers = nn.ModuleList([HERLayer(n_representations, n_outcomes, alphas_layer[i]) for i in range(n_layers)])
        # Response Selection
        self.W_response = nn.Parameter(torch.randn(n_outcomes, n_responses))
        self.response_optimizer = optim.Adam([self.W_response], lr=0.01) # Optimizer for response selection weights



        # Optimizers (one for each WM_Gate)
        self.optimizers = nn.ModuleList([optim.Adam(self.WM_gates[i].parameters(), lr=alphas_gate[i]) for i in range(self.n_layers)])
    
    # Layer 0 is bottom (output) layer
    # Highest layer is input
    def forward(self, s):
        rs = [] # Store r values for each layer (ONE-HOT)
        ps = [] # Store predictions for each layer
        modulation = None # Initialize modulation
        # Process each layer hierarchically
        # iterate in reverse order to get highest layer first
        for i in range(self.n_layers - 1, -1, -1):
            r_prev_index = self.WM[i]  # Get current WM content (index)
            # One-hot encoding for WM (assuming stimuli indices are used)
            r_prev = F.one_hot(r_prev_index.long(), num_classes=self.n_representations).float()  # One-hot encode
            r = self.WM_gates[i](s, r_prev, i) # Pass one-hot r_prev

            # Update WM (store index of the chosen stimulus)
            self.WM[i] = r.argmax() # Store index of active WM unit

            rs.append(r) # Store r for error calculation (on-hot)

            # Update eligibility trace
            # TODO: check the following
            self.eligibility_traces[i] = self.lambdas[i] * self.eligibility_traces[i] + s # Update eligibility trace (Eq. 2.6)
            
            # get output for layer using working memory content
            # (Eq. 2.1)
            # Create WM mask for modulation
            wm_mask = F.one_hot(self.WM[i].long(), num_classes=self.n_representations).float()
            p = self.layers[i](r, wm_mask, modulation) # Pass wm_mask for modulation
            ps.append(p) # Store predictions

            if i > 0: # Set modulation for lower layers
                modulation = p.flatten() # Flatten for modulation of lower layers

        # Response Selection (Eq. 2.12, 2.13)
        response_logits = torch.matmul(ps[-1], self.W_response) # Use predictions from lowest layer
        response_probs = F.softmax(response_logits * self.gamma, dim=-1) # Apply softmax
        return response_probs, rs, ps # Return response probabilities

    def backward(self, s, a_index, o, rs, ps):
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
        o_prime = F.one_hot(o.long(), num_classes=self.n_outcomes).float() # One-hot outcome

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
