import torch
import torch.nn as nn
import torch.nn.functional as F

class WM_Gate(nn.Module):
    def __init__(self, n_stimuli, n_representations, beta=15, bias=0.1, alpha=0.075):
        super(WM_Gate, self).__init__()
        self.X = nn.Parameter(torch.zeros(n_stimuli, n_representations))  # (Eq. 2.4, 2.6)
        self.beta = nn.Parameter(torch.tensor(beta)) # (Eq. 2.5)
        self.bias = nn.Parameter(torch.tensor(bias)) # (Eq. 2.5)
        self.alpha = alpha  # Learning rate for X (Eq. 2.6)

    def forward(self, s, r_prev, eligibility_trace):
        v = torch.matmul(self.X.T, s)  # Value of storing current stimulus s (Eq. 2.4)
        v_prev = torch.matmul(self.X.T, r_prev)  # Value of maintaining previous item (Eq. 2.4)

        # Probabilistic gating (softmax) (Eq. 2.5)
        prob_store = (torch.exp(self.beta * v) + self.bias) / (torch.exp(self.beta * v) + self.bias + torch.exp(self.beta * v_prev))
        
        # Sample from the probability distribution (using Bernoulli)
        store = torch.bernoulli(prob_store)

        # Enforce one-item capacity:
        # r = store * s  # If store = 1, keep s, otherwise WM is empty (one-hot)
        # If you want to maintain the previous item if not storing a new one, uncomment the line below
        r = store * s + (1 - store) * r_prev  # If store = 0, keep r_prev

        return r, prob_store

class HERLayer(nn.Module):
    def __init__(self, n_representations, n_outcomes, alpha=0.075): # n_stimuli not needed here
        super(HERLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(n_representations, n_outcomes))  # (Eq. 2.1, 2.3) Initialize with random values
        self.alpha = alpha # (Eq. 2.3)

    def forward(self, r): # Add top-down modulation
        p = torch.matmul(self.W.T, r)  # Prediction without modulation (Eq. 2.1)

        return p

    def update_weights(self, e, r):
        self.W.data += self.alpha * torch.outer(e, r)  # Update weights (Eq. 2.3)


class HER(nn.Module):
    def __init__(self, n_layers, n_stimuli, n_outcomes, n_representations, n_responses, betas, biases, alphas_gate, alphas_layer, gamma=10, lambdas=None):
        super(HER, self).__init__()
        self.n_layers = n_layers
        self.n_representations = n_representations
        self.n_outcomes = n_outcomes
        self.n_responses = n_responses
        self.gamma = gamma

        # Working Memory and Gating Mechanisms
        self.WM_gates = nn.ModuleList([WM_Gate(n_stimuli, n_representations, betas[i], biases[i], alphas_gate[i]) for i in range(n_layers)])
        self.register_buffer('WM', torch.zeros(n_layers, n_representations))
        self.register_buffer('eligibility_traces', torch.zeros(n_layers, n_stimuli))

        # Layer Modules
        self.layers = nn.ModuleList([HERLayer(n_representations, n_outcomes, alphas_layer[i]) for i in range(n_layers)])
        self.W_response = nn.Parameter(torch.randn(n_outcomes, n_responses)) # Output weights for response selection
        self.lambdas = lambdas if lambdas is not None else [0.1, 0.5, 0.99]


    def forward(self, s, a_index, o):
        errors = [] 
        modulations = [] # upper layer output used to modulate current layer
        rs = [] # values currently held in working memory
        o_prime = None # target values for the upper layer
        n_actions = o.shape[0] # Get number of possible actions from outcome shape


        # Process each layer hierarchically
        for i in range(self.n_layers):
            r_prev = self.WM[i]  # Get current WM content
            r, prob_store = self.WM_gates[i](s, r_prev, self.eligibility_traces[i])
            rs.append(r)
            self.WM[i] = r  # Update working memory
            # Update eligibility trace (Eq. 2.6) Xt+1 = Xt + (etT Wt · rt )dtT. The dot means element-wise multiplication
            self.eligibility_traces[i] = self.lambdas[i] * self.eligibility_traces[i] + s  # Decay and add current stimulus
            # Top-down modulation (Eq. 2.9, 2.10)
            modulation_from_above = None
            if i < self.n_layers - 1:
                p_higher = self.layers[i+1](self.WM[i+1])
                modulation_from_above = p_higher.reshape(self.n_representations, self.n_outcomes)  # Reshape for additive modulation
                modulations.append(modulation_from_above)

            p = self.layers[i](r)  # Predictions *before* modulation
            # Apply top-down modulation to predictions
            if modulation_from_above is not None:
                p = p + torch.flatten(modulation_from_above) # Apply modulation to predictions

            if i == 0: # Lowest layer uses actual outcome (Eq. 2.2)
                # Filter error based on selected action. Create a one-hot vector for the action:
                a = torch.zeros(n_actions)
                a[a_index] = 1.0
                e = a * (o - p)  # (Eq. 2.2)
                o_prime = torch.outer(r, e.T)  # (Eq. 2.7) O' = reT

            else: # Higher layers use error from below as outcome (Eq. 2.8, 2.11)
                o_higher = torch.flatten(o_prime) # Flatten o_prime from previous layer
                a_higher = torch.zeros_like(p) # Placeholder for action vector at higher levels (needs to be defined based on task)
                e = a_higher * (o_higher - p) # Error at higher layers # (Eq. 2.8, 2.11) e' = a'(o' − p' ).
                if i < self.n_layers - 1: # Only calculate o_prime if not the top layer
                    o_prime = torch.outer(r, e.T)

            errors.append(e)
            self.layers[i].update_weights(e, r) # Update layer weights (Eq. 2.3)
            # Update gating weights AFTER error calculation
            self.WM_gates[i].X.data += self.WM_gates[i].alpha * torch.outer(torch.matmul(e, self.layers[i].W.T), self.eligibility_traces[i]) # (Eq. 2.6)

        # Response Selection (using separate predictions for correct/error)
        p_lowest = self.layers[0](rs[0])
        if len(modulations) > 0:
            p_lowest = p_lowest + torch.flatten(modulations[0])

        response_logits = torch.matmul(p_lowest, self.W_response)  # Get logits for each response
        response_probabilities = F.softmax(self.gamma * response_logits, dim=0)  # Softmax over responses (Eq. 2.13)

        return response_probabilities, errors, rs