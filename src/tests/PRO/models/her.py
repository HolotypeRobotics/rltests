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
        r = store * s  # If store = 1, keep s, otherwise WM is empty (one-hot)
        # If you want to maintain the previous item if not storing a new one, uncomment the line below
        # r += (1 - store) * r_prev  # If store = 0, keep r_prev

        return r, prob_store

class HERLayer(nn.Module):
    def __init__(self, n_representations, n_outcomes, alpha=0.075): # n_stimuli not needed here
        super(HERLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(n_representations, n_outcomes))  # (Eq. 2.1, 2.3) Initialize with random values
        self.alpha = alpha # (Eq. 2.3)

    def forward(self, r, m_from_above=None): # Add top-down modulation
        if m_from_above is not None:
            m = torch.matmul(self.W + m_from_above, r)  # Apply top-down modulation before prediction (Eq. 2.9, 2.10)
            p = m #  m is the prediction in this case (Eq. 2.10)
        else:
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
        self.gamma = gamma

        # Working Memory and Gating Mechanisms
        self.WM_gates = nn.ModuleList([WM_Gate(n_stimuli, n_representations, betas[i], biases[i], alphas_gate[i]) for i in range(n_layers)])
        self.register_buffer('WM', torch.zeros(n_layers, n_representations))
        self.register_buffer('eligibility_traces', torch.zeros(n_layers, n_stimuli))

        # Layer Modules
        self.layers = nn.ModuleList([HERLayer(n_representations, n_outcomes, alphas_layer[i]) for i in range(n_layers)])

        # Response Selection (Output Layer - only at the lowest layer)
        self.W_response = nn.Parameter(torch.randn(n_representations, n_responses)) # n_outcomes x n_responses
        self.lambdas = lambdas if lambdas is not None else [0.1, 0.5, 0.99]


    def forward(self, s, a_index, o):
        errors = []
        modulations = []
        rs = []
        o_prime = None

        # Process each layer hierarchically
        for i in range(self.n_layers):
            r_prev = self.WM[i]  # Get current WM content
            r, prob_store = self.WM_gates[i](s, r_prev, self.eligibility_traces[i])
            rs.append(r)
            self.WM[i] = r  # Update working memory
            # Update eligibility trace (Eq. after 2.6)
            self.eligibility_traces[i] = self.lambdas[i] * self.eligibility_traces[i] + (1 - self.lambdas[i]) * s # Update with *s*
            # Top-down modulation (Eq. 2.9, 2.10)
            modulation_from_above = None
            if i < self.n_layers - 1:
                p_higher = self.layers[i+1](self.WM[i+1])
                modulation_from_above = p_higher.reshape(self.n_representations, self.n_outcomes)  # Reshape for additive modulation
                modulations.append(modulation_from_above)

            p = self.layers[i](r, modulation_from_above)


            if i == 0: # Lowest layer uses actual outcome (Eq. 2.2)
                # Filter error based on selected action. Create a one-hot vector for the action:
                a = torch.zeros(self.n_outcomes)  # Assuming n_outcomes = number of possible actions x number of feedback types
                a[a_index] = 1.0
                e = a * (o - p)  # (Eq. 2.2)
                o_prime = torch.outer(r, e)  # (Eq. 2.7)
            else: # Higher layers use error from below as outcome (Eq. 2.8, 2.11)
                o_higher = torch.flatten(o_prime) # Flatten o_prime from previous layer
                e = (o_higher - p) # Error at higher layers # (Eq. 2.8, 2.11)
                if i < self.n_layers - 1: # Only calculate o_prime if not the top layer
                    o_prime = torch.outer(r, e)

            errors.append(e)
            self.layers[i].update_weights(e, r) # Update layer weights (Eq. 2.3)
            # Update gating weights AFTER error calculation
            self.WM_gates[i].X.data += self.WM_gates[i].alpha * torch.outer(e, self.eligibility_traces[i]) # Update gating weights (eq. 2.6)

        # # Response Selection (Eq. 2.12, 2.13)
        # #  Important: The paper uses separate predictions for Correct and Error outcomes to drive response selection.
        # response_correct = self.layers[0](r, modulation_from_above)[:, 0] # Predictions for correct outcome
        # response_error = self.layers[0](r, modulation_from_above)[:, 1] # Predictions for error outcome
        # u = response_correct - response_error # Difference in predictions

        # response_probabilities = F.softmax(self.gamma * u, dim=0) # Softmax for response selection

        # Response Selection (using separate predictions for correct/error)
        p_lowest = self.layers[0](rs[0], modulations[0] if len(modulations) > 0 else None)
        response_logits = torch.matmul(p_lowest, self.W_response)  # Get logits for each response
        response_probabilities = F.softmax(self.gamma * response_logits, dim=0)  # Softmax over responses (Eq. 2.13)

        return response_probabilities, errors, rs