import torch
import torch.nn as nn
import torch.nn.functional as F

class WM_Gate(nn.Module):
    # ... (same as before)

    def forward(self, s, wm_current, eligibility_trace):
        v = torch.matmul(self.X.T, s)  # Value of storing s
        v_current = torch.matmul(self.X.T, wm_current) # Value of maintaining current WM content

        prob_store = (torch.exp(self.beta * v) + self.bias) / (torch.exp(self.beta * v) + self.bias + torch.exp(self.beta * v_current))
        store = torch.bernoulli(prob_store)
        r = store * s  # Store s if store=1, otherwise WM is empty

        return r, prob_store

class HERLayer(nn.Module):
    # ... (same as before)

class HER(nn.Module):
    # ... (same as before)

    def forward(self, s, a_index, o):
        errors = []
        modulations = []
        rs = []
        o_prime = None

        for i in range(self.n_layers):
            r_prev = self.WM[i]  # Get current WM content
            r, prob_store = self.WM_gates[i](s, r_prev, self.eligibility_traces[i])
            rs.append(r)
            self.WM[i] = r  # Update working memory

            self.eligibility_traces[i] = self.lambdas[i] * self.eligibility_traces[i] + (1 - self.lambdas[i]) * s # Update with *s*

            modulation_from_above = None
            if i < self.n_layers - 1:
                p_higher = self.layers[i+1](self.WM[i+1])
                modulation_from_above = p_higher.reshape(self.n_representations, n_outcomes)  # Reshape for additive modulation
                modulations.append(modulation_from_above)

            p = self.layers[i](r, modulation_from_above)

            # ... (Error calculation and weight updates - same logic as before)

        # Response Selection (using W_response - same as your revised code)
        # ...

        return response_probabilities, errors, rs