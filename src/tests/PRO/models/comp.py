import torch
import torch.nn as nn
import torch.nn.functional as F

class OutcomeRepresentation(nn.Module):
    def __init__(self, n_stimuli, n_ro_conjunctions, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(n_stimuli, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_ro_conjunctions)

    def forward(self, stimuli):
        x = F.relu(self.fc1(stimuli))
        badness = F.relu(self.fc2(x))  # Output is "badness" signal (non-negative)
        return badness

class PROControl(nn.Module):
    def __init__(self,
                 n_stimuli,
                 n_responses,
                 n_outcomes,
                 n_delay_units,
                 dt=0.1,
                 theta=1.5,  # Aversion sensitivity
                 alpha_ro=0.1,  # Base learning rate for R-O conjunctions
                 alpha_td=0.1,  # TD learning rate
                 beta=0.1,  # Response gain
                 gamma=0.95,  # Temporal discount
                 lambda_decay=0.95,  # Eligibility trace decay
                 psi=0.1,  # Inhibition scaling
                 phi=0.1,  # Control scaling
                 rho=0.1,  # Excitation scaling
                 response_threshold=0.5,  # Response threshold
                 sigma=0.1, # Noise standard deviation
                 device='cpu'): 
        super().__init__()

        self.device = device
        # ... (rest of the initialization code is the same)
        self.sigma = sigma # added noise parameter

    def compute_ro_prediction(self, stimuli):
        """Predict response-outcome conjunctions"""
        badness = self.outcome_rep(stimuli)
        ro_conj = torch.matmul(self.W_S, stimuli)
        return ro_conj, badness  # Return both RO conjunction and badness
    

    def compute_response_activation(self, stimuli, ro_predictions):
        # ... (Direct pathway and Mutual inhibition are the same)

        # Control pathways
        proactive, reactive = self.compute_control_signals(ro_predictions, self.C)
        control = self.phi * (proactive + reactive)

        # Compute activation change (ADDED NOISE)
        noise = torch.normal(0, self.sigma, self.C.shape, device=self.device)
        delta_C = self.beta * self.dt * (
            excitation * (1 - self.C) -
            (self.C + 0.05) * (inhibition + control) + noise
        )

        # Update activation (REMOVED SIGMOID)
        self.C = self.C + delta_C  # No sigmoid
        return self.C


    def compute_control_signals(self, ro_predictions, response):
        """Compute proactive and reactive control signals (CORRECTED)"""

        # Proactive control (with ReLU, separate excitatory and inhibitory pathways)
        proactive_excitation = F.relu(torch.matmul(ro_predictions, -self.W_F)) # Negative weights for excitation
        proactive_inhibition = F.relu(torch.matmul(ro_predictions, self.W_F)) # Positive weights for inhibition
        proactive = proactive_excitation - proactive_inhibition # Push-pull

        # Reactive control (using BOTH positive and negative surprise)
        omega_p, omega_n = self.compute_surprise(ro_predictions, torch.zeros_like(ro_predictions))
        reactive = (torch.matmul(omega_p, self.W_R) -  # Positive surprise, excitatory
                    torch.matmul(omega_n, self.W_R))     # Negative surprise, inhibitory

        return proactive, reactive

    def forward(self, stimuli, gating_signal=None): # Added gating signal as input
        # ... (Temporal component updates and predictions are the same)

        # Predict R-O conjunctions and badness
        ro_predictions, badness = self.compute_ro_prediction(stimuli)

        # Generate response
        response = self.compute_response_activation(stimuli, ro_predictions)
        response_discrete = (response > self.response_threshold).float()

        # Return all relevant values
        return response, ro_predictions, badness, temporal_prediction, response_discrete

    def update_weights(self, stimuli, response, actual_ro, reward, outcome_valence, gating_signal):
        """Update all weights based on experience (CORRECTED)"""
        # ... (Get predictions and compute surprise - the same)

        # Update R-O conjunction weights (using badness to modulate learning rate)
        effective_lr = self.alpha_ro * (1 / (1 + badness))  # Badness modulates LR
        ro_error = self.theta * actual_ro - ro_predictions[0] # index ro_predictions
        self.W_S.data += gating_signal * effective_lr * torch.outer(ro_error, stimuli)

        # ... (Update temporal prediction weights - the same)

        # Update proactive control weights (with separate excitatory and inhibitory updates)
        response_mask = (response > 0.1).float()
        # Excitation update
        delta_w_f_excite = 0.01 * torch.outer(response_mask, -actual_ro * outcome_valence)
        self.W_F.data -= gating_signal * delta_w_f_excite # Negative update for excitatory connections
        # Inhibition update
        delta_w_f_inhibit = 0.01 * torch.outer(response_mask, actual_ro * outcome_valence)
        self.W_F.data += gating_signal * delta_w_f_inhibit # Positive update for inhibitory connections
        # ... (Update reactive control weights - now uses gating_signal correctly)
        # ... (Update Outcome Representation weights - the same)