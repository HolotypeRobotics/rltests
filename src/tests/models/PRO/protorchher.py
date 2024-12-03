
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

        
        # Model dimensions
        self.n_stimuli = n_stimuli
        self.n_responses = n_responses
        self.n_outcomes = n_outcomes
        self.n_ro_conjunctions = n_responses * n_outcomes
        self.n_delay_units = n_delay_units

        # Parameters
        self.dt = dt
        self.theta = theta
        self.alpha_ro = alpha_ro
        self.alpha_td = alpha_td
        self.beta = beta
        self.gamma = gamma
        self.lambda_decay = lambda_decay
        self.response_threshold = response_threshold
        self.psi = psi
        self.phi = phi
        self.rho = rho
        self.sigma = sigma

        # Initialize learnable weights
        # R-O conjunction prediction weights
        self.W_S = nn.Parameter(torch.abs(
            torch.normal(0.1, 0.05, (self.n_ro_conjunctions, n_stimuli))))
        
        # Fixed stimulus-response weights
        self.register_buffer('W_C', torch.ones((n_responses, n_stimuli)))
        
        # Proactive control weights (equation 12 in paper)
        self.W_F = nn.Parameter(torch.normal(0, 0.1, (self.n_ro_conjunctions, n_responses)))
        
        # Reactive control weights
        self.W_R = nn.Parameter(torch.zeros((n_responses, self.n_ro_conjunctions)))
        
        # Mutual inhibition weights
        self.register_buffer('W_I', torch.zeros((n_responses, n_responses)))
        self.W_I.fill_diagonal_(-1)
        
        # Temporal prediction weights
        self.U = nn.Parameter(torch.zeros(
            (self.n_ro_conjunctions, n_delay_units, n_stimuli)))
        # State buffers
        self.register_buffer('delay_chain', torch.zeros((n_delay_units, n_stimuli)))
        self.register_buffer('eligibility_trace', torch.zeros((n_delay_units, n_stimuli)))
        self.register_buffer('C', torch.zeros(n_responses))
        
        # Normalize W_F weights
        with torch.no_grad():
            norm_factor = torch.sum(torch.abs(self.W_F)) / (n_responses * n_outcomes)
            if norm_factor > 1:
                self.W_F.data /= norm_factor

    def compute_losses(self, response, pred_outcomes, pred_temporal, actual_outcomes, reward, target_valence):  # Include reward and target_valence
        """Compute losses for R-O and temporal predictions"""

        # R-O Loss (Mean Squared Error is a reasonable choice)
        ro_loss = nn.MSELoss()(pred_outcomes, actual_outcomes)

        # Temporal Difference Loss (SmoothL1Loss is often good for TD learning)
        td_loss = nn.SmoothL1Loss()(pred_temporal, target_valence)  # Use target_valence here

        # Reactive Control Loss
        # (Handled in update_weights)
        # omega_p, omega_n = self.compute_surprise(ro_predictions, actual_ro)
        # effective_lr = self.compute_effective_learning_rate(omega_p, omega_n)
        # response_mask = (response > self.response_threshold).float()
        # desired_w_r_update = valence * torch.matmul(response_mask.t(), omega_n.unsqueeze(0)) # Equation 4 without the decay
        # reactive_loss = nn.MSELoss()(self.W_R, self.W_R + desired_w_r_update) # Penalize deviation from desired update

        # Combine losses (weighted if necessary)
        total_loss = ro_loss + td_loss  # Or adjust weights as needed

        return {'ro_loss': ro_loss, 'td_loss': td_loss, 'total_loss': total_loss}

    # PRO Eq. (3)
    def compute_ro_prediction(self, stimuli):
        """Predict response-outcome conjunctions
        returns RO conjunctions shape: n_ro_conjunctions
        """
        ro_conj = torch.matmul(self.W_S, stimuli)
        return ro_conj  # Return RO conjunction
    
    def update_temporal_components(self, stimuli):
        """Update delay chain and eligibility trace"""
        # Roll delay chain. X in PRO Eq. (9)
        self.delay_chain = torch.roll(self.delay_chain, 1, dims=0)
        self.delay_chain[0] = stimuli
        print(f"delay chain: {self.delay_chain}")
        
        # Update eligibility trace PRO Eq. (10)
        self.eligibility_trace = (self.delay_chain + self.lambda_decay * self.eligibility_trace)
        print(f"eligibility trace: {self.eligibility_trace}")

    # PRO Eq. (8)
    def compute_temporal_prediction(self):
        """Compute temporal predictions using eligibility trace
        returns value vector shape: n_ro_conjunctions"""
        return torch.sum(self.U * self.delay_chain, dim=(1, 2))
    
    # PRO Eq. (15-16)
    def compute_surprise(self, predicted_ro, actual_ro):
        """Compute positive and negative surprise (equation 2)
        returns positive and negative surprize tuple with shapes n_ro_conjuctions"""
        omega_p = F.relu(actual_ro - predicted_ro)
        omega_n = F.relu(predicted_ro - actual_ro)
        return omega_p, omega_n
    
    # PRO Eq. (5)
    def compute_effective_learning_rate(self, omega_p, omega_n):
        """Compute surprise-modulated learning rate
        returns vector of learning rates shape: n_ro_conjunctions"""
        return self.alpha_ro / (1 + omega_p + omega_n)


    # PRO-control Eq. (2)
    def compute_excitation(self, stimuli, ro_predictions):
        """Compute excitation term of activation"""
        direct_term = torch.matmul(stimuli, self.W_C)
        proactive_term = torch.clamp(torch.matmul(-ro_predictions, self.W_F), min=0)
        reactive_term = torch.sum(torch.clamp(-self.W_R, min=0), dim=1)
        excitation = self.rho * (direct_term + proactive_term + reactive_term)
        return excitation

    # PRO-control Eq. (3)
    def compute_inhibition(self, stimuli, ro_predictions):
        """Compute inhibition term of activation"""

        # Direct inhibition
        direct_inhib = self.psi * torch.matmul(stimuli, self.W_I)

        # Control Inhibition
        proactive_inhib = torch.clamp(torch.matmul(ro_predictions, self.W_F), min=0)
        reactive_inhib = torch.sum(torch.clamp(self.W_R, min=0), dim=1)
        control_inhib = self.phi * (proactive_inhib + reactive_inhib)

        inhibition = direct_inhib + control_inhib
        return inhibition

    # Pro Eq. (11) and PRO-Control Eq. (2-3)
    # May want to change this later to use different activation function

    def compute_response_activation(self, stimuli, ro_predictions):
        """Compute response activation (equations 5-7)"""

        excitation = self.compute_excitation(stimuli, ro_predictions)
        
        inhibition = self.compute_inhibition(stimuli, ro_predictions)

        # Compute activation change PRO Eq. (11)
        noise = torch.normal(0, self.sigma, self.C.shape, device=self.device)
        delta_C = self.beta * self.dt * (
            excitation * (1 - self.C) - 
            (self.C + 0.05) * (inhibition + 1) + noise
        )
        
        # Update activation
        self.C = torch.clamp(self.C + delta_C, min=0, max=1)
        return self.C

    def forward(self, stimuli):
        # Predict R-O conjunctions
        ro_predictions = self.compute_ro_prediction(stimuli)

        # Update temporal components
        self.update_temporal_components(stimuli)

        # Compute temporal prediction
        temporal_prediction = self.compute_temporal_prediction()
        
        # Generate response
        response = self.compute_response_activation(stimuli, ro_predictions)

        response_discrete = (response > self.response_threshold).float() # Thresholding
        
        return response, ro_predictions, temporal_prediction, response_discrete


    def update_weights(self, stimuli, response, actual_ro, reward, valence, gating_signal):  # Added target_valence and gating_signal
        """Update all weights based on experience"""
        # Get current predictions
        ro_predictions = self.compute_ro_prediction(stimuli)

        # Compute surprise
        omega_p, omega_n = self.compute_surprise(ro_predictions, actual_ro)
        effective_lr = self.compute_effective_learning_rate(omega_p, omega_n)

        # Update W_F (Proactive Control) - within no_grad Eq. (14)
        response_mask = (response > self.response_threshold).float()
        delta_w_f = 0.01 * torch.outer(actual_ro, response_mask) * valence * gating_signal
        with torch.no_grad():
            self.W_F.data += delta_w_f

        # Update reactive control weights (Equation 4)
        delta_w_r = valence * torch.ger(response, omega_n) * gating_signal
        self.W_R.data = 0.25 * (self.W_R + delta_w_r)
        self.W_R.data.clamp_(-1, 1)


class HERModel(nn.Module):
    def __init__(self,
                 n_stimuli,
                 n_responses,
                 n_outcomes,
                 n_delay_units,
                 n_layers=3,
                 **pro_kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.pro_layers = nn.ModuleList()
        for i in range(n_layers):
            self.pro_layers.append(PROControl(n_stimuli, n_responses, n_outcomes, n_delay_units, **pro_kwargs).to(device))

    def forward(self, stimuli):
        layer_outputs = []
        layer_errors = []
        layer_responses = []
        for i in range(self.n_layers):
            if i == 0:
                response, ro_predictions, temporal_prediction, response_discrete = self.pro_layers[i](stimuli)
            else:
                response, ro_predictions, temporal_prediction, response_discrete = self.pro_layers[i](layer_errors[-1])
            layer_outputs.append((response, ro_predictions, temporal_prediction, response_discrete))
            if i < self.n_layers - 1:
                layer_errors.append(ro_predictions)
            layer_responses.append(response_discrete)

        return layer_outputs, layer_responses


    def update_weights(self, stimuli, actual_outcomes, rewards, valences, gating_signals):
        layer_outputs, layer_responses = self.forward(stimuli)
        for i in range(self.n_layers):
            response, ro_predictions, temporal_prediction, response_discrete = layer_outputs[i]
            if i == 0:
                actual_ro = actual_outcomes
                reward = rewards
                valence = valences
                gating_signal = gating_signals
            else:
                actual_ro = layer_errors[i-1]
                reward = layer_errors[i-1] # Placeholder, needs refinement
                valence = layer_errors[i-1] # Placeholder, needs refinement
                gating_signal = layer_errors[i-1] # Placeholder, needs refinement

            self.pro_layers[i].update_weights(stimuli, response, actual_ro, reward, valence, gating_signal)


# Example usage:
n_stimuli = 10
n_responses = 5
n_outcomes = 2
n_delay_units = 5

her_model = HERModel(n_stimuli, n_responses, n_outcomes, n_delay_units, alpha_ro=0.01, alpha_td=0.01, device=device)
optimizer = optim.Adam(her_model.parameters(), lr=0.001)

# Example training loop (replace with your actual data)
stimuli = torch.randn(1, n_stimuli).to(device)
actual_outcomes = torch.randint(0, 2, (1, n_responses * n_outcomes)).float().to(device)
rewards = torch.randn(1, n_responses * n_outcomes).to(device)
valences = torch.randn(1, n_responses * n_outcomes).to(device)
gating_signals = torch.randint(0, 2, (1, n_responses * n_outcomes)).float().to(device)

for epoch in range(1000):
    optimizer.zero_grad()
    layer_outputs, layer_responses = her_model(stimuli)
    loss_dict = her_model.pro_layers[0].compute_losses(layer_responses[0], layer_outputs[0][1], layer_outputs[0][2], actual_outcomes, rewards, valences)
    loss = loss_dict['total_loss']
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')
    # Run Go/No-Go task
    go_nogo_task = GoNoGoTask(p_go=0.7)
    run_experiment(go_nogo_task)

    # Run Change Signal Task
    change_signal_task = ChangeSignalTask()
    run_experiment(change_signal_task)

    # Run Foraging Task
    foraging_task = ForagingTask()
    model, trainer = run_experiment(foraging_task)

    # Run Risk Avoidance Task
    risk_avoidance_task = RiskAvoidanceTask()
    run_experiment(risk_avoidance_task)

    # Run Correlated Outcomes Task
    p_switch_values = np.linspace(0.1, 0.9, 9)  # Example p(switch) values
    correlated_outcomes_task = CorrelatedOutcomesTask(p_switch_values)
    run_experiment(correlated_outcomes_task)