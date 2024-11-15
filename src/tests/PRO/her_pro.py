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
                 device='cpu'):
        super().__init__()
        
        self.device = device

        
        # Model dimensions
        self.n_stimuli = n_stimuli
        self.n_responses = n_responses
        self.n_outcomes = n_outcomes
        self.n_ro_conjunctions = n_responses * n_outcomes
        self.n_delay_units = n_delay_units
        self.outcome_rep = OutcomeRepresentation(n_stimuli, self.n_ro_conjunctions).to(device)
        self.optimizer_or = None # Add optimizer for outcome representation

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
        
        # Initialize learnable weights
        # R-O conjunction prediction weights
        self.W_S = nn.Parameter(torch.abs(
            torch.normal(0.1, 0.05, (self.n_ro_conjunctions, n_stimuli))))
        
        # Fixed stimulus-response weights
        self.register_buffer('W_C', torch.ones((n_responses, n_stimuli)))
        
        # Proactive control weights (equation 12 in paper)
        self.W_F = nn.Parameter(-torch.abs(
            torch.normal(0, 0.1, (self.n_ro_conjunctions, n_responses))))
        
        # Reactive control weights
        self.W_R = nn.Parameter(torch.zeros((n_responses, self.n_ro_conjunctions)))
        
        # Mutual inhibition weights
        self.register_buffer('W_I', torch.zeros((n_responses, n_responses)))
        self.W_I.fill_diagonal_(-1)
        
        # Temporal prediction weights
        self.U = nn.Parameter(torch.zeros(
            (self.n_ro_conjunctions, n_delay_units, n_stimuli)))
        
        # State buffers
        self.register_buffer('delay_chain', 
                           torch.zeros((n_delay_units, n_stimuli)))
        self.register_buffer('eligibility_trace', 
                           torch.zeros((n_delay_units, n_stimuli)))
        self.register_buffer('C', torch.zeros(n_responses))
        
        # Normalize W_F weights
        with torch.no_grad():
            norm_factor = torch.sum(torch.abs(self.W_F)) / (n_responses * n_outcomes)
            if norm_factor > 1:
                self.W_F.data /= norm_factor

    def compute_losses(self, response, pred_ro, pred_temporal, actual_ro, actual_values):
        """Compute the losses for the model."""

        # R-O prediction loss (using MSE loss for now, but could be something else)
        ro_loss = F.mse_loss(pred_ro, actual_ro)

        # Temporal difference loss (using MSE loss)
        td_loss = F.mse_loss(pred_temporal, actual_values)

        # Total loss (weighted sum of individual losses)
        total_loss = ro_loss + td_loss  # Adjust weights as needed
        return {
            'ro_loss': ro_loss,
            'td_loss': td_loss,
            'total_loss': total_loss
        }

    def compute_ro_prediction(self, stimuli):
        """Predict response-outcome conjunctions"""
        valence = self.outcome_rep(stimuli)
        return torch.matmul(self.W_S, stimuli) * valence # Valence scaling
    
    def update_temporal_components(self, stimuli):
        """Update delay chain and eligibility trace"""
        # Roll delay chain
        self.delay_chain = torch.roll(self.delay_chain, 1, dims=0)
        self.delay_chain[0] = stimuli
        
        # Update eligibility trace
        self.eligibility_trace = (self.delay_chain + 
                                self.lambda_decay * self.eligibility_trace)
        
    def compute_temporal_prediction(self):
        """Compute temporal predictions using eligibility trace"""
        return torch.sum(self.U * self.eligibility_trace, dim=(1, 2))
    
    def compute_surprise(self, predicted_ro, actual_ro):
        """Compute positive and negative surprise (equation 2)"""
        omega_p = F.relu(actual_ro - predicted_ro)
        omega_n = F.relu(predicted_ro - actual_ro)
        return omega_p, omega_n
    
    def compute_effective_learning_rate(self, omega_p, omega_n):
        """Compute surprise-modulated learning rate (equation 3)"""
        return self.alpha_ro / (1 + omega_p + omega_n)

    def compute_response_activation(self, stimuli, ro_predictions):
        """Compute response activation (equations 5-7)"""
        # Direct pathway
        excitation = self.rho * torch.matmul(stimuli, self.W_C.t())
        
        # Control pathways
        proactive, reactive = self.compute_control_signals(ro_predictions, self.C)
        control = self.phi * (proactive + reactive)
        
        # Mutual inhibition
        inhibition = self.psi * torch.matmul(self.C, self.W_I)
        
        # Compute activation change
        noise = torch.normal(0, 0.1, self.C.shape, device=self.device)
        delta_C = self.beta * self.dt * (
            excitation * (1 - self.C) - 
            (self.C + 0.05) * (inhibition + control) + noise
        )
        
        # Update activation
        self.C = self.C + delta_C
        self.C = torch.sigmoid(self.C) # Sigmoid activation for regularization
        return self.C

    def compute_control_signals(self, ro_predictions, response):
        """Compute proactive and reactive control signals - CORRECTED"""
        # Proactive control (equations 8-9) - Removed ReLU for push-pull
        proactive = -torch.matmul(ro_predictions, self.W_F) 

        # Reactive control based on *negative* surprise ONLY
        _, omega_n = self.compute_surprise(ro_predictions, torch.zeros_like(ro_predictions))
        reactive = F.relu(torch.matmul(omega_n, self.W_R.t())) # Transpose W_R for correct matrix multiplication

        return proactive, reactive

    def forward(self, stimuli):
        # Update temporal components
        self.update_temporal_components(stimuli)
        
        # Predict R-O conjunctions
        ro_predictions = self.compute_ro_prediction(stimuli)
        
        # Compute temporal prediction
        temporal_prediction = self.compute_temporal_prediction()
        
        # Generate response
        response = self.compute_response_activation(stimuli, ro_predictions)

        response_discrete = (response > self.response_threshold).float() # Thresholding
        
        return response, ro_predictions, temporal_prediction, response_discrete
    
    def update_weights(self, stimuli, response, actual_ro, reward, 
                      outcome_valence, gating_signal):
        """Update all weights based on experience"""
        # Get current predictions
        ro_predictions = self.compute_ro_prediction(stimuli)
        temporal_prediction = self.compute_temporal_prediction()
        
        # Compute surprise
        omega_p, omega_n = self.compute_surprise(ro_predictions, actual_ro)
        
        # Update R-O conjunction weights (equations 1-3)
        effective_lr = self.compute_effective_learning_rate(omega_p, omega_n)
        ro_error = (self.theta * actual_ro - ro_predictions) * outcome_valence
        self.W_S.data += gating_signal * effective_lr * torch.outer(ro_error, stimuli)
        
        # Update temporal prediction weights
        td_error = (reward + 
                   self.gamma * self.compute_temporal_prediction() - 
                   temporal_prediction)
        td_error = td_error.reshape(-1, 1, 1)
        self.U.data += (self.alpha_td * td_error * self.eligibility_trace)
        self.U.data.clamp_(-1, 1)
        
        # Update proactive control weights (equation 12)
        response_mask = (response > 0.1).float()
        delta_w_f = (0.01 * torch.outer(response_mask, actual_ro) * 
                    outcome_valence)
        self.W_F.data += gating_signal * delta_w_f

        # Update reactive control weights (equation 13)
        delta_w_r = (outcome_valence * # Y_i is outcome_valence
                               torch.outer(response_mask, omega_n))
        self.W_R.data = 0.25 * (self.W_R + gating_signal * delta_w_r) # Added gating
        self.W_R.data.clamp_(-1, 1)

        # Update Outcome Representation weights (using MSE loss)
        predicted_valence = self.outcome_rep(stimuli)
        valence_loss = F.mse_loss(predicted_valence, outcome_valence)
        self.optimizer_or.zero_grad() # Zero the optimizer's gradients
        valence_loss.backward(retain_graph=True) # Backpropagate the loss
        self.optimizer_or.step() # Update the weights
        
        return {
            'omega_p': omega_p,
            'omega_n': omega_n,
            'ro_error': ro_error,
            'td_error': td_error,
            'valence_loss': valence_loss
        }