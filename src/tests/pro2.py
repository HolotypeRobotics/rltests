# Key Improvements for PRO-Control Model Implementation

## 1. Weight Initialization and Dimensions

```python
class PROModel:
    def __init__(self, n_stimuli, n_responses, n_outcomes, n_timesteps=100, dt=0.1, 
                 alpha=0.1, beta=0.1, gamma=0.95, psi=0.1, phi=0.1, rho=0.1,
                 noise_sigma=0.1, response_threshold=0.1):
        # Current
        self.W_S = np.random.normal(0, 0.1, (n_outcomes, n_stimuli))
        self.W_C = np.random.full((n_responses, n_stimuli), 1)
        self.W_F = np.random.normal(0, 0.1, (n_responses, n_outcomes))
        
        # Should be:
        # Initialize W_S with small positive values for outcome prediction
        self.W_S = np.abs(np.random.normal(0.1, 0.05, (n_outcomes, n_stimuli)))
        # W_C should be initialized with positive values < 1
        self.W_C = np.random.uniform(0.3, 0.7, (n_responses, n_stimuli))
        # W_F should be initialized with small negative values
        self.W_F = -np.abs(np.random.normal(0.1, 0.05, (n_responses, n_outcomes)))
        
        # Add missing inhibition initialization
        np.fill_diagonal(self.W_I, -1)  # Self-inhibition

    def compute_response_activation(self, stimuli, S, omega_P, omega_N):

        def compute_response_activation(self, stimuli, S, omega_P, omega_N):
            # Compute proactive control signal
            proactive = np.dot(self.W_F, S)
            
            # Compute reactive control signal
            reactive = (np.dot(self.W_omega_P, omega_P) + 
                    np.dot(self.W_omega_N, omega_N))
            
            # Compute direct pathway activation
            direct = np.dot(self.W_C, stimuli)
            
            # Compute noise
            noise = np.random.normal(0, self.noise_sigma, self.n_responses)
            
            # Total control signal
            control = proactive + reactive
            
            # Response dynamics with gating
            E = self.rho * (direct * (1 - np.maximum(0, control)))
            I = self.psi * np.maximum(0, control)
            
            delta_C = self.beta * self.dt * (
                E * (1 - self.C) - 
                I * self.C + 
                noise
            )
            
            self.C = np.clip(self.C + delta_C, 0, 1)

def update_temporal_prediction_weights(self, X, r_t, V_t, V_tp1):
    # Add eligibility trace decay
    lambda_decay = 0.9  # Eligibility trace decay parameter
    
    delta = self.compute_prediction_error(V_t, V_tp1, r_t)
    
    if X.ndim == 1:
        X = X[:, None]
    
    eligibility_trace = np.zeros_like(self.U)
    
    for i in range(self.n_outcomes):
        for j in range(self.n_stimuli):
            if X[j, 0] > 0:
                eligibility_trace[i, j, :] = np.roll(eligibility_trace[i, j, :], 1)
                eligibility_trace[i, j, 0] = X[j, 0]
                # Apply decay to eligibility trace
                eligibility_trace[i, j, :] *= lambda_decay ** np.arange(self.n_timesteps)
                self.U[i, j, :] += self.alpha * delta[i] * eligibility_trace[i, j, :]

def update_reactive_control(self, response, omega_P, omega_N, outcome_valence):
    # Add separate learning rates for positive and negative outcomes
    alpha_pos = 0.1
    alpha_neg = 0.2
    
    learning_rate = alpha_pos if outcome_valence > 0 else alpha_neg
    
    for i in range(self.n_responses):
        if response[i] > self.response_threshold:
            self.W_omega_P[i] += learning_rate * (outcome_valence * omega_P - self.W_omega_P[i])
            self.W_omega_N[i] += learning_rate * (outcome_valence * omega_N - self.W_omega_N[i])
            
            # Maintain asymmetric bounds
            self.W_omega_P[i] = np.clip(self.W_omega_P[i], -0.5, 1.0)
            self.W_omega_N[i] = np.clip(self.W_omega_N[i], -1.0, 0.5)
```

## 4. Task Parameters and Simulation

```python
def create_change_signal_task(n_trials=100, change_prob=0.4):
    # Add SOA (Stimulus Onset Asynchrony) for change signal
    soa_range = [100, 200, 300]  # ms
    
    stimuli = np.zeros((n_trials, 2))
    correct_responses = np.zeros((n_trials, 2))
    soas = np.zeros(n_trials)
    
    # Set go signal
    stimuli[:, 0] = 1
    
    # Add change signals with varying SOAs
    change_trials = (np.random.random(n_trials) < change_prob)
    stimuli[change_trials, 1] = 1
    soas[change_trials] = np.random.choice(soa_range, size=np.sum(change_trials))
    
    # Set correct responses
    correct_responses[~change_trials, 0] = 1
    correct_responses[change_trials, 1] = 1
    
    return stimuli, correct_responses, soas
```

## 5. Model Parameters

```python
# Recommended parameter values based on empirical fits
model_params = {
    'dt': 0.001,          # 1ms timestep
    'alpha': 0.05,        # Learning rate
    'beta': 20.0,         # Response gain
    'gamma': 0.95,        # Temporal discount
    'psi': 1.5,          # Inhibition scaling
    'phi': 1.2,          # Control signal scaling
    'rho': 1.8,          # Excitation scaling
    'noise_sigma': 0.1,   # Response noise
    'response_threshold': 0.25  # Response threshold
}