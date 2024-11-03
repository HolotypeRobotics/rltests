import numpy as np
import matplotlib.pyplot as plt

class PROControlModel:
    def __init__(self,
                n_stimuli,
                n_responses,
                n_outcomes,
                n_timesteps=100,
                dt=0.1,
                alpha=0.1,
                beta=0.1,
                gamma=0.95,
                psi=0.1,
                phi=0.1,
                rho=0.1,
                noise_sigma=0.1,
                response_threshold=0.1,
                debug=False):
        """
        Initialize PRO-Control Model
        
        Args:
            n_stimuli (int): Number of stimulus dimensions
            n_responses (int): Number of possible responses
            n_outcomes (int): Number of possible outcomes
            n_timesteps (int): Length of temporal prediction chain
            dt (float): Time step size
            alpha (float): Base learning rate
            beta (float): Response gain
            gamma (float): Temporal discount parameter
            psi (float): Inhibition scaling
            phi (float): Control signal scaling
            rho (float): Excitation scaling
            noise_sigma (float): Response noise standard deviation
            response_threshold (float): Threshold for response execution
            debug (bool): Enable debug printing
        """
        # Basic parameters
        self.dt = dt
        self.gamma = gamma
        self.alpha = alpha
        self.n_timesteps = n_timesteps
        self.debug = debug
        
        # Dimensions
        self.n_stimuli = n_stimuli
        self.n_responses = n_responses
        self.n_outcomes = n_outcomes
        
        # Response parameters
        self.beta = beta
        self.noise_sigma = noise_sigma
        self.response_threshold = response_threshold
        
        # Scaling parameters
        self.psi = psi
        self.phi = phi
        self.rho = rho
        
        self._initialize_weights()
        self.reset_state()

    def _initialize_weights(self):
        """Initialize model weights"""
        # Core weights
        self.W_S = np.random.normal(0, 0.1, (self.n_outcomes, self.n_stimuli))
        self.W_C = np.random.normal(0, 0.1, (self.n_responses, self.n_stimuli))
        self.W_F = np.random.normal(0, 0.1, (self.n_responses, self.n_outcomes))
        self.W_I = np.zeros((self.n_responses, self.n_responses))
        self.U = np.zeros((self.n_outcomes, self.n_stimuli, self.n_timesteps))
        
        # Reactive control weights
        self.W_omega_P = np.random.normal(0, 0.1, (self.n_responses, self.n_outcomes))
        self.W_omega_N = np.random.normal(0, 0.1, (self.n_responses, self.n_outcomes))

    def reset_state(self):
        """Reset model state"""
        self.C = np.zeros(self.n_responses)
        self.A = self.alpha

    def update_learning_rate(self, omega_P, omega_N):
        """Update effective learning rate based on surprise signals"""
        self.A = self.alpha / (1 + omega_P + omega_N)
        return self.A

    def compute_outcome_prediction(self, stimuli):
        """Compute immediate outcome predictions"""
        return np.dot(self.W_S, stimuli)

    def compute_temporal_prediction(self, X):
        """
        Compute temporal prediction based on stimulus history
        
        Args:
            X: Stimulus array (n_stimuli,) or (n_stimuli, n_timesteps)
        Returns:
            V: Temporal prediction array (n_outcomes,)
        """
        if X.ndim == 1:
            X = X[:, None]
        
        n_timesteps = min(X.shape[1], self.U.shape[2])
        X_padded = np.pad(X, ((0, 0), (0, self.U.shape[2] - X.shape[1])), mode='constant')
        return np.sum(np.sum(self.U * X_padded[None, :, :], axis=1), axis=1)

    def compute_prediction_error(self, V_t, V_tp1, r_t):
        """Compute temporal difference error"""
        return r_t + self.gamma * V_tp1 - V_t

    def compute_surprise(self, predicted, actual):
        """Compute positive and negative surprise"""
        omega_P = np.maximum(0, actual - predicted)
        omega_N = np.maximum(0, predicted - actual)
        return omega_P, omega_N

    def compute_response_activation(self, stimuli, S, omega_P, omega_N):
        """Compute response activation with proactive and reactive control"""
        if not all(dim.shape[0] == expected for dim, expected in 
                zip([stimuli, S, omega_P, omega_N], 
                    [self.n_stimuli, self.n_outcomes, self.n_outcomes, self.n_outcomes])):
            raise ValueError("Input dimension mismatch")
            
        reactive_control = (np.sum(self.W_omega_P[:, omega_P > 0], axis=1) + 
                          np.sum(self.W_omega_N[:, omega_N > 0], axis=1))
        
        noise = np.random.normal(0, self.noise_sigma, self.n_responses)
        
        E = self.rho * (np.dot(self.W_C, stimuli) - np.maximum(0, np.dot(self.W_F, S)))
        I = self.psi * (np.maximum(0, np.dot(self.W_F, S)) + np.maximum(0, reactive_control))

        if self.debug:
            self._debug_print_activation(stimuli, E, I, noise)
        
        delta_C = self.beta * self.dt * (E * (1 - self.C) - (self.C + 0.05) * (I + 1) + noise)
        self.C = np.clip(self.C + delta_C, 0, 1)
        return self.C

    def update_weights(self, stimulus, outcomes, theta=1.5):
        """Update all weights based on trial outcomes"""
        # Update outcome prediction weights
        S = self.compute_outcome_prediction(stimulus)
        omega_P, omega_N = self.compute_surprise(S, outcomes)
        A = self.update_learning_rate(omega_P, omega_N)
        
        delta = A * (theta * outcomes - S)
        self.W_S += np.outer(delta, stimulus)
        
        return omega_P, omega_N

    def update_reactive_control(self, response, omega_P, omega_N, outcome_valence):
        """Update reactive control weights"""
        for i in range(self.n_responses):
            if response[i] > self.response_threshold:
                self.W_omega_P[i] = 0.25 * (self.W_omega_P[i] + outcome_valence * omega_P)
                self.W_omega_N[i] = 0.25 * (self.W_omega_N[i] + outcome_valence * omega_N)
                self.W_omega_P[i] = np.clip(self.W_omega_P[i], -1, 1)
                self.W_omega_N[i] = np.clip(self.W_omega_N[i], -1, 1)

    def update_temporal_prediction_weights(self, X, r_t, V_t, V_tp1):
        """Update temporal prediction weights"""
        delta = self.compute_prediction_error(V_t, V_tp1, r_t)
        
        if X.ndim == 1:
            X = X[:, None]
        
        eligibility_trace = np.zeros_like(self.U)
        for i in range(self.n_outcomes):
            for j in range(self.n_stimuli):
                if X[j, 0] > 0:
                    eligibility_trace[i, j, :] = np.roll(eligibility_trace[i, j, :], 1)
                    eligibility_trace[i, j, 0] = X[j, 0]
                    self.U[i, j, :] += self.alpha * delta[i] * eligibility_trace[i, j, :]

    def _debug_print_activation(self, stimuli, E, I, noise):
        """Print debug information for response activation"""
        print("\nResponse activation components:")
        print(f"Stimuli: {stimuli}")
        print(f"E (excitation): {E}")
        print(f"I (inhibition): {I}")
        print(f"Noise: {noise}")

class ChangeSignalTask:
    def __init__(self, n_trials=100, change_prob=0.4):
        """
        Initialize Change Signal Task
        
        Args:
            n_trials (int): Number of trials
            change_prob (float): Probability of change signal
        """
        self.n_trials = n_trials
        self.change_prob = change_prob
        self.stimuli, self.correct_responses = self._create_trials()
    
    def _create_trials(self):
        """Create trial sequence"""
        stimuli = np.zeros((self.n_trials, 2))
        correct_responses = np.zeros((self.n_trials, 2))
        
        stimuli[:, 0] = 1
        change_trials = (np.random.random(self.n_trials) < self.change_prob)
        stimuli[change_trials, 1] = 1
        
        correct_responses[~change_trials, 0] = 1
        correct_responses[change_trials, 1] = 1
        
        return stimuli, correct_responses

    def create_outcome_vector(self, response, correct_response):
        """Create outcome vector based on response and correct response"""
        outcomes = np.zeros(4)
        if response[0]:  # Go response
            outcomes[0] = float(correct_response[0])  # Go correct
            outcomes[1] = float(not correct_response[0])  # Go error
        elif response[1]:  # Change response
            outcomes[2] = float(correct_response[1])  # Change correct
            outcomes[3] = float(not correct_response[1])  # Change error
        return outcomes

def run_simulation(n_trials=200, change_prob=0.3, debug=False):
    """Run change signal task simulation"""
    # Initialize task and model
    task = ChangeSignalTask(n_trials, change_prob)
    model = PROControlModel(
        n_stimuli=2,
        n_responses=2,
        n_outcomes=4,
        dt=0.05,          # Reduced time step for finer temporal resolution
        alpha=0.1,        # Increased learning rate for faster adaptation
        gamma=0.9,        # Higher temporal discount for stronger future predictions
        beta=15.0,        # Increased response gain for more decisive responses
        noise_sigma=0.02, # Reduced noise for more stable responses
        response_threshold=0.35, # Higher threshold for more selective responses
        psi=1.2,         # Increased inhibition scaling
        phi=1.0,         # Control signal scaling
        rho=1.5,         # Increased excitation scaling for stronger initial responses
        debug=debug
    )
    
    # Recording arrays
    accuracy = np.zeros(n_trials)
    rts = np.zeros(n_trials)
    
    for trial in range(n_trials):
        # Reset model state
        model.reset_state()
        
        # Get trial information
        stimulus = task.stimuli[trial]
        correct_response = task.correct_responses[trial]
        
        # Run trial
        max_steps = 100
        response = None
        rt = 0
        
        for step in range(max_steps):
            S = model.compute_outcome_prediction(stimulus)
            omega_P, omega_N = model.compute_surprise(S, np.zeros_like(S))
            response = model.compute_response_activation(stimulus, S, omega_P, omega_N)
            
            if np.any(response > model.response_threshold):
                rt = step * model.dt
                break
        
        if response is None:
            response = model.C
            rt = max_steps * model.dt
        
        # Record results
        response_made = response > model.response_threshold
        accuracy[trial] = np.array_equal(response_made, correct_response)
        rts[trial] = rt
        
        # Update model
        outcomes = task.create_outcome_vector(response_made, correct_response)
        omega_P, omega_N = model.update_weights(stimulus, outcomes)
        
        outcome_valence = 1.0 if accuracy[trial] else -0.5
        model.update_reactive_control(response_made, omega_P, omega_N, outcome_valence)
        
        if trial < n_trials - 1:
            next_stimulus = task.stimuli[trial + 1]
            V_t = model.compute_temporal_prediction(stimulus)
            V_tp1 = model.compute_temporal_prediction(next_stimulus)
            model.update_temporal_prediction_weights(stimulus, float(accuracy[trial]), V_t, V_tp1)
        
        if debug and trial < 3:
            print(f"\n=== Trial {trial + 1} ===")
            print(f"Stimulus: {stimulus}")
            print(f"Response: {response_made}")
            print(f"Correct response: {correct_response}")
            print(f"RT: {rt:.3f}s")
            print(f"Correct? {accuracy[trial]}")
            input("Press Enter to continue...")
    
    return {'accuracy': accuracy, 'rts': rts}

def plot_results(results):
    """Plot simulation results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot running average accuracy
    window = 20
    acc_smooth = np.convolve(results['accuracy'], 
                            np.ones(window)/window, mode='valid')
    
    ax1.plot(acc_smooth)
    ax1.set_title('Running Average Accuracy')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Accuracy')
    
    # Plot RT distribution
    ax2.hist(results['rts'], bins=20)
    ax2.set_title('Response Time Distribution')
    ax2.set_xlabel('Response Time (s)')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    results = run_simulation(debug=True)
    plot_results(results)
    
    print("\nSummary Statistics:")
    print(f"Mean Accuracy: {np.mean(results['accuracy']):.3f}")
    print(f"Mean RT: {np.mean(results['rts']):.3f}s")