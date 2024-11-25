import numpy as np


class PROControlModel():
    def __init__(self,
                n_stimuli,
                n_responses,
                n_outcomes,
                n_timesteps,
                n_delay_units,
                dt=0.1, 
                alphaRO=0.1,
                alphaTD=0.1,
                beta=0.1,
                gamma=0.95,
                lambda_decay=0.95,
                psi=0.1,
                phi=0.1,
                rho=0.1,
                noise_sigma=0.1,
                response_threshold=0.1
                ):

        # Dimensions
        self.n_stimuli = n_stimuli
        self.n_responses= n_responses
        self.n_outcomes = n_outcomes
        self.n_ro_conjunctions = n_responses * n_outcomes  # Number of response-outcome combinations

        # Basic parameters
        self.dt = dt  # Time step (10ms)
        self.gamma = gamma  # Temporal discount parameter
        self.alphaRO = alphaRO  # Base learning rate for response outcome conjunction learning
        self.alphaTD = alphaTD  # Base learning rate for temporal difference learning
        self.n_timesteps = n_timesteps  # Length of the tapped delay chain
        self.lambda_decay = lambda_decay  # Eligibility trace decay factor

        # Initialize weights
        self.W_S = np.abs(np.random.normal(0.18, 0.1, (self.n_ro_conjunctions, n_stimuli))) # Response Output conjunction weights
        self.W_C = np.full((n_responses, n_stimuli), 1)  # Stimulus to response weights (Non-learning)
        self.W_F = -np.abs(np.random.normal(0, 0.1, (self.n_ro_conjunctions, n_responses))) # Learned top-down control from R-O units to Response units
        self.W_I = np.zeros((n_responses, n_responses))  # Fixed Mutual inhibition weights between response units
        np.fill_diagonal(self.W_I, -1)  # Self-inhibition
        self.delay_chain = np.zeros((n_delay_units, n_stimuli))
        self.U = np.random.normal(0.1, 0.2, (self.n_ro_conjunctions, n_delay_units, n_stimuli))  # Temporal prediction weights 
        self.eligibility_trace = np.zeros((n_delay_units, n_stimuli))
        
        # Normalize WF weights as specified in the paper
        # Sum of absolute values should not exceed number of R-O conjunctions
        norm_factor = np.sum(np.abs(self.W_F)) / (n_responses * n_outcomes)
        if norm_factor > 1:
            self.W_F /= norm_factor

        # Response parameters
        self.beta = beta  # Response gain
        self.noise_sigma = noise_sigma # Response noise
        self.response_threshold = response_threshold # Response threshold

        # Scaling parameters
        self.psi = psi  # Inhibition scaling
        self.phi = phi  # Control signal scaling
        self.rho = rho  # Excitation scaling

        self.C = np.zeros(n_responses)  # Responses
        self.A = self.alphaRO  # Effective learning rate

        self.TD_error = np.zeros(self.n_ro_conjunctions)  # TD error
        self.V = np.zeros(self.n_ro_conjunctions)  # Value predictions
        
        # Extra values for tracking/visualization
        # Surprise vectors for each outcome
        self.omega_P = np.zeros((n_outcomes))  # Positive surprise
        self.omega_N = np.zeros((n_outcomes))  # Negative surprise

        # Weights from surprise to response weights
        self.W_omega_P = np.random.normal(0, 0.1, (n_responses, self.n_ro_conjunctions))  # Positive surprise to response weights
        self.W_omega_N = np.random.normal(0, 0.1, (n_responses, self.n_ro_conjunctions))  # Negative surprise to response weights


    def update_learning_rate(self, omega_P, omega_N):
        self.A = self.alphaRO / (1 + omega_P + omega_N)
        return self.A

    def compute_outcome_prediction(self, stimuli):
        return np.dot(self.W_S, stimuli)
    
    def set_input_stimuli(self, stimuli):
        self.update_delay_chain(stimuli)
        self.update_eligibility_trace(self.delay_chain)
    
    def compute_temporal_prediction(self):
        """Compute temporal prediction (V) based on stimulus history"""
        self.V = np.sum(self.U * self.delay_chain, axis=(1, 2))
        if self.V.shape != (self.n_ro_conjunctions,):
            print("Compute temporal returning incorrect dimensions for value:")
            print(f"V: {self.V.shape}")
            print(f"Should be: ({self.n_ro_conjunctions},)")
            input("Press Enter to continue...")
        return self.V
    
    def update_eligibility_trace(self, X):
        self.eligibility_trace = X + (self.lambda_decay * self.eligibility_trace)
        return self.eligibility_trace
    
    def compute_prediction_error(self, V_t, V_tp1, r_t):
        self.TD_error = r_t + self.gamma * V_tp1 - V_t
        return self.TD_error

    def compute_surprise(self, predicted, actual):
        omega_P = np.maximum(0, actual - predicted)  # Unexpected occurrences
        omega_N = np.maximum(0, predicted - actual)  # Unexpected non-occurrences
        return omega_P, omega_N

    def update_delay_chain(self, stimuli):
        self.delay_chain = np.roll(self.delay_chain, 1, axis=0)
        self.delay_chain[0] = stimuli

    def update_temporal_prediction_weights(self, r_t, V_t=None, V_tp1=None):
        """Updates U_ijk based on the TD error."""
        if V_t is None:
            print("V_t is None")
            V_t = self.compute_temporal_prediction()

        if V_tp1 is None:
            print("V_tp1 is None")
            V_tp1 = self.compute_temporal_prediction()

        delta = self.compute_prediction_error(V_t, V_tp1, r_t)

        # Reshape delta to (4, 1, 1) so it can broadcast across U
        delta = delta[:, np.newaxis, np.newaxis]

        # Update rule for U
        self.U += self.alphaTD * delta * self.eligibility_trace
        # Normalize U between 0 and 1. PRO model eq (9)
        self.U = (self.U-np.min(self.U))/(np.max(self.U)-np.min(self.U))
        return self.U

    def update_surprise_weights(self, stimuli, outcomes, learning=True):
        """Update surprise weights with value modulation"""
        # Gating signal G
        if not learning:
            return
        
        S = self.compute_outcome_prediction(stimuli)
        self.omega_P, self.omega_N = self.compute_surprise(S, outcomes)

    # eq (4)
    def update_outcome_weights(self, stimuli, outcomes, subjective_badness, learning=True):
        """Update outcome prediction weights with value modulation"""
        # Gating signal G
        if not learning:
            return
        
        theta = subjective_badness
        S = self.compute_outcome_prediction(stimuli)
        
        # Learning rate modulation based on surprise
        self.omega_P, self.omega_N = self.compute_surprise(S, outcomes)
        A = self.update_learning_rate(self.omega_P, self.omega_N)
        
        # Update weights with value modulation (theta)
        delta = A * (theta * outcomes - S)
        delta = np.outer(delta, stimuli)
        self.W_S += delta

    def rectify(self, x):
        """Implement the [x]+ rectification function"""
        return np.maximum(0, x)

    def calculate_excitation(self, D, S):
        print("--- Calculate excitation ---")
        # Direct S-R associations
        direct_term = np.dot(D, self.W_C)
        print(f"Direct excitation: {direct_term}")
        
        # Proactive control
        proactive_term = self.rectify(np.dot(-S, self.W_F))
        
        print(f"Proactive excitation: {proactive_term}")
        print("Unrectified proactive excitation: ", np.dot(-S, self.W_F))
        print(S)
        print()


        reactive_term = np.sum(self.rectify(-self.W_omega_N), axis=1)  # Apply weights to each outcome

        # Combine terms with scaling
        E = self.rho * (direct_term + proactive_term + reactive_term)
        print(f"Total Excitation: {E}")
        
        return E

    def calculate_inhibition(self, C, S):
        print("--- Calculate inhibition ---")
        # Direct inhibition
        direct_inhib = self.psi * np.dot(C, self.W_I)
        print(f"Direct inhibition: {direct_inhib}")
        
        # Proactive control inhibition
        proactive_inhib = self.rectify(np.dot(S, self.W_F))
        print(f"Proactive inhibition: {proactive_inhib}")
        print("Unrectified proactive inhibition: ", np.dot(S, self.W_F))

        reactive_inhib = self.rectify(np.sum(self.W_omega_N, axis=1)) # Apply weights to each outcome
        print(f"Reactive inhibition: {reactive_inhib}")
        
        # Combine control terms with scaling
        control_inhib = self.phi * (proactive_inhib + reactive_inhib)
        print(f"Control inhibition: {control_inhib}")
        
        # Total inhibition
        I = direct_inhib + control_inhib
        print(f"Total inhibition: {I}")
        print()

        return I

    def compute_response_activation(self, stimuli, C, ro_conjuction):

        E = self.calculate_excitation(stimuli, ro_conjuction)
        I = self.calculate_inhibition(C, ro_conjuction)
        
        noise = np.random.normal(0, self.noise_sigma, self.n_responses)
        delta_C = self.beta * self.dt * (E * (1 - C) - (C + 0.05) * (I + 1) + noise)
        self.C = np.clip(C + delta_C, 0, 1)
        print(f"Total activation C: {self.C}")
        return self.C

    def update_proactive_WF(self, response, outcomes, valence_signal, learning=True):
        """Update W_F (proactive component)"""
        if learning: # (Gating signal G)
            for i in range(self.n_responses):
                for k in range(self.n_outcomes):
                    if response[i] > self.response_threshold:  # T_i,t
                        self.W_F[i, k] += 0.01 * response[i] * outcomes[k] * valence_signal #( valence y)
    
    def update_reactive_control(self, response, omega_N, outcome_valence):
        executed_responses = (response > self.response_threshold).astype(float)        
        T = executed_responses.reshape(-1, 1)
        # omega = omega_N.reshape(1, -1)
        self.W_omega_N = (0.25 * self.W_omega_N) + outcome_valence * T * omega_N
        self.W_omega_N = np.clip(self.W_omega_N, -1, 1)

