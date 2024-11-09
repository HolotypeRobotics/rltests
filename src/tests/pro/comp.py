import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
This version is based on pro3.py, but removed all comments, and combined pro-control/pro model into single class.
"""
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
        self.W_S = np.abs(np.random.normal(0.1, 0.05, (self.n_ro_conjunctions, n_stimuli))) # Response Output conjunction weights
        self.W_C = np.full((n_responses, n_stimuli), 1)  # Stimulus to response weights (Non-learning)
        self.W_F = -np.abs(np.random.normal(0, 0.1, (n_responses, self.n_ro_conjunctions))) # Learned top-down control from R-O units to Response units
        self.W_I = np.zeros((n_responses, n_responses))  # Fixed Mutual inhibition weights between response units
        np.fill_diagonal(self.W_I, -1)  # Self-inhibition
        self.delay_chain = np.zeros((n_delay_units, n_stimuli))
        self.U = np.zeros((n_outcomes, n_stimuli, n_delay_units))  # Temporal prediction weights 
        self.eligibility_trace = np.zeros_like(self.U)
        
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

        self.TD_error = np.zeros(n_outcomes)  # TD error
        self.V = np.zeros(n_outcomes)  # Value predictions
        
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


    def compute_temporal_prediction(self, X):
        """Compute temporal prediction (V) based on stimulus history"""

        # Roll over the delay chain: insert new stimulus to beginning
        self.delay_chain[1:] = self.delay_chain[:-1]
        self.delay_chain[0] = X
        
        # Reshape delay chain to (timesteps, 1, stimuli) for broadcasting
        delay_chain_reshaped = self.delay_chain[:, np.newaxis, :]

        # Compute V using tensor multiplication
        self.V = np.sum(self.U * delay_chain_reshaped, axis=(1, 2))

        return self.V


    def compute_prediction_error(self, V_t, V_tp1, r_t):
        self.TD_error = r_t + self.gamma * V_tp1 - V_t
        return self.TD_error

    def compute_surprise(self, predicted, actual):
        omega_P = np.maximum(0, actual - predicted)  # Unexpected occurrences
        omega_N = np.maximum(0, predicted - actual)  # Unexpected non-occurrences
        return omega_P, omega_N

    def update_temporal_prediction_weights(self, X, r_t, V_t, V_tp1):
        """Update temporal prediction weights based on TD error"""
        delta = self.compute_prediction_error(V_t, V_tp1, r_t)

        if X.ndim == 1:
            X = X[:, None]

        # Decay eligibility trace FIRST, then update with current stimulus
        self.eligibility_trace *= self.lambda_decay
        self.eligibility_trace = np.roll(self.eligibility_trace, 1, axis=2) # Shift time dimension
        self.eligibility_trace[:, :, 0] = X # Add current stimulus to eligibility

        self.U += self.alphaTD * np.outer(delta, X).reshape(self.U.shape)


    def update_outcome_weights(self, stimuli, outcomes, subjective_badness):
        """Update outcome prediction weights with value modulation"""
        theta = subjective_badness
        S = self.compute_outcome_prediction(stimuli)
        
        # Learning rate modulation based on surprise
        self.omega_P, self.omega_N = self.compute_surprise(S, outcomes)
        A = self.update_learning_rate(self.omega_P, self.omega_N)
        
        # Update weights with value modulation (theta)
        delta = A * (theta * outcomes - S)
        self.W_S += np.outer(delta, stimuli)

    def rectify(self, x):
        """Implement the [x]+ rectification function"""
        return np.maximum(0, x)

    def calculate_excitation(self, D, S):
        # Direct S-R associations
        direct_term = np.dot(D, self.W_C.T)
        print(f"Direct term: {direct_term}")
        # Proactive control
        proactive_term = np.sum(self.rectify(-np.dot(self.W_F, S)))

        print(f"Proactive term: {proactive_term}")

        reactive_term = -self.rectify(np.dot(S, self.W_omega_N.T))  # Apply weights to each outcome
        # Combine terms with scaling
        E = self.rho * (direct_term + proactive_term + reactive_term)
        print(f"Excitation: {E}")
        
        return E

    def calculate_inhibition(self, C, S):
        # Direct inhibition
        direct_inhib = self.psi * np.dot(C, self.W_I)
        print(f"Direct inhibition: {direct_inhib}")
        
        # Proactive control inhibition
        proactive_inhib = -np.clip(np.dot(self.W_F, S), -np.inf, 0)
        print(f"Proactive inhibition: {proactive_inhib}")

        reactive_inhib = self.rectify(np.dot(S, self.W_omega_N.T)) # Apply weights to each outcome
        print(f"Reactive inhibition: {reactive_inhib}")
        
        # Combine control terms with scaling
        control_inhib = self.phi * (proactive_inhib + reactive_inhib)
        print(f"Control inhibition: {control_inhib}")
        
        # Total inhibition
        I = direct_inhib + control_inhib
        print(f"Total inhibition: {I}")

        return I

    def compute_response_activation(self, stimuli, C, ro_conjuction):

        E = self.calculate_excitation(stimuli, ro_conjuction)
        I = self.calculate_inhibition(C, ro_conjuction)
        
        noise = np.random.normal(0, self.noise_sigma, self.n_responses)
        delta_C = self.beta * self.dt * (E * (1 - C) - (C + 0.05) * (I + 1) + noise)
        self.C = np.clip(C + delta_C, 0, 1)
    
        return self.C

    def update_proactive_WF(self, response, outcomes, outcome_valence, learning=True):
        """Update W_F (proactive component)"""
        if learning:
            T = response.reshape(-1, 1) # Reshape for outer product
            self.W_F += 0.01 * outcome_valence * T * outcomes.reshape(1, -1)


    def update_reactive_control(self, response, omega_N, outcome_valence):
        executed_responses = (response > self.response_threshold).astype(float)        
        T = executed_responses.reshape(-1, 1)
        omega = omega_N.reshape(1, -1)
        self.W_omega_N = 0.25 * (self.W_omega_N + outcome_valence * T * omega)
        self.W_omega_N = np.clip(self.W_omega_N, -1, 1) # Clip values as needed




def update_models(pro_control_model, stimulus, correct_response, 
                 pro_control_response, next_stimulus=None):
    # Convert binary correct_response to outcome vector
    pro_control_outcome = np.zeros(pro_control_model.n_ro_conjunctions)
    
    # Determine if response was correct
    response_made = pro_control_response > pro_control_model.response_threshold
    is_correct = np.array_equal(response_made, correct_response)
    # Set appropriate outcome signals
    if response_made[0]:    # If there was a Go response
        pro_control_outcome[0] = float(is_correct)  # Go correct
        pro_control_outcome[1] = float(not is_correct)  # Go error
    elif response_made[1]:  # if there was a Change response
        pro_control_outcome[2] = float(is_correct)  # Change correct
        pro_control_outcome[3] = float(not is_correct)  # Change error
    
    # Update PRO-Control model
    pro_control_model.update_outcome_weights(stimulus, pro_control_outcome, subjective_badness=1.5)
    
    # Compute surprise signals for reactive control
    S = pro_control_model.compute_outcome_prediction(stimulus)
    omega_P, omega_N = pro_control_model.compute_surprise(S, pro_control_outcome)

    # Update reactive control with stronger learning signal
    outcome_valence = 1.0 if is_correct else -0.5  # Stronger punishment for errors
    pro_control_model.update_reactive_control(response_made, omega_N, outcome_valence)
    
    # Update proactive component with stronger learning signal
    pro_control_model.update_proactive_WF(response_made, pro_control_outcome, outcome_valence)

    # Update temporal predictions with actual next state
    if next_stimulus is not None:
        V_t = pro_control_model.compute_temporal_prediction(stimulus)
        V_tp1 = pro_control_model.compute_temporal_prediction(next_stimulus)
    
        # Calculate the reward signal
        if is_correct:
            r_t = 1.0  # Reward for correct response
        else:
            r_t = -0.5  # Penalty for incorrect response (adjust as needed)
        
        # Update temporal predictions with reward
        pro_control_model.update_temporal_prediction_weights(stimulus, r_t, V_t, V_tp1)

def create_change_signal_task(n_trials=100, change_prob=0.4):
    # Initialize task structure
    stimuli = np.zeros((n_trials, 2))  # [go_signal, change_signal]
    correct_responses = np.zeros((n_trials, 2))  # [go_response, change_response]
    
    # Set go signal for all trials
    stimuli[:, 0] = 1
    
    # Randomly add change signals
    # Convert boolean to int by multiplying by 1
    change_trials = (np.random.random(n_trials) < change_prob).astype(int)
    stimuli[np.arange(n_trials), change_trials] = 1
    
    # Set correct responses
    correct_responses[change_trials == 0, 0] = 1  # Go response for no-change trials
    correct_responses[change_trials == 1, 1] = 1   # Change response for change trials
    
    return stimuli, correct_responses

def simulate_trial(model, stimulus, max_steps):
    """Debugging version of trial simulation"""
    model.C = np.zeros(model.n_responses)
    
    # Monitor response accumulation
    for step in range(max_steps):
        old_C = model.C.copy()
        S = model.compute_outcome_prediction(stimulus)

        response = model.compute_response_activation(stimulus, old_C, S)
        
        if step % 10 == 0:  # Print every 10 steps
            print(f"\nStep {step}:")
            print(f"Response activations: {response}")
            print(f"Delta C: {response - old_C}")
            
        if np.any(response > model.response_threshold):
            print(f"\nResponse threshold crossed at step {step}")
            print(f"Final response: {response}")
            return response, step * model.dt
            
        # Prevent response saturation
        model.C = np.clip(model.C, -1, 1)
    
    print("\nMax steps reached without response")
    print(f"Final response activations: {model.C}")
    return model.C, max_steps * model.dt

def diagnostic_pro_control_model():
    """Create a PRO-Control model with diagnostic prints"""
    model = PROControlModel(n_stimuli=2,
        n_responses=2,
        n_outcomes=4,  # 2 for Go, 2 for Change
        dt=0.01,     # Time step
        n_timesteps=20, # Number of time steps
        n_delay_units=20, # Number of delay units
        alphaRO=0.012, # Initial learning rate
        alphaTD=0.1, # Initial learning rate for TD Model
        beta=1.038,  # Response gain
        gamma=0.95, # Temporal discount
        lambda_decay=0.95, # Decay rate for eligibility traces
        noise_sigma=0.005, # Response noise
        response_threshold=0.313, # Response threshold
        psi=0.724,  # Inhibition scaling
        phi=2.246,  # Control signal scaling
        rho=1.764   # Excitation scaling
        )
    
    # Print initial weights
    print("\nInitial Model State:")
    print(f"W_C (stimulus->response):\n{model.W_C}")
    print(f"W_F (outcome->response):\n{model.W_F}")
    print(f"W_S (stimulus->outcome):\n{model.W_S}")
    print(f"W_omega_P:\n{model.W_omega_P}")
    print(f"W_omega_N:\n{model.W_omega_N}")
    
    return model

def run_change_signal_simulation(n_trials=100, steps_per_trial=50, change_prob=0.3):

    # Create task
    stimuli, correct_responses = create_change_signal_task(n_trials, change_prob)
    
    # Initialize model
    model = diagnostic_pro_control_model()
    
    # Recording arrays
    pro_control_accuracy =  np.zeros(n_trials)
    pro_control_rts =       np.zeros(n_trials)
    td_errors =             np.zeros(n_trials)
    predicted_values =      np.zeros(n_trials)
    learning_rate =         np.zeros(n_trials)
    omega_P_values =        np.zeros(n_trials)
    omega_N_values =        np.zeros(n_trials)
    responses =             np.zeros(n_trials)
    responses_got_correct = np.zeros(n_trials)
    W_S_changes =           np.zeros(n_trials)
    W_C_changes =           np.zeros(n_trials)
    W_F_changes =           np.zeros(n_trials)
    
    # Run first few trials with detailed monitoring
    for trial in range(min(5, n_trials)):
        print(f"\n=== Trial {trial + 1} ===")
        print(f"Stimulus: {stimuli[trial]}")
        print(f"Correct response: {correct_responses[trial]}")

        response, rt = simulate_trial(model, stimuli[trial], steps_per_trial)
        response_made = response > model.response_threshold
        
        print(f"Response made: {response_made}")
        print(f"Response Time: {rt}")
        print(f"Correct? {np.array_equal(response_made, correct_responses[trial])}")
        print(f"********")
        print()
        print()
        pro_control_accuracy[trial] = np.array_equal(response_made, correct_responses[trial]) * 100
        print(f"Accuracy: {pro_control_accuracy[trial]}")
        pro_control_rts[trial] = rt
        
        # Update model
        update_models( model, stimuli[trial], correct_responses[trial],
                     response, stimuli[trial + 1] if trial < n_trials - 1 else None)
        
        # Print weight updates
        print("\nWeight updates:")
        print(f"W_C:\n{model.W_C}")
        print(f"W_F:\n{model.W_F}")
        print(f"W_S:\n{model.W_S}")

        input("Press Enter to continue...")

    # Continue with remaining trials
    for trial in range(5, n_trials):
        # Run trial
        response, rt = simulate_trial(model, stimuli[trial], steps_per_trial)
        
        # Record results
        response_made = response > model.response_threshold
        pro_control_accuracy[trial] = np.array_equal(response_made, correct_responses[trial]) * 100
        pro_control_rts[trial] = rt
        learning_rate[trial] = model.A # effective learning rate is a scalar
        td_errors[trial] = np.linalg.norm(model.TD_error)
        predicted_values[trial] = np.linalg.norm(model.V)
        omega_P_values[trial] = np.linalg.norm(model.omega_P)
        omega_N_values[trial] = np.linalg.norm(model.omega_N)
        responses[trial] = np.any(response_made).astype(int)
        responses_got_correct[trial] = np.array_equal(response_made, correct_responses[trial]).astype(int)
        W_S_changes[trial] = np.linalg.norm(model.W_S)
        W_C_changes[trial] = np.linalg.norm(model.W_C)
        W_F_changes[trial] = np.linalg.norm(model.W_F)

        # Update model
        update_models(model, stimuli[trial], correct_responses[trial], response, stimuli[(trial+1) % n_trials])


    return {
        'n_trials': n_trials,
        'pro_control_accuracy': pro_control_accuracy,
        'pro_control_rts': pro_control_rts,
        'learning_rate': learning_rate,
        'td_errors': td_errors,
        'predicted_values': predicted_values,
        'omega_P_values': omega_P_values,
        'omega_N_values': omega_N_values,
        'responses': responses,
        'responses_got_correct': responses_got_correct,
        'W_S_changes': W_S_changes,
        'W_C_changes': W_C_changes,
        'W_F_changes': W_F_changes,
    }


def plot_results(results):
    """
    Plot accuracy and RT results
    """
    try:
        plt.style.use('rose-pine')
        print("Using rose-pine style")
    except:
        print("Style not found")
        print(f"Config dir: {matplotlib.get_configdir()}, Available styles: {plt.style.available}")
    
    trials = np.arange(results['n_trials'])
    fig, axs = plt.subplots(8, 1, figsize=(12, 18), sharex=True)
    fig.suptitle("Factors Affecting Model Accuracy Over Trials")
    
    # Plot running average accuracy
    window = 20
    pro_control_acc_smooth = np.convolve(results['pro_control_accuracy'],
                                        np.ones(window)/window, mode='valid')
    axs[0].plot(trials[:len(pro_control_acc_smooth)], pro_control_acc_smooth, label='PRO-Control Model')
    axs[0].set_title('Running Average Accuracy')
    axs[0].set_xlabel('Trial')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    
    # Plot RT distributions
    axs[1].hist(results['pro_control_rts'], alpha=0.5, label='PRO-Control Model', bins=20)
    axs[1].set_title('Response Time Distributions')
    axs[1].set_xlabel('Response Time (s)')
    axs[1].set_ylabel('Count')
    axs[1].legend()

    # Plot Learning Rate
    axs[2].plot(trials, results['learning_rate'], label="Learning Rate (α)", color="teal")
    axs[2].set_ylabel("Learning Rate (α)")
    axs[2].legend()

    # Plot Temporal Difference Error
    axs[3].plot(trials, results['td_errors'], label="TD Error", color="purple")
    axs[3].set_ylabel("Temporal Difference Error")
    axs[3].legend()

    # Plot Surprise Signals
    axs[4].plot(trials, results['omega_P_values'], label="Positive Surprise (ω_P)", color="blue")
    axs[4].plot(trials, results['omega_N_values'], label="Negative Surprise (ω_N)", color="orange")
    axs[4].set_ylabel("Surprise Signals")
    axs[4].legend()

    # Plot Weight Changes
    axs[5].plot(trials, results['W_S_changes'], label="W_S Changes", color="blue")
    axs[5].plot(trials, results['W_C_changes'], label="W_C Changes", color="orange")
    axs[5].plot(trials, results['W_F_changes'], label="W_F Changes", color="purple")
    axs[5].set_ylabel("Weight Changes")
    axs[5].legend()

    # Plot the predicted value over time
    axs[6].plot(trials, results['predicted_values'], label="Predicted Value", color="green")
    axs[6].set_xlabel("Trials")
    axs[6].set_ylabel("Predicted Value")
    axs[6].legend()

    # Plot the actions made
    axs[7].bar(trials, results['responses'], label="response activations", color="yellow")
    axs[7].bar(trials, results['responses_got_correct'], label="correct", color="green")
    axs[7].set_xlabel("Trials")
    axs[7].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent overlap

    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run simulation
    results = run_change_signal_simulation()
    
    # Plot results
    plot_results(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"PRO-Control Model Mean Accuracy: {np.mean(results['pro_control_accuracy']):.3f}")
    print(f"PRO-Control Model Mean RT: {np.mean(results['pro_control_rts']):.3f}s")