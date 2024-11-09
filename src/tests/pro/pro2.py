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
        self.W_F = -np.abs(np.random.normal(0, 0.1, (self.n_ro_conjunctions, n_responses))) # Learned top-down control from R-O units to Response units
        self.W_I = np.zeros((n_responses, n_responses))  # Fixed Mutual inhibition weights between response units
        np.fill_diagonal(self.W_I, -1)  # Self-inhibition
        self.delay_chain = np.zeros((n_delay_units, n_stimuli))
        self.U = np.zeros((self.n_ro_conjunctions, n_delay_units, n_stimuli))  # Temporal prediction weights 
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
        self.V = np.sum(self.U * self.eligibility_trace, axis=(1, 2))
        if self.V.shape != (self.n_ro_conjunctions,):
            print("Compute temporal returning incorrect dimensions for value:")
            print(f"V: {self.V.shape}")
            print(f"Should be: ({self.n_ro_conjunctions},)")
            input("Press Enter to continue...")
        return self.V
    
    def update_eligibility_trace(self, X):
        self.eligibility_trace = X + (self.lambda_decay * self.eligibility_trace)
        print(f"Updated eligibility trace: {self.eligibility_trace}")
        print()
        return self.eligibility_trace
    
    def compute_prediction_error(self, V_t, V_tp1, r_t):
        return r_t + self.gamma * V_tp1 - V_t

    def compute_surprise(self, predicted, actual):
        omega_P = np.maximum(0, actual - predicted)  # Unexpected occurrences
        omega_N = np.maximum(0, predicted - actual)  # Unexpected non-occurrences
        return omega_P, omega_N

    def update_delay_chain(self, stimuli):
        self.delay_chain = np.roll(self.delay_chain, 1, axis=0)
        self.delay_chain[0] = stimuli
        print(self.delay_chain)

    def update_temporal_prediction_weights(self, r_t, V_t=None, V_tp1=None):
        """Updates U_ijk based on the TD error."""
        print(f"Updating temporal prediction weights V_t: {V_t}, vtp1:{V_tp1}")
        if V_t is None:
            print("V_t is None")
            V_t = self.compute_temporal_prediction()

        if V_tp1 is None:
            print("V_tp1 is None")
            V_tp1 = self.compute_temporal_prediction()


        delta = self.compute_prediction_error(V_t, V_tp1, r_t)

        delta = r_t + self.gamma * V_tp1 - V_t

        # Reshape delta to (4, 1, 1) so it can broadcast across U
        delta = delta[:, np.newaxis, np.newaxis]

        # Update rule for U
        self.U += self.alphaTD * delta * self.eligibility_trace

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
        direct_term = np.dot(D, self.W_C)
        print(f"Direct term: {direct_term}")
        
        # Proactive control
        proactive_term = self.rectify(np.dot(-S, self.W_F))
        print(f"Proactive term: {proactive_term}")

        reactive_term = np.sum(self.rectify(-self.W_omega_N), axis=1)  # Apply weights to each outcome

        # Combine terms with scaling
        E = self.rho * (direct_term + proactive_term + reactive_term)
        print(f"Excitation: {E}")
        
        return E

    def calculate_inhibition(self, C, S):
        # Direct inhibition
        direct_inhib = self.psi * np.dot(C, self.W_I)
        print(f"Direct inhibition: {direct_inhib}")
        
        # Proactive control inhibition
        proactive_inhib = self.rectify(np.dot(S, self.W_F))
        print(f"Proactive inhibition: {proactive_inhib}")

        reactive_inhib = self.rectify(np.sum(self.W_omega_N, axis=1)) # Apply weights to each outcome
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
            for i in range(self.n_responses):
                if response[i] > self.response_threshold:  # T_i,t
                    for k in range(self.n_outcomes):
                        self.W_F[i, k] += 0.01 * response[i] * outcomes[k] * outcome_valence
        
    def update_reactive_control(self, response, omega_N, outcome_valence):
        executed_responses = (response > self.response_threshold).astype(float)        
        T = executed_responses.reshape(-1, 1)
        omega = omega_N.reshape(1, -1)
        self.W_omega_N = 0.25 * (self.W_omega_N + outcome_valence * T * omega)
        self.W_omega_N = np.clip(self.W_omega_N, -1, 1)


def update_model(model, stimulus, correct_response, 
                 actual_response, V_t=None):
    # Convert binary correct_response to outcome vector
    pro_control_outcome = np.zeros(model.n_outcomes * model.n_responses)
    
    # Determine if response was correct
    response_made = actual_response > model.response_threshold
    is_correct = np.array_equal(response_made, correct_response)
    # Set appropriate outcome signals
    if response_made[0]:    # If there was a Go response
        print("response made 0")
        pro_control_outcome[0] = float(is_correct)  # Go correct
        pro_control_outcome[1] = float(not is_correct)  # Go error
    elif response_made[1]:  # if there was a Change response
        print("response made 1")
        pro_control_outcome[2] = float(is_correct)  # Change correct
        pro_control_outcome[3] = float(not is_correct)  # Change error
    
    # Update PRO-Control model
    model.update_outcome_weights(stimulus, pro_control_outcome, subjective_badness=1.5)
    
    # Compute surprise signals for reactive control
    S = model.compute_outcome_prediction(stimulus)
    omega_P, omega_N = model.compute_surprise(S, pro_control_outcome)

    # Update reactive control with stronger learning signal
    outcome_valence = 1.0 if is_correct else -0.5  # Stronger punishment for errors
    model.update_reactive_control(response_made, omega_N, outcome_valence)
    
    # Update proactive component with stronger learning signal
    model.update_proactive_WF(response_made, pro_control_outcome, outcome_valence)

    # Update temporal predictions with actual next state

    model.set_input_stimuli(stimulus)
    V_tp1 = model.compute_temporal_prediction()
    # Calculate the reward signal
    if is_correct:
        r_t = 1.0  # Reward for correct response
    else:
        r_t = -0.5  # Penalty for incorrect response (adjust as needed)
    
    # Update temporal predictions with reward
    model.update_temporal_prediction_weights(r_t, V_t, V_tp1)

    return V_tp1

def create_change_signal_task(n_trials=100, change_prob=0.4):
    print(f"Creating change signal task with {n_trials} trials")
    # Initialize task structure
    stimuli = np.zeros((n_trials, 2))  # [go_signal, change_signal]
    correct_responses = np.zeros((n_trials, 2))  # [go_response, change_response]
    
    # Set go signal for all trials
    stimuli[:, 0] = 1
    
    # Randomly add change signals
    # Convert boolean to int by multiplying by 1
    change_trials = (np.random.random(n_trials) < change_prob)
    stimuli[change_trials, 1] = 1
    
    # Set correct responses
    correct_responses[~change_trials, 0] = 1  # Go response for no-change trials
    correct_responses[change_trials, 1] = 1   # Change response for change trials
    
    return stimuli, correct_responses

def simulate_trial(model, stimulus, max_steps):
    """Debugging version of trial simulation"""
    model.C = np.zeros(model.n_responses)
    
    # Print initial state
    print("\nTrial Start:")
    print(f"Stimulus: {stimulus}")

    # Initial predictions
    S = model.compute_outcome_prediction(stimulus)
    print(f"Initial outcome prediction (S): {S}")
    
    omega_P, omega_N = model.compute_surprise(S, np.zeros_like(S))
    print(f"Initial surprise - P: {omega_P}, N: {omega_N}")
    
    # Monitor response accumulation
    for step in range(max_steps):
        old_C = model.C.copy()
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
        n_outcomes=2,
        dt=0.01,     # Time step
        n_timesteps=250, # Number of time steps
        n_delay_units=10, # Number of delay units
        alphaRO=0.012, # Initial learning rate
        alphaTD=0.1,
        gamma=0.1, # Temporal discount
        lambda_decay=0.95, # Decay rate for eligibility traces
        beta=1.0,  # Response gain
        noise_sigma=0.002, # Response noise
        response_threshold=0.25, # Response threshold
        psi=0.724,  # Inhibition scaling
        phi=2.246,  # Control signal scaling
        rho=1.746   # Excitation scaling
        )
    
    # Print initial weights
    print("\nInitial Model State:")
    print(f"W_C (stimulus->response):\n{model.W_C}")
    print(f"W_F (outcome->response):\n{model.W_F}")
    print(f"W_S (stimulus->outcome):\n{model.W_S}")
    print(f"W_omega_P:\n{model.W_omega_P}")
    print(f"W_omega_N:\n{model.W_omega_N}")
    
    return model

def run_change_signal_simulation():
    """Run simulation with diagnostics"""
    n_trials = 500
    steps_per_trial = 10
    
    # Create task
    stimuli, correct_responses = create_change_signal_task(n_trials, change_prob=0.3)
    
    # Initialize model with modified parameters
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
    
    V_t = None
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
        V_t = update_model( model, stimuli[trial], correct_responses[trial],
                     response, V_t)
        
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
        did_respond = response_made[0] or response_made[1]
        pro_control_accuracy[trial] = np.array_equal(response_made, correct_responses[trial]) * 100
        pro_control_rts[trial] = rt
        learning_rate[trial] = np.linalg.norm(model.A)
        td_errors[trial] = np.linalg.norm(model.TD_error)
        predicted_values[trial] = np.linalg.norm(model.V)
        omega_P_values[trial] = np.linalg.norm(model.omega_P)
        omega_N_values[trial] = np.linalg.norm(model.omega_N)
        responses[trial] = did_respond
        responses_got_correct[trial] = np.array_equal(response_made, correct_responses[trial])
        W_S_changes[trial] = np.linalg.norm(model.W_S)
        W_C_changes[trial] = np.linalg.norm(model.W_C)
        W_F_changes[trial] = np.linalg.norm(model.W_F)

        # Update model
        V_t = update_model(model, stimuli[trial], correct_responses[trial],
                     response, V_t)

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
    axs[0].plot(pro_control_acc_smooth, label='PRO-Control Model')
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