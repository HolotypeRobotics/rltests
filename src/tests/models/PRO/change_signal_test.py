from models.pro_control import PROControlModel
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


"""
This is a test of the numpy based pro-control model using the Change signal task.
"""

def update_model(model, stimulus, correct_response, 
                 actual_response, V_t=None):
    # Convert binary correct_response to outcome vector
    pro_control_outcome = np.zeros(model.n_outcomes * model.n_responses)
    
    # Determine if response was correct
    response_made = actual_response > model.response_threshold
    is_correct = np.array_equal(response_made, correct_response)
    # Set appropriate outcome signals
    if response_made[0]:    # If there was a Go response
        pro_control_outcome[0] = float(is_correct)  # Go correct
        pro_control_outcome[1] = float(not is_correct)  # Go error
    elif response_made[1]:  # if there was a Change response
        pro_control_outcome[2] = float(is_correct)  # Change correct
        pro_control_outcome[3] = float(not is_correct)  # Change error
    
    # Update PRO-Control model
    model.update_outcome_weights(stimulus, pro_control_outcome, subjective_badness=1.5)
    
    # Compute surprise signals for reactive control
    S = model.compute_outcome_prediction(stimulus)
    omega_P, omega_N = model.compute_surprise(S, pro_control_outcome)

    # Update reactive control with stronger learning signal
    valence_signal = -0.1 if is_correct else 1 # (valence y = -0.1 if correct, 1 if incorrect eq 14)
    model.update_reactive_control(response_made, omega_N, valence_signal)
    
    # Update proactive component with stronger learning signal
    model.update_proactive_WF(response_made, pro_control_outcome, valence_signal)

    # Update temporal predictions with actual next state

    model.set_input_stimuli(stimulus)
    V_tp1 = model.compute_temporal_prediction()
    # Calculate the reward signal
    if is_correct:
        r_t = 1.0  # Reward for correct response
    else:
        r_t = -1.0  # Penalty for incorrect response (adjust as needed)
    
    # Update temporal predictions witFh reward
    model.update_temporal_prediction_weights(r_t, V_t, V_tp1)

    return V_tp1

def create_change_signal_task(n_trials=100, change_prob=0.4):
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
        alphaRO=1.5, # Initial learning rate
        alphaTD=0.1,
        gamma=0.095, # Temporal discount
        lambda_decay=0.5, # Decay rate for eligibility traces
        beta=1,  # Response gain
        noise_sigma=0.0001, # Response noise
        response_threshold=0.15, # Response threshold
        psi=1.5,  # Inhibition scaling
        phi=1.5,  # Control signal scaling
        rho=1.0   # Excitation scaling
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
        print("\nWeights:")
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