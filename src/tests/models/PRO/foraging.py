import numpy as np
import matplotlib.pyplot as plt
from models.pro_control import PROControlModel  # Assuming you have a PROControlModel class

def run_foraging_simulation(model, n_trials=1000, foraging_cost=2, n_forage_options=6):
    """Runs the foraging simulation."""

    n_engage_options = 2
    choices = []
    outcomes = []
    rts = []
    controller_activity = []
    negative_surprise = []

    for _ in range(n_trials):
        # Generate engage option values
        engage_values = np.random.randint(1, 101, size=n_engage_options)

        # Initialize forage_values (important for correct relative value calculation)
        forage_values = np.zeros(n_forage_options) # Placeholder, actual values generated later if foraging

        # Calculate relative foraging value (BEFORE choice is made)
        #  Estimate forage values (could be more sophisticated, but mean is reasonable)
        expected_forage_value = 50.5 # Prior expectation of forage value (mean of uniform 1-100 distribution)
        relative_foraging_value = expected_forage_value - np.mean(engage_values) - foraging_cost


        # Determine stimulus bin (using relative foraging value)
        stimulus_bin = int((relative_foraging_value + 60) // 12)  # Discretize to 10 bins [-60, 60]
        stimuli = np.zeros(10)
        if 0 <= stimulus_bin < 10:
            stimuli[stimulus_bin] = 1

        # Model Steps (keep these together for clarity)
        model.set_input_stimuli(stimuli)
        ro_conjunction = model.compute_outcome_prediction(stimuli)
        V_t = model.compute_temporal_prediction() # Get prediction BEFORE action
        response = model.compute_response_activation(stimuli, np.zeros(model.n_responses), ro_conjunction) # Initialize response to zeros
        choice = np.argmax(response)



        # Determine outcome based on choice
        if choice == 0:  # Engage
            outcome = engage_values[np.random.randint(0, n_engage_options)]
            outcome_vector = np.array([outcome, 0])  # Engage outcome, no forage outcome
            choices.append(0)
        else:  # Forage
            forage_values = np.random.randint(1, 101, size=n_forage_options)
            outcome = np.mean(np.sort(forage_values)[-n_engage_options:]) - foraging_cost
            outcome_vector = np.array([0, outcome])  # No engage outcome, forage outcome
            choices.append(1)


        # --- Important: Temporal Difference Learning AFTER outcome ---
        V_tp1 = model.compute_temporal_prediction()  # Prediction AFTER action/outcome
        r_t = outcome_vector
        model.update_temporal_prediction_weights(r_t, V_t, V_tp1)


        outcomes.append(outcome)
        controller_activity.append(np.sum(ro_conjunction))

        # ... (Rest of the code for learning updates is mostly correct)

    return choices, outcomes, rts, controller_activity, negative_surprise



# Example usage (make sure n_stimuli matches the number of bins):
n_stimuli = 10  # 10 stimulus bins for relative foraging value
n_responses = 2  # Engage or forage
n_outcomes = 2  # Outcome for engage and forage
n_timesteps = 100  # Time steps per trial (adjust as needed)
n_delay_units = 20 # Number of delay units (adjust as needed)

model = PROControlModel(n_stimuli, n_responses, n_outcomes, n_timesteps, n_delay_units)
choices, outcomes, rts, controller_activity, negative_surprise = run_foraging_simulation(model)

# --- Analysis and plotting ---
# ... (Add your analysis and plotting code here)