import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.001  # Time step
num_steps = 10000  # Number of simulation steps

# Circuit and learning parameters
capacitance = 1.0  # Capacitance
baseline_resistance = 10.0  # Baseline resistance (ohms)
resistance_slope_factor = 0.5  # Scaling for resistance increase with slope
full_energy_level = 5.0  # Reference energy level
charging_gain = 0.05  # Charging rate from food source
motor_drive_baseline = 0.1  # Motor drive strength (subject to learning)
incentive_gain = 0.01  # Incentive gain from TD error
photosensor_sensitivity = 1.0  # Sensitivity to food signal

# Fatigue modulation for motor drive:
fatigue_factor = 0.5  # Scaling for motor fatigue with slope

learning_rate = 0.001  # Learning rate for updating motor drive

# Initial conditions
energy_level = 1.0  # Initial energy level (capacitor voltage)
expected_future_energy = energy_level  # Initial value estimate

def simulate_step(energy_level, motor_drive_baseline, expected_future_energy, food_signal_intensity, terrain_slope):
    """
    Simulate one time step of the organism's behavior, incorporating terrain slope effects.

    Parameters:
      energy_level          : Current energy level (capacitor voltage)
      motor_drive_baseline  : Baseline motor drive (subject to learning)
      expected_future_energy: Current value estimate (future energy expectation)
      food_signal_intensity : Light intensity representing food signal
      terrain_slope         : Slope angle (in radians)

    Returns:
      Updated energy level, motor drive, expected future energy, reward, TD error, discount factor, resistance, motor drive efficiency.
    """
    # Compute uphill fatigue factor: Only positive slopes contribute to fatigue.
    # In biological system, this would be predicted effort from visual cues.
    uphill_factor = max(0, np.sin(terrain_slope))

    # Compute effective resistance based on terrain slope
    effective_resistance = baseline_resistance * (1 + resistance_slope_factor * uphill_factor)
    time_constant = effective_resistance * capacitance  # Updated time constant

    # Compute gamma from circuit properties:
    discharge_factor = np.exp(-dt / time_constant)

    # Adjust the motor drive for fatigue:
    # In biological system, this would be the estimated work required to move forward in the terrain.
    motor_drive_effective = motor_drive_baseline / (1 + fatigue_factor * uphill_factor)

    # Photosensor response to food signal:
    perceived_food_signal = photosensor_sensitivity * food_signal_intensity

    # Energy intake from food (analogous to charging current)
    energy_intake = charging_gain * perceived_food_signal

    # Baseline motor energy usage (depends on hunger and available food)
    # How much energy should be spent on moving forward toward food?
    hunger_factor = full_energy_level - energy_level # We could have similar senses replicating emotional factors driving behaviors
    baseline_energy_expenditure = motor_drive_effective * perceived_food_signal * hunger_factor

    # Compute the baseline change in energy level
    # How much energy is left after accounting for the baseline expenditure?
    d_energy_dt_baseline = (energy_intake - baseline_energy_expenditure) / capacitance # rate of change of energy level
    energy_level_pre = energy_level + d_energy_dt_baseline * dt # energy level after accounting for baseline expenditure

    # Compute energy stored before and after
    energy_before = 0.5 * capacitance * energy_level**2
    energy_after = 0.5 * capacitance * energy_level_pre**2

    # Immediate reward as the energy increase
    reward = energy_after - energy_before

    # Update the expected future energy
    expected_future_energy_new = energy_level_pre

    # Compute the temporal difference (TD) error:
    td_error = reward + discharge_factor * expected_future_energy_new - expected_future_energy

    # Incentive: extra motor drive when TD error is positive
    incentive = incentive_gain * max(0, td_error)

    # Compute the total motor energy expenditure:
    energy_expenditure = baseline_energy_expenditure + incentive

    # Update the energy level after accounting for the full motor drive
    d_energy_dt = (energy_intake - energy_expenditure) / capacitance
    energy_level_new = energy_level + d_energy_dt * dt

    # Clamp energy level to a valid range
    energy_level_new = np.clip(energy_level_new, 0, 10)

    # Update motor drive baseline with TD error (training the policy)
    motor_drive_baseline_new = motor_drive_baseline + learning_rate * td_error * baseline_energy_expenditure
    motor_drive_baseline_new = np.clip(motor_drive_baseline_new, 0, 10)

    # Update value function estimate (using new energy level)
    expected_future_energy = expected_future_energy_new

    return (
        energy_level_new,
        motor_drive_baseline_new,
        expected_future_energy,
        reward,
        td_error,
        discharge_factor,
        effective_resistance,
        motor_drive_effective
    )

# Prepare arrays for logging
energy_history = []
motor_drive_history = []
expected_energy_history = []
reward_history = []
td_error_history = []
discount_factor_history = []
effective_resistance_history = []
motor_drive_effective_history = []

# Define a sample food availability signal (light intensity)
time_array = np.linspace(0, num_steps * dt, num_steps)
food_signal_sequence = np.sin(2 * np.pi * time_array) + 1.5  # Keep food signal positive

# Define a sample terrain slope signal
terrain_slope_sequence = 0.3 * np.sin(0.5 * 2 * np.pi * time_array)  # Slope in radians

# Run the simulation
for step in range(num_steps):
    food_signal_intensity = food_signal_sequence[step]
    terrain_slope = terrain_slope_sequence[step]
    
    (
        energy_level,
        motor_drive_baseline,
        expected_future_energy,
        reward,
        td_error,
        discount_factor,
        effective_resistance,
        motor_drive_effective
    ) = simulate_step(energy_level, motor_drive_baseline, expected_future_energy, food_signal_intensity, terrain_slope)

    energy_history.append(energy_level)
    motor_drive_history.append(motor_drive_baseline)
    expected_energy_history.append(expected_future_energy)
    reward_history.append(reward)
    td_error_history.append(td_error)
    discount_factor_history.append(discount_factor)
    effective_resistance_history.append(effective_resistance)
    motor_drive_effective_history.append(motor_drive_effective)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(time_array, energy_history, label="Energy Level")
plt.plot(time_array, expected_energy_history, label="Expected Future Energy", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.legend()
plt.title("Energy Level and Future Expectation")

plt.subplot(2, 2, 2)
plt.plot(time_array, td_error_history, label="TD Error", color='orange')
plt.xlabel("Time (s)")
plt.ylabel("TD Error")
plt.legend()
plt.title("Temporal Difference Error")

plt.subplot(2, 2, 3)
plt.plot(time_array, effective_resistance_history, label="Effective Resistance", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Resistance (ohms)")
plt.legend()
plt.title("Effective Resistance with Terrain")

plt.subplot(2, 2, 4)
plt.plot(time_array, motor_drive_effective_history, label="Effective Motor Drive", color='red')
plt.xlabel("Time (s)")
plt.ylabel("Motor Drive")
plt.legend()
plt.title("Motor Drive with Fatigue")

plt.tight_layout()
plt.show()
