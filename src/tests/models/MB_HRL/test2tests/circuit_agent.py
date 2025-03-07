import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Simulation Parameters
# -------------------------------
dt = 0.01            # Time step (s)
num_steps = 200      # Short sequence for debugging

# -------------------------------
# Energy Storage Parameters
# -------------------------------
main_cap_capacity = 1.0      # Big capacitor energy (arbitrary units)
buffer_cap_capacity = 0.2    # Small capacitor (fast power)
tau_transfer = 0.01          # Time constant for energy transfer

# -------------------------------
# Circuit & Control Parameters
# -------------------------------
full_energy_level = 5.0             # Desired full energy level
charging_gain = 0.05                # Energy added per unit food signal (J/s)
motor_drive_baseline = 0.1          # Baseline motor drive (subject to learning)
incentive_gain = 0.01               # Immediate incentive (hunger bias) from TD error
photosensor_sensitivity = 1.0       # Sensor sensitivity
fatigue_factor = 0.5                # How much uphill slopes reduce drive
sensitivity_gain = 1.0              # How much food signal and hunger boost drive
learning_rate = 0.001               # Learning rate for motor drive update

# -------------------------------
# Terrain-Related Parameters
# -------------------------------
baseline_resistance = 10.0          # (Not used for position update in this demo)
resistance_slope_factor = 0.5       # (Not used directly below)

# -------------------------------
# Initial Conditions
# -------------------------------
main_energy = 1.0       # Energy in main capacitor
buffer_energy = 0.2     # Energy in fast buffer
expected_future_energy = main_energy  # Value estimate
position = 5.0          # Start away from food at x = 5 (food is at x = 0)

# -------------------------------
# Helper Functions
# -------------------------------
def get_food_intensity(pos, food_source=0.0, sigma=1.0):
    """Gaussian food intensity: maximum at food_source."""
    return np.exp(-((pos - food_source)**2) / (2 * sigma**2))

# -------------------------------
# Simulation Step Function
# -------------------------------
def simulate_step(main_energy, buffer_energy, motor_drive_baseline,
                  expected_future_energy, position, terrain_slope):
    """
    Simulate one time step.
    
    Parameters:
      main_energy           : Energy in the main capacitor.
      buffer_energy         : Energy in the fast buffer.
      motor_drive_baseline  : Baseline motor drive (subject to learning).
      expected_future_energy: Value estimate (expected future main energy).
      position              : Current position (x, food is at x=0).
      terrain_slope         : Randomly chosen terrain slope (radians).
      
    Returns:
      Updated main_energy, buffer_energy, motor_drive_baseline, expected_future_energy,
      reward, TD error, discount_factor, effective_motor_drive, new_position, hunger_signal, terrain_slope.
    """
    # Compute food signal: higher near x=0.
    food_signal_intensity = photosensor_sensitivity * get_food_intensity(position)
    
    # For fatigue, compute an uphill factor (only positive slopes hurt drive).
    uphill_factor = max(0, np.sin(terrain_slope))
    
    # Compute discount factor from tau_transfer.
    discount_factor = np.exp(-dt / tau_transfer)
    
    # Here we boost motor drive with a term proportional to (food intensity * hunger)
    hunger_signal = full_energy_level - buffer_energy  # Hunger increases when buffer is low.
    # Note: If hunger_signal is high and food is strong, we want drive to increase.
    effective_motor_drive = motor_drive_baseline * (1 + sensitivity_gain * food_signal_intensity * hunger_signal)
    # Then, incorporate fatigue by reducing drive if climbing uphill.
    effective_motor_drive /= (1 + fatigue_factor * uphill_factor)
    
    # Energy intake: add to main energy.
    energy_intake = charging_gain * food_signal_intensity  # J/s
    
    # Transfer energy from main to buffer.
    transfer_rate = (main_energy - buffer_energy) / tau_transfer
    transfer_energy = transfer_rate * dt
    
    # Update main energy.
    main_energy_new = main_energy + dt * (energy_intake - transfer_rate)
    main_energy_new = max(0, main_energy_new)
    
    # Update buffer energy (capped at buffer capacity).
    buffer_energy_new = buffer_energy + transfer_energy
    buffer_energy_new = min(buffer_energy_new, buffer_cap_capacity)
    
    # Reward is the change in main energy.
    reward = main_energy_new - main_energy
    expected_future_energy_new = main_energy_new
    
    # Compute TD error.
    td_error = reward + discount_factor * expected_future_energy_new - expected_future_energy
    
    # Immediate incentive: boost if TD error is positive.
    incentive = incentive_gain * max(0, td_error)
    
    # Total motor drive (we add incentive to effective motor drive for immediate response).
    total_motor_drive = effective_motor_drive + incentive
    
    # Update motor drive baseline via learning.
    motor_drive_baseline_new = motor_drive_baseline + learning_rate * td_error * effective_motor_drive
    motor_drive_baseline_new = np.clip(motor_drive_baseline_new, 0, 10)
    
    # The motor draws energy from the buffer.
    # For simplicity, assume the total energy expenditure (per time step) equals total_motor_drive * dt,
    # but cannot exceed the available buffer energy.
    energy_expenditure = min(total_motor_drive * dt, buffer_energy_new)
    buffer_energy_new -= energy_expenditure
    buffer_energy_new = max(0, buffer_energy_new)
    
    # Update expected future energy.
    expected_future_energy = expected_future_energy_new
    
    # --- Movement Update ---
    # Use total_motor_drive as a velocity (the higher it is, the faster the organism moves toward food).
    # Here, since food is at x=0 and we start at a positive x, we subtract velocity * dt.
    new_position = position - total_motor_drive * dt
    # Ensure position doesn't go below 0 (overshooting the food source).
    new_position = max(0, new_position)
    
    return (main_energy_new, buffer_energy_new, motor_drive_baseline_new,
            expected_future_energy, reward, td_error, discount_factor,
            effective_motor_drive, new_position, hunger_signal, terrain_slope)

# -------------------------------
# Data Logging Arrays
# -------------------------------
main_energy_history = []
buffer_energy_history = []
motor_drive_history = []
expected_energy_history = []
reward_history = []
td_error_history = []
discount_factor_history = []
effective_motor_drive_history = []
position_history = []
hunger_history = []
terrain_slope_history = []

# -------------------------------
# Simulation Loop
# -------------------------------
for step in range(num_steps):
    # Generate a random terrain slope each step (in radians, e.g., between -0.5 and 0.5)
    random_slope = np.random.uniform(-0.5, 0.5)
    
    (main_energy, buffer_energy, motor_drive_baseline, expected_future_energy,
     reward, td_error, discount_factor, effective_motor_drive, position,
     hunger_signal, used_terrain_slope) = simulate_step(
         main_energy, buffer_energy, motor_drive_baseline, expected_future_energy, position, random_slope)
    
    main_energy_history.append(main_energy)
    buffer_energy_history.append(buffer_energy)
    motor_drive_history.append(motor_drive_baseline)
    expected_energy_history.append(expected_future_energy)
    reward_history.append(reward)
    td_error_history.append(td_error)
    discount_factor_history.append(discount_factor)
    effective_motor_drive_history.append(effective_motor_drive)
    position_history.append(position)
    hunger_history.append(hunger_signal)
    terrain_slope_history.append(used_terrain_slope)

# -------------------------------
# Plotting the Results
# -------------------------------
time = np.arange(num_steps) * dt

plt.figure(figsize=(12, 10))

# Plot Hunger over Time
plt.subplot(5, 1, 1)
plt.plot(time, hunger_history, label="Hunger Signal")
plt.xlabel("Time (s)")
plt.ylabel("Hunger (Full - Buffer)")
plt.legend()

# Plot Effective Motor Drive over Time
plt.subplot(5, 1, 2)
plt.plot(time, effective_motor_drive_history, label="Effective Motor Drive", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Motor Drive")
plt.legend()

# Plot Total Motor Drive (with incentive) over Time
total_drive = (np.array(effective_motor_drive_history) + incentive_gain * np.maximum(0, np.array(td_error_history)))
plt.subplot(5, 1, 3)
plt.plot(time, total_drive, label="Total Motor Drive", color="magenta")
plt.xlabel("Time (s)")
plt.ylabel("Total Drive")
plt.legend()

# Plot Terrain Slope over Time
plt.subplot(5, 1, 4)
plt.plot(time, terrain_slope_history, label="Terrain Slope", color="purple")
plt.xlabel("Time (s)")
plt.ylabel("Slope (radians)")
plt.legend()

# Plot Position over Time (should be decreasing toward 0)
plt.subplot(5, 1, 5)
plt.plot(time, position_history, label="Position", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Position (x)")
plt.legend()

plt.tight_layout()
plt.show()
