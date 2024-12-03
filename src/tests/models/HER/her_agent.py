from her_environment import Environment
from her_copy import HER
import numpy as np

n_layers = 3
n_actions = 3 # turn left, turn right, forward
n_trials = 100
n_steps_per_trial = 100

# All values should be between 0 and 1
# Agent should take as input the distances to objects
# The current position as coordinates
# The current orientation as angle
# The previous action 1 hot encoded

env = Environment()
model = HER()
done = False
total_reward = 0
total_loss = 0

def step():
    response_probs, rs, ps = model(s) # Forward pass
    a_index = np.random.choice(np.arange(n_actions), p=response_probs) # Sample an action

    s, r, done = env.step(a_index) # Step the environment
    print(f"s: {s}, r: {r}, done: {done}")
    model.backward(s, a_index, rs, ps) # Backward pass
    return done

for i in range(n_trials):
    s = env.reset()
    for j in range(n_steps_per_trial):
        done = step()
        if done:
            print(f"Trial {i} step {j} done")
            break

    print(f"Trial {i} done")
    print(f"Reward: {total_reward}")
    print(f"Loss: {total_loss}")
    print()