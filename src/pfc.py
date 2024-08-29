import torch
import torch.nn as nn
import numpy as np
from environment import Environment

# HP to PL:
# fine allocentric position
# broad allocentric position
# place
# broad allocentric direction
# egocentric goal distances
# possible egocentric transition points

# place is predicted based on transition from environment to environment
# requies RNN for pattern completion and sequence learning to predict the place.
# the rnn should take in the speed and dirction, or the grid cells as imput

# Resembles the Prelimbic Cortex (PL) in the rodent brain
# predicts next step in plan
class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        # Forward pass through the GRU
        output, hidden = self.gru(x, hidden)
        return output, hidden

# Resembles the projections to M2
# predicts the action using the plan step
class ActorNetwork(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, hidden_state):
        action_probs = torch.softmax(self.fc(hidden_state), dim=-1)
        return action_probs

# Resembles the value prediction in the ACC
class CriticNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state):
        value = self.fc(hidden_state)
        return value

# Predicts resulting state from plan. ACC does this maybe?
# TODO: research more about this
class StatePredictorNetwork(nn.Module):
    def __init__(self, hidden_size, state_size):
        super(StatePredictorNetwork, self).__init__()
        self.fc = nn.Linear(hidden_size, state_size)

    def forward(self, hidden_state):
        predicted_state = self.fc(hidden_state)
        return predicted_state

# predicts the effort of the action in the ACC using motor feedback
class EffortNetwork(nn.Module):
    def __init__(self, input_size):
        super(EffortNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, allocentric_place, egocentric_goal_distances, action):
        x = torch.cat((allocentric_place, egocentric_goal_distances, action), dim=-1)
        effort = self.fc(x)
        return effort

# Predicts potential the task sets from hippocampal input
# Resembles the index that the ACC uses for potential task sets
class TaskPredictor(nn.Module):
    def __init__(self, input_size, num_task_sets, lr=0.01):
        super(TaskPredictor, self).__init__()
        self.fc = nn.Linear(input_size, num_task_sets)
        self.lr = lr

    def forward(self, allocentric_place, egocentric_goal_distances):
        # Concatenate inputs
        x = torch.cat((allocentric_place, egocentric_goal_distances), dim=-1)

        # Predict task set probabilities
        task_set_probs = F.softmax(self.fc(x), dim=-1)

        return task_set_probs

    def hebbian_update(self, allocentric_place, egocentric_goal_distances, selected_task):
        # Concatenate inputs
        x = torch.cat((allocentric_place, egocentric_goal_distances), dim=-1)
    
        # Apply Hebbian update to self.fc weights
        with torch.no_grad():  # Ensure this is a manual weight update, not part of autograd
            weight_update = self.hebbian_lr * torch.outer(selected_task.float(), x)
            self.fc.weight[selected_task] += weight_update

class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, state_size, num_task_sets):
        super(RLAgent, self).__init__()
        
        self.gru = GRUNetwork(input_size, hidden_size)
        self.actor = ActorNetwork(hidden_size, action_size)
        self.critic = CriticNetwork(hidden_size)\
        self.effort_net = EffortNetwork(hidden_size)
        self.state_predictor = StatePredictorNetwork(hidden_size, state_size)
        self.task_predictor = TaskPredictor(hidden_size, num_task_sets=4)
        self.task_sets = [num_task_sets]

    def switch_task_set():
        # switch out the weights in the current GRU with the next best one

    def task_switch_needed():
        # The need for a task switch is weighed based on the amount of errors from the current task set,
        # and the amount of conflict with another task set. 
        
        return False
    
    def handl_task_sets(self, task_sets):
        #  if there 
        
        if self.task_switch_needed() = True:
            # This will check how the current task set is doing, and if another could do it better.
            self.task_switch_needed() = False
            # This will handle switching to the next best one already lined up
            self.switch_task_set()



    

        # Inputs:
        # Current fine allocentric place
        #     using overlapping grids of increasing resolution, and at different offsets
        
        # Current broad allocentric place is like a behavioral marker that can change position depending on the objects around it.
        # This is used to help predict the task set.
        # They can be manipulated mentally through planning.

        # Current egocentroic goal distances example intensity of different smells when hungry giving their distance.
        #     The intensity of smell can change depending on the state of the agent.
        #     The agent can be in a hungry state. this is similar to Chemosensation

        # Current possible egocentric transition point options
        # (using allocentric and egocentric data) e,g (x,y) - (x_self,y_self), but respecting immediate boundaries.
    
    def forward(self,
                egocentric_goal_distances,
                egocentric_transition_points,
                allocentric_place,
                
                inputs, hidden=None):
        
        # Predict next step in plan
        gru_output, hidden = self.gru(inputs, hidden)

        # Take the last hidden state
        last_hidden_state = gru_output[:, -1, :]

        # Predict the action using the plan step
        action_probs = self.actor(last_hidden_state)

        # Predict the expected value for the next step in the plan
        value = self.critic(last_hidden_state)

        # Predict the effort of taking the next step in the plan
        effort = self.effort_net(last_hidden_state)

        # Predict the resulting state from the next step in the plan
        predicted_state = self.state_predictor(last_hidden_state)


        # Get probabilities to choose task set from
        task_sets = self.task_predictor(last_hidden_state, egocentric_goal_distances)

        # switch task set if necessary
        self.handle_task_sets(task_sets)

        # Calculate conflict signal
        conflict_signal = calculate_conflict(action_probs)

        return action_probs, value, effort, predicted_state, conflict_signal, hidden

task_sets = []
num_episodes = 100

rewards = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 10]
]

grid_size = len(rewards)
env = Environment(rewards)
env.set_exit(grid_size - 1, grid_size - 1)

