
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# The objective of the ACC is by telling the Prelimbic cortex
# when to switch out of the current trajectory.
# This can be accomlished if there are multple competing goals in a hierarchy
# (i.e. a distrobution of goals in the PL), and the ACC inhibits the
# current goal. The ACC can also inhibit the next action,
# The ACC needs to keep track of the reward of the actions not taken
# The ACC should Bias the Goal in the PL
# The Goal would be a state at the end of a rollout
# Studies show ACC predicts and calculates cost-benefit arbitration into a single prospective decision variable

# Define the recurrent module
class RecurrentModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RecurrentModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden_state):
        output, hidden_state = self.gru(x, hidden_state)
        return output, hidden_state

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


# Define the value network
class MLP(nn.Module):
    def __init__(self, obs_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.val_l1 = nn.Linear(obs_size, hidden_size)
        self.val_l2 = nn.Linear(hidden_size, output_size)

    def forward(self, obs, hidden_state):
        # Concatenate observation and hidden state
        x = F.relu(self.val_l1(x))
        state_value = self.val_l2(x)
        return F.relu(state_value)


class MetaActor(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(MetaActor, self).__init__()
        self.recurrent_module = RecurrentModule(state_size, hidden_size)
        self.policy_net = MLP(state_size, hidden_size, action_size)
        self.reward_predictor = MLP(state_size, hidden_size, 1)
        self.risk_predictor = MLP(state_size, hidden_size, 1)

    # Task is the hidden state of the GRU
    def forward(self, state, task):
        gru_out, task = self.recurrent_module(state, task)
        action_logits = self.policy_net(state, gru_out)
        risk = self.risk_predictor(state, gru_out)
        reward = self.reward_predictor(state, gru_out)
        
        return action_logits, risk, reward, task

    def init_hidden(self, batch_size):
        return self.recurrent_module.init_hidden(batch_size)
    

class WorldModel(nn.Module):
    def __init__(self, obs_size, hidden_size):
        super(WorldModel, self).__init__()
        self.recurrent_module = RecurrentModule(obs_size, hidden_size)
        self.state_predictor = MLP(obs_size, hidden_size, obs_size)
        self.cost_predictor = MLP(obs_size, hidden_size, 1)
        self.benefit_predictor = MLP(obs_size, hidden_size, 1)

    # Goal is the hidden state of the GRU
    # GRU is needed for accept, reject, and persist action for goals
    def forward(self, obs, goal):
        # Forward pass through the GRU
        gru_output, hidden_state = self.recurrent_module(obs, goal)
        x = torch.cat((obs, gru_output), dim=1)
        state = self.state_predictor(x)
        cost = self.cost_predictor(x)
        benefit = self.benefit_predictor(x)
        error = self.error_predictor(x)
        
        return state, cost, benefit, error, hidden_state


# Define the actor and critic
state_size = 9
hidden_size = 10
action_size = 3

meta_actor = MetaActor(state_size, hidden_size, action_size)
meta_critic = WorldModel(state_size, hidden_size)

actor_hidden = None
critic_hidden = None
energy = 0

# Define the optimizer
optimizer = optim.Adam(meta_actor.parameters(), lr=0.001)

def motivation(cost, benefit):
    return benefit - cost

# TODO:
# - Based on the predicted error, compute the learning rate,
#   an create a temporary contrast in activity for the actor
# - Use the motivation to bias reward to drive actor towards 
#   the right direction, by incorperating it in the loss function

# Define the meta critic loss function
def critic_loss_fn(predicted_state, advantage):
    return F.mse_loss(predicted_state, state) + F.mse_loss(advantage, advantage)

# define the meta actor loss function
def actor_loss_fn(action_logits, action):
    return -torch.distributions.Categorical(action_logits).log_prob(action)

# Training loop
for state, action, reward, effort, next_state, done in batch:
    # Forward pass through the actor network
    action_logits, risk_pred, reward_pred, task = meta_actor(state, task) #task is the hidden state of the GRU
    # Sample an action from the action logits
    action = torch.distributions.Categorical(action_logits).sample().item()
    # Forward pass through the critic network
    predicted_state, cost, predicted_error, critic_hidden = meta_critic(state, action, critic_hidden)

    actual_reward = reward * motivation(cost, predicted_error)


    # Compute the loss
    loss += actor_loss_fn(action_logits, action)
    loss = critic_loss_fn(predicted_state, advantage)
    # Backpropagate the loss
    loss.backward()
    # Update the parameters
    optimizer.step()
    # Reset the gradients
    optimizer.zero_grad()