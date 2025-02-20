import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class SequenceEnv:
    def __init__(self, seq, topology):
        self.seq = np.array(seq, dtype=np.float32)
        self.topology = np.array(topology, dtype=np.float32)
        self.seq_len = len(seq)
        self.max_index = self.seq_len - 1
        self.reset()
    
    def reset(self):
        self.index = 0
        self.done = False
        self.prev_action = np.zeros(2, dtype=np.float32)
        return self._get_obs()
    
    def _get_obs(self):
        pos_onehot = np.zeros(self.seq_len, dtype=np.float32)
        pos_onehot[self.index] = 1.0
        obs = np.concatenate([pos_onehot, self.prev_action])
        return obs

    # The step function now returns the effort experienced.
    # effort is computed as the difference in topology between consecutive indices.
    def step(self, action, terminate):
        if self.done:
            raise RuntimeError("Episode already terminated.")
        # default minimal effort when not moving
        effort = 0.0001
        reward = 0.0

        # If the agent moves right, it incurs an effort cost and gets a reward.
        if action == 1 and self.index < self.max_index:
            prev_topology = self.topology[self.index]
            self.index += 1
            effort = self.topology[self.index] - prev_topology
            reward = self.seq[self.index]

        # Update previous action (one-hot)
        self.prev_action = np.eye(2, dtype=np.float32)[action]

        # Terminate if signaled or if at the end.
        if terminate or self.index == self.max_index:
            self.done = True

        next_obs = self._get_obs() if not self.done else None
        return next_obs, reward, effort, self.done, self.index

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_actions=2):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Policy (action logits)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        # State-value
        self.value_head = nn.Linear(hidden_dim, 1)
        # Termination probability
        self.term_head = nn.Linear(hidden_dim, 1)
        # Predicted effort for transitioning to the next state.
        self.effort_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        value = self.value_head(x)
        term_prob = torch.sigmoid(self.term_head(x))
        # Use softplus to ensure predicted effort is positive.
        predicted_effort = F.softplus(self.effort_head(x))
        return action_logits, value, term_prob, predicted_effort

def train_episode_online(env, model, optimizer, init_energy):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    total_reward = 0.0
    done = False
    energy = init_energy

    # Coefficient for the effort loss term
    effort_coef = 1.0

    while not done:
        # Get outputs from current state.
        action_logits, value, term_prob, predicted_effort = model(obs)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        term_dist = torch.distributions.Bernoulli(term_prob)
        term_sample = term_dist.sample()
        term_log_prob = term_dist.log_prob(term_sample)
        terminate = term_sample.item() > 0.5
        
        # Take a step in the environment.
        obs_next, reward, actual_effort, done, index = env.step(action.item(), terminate)
        total_reward += reward

        # Bootstrap value if not terminal.
        if not done:
            next_obs = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0)
            _, next_value, _, _ = model(next_obs)
            next_value = next_value.squeeze().detach()
        else:
            next_value = torch.tensor(0.0)

        # Use the predicted effort from the current state to discount the next value.
        discount = torch.exp(-predicted_effort.squeeze() / energy)
        td_target = reward + discount * next_value
        delta = td_target - value.squeeze()
        
        # Loss for the policy and termination heads using the TD error.
        policy_loss = -log_prob * delta.detach()
        term_loss = -term_log_prob * delta.detach()
        value_loss = F.mse_loss(value.squeeze(), td_target.detach())
        
        # Effort prediction loss: match predicted effort to actual effort experienced.
        effort_target = torch.tensor(actual_effort, dtype=torch.float32)
        effort_loss = F.mse_loss(predicted_effort.squeeze(), effort_target)
        
        loss = policy_loss + term_loss + value_loss + effort_coef * effort_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update energy: subtract the actual effort and add the reward.
        energy = max(energy - actual_effort + reward, 0.1)
        
        if not done:
            obs = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0)
    
    return total_reward, index

if __name__ == '__main__':
    def generate_random_sequence(length=7):
        return np.random.uniform(-10, 10, size=length)
    
    # Example sequences.
    seq = [1, 3, 10, 10, -10, 6, 7]
    topology = [0, 0, 4, 1, 2, 5, 6]
    print("Reward sequence:", seq)
    print("Topology:", topology)
    
    env = SequenceEnv(seq, topology)
    input_dim = env.seq_len + 2  # one-hot for position plus previous action.
    model = ActorCriticNet(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_episodes = 1000
    for episode in range(num_episodes):
        total_reward, index = train_episode_online(env, model, optimizer, init_energy=10.0)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Index: {index}")
