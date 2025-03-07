import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# Ensure results directory exists
os.makedirs('results', exist_ok=True)


class SequenceEnv:
    def __init__(self, seq, topology):
        self.seq = np.array(seq, dtype=np.float32)
        self.topology = np.array(topology, dtype=np.float32)
        self.seq_len = len(seq)
        self.max_index = self.seq_len - 1
        self.index = 0
        self.reset()

    def reset(self):
        self.index = 0
        self.done = False
        self.prev_action = np.zeros(2, dtype=np.float32)  # [move, terminate]
        return self._get_obs()
    
    def _get_obs(self):
        pos_onehot = np.zeros(self.seq_len, dtype=np.float32)
        pos_onehot[self.index] = 1.0
        obs = np.concatenate([pos_onehot, self.prev_action])
        return obs

    def step(self, action):
        """
        Take a step in the environment. Actions:
        0: terminate
        1: move forward
        """
        if self.done:
            raise RuntimeError("Episode already terminated.")
        
        effort = 0.0001
        reward = 0.0
        
        terminate = (action == 0)
        
        # If the agent moves right (action 1), update position
        if action == 1 and self.index < self.max_index:
            prev_topology = self.topology[self.index]
            prev_reward = self.seq[self.index]
            self.index += 1
            effort = self.topology[self.index] - prev_topology
            reward = self.seq[self.index] - prev_reward
        
        # Record the action
        self.prev_action = np.eye(2, dtype=np.float32)[action]
        
        # Check if episode should terminate
        if terminate or self.index == self.max_index:
            self.done = True
        
        next_obs = self._get_obs() if not self.done else None
        return next_obs, reward, effort, self.done, self.index


class VTEAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(VTEAgent, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Main network - extract features
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy head (actions: terminate=0, move=1)
        self.action_head = nn.Linear(hidden_dim, 2)
        
        # Reward and Effort prediction head
        self.effort_head = nn.Linear(hidden_dim, 1)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Prediction heads (predict next state, reward from current state and action)
        self.predict_next_state = nn.Linear(hidden_dim , input_dim)  # Hidden + one-hot action

        self.surprise_level = 0.0
        self.surprise_threshold = 0.3

        
    def get_hidden_features(self, x):
        """Extract features from state"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def forward(self, x):
        """Forward pass: policy, value, and uncertainty"""
        hidden = self.get_hidden_features(x)
        
        action_logits = self.action_head(hidden)
        predicted_reward = self.reward_head(hidden)
        predicted_effort = F.softplus(self.effort_head(hidden))
        predicted_state = self.predict_next_state(hidden)
        predicted_value = self.value_head(hidden)
        
        return action_logits, predicted_reward, predicted_effort, predicted_value, predicted_state, hidden


    def resolve_best_action(self, state, ):
        action_logits, predicted_reward, predicted_effort, predicted_value, predicted_state_in_branch, hidden_features = self.forward(state)
        
        # Get action probabilities
        action_probs = F.softmax(action_logits, dim=1)

        # Calculate uncertainty in imagined state prediction
        # This calculation may be wrong
        state_uncertainty = -torch.sum(predicted_state_in_branch * torch.log(predicted_state_in_branch + 1e-8)).item()

        # Calculate uncertainty in imagined actions
        action_uncertainty = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1).item()
    
        # Use Upper Confidence Bound (UCB) to select action
        UCB_Score = predicted_value + 
        if last_best_UCB_Score is None or UCB_Score > last_best_UCB_Score:
            last_best_UCB_Score = UCB_Score
            last_best_action = action_probs
    
    def vte_rollout(self, state, max_depth=3, discount=0.95):
        """
        Perform mental simulation (VTE) to evaluate possible actions
        Returns: best_action, info_gains, trajectory
        """
        # Initial evaluation of current state
        with torch.no_grad():

            # Track the best action and its expected value
            competing_action_probs = None
            predicted_state_in_branch = None

            last_best_action = None
            last_best_UCB_Score = None
            action_uncertainty = 1.0
            last_best_value = None
            last_best_effort = None
            last_best_reward = None
            
            # Todo: Somehow implement branching.
            # It should keep only the first action probs, predicted state, value, effort, and reward, and they should get updated over the course o deliberation.

            last_best_UCB_Score,  ... = self.resolve_best_action(state, last_best_UCB_Score, last_best_action, action_uncertainty)
           last_best_
            while action_uncertainty > self.action_uncertainty_threshold:
                UCB_Score, ... = self.resolve_best_action(state, last_best_UCB_Score, last_best_action, action_uncertainty)
                if UCB_Score > last_best_UCB_Score:
                    last_best_UCB_Score = UCB_Score
                    last_best_action = action_probs
                    last_best_state = predicted_state_in_branch
                    last_best_value = predicted_value
                    last_best_effort = predicted_effort
                    last_best_reward = predicted_reward

            return last_best_action, predicted_reward, predicted_effort, predicted_state_in_branch, hidden_features
            


def train_episode(env, agent, optimizer, use_vte=True, vte_threshold=0.3):
    """Train agent for one episode"""
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0
    energy = 10.0  # Initial energy
    
    # For tracking
    vte_count = 0
    vte_trajectories = []
    all_info_gains = []
    
    # Keep track of previous predictions to compute prediction errors
    prev_state_pred = None
    prev_reward_pred = None
    


    while not done:
        # We only really need the rollout since forward is already implemented inside of it as the first step
        action_logits, predicted_reward, predicted_effort, predicted_state, hidden = agent.vte_rollout(state)
        action_probs = F.softmax(action_logits, dim=1)
        

        # TODO: come up with logic flow based on combinations of different types of uncertainty.
            #Utilize: state_prediction_loss, reward_loss, effort_loss, policy_entropy, state_entropy

        # If we are unable to predict the next state, then the environment is not known, and we should default to domain specific exploration behaviors.
        #   - Take actions that are predicted to increase information gain, and reduce uncertainty.
        # If the action confidence is low, then the agent would typically...

        """
        
        The Agent should engage in structured exploratory behaviors, maximizing novelty during outbound exploration and employ retreat behaviors when novelty is overwhelming(Gordon et al., 2014).
        Exploratory actions are organized into sequences, with outward tours followed by direct returns to a refuge, highlighting a systematic approach to exploration(Wallace et al., 2006).
        Characteristic movement patterns during exploratory behavior include reiterated roundtrips of increasing amplitude, transitioning from zero-dimension movements (staying-in-place) to one, two, and three-dimensional movements, reflecting a progressive increase in freedom and complexity of motion
        These are all domain mapping behaviors that allow sampling of the environment until predictions are accurate enough to make informed decisions.
        
        In stressful situation:
        In a corridor with 3 branches, enter each sequentially, only a little to observe it, then return to the main corridor.
        in unstressful environment:
        enter each branch and explore it fully until dead end, then return to the main corridor.
        The difference is that in a stressful situation, the agent needs to make decisions fast and reliably.
        """


# ...



        # Sample action from policy
        action_dist = torch.distributions.Categorical(probs=action_probs)
        action = action_dist.sample().item()

        
        # Take action in environment
        next_state, reward, effort, done, index = env.step(action)
        total_reward += reward
        
        # Learn from experience
        if not done:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            next_value = value.item()
        else:
            next_value = 0.0
        
        # Calculate TD target and error
        discount = torch.exp(-predicted_effort.squeeze() / energy)
        td_target = reward + discount.item() * next_value
        
        # Policy gradient loss
        log_prob = torch.log(action_probs[0, action] + 1e-8)
        advantage = td_target - value.item()
        policy_loss = -log_prob * advantage
        
        # Value loss
        value_loss = F.mse_loss(value, torch.tensor([[td_target]]))
        
        # Prediction losses
        if not done:
            # State prediction loss
            state_pred_loss = F.mse_loss(next_state_pred, next_state_tensor)
            
            # Reward prediction loss
            reward_pred_loss = F.mse_loss(reward_pred, torch.tensor([[reward]]))
            
            # Effort prediction loss
            effort_pred_loss = F.mse_loss(predicted_effort, torch.tensor([[effort]]))
            
            # Combined prediction loss
            prediction_loss = state_pred_loss + reward_pred_loss + effort_pred_loss
        else:
            prediction_loss = torch.tensor(0.0)
        
        # Uncertainty loss - should be high when prediction error is high
        uncertainty_target = torch.tensor([[min(1.0, prediction_error * 5)]])
        uncertainty_loss = F.mse_loss(uncertainty, uncertainty_target)
        
        # Total loss
        loss = policy_loss + value_loss + 0.5 * prediction_loss + 0.3 * uncertainty_loss
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update energy based on effort and reward
        energy = max(0.1, energy - effort + reward)
        
        # Store predictions for next step
        if not done:
            prev_state_pred = next_state_pred.detach()
            prev_reward_pred = reward_pred.detach()
            state = next_state_tensor
    
    return {
        'reward': total_reward,
        'position': index,
        'vte_count': vte_count,
        'vte_trajectories': vte_trajectories,
        'info_gains': all_info_gains
    }

def plot_vte_trajectory(vte_trajectories, info_gains, episode):
    """Plot VTE trajectories with error handling for empty data"""
    if not vte_trajectories or len(vte_trajectories) == 0:
        print("No VTE trajectories to plot yet")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Try to extract state data
    try:
        # Check if trajectory contains state data
        states = np.array([step[0][0] for step in vte_trajectories[0]])
        if states.size == 0:
            print("Warning: Empty state data in trajectory")
            return
            
        plt.subplot(2, 1, 1)
        plt.imshow(states.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Activation')
        plt.title(f'VTE Trajectory (Episode {episode})')
        plt.xlabel('Simulation Step')
        plt.ylabel('State Element')
        
        # Plot actions and info gains if available
        plt.subplot(2, 1, 2)
        actions = [step[1] for step in vte_trajectories[0]]
        plt.bar(range(len(actions)), actions, alpha=0.6, label='Actions')
        
        if info_gains and len(info_gains) > 0:
            # Make sure we don't try to plot more info gains than actions
            plot_gains = info_gains[0][:len(actions)] if len(info_gains[0]) > len(actions) else info_gains[0]
            plt.plot(range(len(plot_gains)), plot_gains, 'r-', linewidth=2, label='Info Gain')
        
        plt.title('VTE Actions & Info Gain')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/vte_trajectory_ep{episode}.png')
        plt.close()
    
    except (IndexError, ValueError, TypeError) as e:
        print(f"Error plotting VTE trajectory: {e}")
        plt.close()


def train_and_evaluate(env, agent, optimizer, num_episodes=1000, use_vte=True):
    """Train agent and track performance"""
    rewards = []
    positions = []
    vte_counts = []
    
    for episode in range(num_episodes):
        # Train for one episode
        result = train_episode(env, agent, optimizer, use_vte=use_vte)
        
        # Record metrics
        rewards.append(result['reward'])
        positions.append(result['position'])
        vte_counts.append(result['vte_count'])
        
        # Plot VTE trajectory occasionally
        if episode % 100 == 0 and result['vte_trajectories']:
            plot_vte_trajectory(result['vte_trajectories'], result['info_gains'], episode)
            
        # Log progress
        if episode % 50 == 0:
            print(f"Episode {episode}: Reward = {result['reward']:.2f}, "
                  f"Position = {result['position']}, VTE Count = {result['vte_count']}")
    
    return {
        'rewards': rewards,
        'positions': positions,
        'vte_counts': vte_counts
    }


def plot_results(results_vte, results_no_vte=None):
    """Plot training results"""
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 3, 1)
    plt.plot(results_vte['rewards'], label='With VTE')
    if results_no_vte:
        plt.plot(results_no_vte['rewards'], label='Without VTE')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot positions
    plt.subplot(1, 3, 2)
    plt.plot(results_vte['positions'], label='With VTE')
    if results_no_vte:
        plt.plot(results_no_vte['positions'], label='Without VTE')
    plt.title('Final Positions')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.legend()
    
    # Plot VTE count
    plt.subplot(1, 3, 3)
    window = min(50, len(results_vte['vte_counts']))
    vte_ma = [sum(results_vte['vte_counts'][max(0, i-window):i+1])/min(i+1, window) 
              for i in range(len(results_vte['vte_counts']))]
    plt.plot(vte_ma)
    plt.title('VTE Events (Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('results/training_results.png')
    plt.close()


def run_experiment(seq, topology, num_episodes=1000):
    """Run experiment with and without VTE"""
    # Initialize environment
    env = SequenceEnv(seq, topology)
    input_dim = env.seq_len + 2  # State + prev action
    
    # Create agent with VTE
    agent_vte = VTEAgent(input_dim)
    optimizer_vte = optim.Adam(agent_vte.parameters(), lr=0.001)
    
    # Train with VTE
    print("Training agent with VTE...")
    results_vte = train_and_evaluate(env, agent_vte, optimizer_vte, num_episodes=num_episodes, use_vte=True)
    
    # Create agent without VTE
    agent_no_vte = VTEAgent(input_dim)
    optimizer_no_vte = optim.Adam(agent_no_vte.parameters(), lr=0.001)
    
    # Train without VTE
    print("Training agent without VTE...")
    results_no_vte = train_and_evaluate(env, agent_no_vte, optimizer_no_vte, num_episodes=num_episodes, use_vte=False)
    
    # Plot results
    plot_results(results_vte, results_no_vte)
    
    return agent_vte, agent_no_vte, results_vte, results_no_vte


if __name__ == "__main__":
    # Define sequence with reward structure that requires exploration
    # Non-monotonic: has a local maximum at position 5
    seq = [1, 3, 10, 11, 9, 20, -100, -100]  # Rewards
    topology = [0, 0, 4, 1, 1, 5, 100, 2]    # Topology affects effort
    
    print("Reward sequence:", seq)
    print("Topology sequence:", topology)
    
    # Run experiment
    agent_vte, agent_no_vte, results_vte, results_no_vte = run_experiment(seq, topology, num_episodes=500)