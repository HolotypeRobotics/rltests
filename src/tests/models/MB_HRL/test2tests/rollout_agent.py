import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
        self.prev_action = np.zeros(2, dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        pos_onehot = np.zeros(self.seq_len, dtype=np.float32)
        pos_onehot[self.index] = 1.0
        obs = np.concatenate([pos_onehot, self.prev_action])
        return obs

    def step(self, action):
        """
        Take a step in the environment. Actions:
        0: terminate, reward = 0, effort = 0
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

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_actions=2):
        super(ActorCriticNet, self).__init__()
        # Shared network for feature extraction
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy and value heads
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.term_head = nn.Linear(hidden_dim, 1)
        self.effort_head = nn.Linear(hidden_dim, 1)
        
        # Execution head - decides when to stop simulation and execute
        self.exec_head = nn.Linear(hidden_dim, 1)
        
        # Additional network for belief state representation
        self.belief_output = nn.Linear(hidden_dim, input_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        
        action_logits = self.action_head(features)
        value = self.value_head(features)
        term_prob = torch.sigmoid(self.term_head(features))
        predicted_effort = F.softplus(self.effort_head(features))
        exec_prob = torch.sigmoid(self.exec_head(features))  # Probability to execute vs continue simulating
        belief = F.relu(self.belief_fc(features))
        belief = self.belief_output(belief)
        
        return action_logits, value, term_prob, predicted_effort, exec_prob, belief

def kl_divergence(p, q):
    """
    Calculate KL divergence between two probability distributions
    For continuous distributions, use a simplified approach
    """
    # Normalize to ensure they sum to 1
    
    p_normalized = F.softmax(p, dim=1)
    q_normalized = F.softmax(q, dim=1)
    return F.kl_div(q_normalized.log(), p_normalized, reduction='batchmean')


def calculate_belief_kl(new_belief, old_belief):
    """
    Calculate information gain as KL divergence between beliefs
    This simulates the "expected information gain" from the papers
    """
    return kl_divergence(new_belief, old_belief).item()


def simulate_internal_rollout(model, forward_model, obs, max_depth=5, discount=0.95, evidence_threshold=0.7):
    """
    Implements VTE-like behavior with sequential evaluation based on information foraging
    This models the "hippocampal sweeps" described in the papers, in a more biologically plausible way
    """
    # Initialize with current observation
    current_obs = obs.clone()
    
    # Get initial predictions from policy
    action_logits, initial_value, _, _, initial_exec_prob, initial_belief = model(current_obs)
    action_probs = F.softmax(action_logits, dim=1)
    
    # Initial belief state about the environment (used for information gain)
    initial_belief = model.get_belief_state(current_obs)
    
    # Instead of expanding all branches in parallel and comparing them at the end,
    # we'll evaluate one action at a time, comparing it to the best so far
    
    # Define variables to track best action seen
    best_action = None
    best_score = initial_value.item()  # Start with baseline value
    best_trajectory = []
    best_info_gains = []
    
    # Dictionary to keep all evaluated branches (for visualization)
    branches = {}
    
    # Define execution threshold based on confidence and accumulated evidence
    # Higher values will cause longer VTE sequences
    evidence_accumulation = 0.0
    
    # Sequentially evaluate each possible action
    for action_idx in range(action_probs.size(1)):
        # Reset for this branch
        branch_obs = current_obs.clone()
        action_onehot = torch.eye(action_probs.size(1))[action_idx].unsqueeze(0).float()
        
        # Initialize branch trajectory
        trajectory = [(branch_obs.detach().numpy(), action_idx, 0.0)]
        branch_info_gains = []
        
        # Predict first step
        next_obs, pred_reward, pred_effort = forward_model(branch_obs, action_onehot)
        next_obs_belief = model.get_belief_state(next_obs)
        
        # Calculate initial information gain
        info_gain = calculate_belief_kl(next_obs_belief, initial_belief)
        branch_info_gains.append(info_gain)
        
        # Initialize branch metrics
        cumulative_reward = pred_reward.item()
        cumulative_effort = pred_effort.item()
        cumulative_info_gain = info_gain
        branch_depth = 1
        
        # Check execution probability for first step
        _, next_value, _, _, exec_prob, confidence, _ = model(next_obs)
        branch_terminated = False
        
        # Update branch observation
        branch_obs = next_obs
        
        # Expand this branch with forward simulation steps
        while branch_depth < max_depth and not branch_terminated:
            # Get policy prediction for current state in this branch
            branch_logits, branch_value, _, branch_effort, branch_exec, branch_conf, _ = model(branch_obs)
            branch_probs = F.softmax(branch_logits, dim=1)
            
            # Decide if we should execute or continue simulating
            evidence_delta = branch_value.item() - best_score
            evidence_accumulation += evidence_delta if evidence_delta > 0 else 0
            
            # Check termination conditions:
            # 1. Execution probability exceeds threshold
            # 2. Accumulated evidence exceeds threshold
            # 3. Confidence in action exceeds threshold
            should_execute = (branch_exec.item() > 0.7 or 
                             evidence_accumulation > evidence_threshold or
                             branch_conf.item() > 0.9)
            
            if should_execute:
                # Terminate this branch's simulation
                branch_terminated = True
                continue
            
            # Get next action for this branch
            next_action = torch.argmax(branch_probs, dim=1).item()
            action_onehot = torch.eye(action_probs.size(1))[next_action].unsqueeze(0).float()
            
            # Predict next step
            next_obs, pred_reward, pred_effort = forward_model(branch_obs, action_onehot)
            next_obs_belief = model.get_belief_state(next_obs)
            
            # Calculate information gain
            info_gain = calculate_belief_kl(next_obs_belief, initial_belief)
            branch_info_gains.append(info_gain)
            
            # Update metrics
            branch_depth += 1
            cumulative_reward += discount**(branch_depth-1) * pred_reward.item()
            cumulative_effort += pred_effort.item()
            cumulative_info_gain += info_gain
            
            # Add to trajectory for visualization
            trajectory.append((next_obs.detach().numpy(), next_action, info_gain))
            
            # Update branch observation
            branch_obs = next_obs
        
        # Calculate final branch score
        branch_score = cumulative_reward + 0.5 * cumulative_info_gain - 0.3 * cumulative_effort
        
        # Store branch data (for visualization)
        branches[action_idx] = {
            'action': action_idx,
            'score': branch_score,
            'cumulative_reward': cumulative_reward,
            'cumulative_effort': cumulative_effort,
            'cumulative_info_gain': cumulative_info_gain,
            'depth': branch_depth,
            'path': trajectory,
            'info_gains': branch_info_gains
        }
        
        # Compare with best so far
        if branch_score > best_score:
            best_score = branch_score
            best_action = action_idx
            best_trajectory = trajectory
            best_info_gains = branch_info_gains
    
    # If no action improved over baseline, pick the highest probability action
    if best_action is None:
        best_action = torch.argmax(action_probs, dim=1).item()
        # Get data from this branch if available
        if best_action in branches:
            best_trajectory = branches[best_action]['path']
            best_info_gains = branches[best_action]['info_gains']
    
    # Return best action, whether it's beneficial, info gains, and trajectory
    return best_action, best_score > initial_value.item(), best_info_gains, best_trajectory, branches


def train_episode(env, model, optimizer, init_energy=10.0, 
                 forward_coef=1.0, use_vte=True, base_vte_threshold=0.3):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    total_reward = 0.0
    total_loss = 0.0
    done = False
    energy = init_energy
    effort_coef = 1.0
    vte_count = 0
    vte_trajectories = []
    info_gains_history = []
    
    # Initialize prediction error tracking with smoothing
    recent_pred_errors = []
    max_error_history = 10  # Keep track of this many recent errors
    pred_error_weight = 0.5  # Weight for prediction error's influence on VTE
    error_smoothing = 0.7  # Exponential smoothing factor (higher = more smoothing)
    current_smoothed_error = 0.0  # Running smoothed error
    
    # Last prediction for initial state (no previous prediction exists)
    last_state_pred = None
    last_reward_pred = None
    last_effort_pred = None

    # Tracking for position-based prediction errors
    position_error_map = {}  # Maps positions to their recent prediction errors

    while not done:
        # Decide whether to use VTE based on uncertainty and prediction errors

        action_logits, value, term_prob, predicted_effort, exec_prob, beleif_state = model(obs)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        entropy = action_dist.entropy().item()

        # Position-specific prediction error (if we've been here before)
        position_specific_error = position_error_map.get(env.index, 0.0)
        
        # Calculate adjusted VTE threshold based on position-specific prediction errors
        # Higher prediction errors -> lower threshold -> more VTE
        vte_threshold = max(0.1, base_vte_threshold - pred_error_weight * position_specific_error)
    

        # Use VTE when uncertainty is high or prediction errors are high
        should_use_vte = (use_vte and 
                         (entropy > vte_threshold or position_specific_error > 0.5) and
                         confidence.item() < 0.7 and
                         energy > 2.0)
        
        if should_use_vte:
            # Perform mental simulation (VTE) - sequential evaluation of options
            sim_action, is_beneficial, info_gains, rollout_steps, branches = simulate_internal_rollout(
                model, forward_model, obs)
            
            # Only count VTE if it was beneficial
            if is_beneficial:
                action = sim_action
                vte_count += 1
                vte_trajectories.append(rollout_steps)
                info_gains_history.append(info_gains)
                
                # Cost of mental simulation - VTE requires energy too
                vte_cost = 0.1 * len(rollout_steps)
                energy -= vte_cost
                
                print(f"Used VTE at position {env.index}, chose action {action}, " 
                      f"info gain: {sum(info_gains):.3f}, pred error: {position_specific_error:.3f}")
            else:
                # If VTE wasn't beneficial, use direct policy
                action_dist = torch.distributions.Categorical(logits=action_logits)
                action = action_dist.sample().item()
        else:
            # Direct action using current policy (no VTE needed)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample().item()
            
        # Make predictions for current state and action
        action_onehot = torch.eye(2)[action].float().unsqueeze(0)
        with torch.no_grad():
            next_obs_pred, reward_pred, effort_pred = forward_model(obs, action_onehot)
        
        # Execute action in environment
        term_dist = torch.distributions.Bernoulli(term_prob)
        term_sample = term_dist.sample()
        terminate = term_sample.item() > 0.5

        obs_next, reward, actual_effort, done, index = env.step(action, terminate)
        total_reward += reward
        
        # Calculate prediction errors if we have previous predictions
        if last_state_pred is not None:
            # Calculate error between previous prediction and current observation
            current_obs_tensor = obs.clone().detach()
            
            # Ensure shapes match before calculating MSE
            if last_state_pred.shape == current_obs_tensor.shape:
                state_pred_error = F.mse_loss(last_state_pred, current_obs_tensor).item()
            else:
                # Handle shape mismatch - flatten both tensors to 1D
                state_pred_error = F.mse_loss(
                    last_state_pred.view(-1), 
                    current_obs_tensor.view(-1)
                ).item()
                
            reward_pred_error = abs(last_reward_pred - reward) if last_reward_pred is not None else 0
            effort_pred_error = abs(last_effort_pred - actual_effort) if last_effort_pred is not None else 0
            
            # Combined prediction error (normalized)
            # Scale state error down as it tends to dominate
            state_weight = 0.2
            reward_weight = 0.4
            effort_weight = 0.4
            total_pred_error = (state_weight * state_pred_error + 
                               reward_weight * reward_pred_error + 
                               effort_weight * effort_pred_error)
            
            # Apply exponential smoothing to prediction error
            current_smoothed_error = error_smoothing * current_smoothed_error + (1 - error_smoothing) * total_pred_error
            
            # Update position-specific error map
            if env.index not in position_error_map:
                position_error_map[env.index] = current_smoothed_error
            else:
                # Smooth update for position-specific errors
                position_error_map[env.index] = error_smoothing * position_error_map[env.index] + (1 - error_smoothing) * total_pred_error
            
            # Update global error history
            recent_pred_errors.append(current_smoothed_error)
            if len(recent_pred_errors) > max_error_history:
                recent_pred_errors.pop(0)  # Remove oldest error
        
        # Store predictions for next step comparison
        if not done:
            last_state_pred = next_obs_pred.detach()
            last_reward_pred = reward_pred.item()
            last_effort_pred = effort_pred.item()
        
        # Learn from experience
        if not done:
            next_obs = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0)
            _, next_value, _, next_effort, _, _, _ = model(next_obs)
            next_value = next_value.squeeze().detach()
            next_effort = next_effort.squeeze().detach()
        else:
            next_value = torch.tensor(0.0)
            next_effort = torch.tensor(0.0)
            
        # Discount factor based on NEXT state's predicted effort  
        discount = torch.exp(-next_effort / energy)
        td_target = reward - actual_effort + discount * next_value
        delta = td_target - value.squeeze()
        
        # Get the log probability of the action
        log_prob = action_dist.log_prob(torch.tensor(action))
        term_log_prob = term_dist.log_prob(term_sample)
        
        # Calculate losses
        policy_loss = -log_prob * delta.detach()
        term_loss = -term_log_prob * delta.detach()
        value_loss = F.mse_loss(value.squeeze(), td_target.detach())
        effort_target = torch.tensor(actual_effort, dtype=torch.float32)
        effort_loss = F.mse_loss(predicted_effort.squeeze(), effort_target)
        entropy_loss = -0.01 * action_dist.entropy()  # Entropy regularization
        
        # Execution and confidence loss
        # Train execution head based on prediction accuracy and value improvement
        exec_target = torch.sigmoid(torch.tensor([delta.item() * 2.0]))
        if current_smoothed_error > 0:
            # Reduce execution probability when prediction errors are high
            exec_target = exec_target * (1.0 - min(0.8, current_smoothed_error))
        exec_loss = F.binary_cross_entropy(exec_prob.view(-1), exec_target)
        
        
        # Forward model loss with L2 regularization
        if not done:
            # Create one-hot action for forward model
            next_obs_target = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0)
            
            # Loss for next state, reward and effort prediction
            forward_loss = F.mse_loss(next_obs_pred, next_obs_target)
            reward_loss = F.mse_loss(reward_pred.squeeze(), torch.tensor(reward, dtype=torch.float32))
            effort_pred_loss = F.mse_loss(effort_pred.squeeze(), torch.tensor(actual_effort, dtype=torch.float32))
            
            # Add L2 regularization for the forward model to prevent overfitting
            l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in forward_model.parameters())
            
            forward_total_loss = forward_loss + reward_loss + effort_pred_loss + l2_reg
        else:
            forward_total_loss = torch.tensor(0.0)
        
        # Total loss
        loss = (policy_loss + 0.5 * term_loss + value_loss + effort_coef * effort_loss + 
                0.5 * exec_loss + entropy_loss + forward_coef * forward_total_loss)
        
        total_loss += loss.item()
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent large updates
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(forward_model.parameters()), max_norm=1.0)
        
        optimizer.step()
        
        # Update energy based on reward and effort
        energy = max(energy - actual_effort + reward, 0.1)
        
        # Update observation for next step
        if not done:
            obs = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0)
    
    # Convert position error map to a list for analysis
    position_pred_errors = [position_error_map.get(i, 0.0) for i in range(env.seq_len)]
    
    return total_reward, total_loss, index, vte_count, vte_trajectories, info_gains_history, position_pred_errors


def plot_vte_trajectories(vte_trajectories, info_gains_history):
    """Visualize VTE mental simulations with information gain"""
    if not vte_trajectories:
        print("No VTE trajectories to plot")
        return
    
    # Select up to 3 trajectories to visualize
    indices = [0, len(vte_trajectories)//2, len(vte_trajectories)-1]
    indices = [i for i in indices if i < len(vte_trajectories)]
    
    plt.figure(figsize=(15, 5*len(indices)))
    
    for plot_idx, traj_idx in enumerate(indices):
        trajectory = vte_trajectories[traj_idx]
        info_gains = info_gains_history[traj_idx] if traj_idx < len(info_gains_history) else []
        
        # Plot the observation states (state activation)
        plt.subplot(len(indices), 2, plot_idx*2 + 1)
        states = np.array([step[0][0] for step in trajectory])
        plt.imshow(states.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Activation')
        plt.title(f'VTE Mental Simulation {traj_idx+1}')
        plt.xlabel('Simulation Step')
        plt.ylabel('State Element')
        
        # Plot actions and information gain
        plt.subplot(len(indices), 2, plot_idx*2 + 2)
        actions = [step[1] for step in trajectory]
        plt.bar(range(len(actions)), actions, alpha=0.6, label='Actions')
        
        if info_gains:
            # Ensure info_gains matches the length of actions
            info_gains_padded = info_gains + [0] * (len(actions) - len(info_gains))
            plt.plot(range(len(info_gains_padded)), info_gains_padded, 'r-', linewidth=2, label='Information Gain')
        
        plt.title(f'VTE Actions & Information Gain {traj_idx+1}')
        plt.xlabel('Simulation Step')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/vte_visualization.png')
    plt.close()


def plot_vte_frequency_vs_learning(vte_counts, rewards, positions, pred_errors=None):
    """Plot how VTE frequency changes as learning progresses"""
    # Create a figure with 3 subplots if we have prediction errors
    num_plots = 3 if pred_errors is not None else 2
    plt.figure(figsize=(15, 5))
    
    # First subplot: VTE frequency over time
    plt.subplot(1, num_plots, 1)
    # Compute moving average for VTE counts
    window = min(50, len(vte_counts))
    vte_ma = [sum(vte_counts[max(0, i-window):i+1])/min(i+1, window) for i in range(len(vte_counts))]
    plt.plot(vte_ma, 'b-', linewidth=2)
    plt.title('VTE Frequency During Learning')
    plt.xlabel('Episode')
    plt.ylabel('VTE Events (Moving Average)')
    
    # Second subplot: Relationship between VTE and performance
    plt.subplot(1, num_plots, 2)
    # Apply more smoothing for visualization
    window = min(100, len(rewards))
    reward_ma = [sum(rewards[max(0, i-window):i+1])/min(i+1, window) for i in range(len(rewards))]
    
    # Create color-coded points based on training progress
    num_episodes = len(rewards)
    colors = np.zeros((num_episodes, 3))
    # Start with red (early in training)
    colors[:, 0] = 1.0
    # Transition to blue (late in training)
    for i in range(num_episodes):
        progress = i / num_episodes
        colors[i, 0] = max(0, 1 - 2*progress)  # Red decreases
        colors[i, 2] = min(1, 2*progress)      # Blue increases
    
    # Plot VTE frequency vs reward with color indicating training progress
    plt.scatter(vte_ma, reward_ma, c=colors, s=10)
    plt.title('VTE Frequency vs Performance')
    plt.xlabel('VTE Events (Moving Average)')
    plt.ylabel('Reward (Moving Average)')
    
    # Third subplot: Prediction errors and VTE frequency (if available)
    if pred_errors is not None and len(pred_errors) > 0:
        plt.subplot(1, num_plots, 3)
        # Smooth prediction errors
        window = min(50, len(pred_errors))
        pred_error_ma = [sum(pred_errors[max(0, i-window):i+1])/min(i+1, window) for i in range(len(pred_errors))]
        
        # Plot both prediction errors and VTE frequency
        plt.plot(pred_error_ma, 'r-', linewidth=2, label='Prediction Error')
        plt.plot(vte_ma, 'b-', linewidth=2, label='VTE Frequency')
        plt.title('Prediction Errors vs VTE')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/vte_vs_learning.png')
    plt.close()


def run_experiment(seq, topology, num_episodes=1000, use_vte=True, base_vte_threshold=0.3):
    """Run full experiment with visualization of results"""
    env = SequenceEnv(seq, topology)
    input_dim = env.seq_len + 2  # State dimension + previous action
    
    model = ActorCriticNet(input_dim)
    optimizer = optim.Adam(list(model.parameters()), lr=0.0005, weight_decay=0.0001)  # Added weight decay
    
    # Training metrics
    rewards = []
    vte_counts = []
    positions = []
    mean_pred_errors = []  # Store prediction error history
    position_errors = []   # Store position-specific errors
    
    all_vte_trajectories = []
    all_info_gains = []
    
    # Training loop
    for episode in range(num_episodes):
        total_reward, total_loss, final_pos, vte_count, vte_trajectories, info_gains, pos_errors = train_episode(
            env, model, optimizer, init_energy=10.0, 
            use_vte=use_vte, base_vte_threshold=base_vte_threshold)
        
        rewards.append(total_reward)
        vte_counts.append(vte_count)
        positions.append(final_pos)
        position_errors.append(pos_errors)
        
        # Calculate average prediction error for this episode
        mean_pred_error = np.mean(pos_errors) if pos_errors else 0.0
        mean_pred_errors.append(mean_pred_error)
        
        # Save VTE data
        all_vte_trajectories.extend(vte_trajectories)
        all_info_gains.extend(info_gains)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Effort: {total_effort} Position: {final_pos}, VTE Count: {vte_count}")
            # Visualize some VTE trajectories periodically
            if vte_count > 0 and episode % 200 == 0:
                plot_vte_trajectories(vte_trajectories, info_gains)
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(positions)
    plt.title('Final Positions')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    
    plt.subplot(1, 3, 3)
    plt.plot(vte_counts)
    plt.title('VTE Events per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('results/training_progress.png')
    plt.close()
    
    # Plot prediction errors over time
    plt.figure(figsize=(10, 5))
    plt.plot(mean_pred_errors)
    plt.title('Prediction Errors Over Training')
    plt.xlabel('Episode')
    plt.ylabel('Mean Prediction Error')
    plt.tight_layout()
    plt.savefig('results/prediction_errors.png')
    plt.close()
    
    # Plot position-specific prediction errors
    if len(position_errors) > 0 and len(position_errors[-1]) > 0:
        plt.figure(figsize=(10, 5))
        latest_pos_errors = position_errors[-1]
        plt.bar(range(len(latest_pos_errors)), latest_pos_errors)
        plt.title('Prediction Error by Position (Final Episode)')
        plt.xlabel('Position')
        plt.ylabel('Error')
        plt.tight_layout()
        plt.savefig('results/position_errors.png')
        plt.close()
    
    # Plot VTE frequency vs learning progress including prediction errors
    plot_vte_frequency_vs_learning(vte_counts, rewards, positions, mean_pred_errors)
    
    # Final visualization of representative VTE trajectories
    if all_vte_trajectories:
        # Select a few trajectories from different training stages
        sample_indices = [0, len(all_vte_trajectories)//4, len(all_vte_trajectories)//2, 
                          3*len(all_vte_trajectories)//4, len(all_vte_trajectories)-1]
        sample_indices = [i for i in sample_indices if i < len(all_vte_trajectories)]
        
        sample_trajectories = [all_vte_trajectories[i] for i in sample_indices]
        sample_info_gains = [all_info_gains[i] for i in sample_indices if i < len(all_info_gains)]
        
        plot_vte_trajectories(sample_trajectories, sample_info_gains)
    
    return model, forward_model, rewards, vte_counts, positions, mean_pred_errors, position_errors


def visualize_agent_state(env, episode, step_count, action, vte_active, position_errors):
    """Visualize the current state of the agent in real time"""
    plt.figure(figsize=(12, 6))
    plt.clf()
    
    # Create a bar chart for the reward sequence
    plt.subplot(2, 2, 1)
    plt.bar(range(len(env.seq)), env.seq, color='skyblue')
    plt.axvline(x=env.index, color='red', linestyle='-', linewidth=2)
    plt.title(f'Reward Structure - Position: {env.index}')
    plt.xlabel('Position')
    plt.ylabel('Reward')
    
    # Show the agent's state
    plt.subplot(2, 2, 2)
    status = "THINKING (VTE)" if vte_active else "MOVING" if action == 1 else "STOPPED"
    plt.text(0.5, 0.5, f"Episode: {episode}\nStep: {step_count}\nPosition: {env.index}\nStatus: {status}", 
             horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.axis('off')
    
    # Show prediction errors by position
    plt.subplot(2, 2, 3)
    if position_errors:
        plt.bar(range(len(position_errors)), position_errors, color='salmon')
        plt.title('Prediction Errors by Position')
        plt.xlabel('Position')
        plt.ylabel('Error')
    
    # Show the agent's trajectory
    plt.subplot(2, 2, 4)
    agent_positions = [0] * (env.index) + [1]  # 0s for past positions, 1 for current
    plt.plot(range(env.index + 1), agent_positions, 'ro-')
    plt.title('Agent Trajectory')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.yticks(range(len(env.seq)))
    
    plt.tight_layout()
    plt.pause(0.1)  # Pause for a short time to update the display

if __name__ == '__main__':
    # Define sequence with reward structure that requires exploration
    # This creates a non-monotonic reward landscape with a trap and delayed reward
    seq = [1, 3, 10, 11, 9, 20, 9, 20]  # Rewards
    topology = [0, 0, 4, 1, 1, 5, 8, 1]  # Topology affects effort
    
    print("Reward sequence:", seq)
    print("Topology:", topology)
    
    # Run experiment with VTE enabled
    model_vte, forward_model_vte, rewards_vte, vte_counts_vte, positions_vte, pred_errors_vte, position_errors_vte = run_experiment(
        seq, topology, num_episodes=1000, use_vte=True, base_vte_threshold=0.3)
    
    # Optional: Run comparison experiment without VTE
    model_no_vte, forward_model_no_vte, rewards_no_vte, vte_counts_no_vte, positions_no_vte, pred_errors_no_vte, position_errors_no_vte = run_experiment(
        seq, topology, num_episodes=1000, use_vte=False)
    
    # Compare VTE vs No-VTE performance
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_vte, label='With VTE')
    plt.plot(rewards_no_vte, label='Without VTE')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(positions_vte, label='With VTE')
    plt.plot(positions_no_vte, label='Without VTE')
    plt.title('Final Positions')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.legend()
    
    # Plot moving average of rewards for clearer comparison
    window = 50
    rewards_vte_ma = [sum(rewards_vte[max(0, i-window):i+1])/min(i+1, window) for i in range(len(rewards_vte))]
    rewards_no_vte_ma = [sum(rewards_no_vte[max(0, i-window):i+1])/min(i+1, window) for i in range(len(rewards_no_vte))]
    
    plt.subplot(1, 3, 3)
    plt.plot(rewards_vte_ma, label='With VTE')
    plt.plot(rewards_no_vte_ma, label='Without VTE')
    plt.title('Moving Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/vte_vs_no_vte_comparison.png')
    
    # Compare prediction errors vs VTE usage
    plt.figure(figsize=(10, 5))
    
    # Smooth the data for clearer visualization
    window = 50
    vte_ma = [sum(vte_counts_vte[max(0, i-window):i+1])/min(i+1, window) for i in range(len(vte_counts_vte))]
    pred_errors_ma = [sum(pred_errors_vte[max(0, i-window):i+1])/min(i+1, window) for i in range(len(pred_errors_vte))]
    
    plt.plot(vte_ma, label='VTE Frequency')
    plt.plot(pred_errors_ma, label='Prediction Errors')
    plt.title('VTE Usage vs Prediction Errors')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/vte_vs_prediction_errors.png')
    plt.show()