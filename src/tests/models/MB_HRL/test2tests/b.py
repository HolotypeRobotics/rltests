import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from environment import Environment
class Action:
    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2

class Agent(nn.Module):
    def __init__(self, input_dim, pos_pred_dim, hidden_dim=128, action_dim=3, head_dir_dim=4):
        """
        input_dim: dimensionality of the network input
        pos_pred_dim: number of classes for the position prediction (typically width*height)
        action_dim: number of discrete actions (turn left, turn right, move forward)
        head_dir_dim: number of compass directions (NORTH, EAST, SOUTH, WEST)
        """
        super(Agent, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        # Predict the next position
        self.fc_position = nn.Linear(hidden_dim, pos_pred_dim)
        # Predict the next head direction
        self.fc_head_dir = nn.Linear(hidden_dim, head_dir_dim)
        # Predict the information value
        self.fc_info_value = nn.Linear(hidden_dim, 1)
        # Temperature parameter for confidence calibration
        self.log_temperature = nn.Parameter(torch.zeros(1))
        # Set a confidence threshold for performing VTE
        self.conf_threshold = 0.9
    
    def forward(self, state_input, belief=None):
        """
        Inputs:
          - state_input: full state tensor from environment (batch, input_dim)
          - belief: optional belief vector (batch, belief_dim)
        """
        # Handle the input - state_input already contains position, direction, actions, etc.
        if belief is not None:
            x = torch.cat([state_input, belief], dim=-1)
        else:
            x = state_input
        
        # Prepare for GRU (add time dimension if needed)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Process through GRU
        out, _ = self.gru(x)
        h = out.squeeze(1)
        
        # Generate outputs
        action_logits = self.fc_actor(h)
        value = self.fc_critic(h)
        pos_logits = self.fc_position(h)
        head_logits = self.fc_head_dir(h)
        info_value = self.fc_info_value(h).squeeze(-1)

        # Apply temperature scaling for confidence calibration
        temperature = torch.exp(self.log_temperature)
        calibrated_action_logits = action_logits / temperature
        calibrated_pos_logits = pos_logits / temperature
        calibrated_head_logits = head_logits / temperature
        
        # Calculate confidence as maximum probability
        action_conf = F.softmax(calibrated_action_logits, dim=-1).max(dim=-1)[0]
        pos_conf = F.softmax(calibrated_pos_logits, dim=-1).max(dim=-1)[0]
        head_conf = F.softmax(calibrated_head_logits, dim=-1).max(dim=-1)[0]
        
        # Combined confidence: average of all confidences
        confidence = (action_conf + pos_conf + head_conf) / 3.0

        return action_logits, value, pos_logits, head_logits, confidence, info_value

    def VTE(self, state_input, top_k=2, belief=None):
        """
        VTE (vicarious trial-and-error) for simulating potential actions
        """
        with torch.no_grad():
            # Clone inputs to ensure we don't modify the originals
            state_input = state_input.clone().detach()
            
            if belief is not None:
                belief = belief.clone().detach()

            # Get action probabilities
            action_logits, _, _, _, _, _ = self.forward(
                state_input=state_input,
                belief=belief
            )
            
            probs = F.softmax(action_logits, dim=-1)
            
            # Get top_k candidate action indices
            top_options = torch.topk(probs, top_k, dim=-1).indices.squeeze(0)
            best_score = -float('inf')
            best_action = None

            # Evaluate each candidate action
            for option in top_options:
                # Form a one-hot encoding for the candidate option
                candidate = torch.zeros(1, action_logits.size(-1))
                candidate[0, option] = 1.0
                
                # This is a simplified approach since we can't easily modify the state
                # In a real implementation, we would want to simulate the effect of the action
                # on the state, but for now we'll just use the original state
                
                # Simulate taking this action
                _, sim_value, _, _, sim_confidence, sim_info_value = self.forward(
                    state_input=state_input,
                    belief=belief
                )
                
                # Combine value and information value, weighted by confidence
                score = (sim_value + sim_info_value) * sim_confidence
                
                if score > best_score:
                    best_score = score
                    best_action = option
                    
            return best_action

def train_hierarchical(env, num_episodes=1000):
    # Environment dimensions
    height, width = env.height, env.width
    pos_dim = height * width  # Total number of positions
    
    # Action space
    action_dim = 3  # TURN_LEFT, TURN_RIGHT, MOVE_FORWARD
    
    # Direction space
    head_dir_dim = 4  # NORTH, EAST, SOUTH, WEST
    
    # Belief dimension for high-level agent
    belief_dim = 2
    
    # Calculate input dimensions
    n_objects = env.n_objects
    bins_per_object = 3  # Assuming 3 bins per object for proximity
    
    # Base dimensions for components
    pos_dim_input = pos_dim
    dir_dim_input = head_dir_dim
    action_dim_input = action_dim
    obj_dim_input = n_objects * bins_per_object
    
    # Print dimensions for debugging
    print(f"pos_dim: {pos_dim}, dir_dim: {head_dir_dim}, action_dim: {action_dim}, obj_dim: {obj_dim_input}")
    
    # Verify against actual tensor sizes
    state, _, _, _ = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    print(f"State tensor size: {state_tensor.size()}")
    
    # Extract actual dimensions from the state tensor
    state_size = state_tensor.size(-1)
    
    # Use actual state size for input dimensions
    high_input_dim = state_size
    low_input_dim = state_size + belief_dim
    
    print(f"High input dim: {high_input_dim}, Low input dim: {low_input_dim}")
    
    # Initialize agents
    high_agent = Agent(input_dim=high_input_dim, pos_pred_dim=pos_dim, hidden_dim=128, action_dim=belief_dim)
    low_agent = Agent(input_dim=low_input_dim, pos_pred_dim=pos_dim, hidden_dim=128, action_dim=action_dim)
    
    # Initialize optimizers
    high_optimizer = optim.Adam(high_agent.parameters(), lr=1e-2)
    low_optimizer = optim.Adam(low_agent.parameters(), lr=1e-2)
    
    # Training parameters
    gamma = 0.99
    pos_loss_weight = 1.0
    info_loss_weight = 0.5
    policy_loss_weight = 0.45
    error_threshold = 1.0  # Threshold for triggering a task switch
    
    # Storage for episode statistics
    episode_rewards = []
    episode_steps = []
    episode_positions = []
    
    for episode in range(num_episodes):
        # Reset environment
        state, reward, effort, done = env.reset()
        
        # Convert numpy arrays to PyTorch tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Initialize belief for high-level agent
        belief = torch.zeros(1, belief_dim)
        
        # Episode statistics
        vte_freq = 0
        cumulative_reward_episode = 0.0
        steps = 0
        episode_done = False
        
        # Store positions visited
        pos_log = [env.state]
        
        # Outer loop: entire episode
        while not episode_done:
            # ----- High-Level: Produce a new belief -----
            high_logits, high_value, high_pos_logits, high_head_logits, high_conf, high_info_value = high_agent(
                state_input=state_tensor
            )
            
            # Use VTE if confidence is below threshold
            if high_conf.item() < high_agent.conf_threshold:
                chosen_high_action = high_agent.VTE(
                    state_input=state_tensor,
                    top_k=2
                )
                vte_freq += 1
            else:
                chosen_high_action = torch.argmax(high_logits)
                
            # Convert chosen high-level action to one-hot belief
            new_belief = torch.zeros_like(high_logits)
            new_belief[0, chosen_high_action] = 1.0
            belief = new_belief
            
            segment_reward = 0.0
            task_switch = False
            low_done = False
            prev_low_conf = None
            
            # ----- Low-Level Nested Loop -----
            while not low_done and not task_switch:
                low_logits, low_value, low_pos_logits, low_dir_logits, low_conf, low_voi = low_agent(
                    state_input=state_tensor,
                    belief=belief
                )
                
                low_probs = F.softmax(low_logits, dim=-1)
                
                # Choose action with VTE if confidence is low
                if low_conf.item() < low_agent.conf_threshold:
                    chosen_action = low_agent.VTE(
                        state_input=state_tensor,
                        belief=belief, 
                        top_k=2
                    )
                    vte_freq += 1
                else:
                    chosen_action = torch.distributions.Categorical(low_probs).sample()
                    
                low_log_prob = F.log_softmax(low_logits, dim=-1).squeeze(0)[chosen_action]
                
                # Take a step in the environment
                next_state, reward, effort, env_terminate = env.step(chosen_action.item())
                
                # Convert the next state to PyTorch tensors
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                
                # Update statistics
                segment_reward += reward - effort  # Apply effort penalty
                cumulative_reward_episode += reward - effort
                steps += 1
                pos_log.append(env.state)
                
                # Calculate next value for advantage computation
                if not env_terminate:
                    _, next_low_value, _, _, _, _ = low_agent(
                        state_input=next_state_tensor,
                        belief=belief
                    )
                else:
                    next_low_value = torch.zeros(1)
                
                # Compute advantage and losses
                net_reward = reward - effort
                low_advantage = net_reward + gamma * next_low_value - low_value
                low_policy_loss = -low_log_prob * low_advantage.detach()
                low_value_loss = low_advantage.pow(2)
                
                # Calculate position and direction targets 
                # Extract the position one-hot from the next state
                # Assuming the first env.height * env.width elements are the position one-hot
                pos_dim = env.height * env.width
                head_dir_dim = 4
                
                # Extract targets from next state tensor
                pos_one_hot = next_state_tensor[:, :pos_dim]
                dir_one_hot = next_state_tensor[:, pos_dim:pos_dim+head_dir_dim]
                
                pos_target = torch.argmax(pos_one_hot, dim=1)
                pos_loss = F.cross_entropy(low_pos_logits, pos_target)
                
                head_target = torch.argmax(dir_one_hot, dim=1)
                head_loss = F.cross_entropy(low_dir_logits, head_target)
                
                low_total_state_loss = pos_loss + head_loss
                
                # Calibrate confidence
                pred_index = torch.argmax(low_pos_logits, dim=1)
                correct = (pred_index == pos_target).float()
                calibration_loss = F.binary_cross_entropy(low_conf, correct)
                low_total_state_loss += calibration_loss
                
                # Compute information value loss
                if prev_low_conf is not None:
                    low_info_gain = torch.abs(low_conf - prev_low_conf)
                    if low_info_gain.dim() == 0:
                        low_info_gain = low_info_gain.unsqueeze(0)
                else:
                    low_info_gain = torch.tensor([0.0])
                    
                low_info_loss = F.mse_loss(low_voi, low_info_gain)
                
                # Combine all losses
                low_loss = (
                    policy_loss_weight * low_policy_loss + 
                    low_value_loss + 
                    pos_loss_weight * low_total_state_loss + 
                    info_loss_weight * low_info_loss
                )
                
                # Update low-level agent
                low_optimizer.zero_grad()
                low_loss.backward()
                low_optimizer.step()
                
                # Store confidence for information gain calculation
                prev_low_conf = low_conf.detach()
                
                # Check for task switch conditions
                if pos_loss.item() > error_threshold and low_conf.item() > high_conf.item():
                    task_switch = True
                
                # Check for episode termination
                if env_terminate:
                    low_done = True
                    episode_done = True
                
                # Prepare for next step
                state_tensor = next_state_tensor
            
            # ----- High-Level Update (per segment) -----
            # Calculate next high-level value for advantage computation
            _, next_high_value, _, _, _, _ = high_agent(
                state_input=state_tensor
            )
            
            # Compute advantage and losses
            high_advantage = segment_reward + gamma * next_high_value - high_value
            high_log_prob = torch.distributions.Categorical(F.softmax(high_logits, dim=-1)).log_prob(chosen_high_action)
            high_policy_loss = -high_log_prob * high_advantage.detach()
            high_value_loss = high_advantage.pow(2)
            
            # Calculate position and direction targets for high-level
            # Extract position and direction one-hot vectors
            pos_dim = env.height * env.width
            head_dir_dim = 4
            
            pos_one_hot = state_tensor[:, :pos_dim]
            dir_one_hot = state_tensor[:, pos_dim:pos_dim+head_dir_dim]
            
            pos_target_high = torch.argmax(pos_one_hot, dim=1)
            high_pos_loss = F.cross_entropy(high_pos_logits, pos_target_high)
            
            head_target_high = torch.argmax(dir_one_hot, dim=1)
            high_head_loss = F.cross_entropy(high_head_logits, head_target_high)
            
            high_total_state_loss = high_pos_loss + high_head_loss
            
            # Calibrate high-level confidence
            high_pred_index = torch.argmax(high_pos_logits, dim=1)
            high_correct = (high_pred_index == pos_target_high).float()
            high_calib_loss = F.binary_cross_entropy(high_conf, high_correct)
            high_total_state_loss += high_calib_loss
            
            # Compute information value loss for high-level
            high_info_gain = torch.abs(high_conf - low_conf.detach())
            high_info_loss = F.mse_loss(high_info_value, high_info_gain)
            
            # Combine all high-level losses
            high_loss = (
                policy_loss_weight * high_policy_loss + 
                high_value_loss + 
                pos_loss_weight * high_total_state_loss + 
                info_loss_weight * high_info_loss
            )
            
            # Update high-level agent
            high_optimizer.zero_grad()
            high_loss.backward()
            high_optimizer.step()
        
        # Store episode statistics
        episode_rewards.append(cumulative_reward_episode)
        episode_steps.append(steps)
        episode_positions.append(pos_log)
        
        # Print progress periodically
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Reward: {cumulative_reward_episode:.2f}, "
                  f"Steps: {steps}, "
                  f"Final Pos: {env.state}, "
                  f"High Conf: {high_conf.item():.3f}, "
                  f"Low Conf: {low_conf.item():.3f}, "
                  f"VTE Freq: {vte_freq}")
    
    # Return the trained agents and statistics
    return high_agent, low_agent, episode_rewards, episode_steps, episode_positions

def setup_environment(width=5, height=5, n_objects=3):
    # Create environment with specified dimensions
    env = Environment(width=width, height=height, n_efforts=np.ones((height, width)), n_objects=n_objects)
    
    # Set rewards
    for x in range(width):
        for y in range(height):
            if (y, x) == env.exit:
                env.set_reward(x, y, 10.0)  # High reward at exit
            elif env.objects[y, x] > 0:
                env.set_reward(x, y, 2.0)   # Medium reward at objects
            else:
                env.set_reward(x, y, 0.0)   # No reward elsewhere
    
    # Set higher effort in some cells
    for x in range(width):
        for y in range(height):
            if y > height // 2 and x < width // 2:
                env.set_effort(x, y, 2.0)  # Higher effort in bottom-left quadrant
    
    return env

def run_and_visualize(env, high_agent, low_agent, num_episodes=5):
    for episode in range(num_episodes):
        state, _, _, _ = env.reset()
        env.reset_path()
        
        # Convert state to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Initialize belief
        belief = torch.zeros(1, 2)
        done = False
        
        while not done:
            # Get high-level belief
            high_logits, _, _, _, high_conf, _ = high_agent(
                state_input=state_tensor
            )
            
            if high_conf.item() < high_agent.conf_threshold:
                chosen_high_action = high_agent.VTE(
                    state_input=state_tensor,
                    top_k=2
                )
            else:
                chosen_high_action = torch.argmax(high_logits)
                
            # Convert to one-hot belief
            new_belief = torch.zeros_like(high_logits)
            new_belief[0, chosen_high_action] = 1.0
            belief = new_belief
            
            # Get low-level action
            low_logits, _, _, _, low_conf, _ = low_agent(
                state_input=state_tensor,
                belief=belief
            )
            
            if low_conf.item() < low_agent.conf_threshold:
                chosen_action = low_agent.VTE(
                    state_input=state_tensor,
                    belief=belief, 
                    top_k=2
                )
            else:
                low_probs = F.softmax(low_logits, dim=-1)
                chosen_action = torch.distributions.Categorical(low_probs).sample()
            
            # Take a step in the environment
            next_state, reward, effort, done = env.step(chosen_action.item())
            print(f"Action: {env.action_to_string(chosen_action.item())}, Reward: {reward}, Effort: {effort}")
            
            # Update state tensor
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            if env.state == env.exit or len(env.path) > 20:  # Limit path length
                done = True
        
        # Visualize the path
        print(f"Episode {episode+1} completed. Total reward: {sum([env.rewards[y, x] for y, x in env.path])}")
        env.plot_path(None)  # We don't need to pass in agent for just visualizing rewards

def main():
    # Set up environment
    env = setup_environment(width=5, height=5, n_objects=3)
    
    # Train agents
    high_agent, low_agent, rewards, steps, positions = train_hierarchical(env, num_episodes=1000)
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title('Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.show()
    
    # Run and visualize a few episodes
    run_and_visualize(env, high_agent, low_agent, num_episodes=3)

if __name__ == "__main__":
    main()