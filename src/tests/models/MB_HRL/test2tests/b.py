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
    """Environment with a sequence of states, rewards, and effort costs"""
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
        """Create one-hot encoding of position (place cell representation)"""
        pos_onehot = np.zeros(self.seq_len, dtype=np.float32)
        pos_onehot[self.index] = 1.0
        obs = np.concatenate([pos_onehot, self.prev_action])
        return obs

    def step(self, action, terminate=False): # Ensure terminate=False default and use terminate arg
        """Take a step in the environment"""
        if self.done:
            raise RuntimeError("Episode already terminated.")
        effort = 0.0001
        reward = 0.0
        if action == 1 and self.index < self.max_index:
            prev_topology = self.topology[self.index]
            prev_reward = self.seq[self.index]
            self.index += 1
            effort = self.topology[self.index] - prev_topology
            reward = self.seq[self.index] - prev_reward
        self.prev_action = np.eye(2, dtype=np.float32)[action]
        if terminate or self.index == self.max_index: # Use the terminate signal
            self.done = True
        next_obs = self._get_obs() if not self.done else None
        return next_obs, reward, effort, self.done, self.index


class NeuralVTEAgent(nn.Module):
    """
    Neural VTE agent with bio-inspired computations:
    - Hippocampal-like place representations
    - Striatum-like evidence accumulation
    - PFC-like action evaluation
    - BG-like action gating
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(NeuralVTEAgent, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Policy and value (OFC/vmPFC/Accumbens)
        self.policy_head = nn.Linear(hidden_dim, 1)  # Action selection (move=1)
        # Termination probability head
        self.term_head = nn.Linear(hidden_dim, 1)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.effort_head = nn.Linear(hidden_dim, 1)


        # Forward model (Hippocampal-PFC predictive code)
        self.forward_net = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

        # Uncertainty estimation (mPFC/OFC uncertainty tracking)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

        # Meta-control (PFC-BG evaluation)
        self.evidence_threshold = nn.Parameter(torch.tensor(2.0))  # Base threshold
        self.uncertainty_factor = nn.Parameter(torch.tensor(0.5))  # Threshold modulation by uncertainty
        self.reward_factor = nn.Parameter(torch.tensor(0.2))       # Threshold modulation by reward

        # Computational resource allocation
        self.max_computation = 10.0
        self.base_temperature = 0.1  # For softmax-based decision making


    def get_uncertainty(self, features):
        """Estimate uncertainty (analogous to mPFC confidence)"""
        # Direct uncertainty estimate
        uncertainty = torch.sigmoid(self.uncertainty_head(features))

        # Also incorporate policy entropy
        logits = self.policy_head(features)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)

        # Combined uncertainty
        combined_uncertainty = (uncertainty + 0.2 * entropy)
        return combined_uncertainty

    def forward(self, state):
        """Forward pass through model: features, policy, value, uncertainty"""
        features = F.relu(self.fc1(state))
        logits, value = self.get_policy_value(features)
        uncertainty = self.get_uncertainty(features)
        term_prob = torch.sigmoid(self.term_head(features)) # Add termination prob output
        return logits, value, term_prob, uncertainty, features # Modified return

    def predict_next(self, features, action_onehot):
        """Predict next state, reward and effort (hippocampal forward sweep)"""
        inputs = torch.cat([features, action_onehot], dim=1)
        forward_features = self.forward_net(inputs)
        next_state = self.next_state_head(forward_features)
        reward = self.reward_head(forward_features)
        effort = F.softplus(self.effort_head(forward_features))
        return next_state, reward, effort

    def wasserstein_distance(self, belief1, belief2):
        """
        Calculate Wasserstein distance between belief distributions
        (measure of belief update magnitude)
        """
        # Sort beliefs (to simplify Wasserstein calculation for 1D)
        sorted1, _ = torch.sort(belief1, dim=1)
        sorted2, _ = torch.sort(belief2, dim=1)

        # Calculate 2-Wasserstein distance
        wasserstein = torch.sqrt(torch.sum((sorted1 - sorted2) ** 2, dim=1))

        return wasserstein

    def accumulate_evidence(self, prior_evidence, value_diff, uncertainty, precision=1.0):
        """
        Accumulate evidence for action selection using Bayesian updating
        (striatal evidence accumulation)
        """
        # Scale precision by uncertainty (less certain, less precise)
        scaled_precision = precision / (uncertainty.item() + 0.1)

        # Calculate log likelihood ratio
        log_lr = scaled_precision * value_diff

        # Update evidence (log posterior odds)
        updated_evidence = prior_evidence + log_lr

        return updated_evidence

    def calculate_threshold(self, uncertainty, expected_reward):
        """
        Calculate dynamic threshold for action execution
        (BG threshold adjustment)
        """
        # Base threshold
        threshold = self.evidence_threshold

        # Modulate by uncertainty (higher uncertainty -> higher threshold)
        threshold = threshold + self.uncertainty_factor * uncertainty

        # Modulate by reward magnitude (higher reward -> lower threshold)
        threshold = threshold - self.reward_factor * expected_reward

        return threshold

    def allocate_computation(self, actions, probs, uncertainty, total_budget):
        """
        Allocate computational resources across potential actions
        (PFC resource-rational computation)
        """
        n_actions = len(actions)

        # Information-theoretic allocation based on entropy and probability
        uncertainties = uncertainty.expand(n_actions, 1).detach().numpy()
        probabilities = probs[0, actions].detach().numpy()

        # Calculate value of computation for each action
        # Higher probability and higher uncertainty -> more computation
        voc_values = []
        for i in range(n_actions):
            prob = probabilities[i]
            uncert = uncertainties[i][0]

            # Balance exploitation (high prob) with exploration (high uncertainty)
            exploration = -prob * np.log(prob + 1e-8)  # Entropy-weighted
            exploitation = prob * uncert               # Probability-weighted uncertainty

            voc = 0.4 * exploration + 0.6 * exploitation
            voc_values.append(float(voc))

        # Normalize to sum to total budget
        total_voc = sum(voc_values) if sum(voc_values) > 0 else 1.0
        allocations = [v * total_budget / total_voc for v in voc_values]

        return allocations

    def vte_rollout(self, state, max_depth=5, discount=0.95):
        """
        Perform VTE rollout - sequential action evaluation
        (hippocampal forward sweeps with PFC/BG evaluation)
        """
        # Initial state evaluation
        logits, initial_value, term_prob, uncertainty, features = self.forward(state) # Modified forward call
        probs = F.softmax(logits, dim=1)
        initial_belief = self.get_belief(features)
        term_prob_val = term_prob.item() # Get termination probability value

        # Allocate computational budget across actions (include action 0 and 1)
        actions = list(range(probs.size(1))) # actions = [0, 1]
        computation_budgets = self.allocate_computation(
            actions, probs, uncertainty, self.max_computation
        )

        # Track best action and evidence
        best_action = None
        best_value = initial_value.item()
        best_trajectory = []
        best_info_gains = []

        # Evidence accumulation for each action (striatal integration)
        action_evidence = {}

        # Sequentially evaluate each action with its allocated budget
        for action_idx, action_budget in zip(actions, computation_budgets):
            if action_budget < 0.5:  # Skip if budget too small
                continue

            # Initialize trajectory for this action
            branch_obs = state.clone()
            branch_features = features.clone()
            branch_belief = initial_belief.clone()

            # Initialize metrics for this action
            branch_rewards = []
            branch_efforts = []
            branch_info_gains = []
            branch_values = []
            branch_uncertainties = []
            branch_depth = 0
            branch_done = False
            branch_trajectory = []

            # Initialize evidence (log odds) for this action
            evidence = 0.0

            # Action one-hot encoding
            action_onehot = F.one_hot(torch.tensor([action_idx]), num_classes=2).float()

            # Simulate trajectory for this action
            while branch_depth < max_depth and not branch_done and action_budget > 0.5:
                # Predict next state, reward, effort
                next_obs, reward, effort = self.predict_next(branch_features, action_onehot)
                next_features = self.get_features(next_obs)
                next_belief = self.get_belief(next_features)

                # Get predicted value, uncertainty, and termination prob for next state
                next_logits, next_value, next_term_prob, next_uncertainty, _ = self.forward(next_obs) # Get term_prob
                next_term_prob_val = next_term_prob.item() # Get termination prob value


                # Calculate information gain (belief change)
                info_gain = self.wasserstein_distance(next_belief, branch_belief).item()

                # Accumulate trajectory metrics
                branch_rewards.append(reward.item())
                branch_efforts.append(effort.item())
                branch_info_gains.append(info_gain)
                branch_values.append(next_value.item())
                branch_uncertainties.append(next_uncertainty.item())
                branch_depth += 1
                action_budget -= 1.0  # Computational cost

                # Track trajectory
                current_action = action_idx if branch_depth == 1 else torch.argmax(next_logits, dim=1).item()
                branch_trajectory.append((next_obs.detach().numpy(), current_action, info_gain))

                # Calculate discounted value so far
                total_reward = sum([r * (discount ** i) for i, r in enumerate(branch_rewards)])
                total_effort = sum([e * (discount ** i) for i, e in enumerate(branch_efforts)])
                total_info_gain = sum(branch_info_gains)

                # Integrated value (reward + information - effort)
                branch_value = total_reward + 0.2 * total_info_gain - 0.8 * total_effort
                branch_uncertainty_val = next_uncertainty.item()
                threshold = self.calculate_threshold(branch_uncertainty_val, total_reward) # Moved threshold calculation here

                # Accumulate evidence for this action
                value_diff = branch_value - best_value
                evidence = self.accumulate_evidence(
                    evidence, value_diff, torch.tensor(branch_uncertainty_val))

                # Check if we should terminate this branch based on predicted termination probability
                # OR based on evidence.  Prioritize explicit termination prediction for now.
                if next_term_prob_val > 0.5: # Terminate if term_prob > 0.5 in rollout
                    branch_done = True
                    break
                if evidence < -threshold or (evidence > threshold and branch_depth == 1): # Keep evidence thresholding as backup
                    branch_done = True
                    break


                # Update best action if this branch is better
                if branch_value > best_value:
                    best_action = action_idx
                    best_value = branch_value
                    best_trajectory = branch_trajectory.copy()
                    best_info_gains = branch_info_gains.copy()

                # Choose next action within this branch based on policy
                next_action = torch.argmax(next_logits, dim=1).item()
                action_onehot = F.one_hot(torch.tensor([next_action]), num_classes=2).float()

                # Update for next step
                branch_obs = next_obs
                branch_features = next_features
                branch_belief = next_belief

            # Store final evidence for this action
            action_evidence[action_idx] = evidence

        # If no action was evaluated (all budgets too small), use policy
        if best_action is None:
            logits, _, _, _, _ = self.forward(state) # Get logits again to sample policy if VTE failed
            probs = F.softmax(logits, dim=1)
            best_action = torch.argmax(probs, dim=1).item()

        # Calculate deliberation value (improvement over baseline)
        logits, initial_value, _, _, _ = self.forward(state) # Get initial value again
        deliberation_value = best_value - initial_value.item()

        # Calculate execution probability based on evidence and uncertainty
        # (BG action gating)
        logits, _, term_prob, uncertainty, features = self.forward(state) # Get term_prob and uncertainty again for gating
        uncertainty_val = uncertainty.item()
        if best_action in action_evidence:
            best_evidence = action_evidence[best_action]
            execute_prob = torch.sigmoid(
                torch.tensor((best_evidence - self.evidence_threshold) /
                           max(0.1, uncertainty_val))
            ).item()
        else:
            execute_prob = 0.9  # Default high probability if using policy

        # Return chosen action, whether deliberation helped, and trajectory info
        return best_action, deliberation_value > 0.0, best_info_gains, best_trajectory, execute_prob



def train_agent(env, agent, optimizer, num_episodes=1000, use_vte=True):
    """Train agent with VTE or without VTE"""
    rewards_history = []
    positions_history = []
    vte_count_history = []
    surprise_history = []

    all_vte_trajectories = []
    all_vte_info_gains = []

    # Track surprise at each position (prediction error by location)
    position_surprise = np.zeros(env.seq_len)
    surprise_count = np.zeros(env.seq_len)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False

        episode_reward = 0
        episode_effort = 0
        episode_vte_count = 0
        episode_steps = 0

        # Storage for previous prediction
        prev_state_pred = None
        position_specific_errors = np.zeros(env.seq_len)

        while not done:
            # Get policy, value, and uncertainty estimates
            logits, value, term_prob, uncertainty, features = agent.forward(state) # Modified forward call

            # Calculate prediction error if we have a previous prediction
            prediction_error = 0.0
            if prev_state_pred is not None:
                prediction_error = F.mse_loss(prev_state_pred, state).item()
                position_specific_errors[env.index] = prediction_error

            # Decide whether to use VTE
            use_vte_now = False
            if use_vte:
                # Either use VTE when uncertainty is high or prediction error is high
                uncertainty_val = uncertainty.item()
                vte_threshold = 0.3 - 0.5 * prediction_error  # Lower threshold when surprised
                use_vte_now = uncertainty_val > vte_threshold

            terminate_action = False # Initialize terminate_action
            # Action selection
            if use_vte_now:
                # Perform VTE rollout (mental simulation)
                action, is_beneficial, info_gains, trajectory, execute_prob = agent.vte_rollout(state)

                if is_beneficial:
                    # VTE was helpful - use its recommendation
                    episode_vte_count += 1
                    all_vte_trajectories.append(trajectory)
                    all_vte_info_gains.append(info_gains)

                    # Print VTE details
                    if episode % 50 == 0:
                        print(f"Episode {episode}, Step {episode_steps}: Used VTE at position {env.index}, "
                              f"chose action {action}, exec_prob {execute_prob:.2f}")
                else:
                    # VTE wasn't helpful - sample action based on policy AND termination probability
                    logits, value, term_prob, uncertainty, features = agent.forward(state)
                    probs = F.softmax(logits, dim=1)
                    term_dist = torch.distributions.Bernoulli(term_prob) # Sample termination
                    term_sample = term_dist.sample()
                    terminate_action = term_sample.item() > 0.5 # Decide to terminate or not
                    if terminate_action:
                        action = 0 # Stay action (terminate)
                    else:
                        action = torch.multinomial(probs, 1).item() # Sample from policy


            else: # No VTE case
                # Sample action based on policy AND termination probability
                logits, value, term_prob, uncertainty, features = agent.forward(state)
                probs = F.softmax(logits, dim=1)
                term_dist = torch.distributions.Bernoulli(term_prob) # Sample termination
                term_sample = term_dist.sample()
                terminate_action = term_sample.item() > 0.5 # Decide to terminate or not
                if terminate_action:
                    action = 0 # Stay action (terminate)
                else:
                    action = torch.multinomial(probs, 1).item() # Sample from policy


            # Make predictions for current state
            with torch.no_grad():
                action_onehot = F.one_hot(torch.tensor([action]), num_classes=2).float()
                next_state_pred, reward_pred, effort_pred = agent.predict_next(features, action_onehot)

            # Execute action in environment (pass terminate=terminate_action to env.step)
            next_state, reward, effort, done, position = env.step(action, terminate=terminate_action) # Pass terminate signal
            episode_reward += reward
            episode_effort += effort
            episode_steps += 1

            # Calculate surprise for this position
            if prev_state_pred is not None:
                surprise = F.mse_loss(prev_state_pred, state).item()
                position_surprise[position] = 0.9 * position_surprise[position] + 0.1 * surprise
                surprise_count[position] += 1

            # Learn from experience
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                next_logits, next_value, next_term_prob, _, _ = agent.forward(next_state_tensor) # Get next_term_prob
                next_value = next_value # No .item() here, we need the tensor for discount calculation in dynamic discount case

                # TD target with effort cost (using fixed discount)
                target = reward - effort + 0.95 * next_value.item() # Fixed discount
                # Dynamic discount case (if you want to implement dynamic discount later):
                # discount = torch.exp(-effort_pred.squeeze() / energy) # Example dynamic discount
                # target = reward - effort + discount * next_value.item()


            else:
                target = reward - effort

            # Calculate losses
            value_loss = F.mse_loss(value, torch.tensor([[target]]))

            # Policy gradient loss with advantage
            advantage = target - value.item()
            probs = F.softmax(logits, dim=1)
            log_prob = torch.log(probs.gather(1, torch.tensor([[action]])))
            policy_loss = -log_prob * advantage

            # Termination loss - use binary cross-entropy or similar.
            term_target = torch.tensor([[float(terminate_action)]]) # Target is 1 if terminated, 0 otherwise
            term_loss_fn = nn.BCELoss() # Binary Cross Entropy Loss
            term_loss = term_loss_fn(term_prob, term_target) # Calculate termination loss


            # Uncertainty loss - should be high when prediction error is high
            uncertainty_target = torch.tensor([[min(1.0, prediction_error * 5)]])
            uncertainty_loss = F.mse_loss(uncertainty, uncertainty_target)

            # Forward model losses (next state prediction)
            if not done:
                forward_loss = F.mse_loss(next_state_pred, next_state_tensor)
                reward_loss = F.mse_loss(reward_pred, torch.tensor([[reward]]))
                effort_loss = F.mse_loss(effort_pred, torch.tensor([[effort]]))
                predict_loss = forward_loss + reward_loss + effort_loss
            else:
                predict_loss = torch.tensor(0.0)

            # Entropy regularization for exploration
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            entropy_loss = -0.01 * entropy

            # Total loss (add term_loss)
            loss = policy_loss + term_loss + value_loss + 0.5 * uncertainty_loss + predict_loss + entropy_loss

            # Update model
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            # Save prediction for next step
            if not done:
                prev_state_pred = next_state_pred.detach()
                state = next_state_tensor

        # End of episode - record metrics
        rewards_history.append(episode_reward)
        positions_history.append(position)
        vte_count_history.append(episode_vte_count)

        # Calculate average surprise for this episode
        mean_surprise = np.sum(position_surprise) / np.sum(surprise_count) if np.sum(surprise_count) > 0 else 0
        surprise_history.append(mean_surprise)

        # Log progress
        if episode % 50 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Effort = {episode_effort:.2f}, "
                  f"Position = {position}, VTE Count = {episode_vte_count}")

        # Step the scheduler
        scheduler.step()

    return {
        'rewards': rewards_history,
        'positions': positions_history,
        'vte_counts': vte_count_history,
        'surprise': surprise_history,
        'vte_trajectories': all_vte_trajectories,
        'vte_info_gains': all_vte_info_gains,
        'position_surprise': position_surprise
    }


def plot_vte_trajectories(vte_trajectories, info_gains_history):
    """Visualize sample VTE trajectories"""
    if not vte_trajectories:
        print("No VTE trajectories to plot")
        return

    # Select representative trajectories
    indices = [0, len(vte_trajectories)//2, len(vte_trajectories)-1]
    indices = [i for i in indices if i < len(vte_trajectories)]

    plt.figure(figsize=(15, 5 * len(indices)))

    for idx, traj_idx in enumerate(indices):
        trajectory = vte_trajectories[traj_idx]
        info_gains = info_gains_history[traj_idx] if traj_idx < len(info_gains_history) else []

        # Plot state activations
        plt.subplot(len(indices), 2, idx*2 + 1)
        states = np.array([step[0][0] for step in trajectory])
        plt.imshow(states.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Activation')
        plt.title(f'VTE Trajectory {traj_idx+1}')
        plt.xlabel('Simulation Step')
        plt.ylabel('State Element')

        # Plot actions and info gain
        plt.subplot(len(indices), 2, idx*2 + 2)
        actions = [step[1] for step in trajectory]
        plt.bar(range(len(actions)), actions, alpha=0.6, label='Actions')

        if info_gains:
            plt.plot(range(len(info_gains)), info_gains, 'r-', linewidth=2, label='Info Gain')

        plt.title(f'VTE Actions & Info Gain {traj_idx+1}')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.savefig('results/vte_trajectories.png')
    plt.close()


def plot_training_results(results_vte, results_no_vte=None):
    """Plot training results with comparison if no-VTE results provided"""

    # Determine number of plots needed
    num_plots = 3 if results_no_vte else 2
    fig_width = 15 if results_no_vte else 12

    plt.figure(figsize=(fig_width, 10))

    # Plot rewards
    plt.subplot(2, num_plots, 1)
    plt.plot(results_vte['rewards'], label='With VTE')
    if results_no_vte:
        plt.plot(results_no_vte['rewards'], label='Without VTE')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Plot positions
    plt.subplot(2, num_plots, 2)
    plt.plot(results_vte['positions'], label='With VTE')
    if results_no_vte:
        plt.plot(results_no_vte['positions'], label='Without VTE')
    plt.title('Final Positions')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.legend()

    # Plot VTE frequency
    plt.subplot(2, num_plots, 3)
    plt.plot(results_vte['vte_counts'], label='VTE Count')
    plt.title('VTE Events per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Count')

    # Plot surprise and VTE relationship
    plt.subplot(2, num_plots, 4)
    window = min(50, len(results_vte['surprise']))
    surprise_ma = [sum(results_vte['surprise'][max(0, i-window):i+1])/min(i+1, window)
                  for i in range(len(results_vte['surprise']))]
    vte_ma = [sum(results_vte['vte_counts'][max(0, i-window):i+1])/min(i+1, window)
             for i in range(len(results_vte['vte_counts']))]

    plt.plot(surprise_ma, label='Surprise')
    plt.plot(vte_ma, label='VTE Frequency')
    plt.title('Surprise vs VTE Frequency')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()

    # Plot final surprise by position
    plt.subplot(2, num_plots, 5)
    positions = np.arange(len(results_vte['position_surprise']))
    plt.bar(positions, results_vte['position_surprise'])
    plt.title('Surprise by Position')
    plt.xlabel('Position')
    plt.ylabel('Surprise')

    # Plot VTE vs Performance
    if results_no_vte:
        plt.subplot(2, num_plots, 6)
        # Apply smoothing for better visualization
        window = 50
        vte_rewards_ma = [sum(results_vte['rewards'][max(0, i-window):i+1])/min(i+1, window)
                        for i in range(len(results_vte['rewards']))]
        no_vte_rewards_ma = [sum(results_no_vte['rewards'][max(0, i-window):i+1])/min(i+1, window)
                           for i in range(len(results_no_vte['rewards']))]

        plt.plot(vte_rewards_ma, label='With VTE')
        plt.plot(no_vte_rewards_ma, label='Without VTE')
        plt.title('Moving Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_results.png')
    plt.close()


def run_experiment(seq, topology, num_episodes=1000):
    """Run experiment with and without VTE"""
    # Initialize environment
    env = SequenceEnv(seq, topology)
    input_dim = env.seq_len + 2  # State + prev action

    # Create agent with VTE
    agent_vte = NeuralVTEAgent(input_dim)
    optimizer_vte = optim.Adam(agent_vte.parameters(), lr=0.001, weight_decay=1e-5)

    # Train with VTE
    print("Training agent with VTE...")
    results_vte = train_agent(env, agent_vte, optimizer_vte, num_episodes=num_episodes, use_vte=True)

    # Create agent without VTE
    agent_no_vte = NeuralVTEAgent(input_dim)
    optimizer_no_vte = optim.Adam(agent_no_vte.parameters(), lr=0.001, weight_decay=1e-5)

    # Train without VTE
    print("Training agent without VTE...")
    results_no_vte = train_agent(env, agent_no_vte, optimizer_no_vte, num_episodes=num_episodes, use_vte=False)

    # Visualize VTE trajectories
    if results_vte['vte_trajectories']:
        plot_vte_trajectories(results_vte['vte_trajectories'], results_vte['vte_info_gains'])

    # Plot training results
    plot_training_results(results_vte, results_no_vte)

    return agent_vte, agent_no_vte, results_vte, results_no_vte


if __name__ == "__main__":
    # Define sequence with reward structure that requires exploration
    # This creates a reward landscape with a non-monotonic structure
    # and a "trap" followed by a large reward (similar to detour problems)
    seq = [1, 3, 10, 11, 9, 20, -100, 0]  # Rewards
    topology = [0, 0, 4, 1, 1, 5, 100, 2]  # Topology affects effort

    print("Reward sequence:", seq)
    print("Topology sequence:", topology)

    # Run experiment
    agent_vte, agent_no_vte, results_vte, results_no_vte = run_experiment(seq, topology, num_episodes=500)