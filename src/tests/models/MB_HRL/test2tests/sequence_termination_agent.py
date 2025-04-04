import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

"""
So far:
- Agent learns to navigate a sequence of states and terminate at the optimal time.
- Learns 2 different sequences now instead of just one.
 * A 1-hot context is given to the agent to say which sequence it is learning.
- Predicts the 1-hot index of the next position in the sequence.
- Gives its confidence in the predicted state using temperature scaling confidence calibration.
 * That way it can learn to predict the next state with a confidence that matches the accuracy of the prediction
- Added a head to the model for information value prediction
  * Based on the difference of confidence of the next position prediction instead of actual difference in state information
- Made the second sequence give different states so that the agent is forced to learn 2 different sequences to associate with the respoective contexts
- Added a higher level net that does the same as the lower level, but uses options in place of actions, which represent the belief
- Introduced an inferred latent context representation into the model, which is learned in an unsupervised manner.
  * This inferred context is used in place of the externally provided context.
  * This latent context module predicts:
    A belief (latent context) about which sequence the agent is in.
    A predicted terminal state (i.e., the next state that it picks back up at when the lower layer returns. could also be goal state).
    A confidence temperature head to express uncertainty in its state prediction.
    The value integrated over the course of the lower layer episode.
- Implemented task switching based on prediction error.
    * The layer that has the least confidence in its prediction gets updated. The layers below it terminate.

Todo:
- control/habit network interactions
 * The control output should be used as a target
- implement confidence for the Action(or use 1 confidence for all?), so that VTE will get triggered when action confidence is low, since VTE is all about building confidence in an action.
    * if there are 2  confidences, one for the action and one for the state, then, during vte, exploration can be used to build confidence in the state prediction firs
    so that when we use vte for actions, it has an accurate state prediction to work with.
- Combine voi and value into a decision variable to choose the action.
    * choosing should use drift diffusion model, and binary comparisons.
     - drift rate controlled by the level of 
    * choosing should use control.
    * May need to use the chosen vector as the target

- Finally test the exploration of 3 different branches by walking down, and returning to homebase.

"""

# Multi-sequence environment.

class MultiSeqEnv:
    def __init__(self, reward_seqs, effort_seqs):
        # reward_seqs and effort_seqs: lists of sequences.
        assert len(reward_seqs) == len(effort_seqs), "Mismatch in sequence counts"
        self.reward_seqs = reward_seqs
        self.effort_seqs = effort_seqs
        self.seq_len = len(reward_seqs[0])
        # For hierarchical control, our observation is just the one-hot position.
        self.seq_idx = 1

    def pick_sequence(self, episode):

        if episode < 300:
            if episode % 2 == 0:
                self.seq_idx = 0
            elif episode % 2 == 1:
                self.seq_idx = 1

        elif episode < 500:
            self.seq_idx = 0
        
        elif episode < 700:
            self.seq_idx = 1

        else:
            self.seq_idx = 1 if np.random.rand() < 0.5 else 2

    
    def reset(self, episode):

        self.pick_sequence(episode)  # one-hot vector (length 2)
        self.reward_seq = self.reward_seqs[self.seq_idx]
        self.effort_seq = self.effort_seqs[self.seq_idx]
        # For seq 0, start at beginning; for seq 1, start at end (reverse)
        self.pos = 0 if self.seq_idx == 0 else self.seq_len - 1
        self.done = False
        self.prev_action = np.zeros(2, dtype=np.float32)
        return self._get_obs()
    
    def _get_obs(self):
        # External state: one-hot encoding of current position.
        index_onehot = np.zeros(self.seq_len, dtype=np.float32)
        index_onehot[self.pos] = 1.0
        return index_onehot  # shape: (seq_len,)
    
    def step(self, action):
        if self.done:
            raise Exception("Episode terminated. Call reset() to restart.")
        # Compute net reward.
        reward = self.reward_seq[self.pos] - self.effort_seq[self.pos]
        if action == 1:  # Terminate action.
            self.done = True
        else:
            if self.seq_idx == 0 and self.pos < self.seq_len - 1:
                self.pos += 1
            elif self.seq_idx == 1 and self.pos > 0:
                self.pos -= 1
            else:
                self.done = True
        # Update previous action (one-hot)
        prev_action_onehot = np.zeros(2, dtype=np.float32)
        prev_action_onehot[action] = 1.0
        self.prev_action = prev_action_onehot
        return self._get_obs(), reward, self.done, {}


class Agent(nn.Module):
    def __init__(self, input_dim, state_pred_dim, hidden_dim=128, action_dim=2, conf_threshold=0.5):
        """
        input_dim: dimensionality of the network input.
        state_pred_dim: dimension of the predicted state (e.g. external state dim).
        action_dim: number of discrete actions.
        """
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        self.fc_position = nn.Linear(hidden_dim, state_pred_dim)
        self.fc_info_value = nn.Linear(hidden_dim, 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.conf_threshold = conf_threshold
    
    def forward(self, x):
        # x: network input.
        h = F.relu(self.fc1(x))
        logits = self.fc_actor(h)
        value = self.fc_critic(h)
        next_pos_logits = self.fc_position(h)
        info_value = self.fc_info_value(h).squeeze(-1)
        temperature = torch.exp(self.log_temperature)
        calibrated_logits = next_pos_logits / temperature
        confidence = F.softmax(calibrated_logits, dim=-1).max(dim=-1)[0]
        return logits, value, next_pos_logits, calibrated_logits, confidence, info_value

    def VTE(self, current_input, top_k=2):
        """
        Given the current input to the agent, this function:
        1. Computes the agent's forward pass.
        2. Gets the top_k candidate actions from the agent's policy.
        3. For each candidate action:
            a. Forms a simulated (imagined) next input by taking the agent's predicted next state
                (using the fc_position head) and concatenating it with a one-hot encoding of the candidate action.
            b. Feeds this simulated input back into the agent to obtain a predicted value and info value.
        4. Returns the candidate action that maximizes (predicted value + info value).
        """
        # Run forward pass on current_input.
        action_logits, value, next_pos_logits, calibrated_logits, confidence, info_value = self.forward(current_input)
        probs = F.softmax(action_logits, dim=-1)
        # Get indices of top_k candidate actions.
        top_actions = torch.topk(probs, top_k, dim=-1).indices.squeeze(0)  # shape: (top_k,)
        
        best_score = -float('inf')
        best_action = None
        
        # Get the predicted next state as a probability distribution.
        # (We assume here that the fc_position output can be interpreted as a distribution over external states.)

        print(f"VTE next state probs: {next_pos_logits}")
        
        for action in top_actions:
            # Form a one-hot encoding for candidate action.
            candidate_one_hot = torch.zeros(1, action_logits)
            candidate_one_hot[0, action] = 1.0
            print(f"VTE candidate action: {candidate_one_hot}")
            # Form a simulated new input by concatenating the predicted next state with the candidate action.
            simulated_input = torch.cat([next_pos_logits, candidate_one_hot], dim=-1)
            print(f"VTE simulated input: {simulated_input}")
            # Get the predicted value and info value for this simulated input.
            _, sim_value, _, _, _, sim_info_value = self.forward(simulated_input)
            score = sim_value + sim_info_value  # Combine as desired.
            if score > best_score:
                best_score = score
                best_action = action
            print(f"VTE action value: {sim_value}, info value: {sim_info_value} , score: {score}")
            input()
        return best_action

def train_hierarchical():
    # Define two example sequences.
    reward_seq0 = [1.0, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.0]
    effort_seq0 = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    reward_seq1 = [0.5, 1.5, 2.5, 3.5, 3.0, 2.5, 2.0, 1.5]
    effort_seq1 = [0.2, 0.3, 5.5, 1.1, 3.9, 20.0, 1.0, 1.0]
    
    env = MultiSeqEnv([reward_seq0, reward_seq1], [effort_seq0, effort_seq1])
    seq_len = env.seq_len
    # Define dimensions:
    # External state: one-hot (seq_len) plus a previous action placeholder (2).
    ext_state_dim = seq_len + 2
    belief_dim = 2   # High-level action output becomes the belief.
    action_dim = 2   # Actions: continue (0) or terminate (1).
    
    # High-level agent: receives external state concatenated with belief.
    high_input_dim = ext_state_dim + belief_dim
    high_agent = Agent(high_input_dim, ext_state_dim, hidden_dim=128, action_dim=action_dim)
    # Low-level agent: receives external state concatenated with previous low-level action (2).
    low_input_dim = ext_state_dim + action_dim
    # Its state prediction output is of dimension belief_dim.
    low_agent = Agent(low_input_dim, belief_dim, hidden_dim=128, action_dim=action_dim)
    
    high_optimizer = optim.Adam(high_agent.parameters(), lr=1e-2)
    low_optimizer = optim.Adam(low_agent.parameters(), lr=1e-2)
    
    num_episodes = 1000
    gamma = 0.99
    pos_loss_weight = 1.0
    info_loss_weight = 0.5
    error_threshold = 1.0  # threshold for triggering a task switch
    
    for episode in range(num_episodes):
        # Reset environment for new episode.

        obs_np = env.reset(episode)  # external state: one-hot position (length=seq_len)
        prev_action = np.zeros(2, dtype=np.float32)  # initial previous action placeholder
        ext_state = torch.FloatTensor(np.concatenate([obs_np, prev_action])).unsqueeze(0)
        
        # Initialize belief for high-level (could be zeros).
        belief = torch.zeros(1, belief_dim)
        cumulative_reward_episode = 0.0
        episode_done = False
        
        # Outer loop: entire episode (only ends when env.done becomes True).
        while not episode_done:
            # ----- High-Level: Produce a belief -----
            high_input = torch.cat([ext_state, belief], dim=-1)
            high_logits, high_value, high_next_logits, high_calibrated, high_conf, high_info_value = high_agent(high_input)
            if high_conf.item() < high_agent.conf_threshold:
                high_action = high_agent.VTE(high_input)
            else:
                high_probs = F.softmax(high_logits, dim=-1)
                high_action = torch.distributions.Categorical(high_probs).sample()
            # Convert high-level action to one-hot belief.
            new_belief = torch.zeros_like(high_logits)
            new_belief[0, high_action] = 1.0
            # (Optionally, one could update belief slowly; here we replace it.)
            belief = new_belief
            
            segment_reward = 0.0
            task_switch = False
            low_done = False
            # Initialize low-level input: external state + previous low-level action.
            # Use env.prev_action for low-level action placeholder.
            low_input = torch.cat([ext_state, torch.FloatTensor(env.prev_action).unsqueeze(0)], dim=-1)
            prev_low_conf = None
            
            # Inner loop: low-level segment.
            while not low_done and not task_switch:
                low_logits, low_value, low_next_logits, low_calibrated, low_conf, low_info_value = low_agent(low_input)              # Low-level action selection:
                if low_conf.item() < low_agent.conf_threshold:
                    low_action = low_agent.VTE(low_input)
                else:
                    low_probs = F.softmax(low_logits, dim=-1)
                    low_action = torch.distributions.Categorical(low_probs).sample()
                low_log_prob = torch.distributions.Categorical(low_probs).log_prob(low_action)
                
                next_obs_np, reward, env_terminate, _ = env.step(low_action.item())
                segment_reward += reward
                cumulative_reward_episode += reward
                
                # Update external state with new observation and env.prev_action.
                next_ext_np = np.concatenate([next_obs_np, env.prev_action])
                next_ext = torch.FloatTensor(next_ext_np).unsqueeze(0)
                
                # Low-level TD target.
                if not env_terminate:
                    next_low_input = torch.cat([next_ext, torch.FloatTensor(env.prev_action).unsqueeze(0)], dim=-1)
                    _, next_low_value, _, _, next_low_conf, _ = low_agent(next_low_input)
                else:
                    next_low_value = torch.zeros(1)
                    next_low_conf = torch.tensor(0.0)
                
                low_advantage = reward + gamma * next_low_value - low_value
                low_policy_loss = -low_log_prob * low_advantage.detach()
                low_value_loss = low_advantage.pow(2)
                
                # Next-state prediction loss (low-level): we predict the next external state.
                target_index = torch.argmax(belief, dim=1)
                pos_loss = F.cross_entropy(low_next_logits, target_index)
                
                pred_index = torch.argmax(low_next_logits, dim=1)
                correct = (pred_index == target_index).float()
                calibration_loss = F.binary_cross_entropy(low_conf, correct)
                low_total_pos_loss = pos_loss + calibration_loss
                
                if prev_low_conf is not None:
                    low_info_gain = torch.abs(low_conf - prev_low_conf)
                    if low_info_gain.dim() == 0:
                        low_info_gain = low_info_gain.unsqueeze(0)
                else:
                    low_info_gain = torch.tensor([0.0])
                low_info_loss = F.mse_loss(low_info_value, low_info_gain)
                
                low_loss = low_policy_loss + low_value_loss + pos_loss_weight * low_total_pos_loss + info_loss_weight * low_info_loss
                low_optimizer.zero_grad()
                low_loss.backward()
                low_optimizer.step()
                
                prev_low_conf = low_conf.detach()
                # Prepare input for next low-level step.
                low_input = torch.cat([next_ext, torch.FloatTensor(env.prev_action).unsqueeze(0)], dim=-1)
                # Update external state for high-level.
                ext_state = next_ext
                
                # Check for task switch: if prediction error is high and low_conf exceeds high_conf.
                if pos_loss.item() > error_threshold and low_conf.item() > high_conf.item():
                    task_switch = True
                    print(f"Task switch at episode {episode}, pos_loss: {pos_loss.item()},layer 1 conf: {low_conf.item()}, layer 2 conf: {high_conf.item()}")
                # Also, if environment terminates, mark low-level done.
                if env_terminate:
                    low_done = True
                    episode_done = True
            
            # ----- High-Level Update (per segment) -----
            high_advantage = segment_reward + gamma * 0.0 - high_value  # no bootstrapping beyond segment
            high_log_prob = torch.distributions.Categorical(F.softmax(high_logits, dim=-1)).log_prob(high_action)
            high_policy_loss = -high_log_prob * high_advantage.detach()
            high_value_loss = high_advantage.pow(2)
            
            target_index_high = torch.argmax(ext_state, dim=1)
            high_pos_loss = F.cross_entropy(high_next_logits, target_index_high)
            
            high_info_gain = torch.abs(high_conf - low_conf.detach())
            high_info_loss = F.mse_loss(high_info_value, high_info_gain)
            
            high_loss = high_policy_loss + high_value_loss + pos_loss_weight * high_pos_loss + info_loss_weight * high_info_loss
            high_optimizer.zero_grad()
            high_loss.backward()
            high_optimizer.step()
            
            # After task switch, the high-level agent re-predicts a new belief in the next outer iteration.
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode}, Context {env.seq_idx} Cumulative Reward: {cumulative_reward_episode:.2f}, Final Env Pos: {env.pos}, High Conf: {high_conf.item():.3f}, Low Conf: {low_conf.item():.3f}")

        elif episode % 100 == 0:
            print(f"Episode {episode}, seq_idx {env.seq_idx} Cumulative Reward: {cumulative_reward_episode:.2f}, Final Env Pos: {env.pos}, High Conf: {high_conf.item():.3f}, Low Conf: {low_conf.item():.3f}")

if __name__ == "__main__":
    train_hierarchical()
