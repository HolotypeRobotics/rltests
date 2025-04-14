import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



"""
So far:

    Basic Navigation & Termination:
        The agent learns to navigate a sequence of states and terminate at the optimal time.

    Learning Multiple Sequences:
        The agent learns two different sequences instead of just one by receiving a one‑hot context that indicates which sequence it should follow.

    Next-State Prediction:
        The network predicts the one‑hot index of the next position in the sequence.

    Confidence Calibration via Temperature Scaling:
        The agent outputs a confidence measure for its next‑state prediction, calibrated using a temperature parameter so that its confidence better matches the accuracy of its prediction.

    Information Value Prediction:
        An additional head is added to predict the information value—computed based on changes in the prediction confidence (rather than raw state differences).

    Combine Value and Information Value:
        Merge the value and the information value into a single decision variable (e.g., using a drift diffusion model or binary comparisons) to guide action selection.
        The chosen vector may also need to be used as the target in learning.

    Distinct Sequence Dynamics:
        The second sequence is designed to produce different states, forcing the agent to learn to associate distinct state transitions with each context.

    Hierarchical Architecture:
        A higher‑level network is introduced that mirrors the lower‑level functionality but outputs “options” (i.e. a belief about the sequence) instead of raw actions.

    Latent Context Representation:
        An inferred latent context (or belief) is learned in an unsupervised manner to replace the externally provided context. This module predicts:
            A belief about which sequence the agent is in,
            A predicted terminal (or goal) state that the lower layer will return to,
            A unified confidence (via a temperature head) reflecting uncertainty in state prediction, and
            A value integrated over the course of the lower‑level episode.

    Task Switching Mechanism:
        A mechanism is implemented in which the layer with the least confidence (i.e. highest prediction error) triggers a task switch, causing lower layers to terminate their current episode and update accordingly.

    GRU‑Based Temporal Processing:
        The network architecture is updated to use GRUs instead of simple feedforward layers, allowing it to capture temporal dependencies.

    Explicit Head Direction & Expanded Action Space:
        The external input is augmented with an explicit one‑hot head direction signal (left/right), and the action space is expanded to include a reverse action.

    Separate Prediction Heads for Position and Head Direction:
        The model now has separate output heads: one for predicting the next position and another for predicting head direction, reducing ambiguity and enabling separate loss computations.

    Simultaneous Forward & Reverse Predictions:
        The agent now predicts two different next positions by feeding two different fixed head directions (forward and reverse) into the network. Both predictions are trained simultaneously using targets derived from the current (or previous) state, allowing the network to learn the dynamics of both directions.

    Control/Habit Network Interactions:
        Control outputs should not be used as direct training targets for the habit net, instead, it should indirectly train the net by choosing options not normally chosen, such as effortful, or informative options
        The habitual net then picks up on the more frequently chosen options/actions.
        This allows the habitual net to still be based on habit, and not on measured value, thus saving computation.

    # TODO:
    Test Multi-Branch Exploration:
        Finally, test the agent’s ability to explore 2-3 different branches by allowing it to “walk down” a branch and then return to a home base.
"""


class MultiSeqEnv:
    def __init__(self, reward_seq, effort_seq, max_steps=6):
        # reward_seq and effort_seq: a single sequence each.
        assert len(reward_seq) == len(effort_seq), "Mismatch in sequence lengths"
        self.reward_seq = reward_seq
        self.effort_seq = effort_seq
        self.seq_len = len(reward_seq)
        self.max_steps = max_steps
        self.num_steps = 0
        # For hierarchical control, our observation is just the one-hot position.

    def reset(self, episode=None):
        # In this single-sequence scenario, we ignore episode for sequence switching.
        # Start in the middle of the sequence.
        self.pos = self.seq_len // 2
        self.recent_pos = self.pos
        self.done = False
        self.num_steps = 0
        # Previous action: now length 3 (one-hot for 3 actions).
        self.prev_action = torch.zeros(3, dtype=torch.float32)
        # Head direction: one-hot for left/right. Let's default to "right" ([0, 1]).
        self.head_direction = torch.tensor([0, 1], dtype=torch.float32)
        return self._get_obs()
    
    def _get_obs(self):
        # Position: one-hot vector (length = seq_len).
        pos_onehot = torch.zeros(self.seq_len, dtype=torch.float32)
        pos_onehot[self.pos] = 1.0
        # External state: concatenate position, previous action, and head direction.
        return pos_onehot.unsqueeze(0), self.head_direction.unsqueeze(0), self.prev_action.unsqueeze(0)

    def step(self, action):
        """
        Action space:
         0: move forward
         1: move reverse  
         2: terminate  
        """
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            self.done = True

        # Compute net reward.
        reward = self.reward_seq[self.pos] - self.reward_seq[self.recent_pos]
        effort = self.effort_seq[self.pos] - self.effort_seq[self.recent_pos]
        reward -= effort  # Effort penalty.

        if action == 2:  # Terminate action.
            pass
        else:
            # Action 0: move forward; Action 1: move reverse.
            if action == 0 and self.pos < self.seq_len - 1:
                self.pos += 1
            elif action == 1 and self.pos > 0:
                self.pos -= 1

        # Update previous action (one-hot vector of length 3).
        prev_action_onehot = torch.zeros(3, dtype=torch.float32)
        prev_action_onehot[action] = 1.0
        self.prev_action = prev_action_onehot
        # (Optionally, update head direction based on the action.)
        # For example, if moving reverse, flip head direction.

        if action == 1: # reverse
            self.head_direction = torch.tensor([1, 0], dtype=torch.float32)
        elif action == 0: # forward
            self.head_direction = torch.tensor([0, 1], dtype=torch.float32)
        
        return self._get_obs(), reward, self.done


class Agent(nn.Module):
    def __init__(self, input_dim, pos_pred_dim, hidden_dim=128, action_dim=3, head_dir_dim=2):
        """
        input_dim: dimensionality of the network input.
        pos_pred_dim: number of classes for the position prediction (typically seq_len).
        action_dim: number of discrete actions.
        head_dir_dim: number of classes for head direction (2).
        """

        super(Agent, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        # This head predicts the next external state.
        self.fc_position = nn.Linear(hidden_dim, pos_pred_dim)
        self.fc_head_dir = nn.Linear(hidden_dim, head_dir_dim)
        self.fc_info_value = nn.Linear(hidden_dim, 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        # Set a confidence threshold for preforming VTE.
        self.conf_threshold = 0.9
    
    def forward(self, position, direction, prev_option, belief=None):
        """
        Inputs:
          - position: one-hot position (batch, pos_pred_dim)
          - direction: head direction (batch, head_dir_dim)
          - prev_option: previous action (batch, action_dim)
          - belief: optional belief vector (batch, belief_dim)
        """
        if belief is not None:
            x = torch.cat([position, direction, prev_option, belief], dim=-1)
        else:            
            x = torch.cat([position, direction, prev_option], dim=-1)
        # Expect x shape: (batch, input_dim). Unsqueeze for GRU.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        h = out.squeeze(1)
        action_logits = self.fc_actor(h)
        value = self.fc_critic(h)
        pos_logits = self.fc_position(h)
        head_logits = self.fc_head_dir(h)
        info_value = self.fc_info_value(h).squeeze(-1)

        temperature = torch.exp(self.log_temperature)
        # Use the same temperature for both action and state prediction.
        calibrated_action_logits = action_logits / temperature
        calibrated_pos_logits = pos_logits / temperature
        calibrated_head_logits = head_logits / temperature
        
        action_conf = F.softmax(calibrated_action_logits, dim=-1).max(dim=-1)[0]
        pos_conf    = F.softmax(calibrated_pos_logits, dim=-1).max(dim=-1)[0]
        head_conf   = F.softmax(calibrated_head_logits, dim=-1).max(dim=-1)[0]
        # Combined confidence: simple average of all three.
        confidence = (action_conf + pos_conf + head_conf) / 3.0

        return action_logits, value, pos_logits, head_logits, confidence, info_value

    def VTE(self, position, direction, top_k=2, prev_option=None, belief=None):
        """
        VTE (vicarious trial-and-error) for this layer.
        Given the current input, this function:
          1. Runs a forward pass to get the policy logits.
          2. Extracts the top_k candidate actions.
          3. For each candidate, forms a simulated input by concatenating the *raw* next_pos_logits
             (as a proxy for the resulting state) with the candidate action (one-hot encoded).
          4. Runs a forward pass on the simulated input to obtain the predicted value and info value.
          5. Returns the candidate action with the highest combined (value + info value).
        """

        with torch.no_grad():
            # Forward pass on the current input.
            position = position.clone().detach()
            direction = direction.clone().detach()
            # High level and low level
            if belief is not None:
                belief = belief.clone().detach()

            # Low level
            if prev_option is not None:
                prev_option = prev_option.clone().detach()

            # get the option probabilities only
            action_logits, _, _, _, _, _ = self.forward(position, direction, prev_option, belief)
            probs = F.softmax(action_logits, dim=-1)
            # Get top_k candidate action indices.
            top_options = torch.topk(probs, top_k, dim=-1).indices.squeeze(0)
            best_score = -float('inf')
            best_action = None

            # look at each action/belief individually
            for option in top_options:
                # Form a one-hot encoding for candidate option.
                candidate = torch.zeros(1, action_logits.size(-1))
                candidate[0, option] = 1.0
                # Form a simulated new input by concatenating the predicted next state with the candidate option, then get the value
                # Get the predicted value and info value by plugging in the candidate action
                # Simulate forward prediction.
                _, sim_value, _, _, sim_confidence, sim_info_value = self.forward(position, direction, candidate, belief)
                # TODO: try out different ways of combining the values
                score = (sim_value + sim_info_value) * sim_confidence
                if score > best_score:
                    best_score = score
                    best_action = option
            return best_action

def train_hierarchical():
    # Define a single sequence.
    reward_seq = [1.0, 2.0, 2.0, 1.0, 0.1, 0.0, 0.2, 2.0, -1.0, 25]
    effort_seq = [0.5, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]


    env = MultiSeqEnv(reward_seq, effort_seq)
    seq_len = env.seq_len
    # External state: one-hot position (seq_len) + previous action (3) + head direction (2).
    belief_dim = 2   # For high-level belief.
    action_dim = 3   # 3 discrete actions.
    head_dir_dim = 2  # 2 head directions.
    pos_dim = env.seq_len
    # High-level agent: input is external state concatenated with belief.
    high_input_dim = pos_dim + head_dir_dim + belief_dim
    low_input_dim = pos_dim + head_dir_dim + action_dim + belief_dim
    high_agent = Agent(input_dim=high_input_dim, pos_pred_dim=pos_dim, hidden_dim=128, action_dim=belief_dim)
    low_agent = Agent(input_dim=low_input_dim, pos_pred_dim=pos_dim, hidden_dim=128, action_dim=action_dim)

    high_optimizer = optim.Adam(high_agent.parameters(), lr=1e-2)
    low_optimizer = optim.Adam(low_agent.parameters(), lr=1e-2)

    num_episodes = 1000
    gamma = 0.99
    pos_loss_weight = 1.0
    info_loss_weight = 0.5
    policy_loss_weight = 0.45
    error_threshold = 1.0  # threshold for triggering a task switch
    episode_positions = []
    # Define two fixed head directions for training both forward and reverse predictions.
    forward_hd = torch.tensor([[0, 1]], dtype=torch.float32)
    reverse_hd = torch.tensor([[1, 0]], dtype=torch.float32)
    for episode in range(num_episodes):
        position, direction, prev_action = env.reset(episode)
        vte_freq = 0
        belief = torch.zeros(1, belief_dim)
        cumulative_reward_episode = 0.0
        episode_done = False

        pos_log = [env.pos]

        # Outer loop: entire episode.
        while not episode_done:
            # ----- High-Level: Produce a new belief -----
    
            high_logits, high_value, high_pos_logits, high_head_logits, high_conf, high_info_value = high_agent(position=position, direction=direction, prev_option=belief)
            # Use VTE only if high_conf is low.
            if high_conf.item() < high_agent.conf_threshold:
                chosen_high_action = high_agent.VTE(position=position, direction=direction, prev_option=belief, top_k=2, belief=None)
                vte_freq += 1
            else:
                chosen_high_action = torch.argmax(high_logits)
            # Convert chosen high-level action to one-hot belief.
            new_belief = torch.zeros_like(high_logits)
            new_belief[0, chosen_high_action] = 1.0
            belief = new_belief

            segment_reward = 0.0
            task_switch = False
            low_done = False
            prev_low_conf = None

            # ----- Low-Level Nested Loop -----
            while not low_done and not task_switch:
                low_logits, low_value, low_pos_logits, low_dir_logits, low_conf, low_voi = low_agent(position=position, direction=direction, prev_option=prev_action, belief=belief)
                low_probs = F.softmax(low_logits, dim=-1)

                if low_conf.item() < low_agent.conf_threshold:
                    chosen_action = low_agent.VTE(position=position, prev_option=prev_action, direction=direction, top_k=2, belief=belief)
                    vte_freq += 1
                else:
                    chosen_action = torch.distributions.Categorical(low_probs).sample()
                low_log_prob = F.log_softmax(low_logits, dim=-1).squeeze(0)[chosen_action]

                (position, direction, prev_action), reward, env_terminate = env.step(chosen_action.item())
                segment_reward += reward
                cumulative_reward_episode += reward
                pos_log.append(env.pos)

                if not env_terminate:
                    _, next_low_value, _, _, _, _ = low_agent(position=position, direction=direction, prev_option=prev_action, belief=belief)
                else:
                    next_low_value = torch.zeros(1)
                
                low_advantage = reward + gamma * next_low_value - low_value
                low_policy_loss = -low_log_prob * low_advantage.detach()
                low_value_loss = low_advantage.pow(2)
                
                pos_target = torch.argmax(position, dim=1)
                pos_loss = F.cross_entropy(low_pos_logits, pos_target)
                head_target = torch.argmax(direction, dim=1)
                head_loss = F.cross_entropy(low_dir_logits, head_target)
                low_total_state_loss = pos_loss + head_loss

                # TODO: calibrate the confidence for the rest of the prediction varaibles
                pred_index = torch.argmax(low_pos_logits, dim=1)
                correct = (pred_index == pos_target).float()
                calibration_loss = F.binary_cross_entropy(low_conf, correct)
                low_total_state_loss += calibration_loss

                if prev_low_conf is not None:
                    low_info_gain = torch.abs(low_conf - prev_low_conf)
                    if low_info_gain.dim() == 0:
                        low_info_gain = low_info_gain.unsqueeze(0)
                else:
                    low_info_gain = torch.tensor([0.0])
                low_info_loss = F.mse_loss(low_voi, low_info_gain)

                low_loss = policy_loss_weight * low_policy_loss + low_value_loss + pos_loss_weight * low_total_state_loss + info_loss_weight * low_info_loss
                low_optimizer.zero_grad()
                low_loss.backward()
                low_optimizer.step()

                prev_low_conf = low_conf.detach()

                if pos_loss.item() > error_threshold and low_conf.item() > high_conf.item():
                    task_switch = True
                if env_terminate:
                    low_done = True
                    episode_done = True

            # ----- High-Level Update (per segment) -----
            # Here we now use the next high value if available.
            # We form new high-level input from the updated external state and current belief.

            _, next_high_value, _, _, _, _ = high_agent(position=position, direction=direction, prev_option=belief, belief=None)
            high_advantage = segment_reward + gamma * next_high_value - high_value
            high_log_prob = torch.distributions.Categorical(F.softmax(high_logits, dim=-1)).log_prob(chosen_high_action)
            high_policy_loss = -high_log_prob * high_advantage.detach()
            high_value_loss = high_advantage.pow(2)
            pos_target_high = torch.argmax(position, dim=1)
            high_pos_loss = F.cross_entropy(high_pos_logits, pos_target_high)
            head_target_high = torch.argmax(direction, dim=1)
            high_head_loss = F.cross_entropy(high_head_logits, head_target_high)
            high_total_state_loss = high_pos_loss + high_head_loss

            # Calibrate high-level confidence.
            high_pred_index = torch.argmax(high_pos_logits, dim=1)
            high_correct = (high_pred_index == pos_target_high).float()
            high_calib_loss = F.binary_cross_entropy(high_conf, high_correct)
            high_total_state_loss += high_calib_loss

            high_info_gain = torch.abs(high_conf - low_conf.detach())
            high_info_loss = F.mse_loss(high_info_value, high_info_gain)
            
            high_loss = policy_loss_weight * high_policy_loss + high_value_loss + pos_loss_weight * high_total_state_loss + info_loss_weight * high_info_loss
            high_optimizer.zero_grad()
            high_loss.backward()
            high_optimizer.step()
            # After a task switch, the high-level agent re-predicts a new belief.
        episode_positions.append(pos_log)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode}, Cumulative Reward: {cumulative_reward_episode:.2f}, Final Env Pos: {env.pos}, High Conf: {high_conf.item():.3f}, Low Conf: {low_conf.item():.3f}, VTE Freq: {vte_freq} / {episode+1}")

if __name__ == "__main__":
    train_hierarchical()
