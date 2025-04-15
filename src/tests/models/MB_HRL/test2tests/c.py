import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from environment import Environment


class Agent(nn.Module):
    def __init__(self, input_dim, pos_pred_dim, hidden_dim=128, action_dim=3, head_dir_dim=2, obj_distance_dim=3):
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
        self.fc_obj_dist = nn.Linear(hidden_dim, obj_distance_dim)
        self.fc_info_value = nn.Linear(hidden_dim, 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        # Set a confidence threshold for preforming VTE.
        self.conf_threshold = 0.9
        print(f"obj_distance_dim: {obj_distance_dim}, pos_pred_dim: {pos_pred_dim}, head_dir_dim: {head_dir_dim}")

    def forward(self, position, direction, obj_distances, prev_option, belief=None):
        """
        Inputs:
          - position: one-hot position (batch, pos_pred_dim)
          - direction: head direction (batch, head_dir_dim)
          - prev_option: previous action (batch, action_dim)
          - belief: optional belief vector (batch, belief_dim)
        """
        if belief is not None:
            x = torch.cat([position, direction, obj_distances, prev_option, belief], dim=-1)
        else:
            x = torch.cat([position, direction, obj_distances, prev_option], dim=-1)
        # Expect x shape: (batch, input_dim). Unsqueeze for GRU.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        h = out.squeeze(1)
        action_logits = self.fc_actor(h)
        value = self.fc_critic(h)
        pos_logits = self.fc_position(h)
        head_logits = self.fc_head_dir(h)
        obj_distance_logits = self.fc_obj_dist(h)
        info_value = self.fc_info_value(h).squeeze(-1)

        temperature = torch.exp(self.log_temperature)
        # Use the same temperature for both action and state prediction.
        calibrated_action_logits = action_logits / temperature
        calibrated_pos_logits = pos_logits / temperature
        calibrated_head_logits = head_logits / temperature
        calibrated_obj_distance_logits = obj_distance_logits / temperature
        
        action_conf = F.softmax(calibrated_action_logits, dim=-1).max(dim=-1)[0]
        pos_conf    = F.softmax(calibrated_pos_logits, dim=-1).max(dim=-1)[0]
        head_conf   = F.softmax(calibrated_head_logits, dim=-1).max(dim=-1)[0]
        obj_distance_conf = F.softmax(calibrated_obj_distance_logits, dim=-1).max(dim=-1)[0]
        # Combined confidence: simple average of all three.
        confidence = (action_conf + pos_conf + head_conf + obj_distance_conf) / 4.0

        return action_logits, value, pos_logits, head_logits, obj_distance_logits, confidence, info_value

    def VTE(self, position, direction, obj_distances, top_k=2, prev_option=None, belief=None):
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
            obj_distances = obj_distances.clone().detach()
            # High level and low level
            if belief is not None:
                belief = belief.clone().detach()

            # Low level
            if prev_option is not None:
                prev_option = prev_option.clone().detach()

            # get the option probabilities only
            action_logits, _, _, _, _, _, _ = self.forward(position=position, direction=direction, obj_distances=obj_distances, prev_option=prev_option, belief=belief)
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
                _, sim_value, _, _, _, sim_confidence, sim_info_value = self.forward(position=position, direction=direction, obj_distances=obj_distances, prev_option=candidate, belief=belief)
                # TODO: try out different ways of combining the values
                score = (sim_value + sim_info_value)
                if score > best_score:
                    best_score = score
                    best_action = option
            return best_action

def train_hierarchical():
    # Define a single sequence.

    n_objects = 3
    env = Environment(rewards=[[-1, 0.1, 0, .1, 1],
                            [0, 0.1, .2, .2, 0],
                            [.05, .1, .5, 0, -1],
                            [0, .1, 0, .6, 0],
                            [0, 0, 0, 0, 2]],
                    efforts=[[.5, .1, .2, .1, .1],
                            [.1, .1, .1, .5, .1],
                            [.1, .5, .2, .1, .1],
                            [.1, .1, .1, .1, .1],
                            [2, 1, 2, 1, 0]],
                    n_objects=n_objects)

    # External state: one-hot position (seq_len) + previous action (3) + head direction (2).
    belief_dim = 2   # For high-level belief.
    action_dim = 3   # left right forward
    head_dir_dim = 4
    env.set_exit(env.width, env.height)
    pos_dim = env.width * env.height

    # High-level agent: input is external state concatenated with belief.
    high_input_dim = pos_dim + head_dir_dim + n_objects + belief_dim
    low_input_dim = pos_dim + head_dir_dim + action_dim + n_objects + belief_dim
    high_agent = Agent(input_dim=high_input_dim, pos_pred_dim=pos_dim, hidden_dim=128, obj_distance_dim=n_objects, action_dim=belief_dim, head_dir_dim=head_dir_dim)
    low_agent = Agent(input_dim=low_input_dim, pos_pred_dim=pos_dim, hidden_dim=128, obj_distance_dim=n_objects, action_dim=action_dim, head_dir_dim=head_dir_dim)

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
    for episode in range(num_episodes):
        (position, direction, obj_distances, prev_action), _, _ = env.reset()
        vte_freq = 0
        belief = torch.zeros(1, belief_dim)
        prev_high_conf = None
        cumulative_reward_episode = 0.0
        episode_done = False
        pos_log = []

        # Outer loop: entire episode.
        while not episode_done:
            # ----- High-Level: Produce a new belief -----
            high_logits, high_value, high_pos_logits, high_dir_logits, high_obj_dist_logits, high_conf, high_voi = high_agent(position=position, direction=direction, obj_distances=obj_distances, prev_option=belief)
            # Use VTE only if high_conf is low.
            if high_conf.item() < high_agent.conf_threshold:
                chosen_high_action = high_agent.VTE(position=position, direction=direction, obj_distances=obj_distances, prev_option=belief, top_k=2, belief=None)
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
                low_logits, low_value, low_pos_logits, low_dir_logits, low_obj_dist_logits, low_conf, low_voi = low_agent(position=position, direction=direction, obj_distances=obj_distances, prev_option=prev_action, belief=belief)
                low_probs = F.softmax(low_logits, dim=-1)

                if low_conf.item() < low_agent.conf_threshold:
                    chosen_action = low_agent.VTE(position=position, prev_option=prev_action, obj_distances=obj_distances, direction=direction, top_k=2, belief=belief)
                    vte_freq += 1
                else:
                    chosen_action = torch.distributions.Categorical(low_probs).sample()

                (position, direction, obj_distances, prev_action), reward, env_terminate = env.step(chosen_action.item())
                segment_reward += reward
                cumulative_reward_episode += reward
                pos_log.append(env.state)


                # Low level training

                # Get the next low-level value.
                if not env_terminate:
                    _, next_low_value, _, _, _, _, _ = low_agent(position=position, direction=direction, obj_distances=obj_distances, prev_option=prev_action, belief=belief)
                else:
                    next_low_value = torch.zeros(1)

                # Find low level targets
                low_pred_pos = torch.argmax(low_pos_logits, dim=-1)
                low_pred_dir = torch.argmax(low_dir_logits, dim=-1)
                low_pred_obj_dist = F.softmax(low_obj_dist_logits, dim=-1)

                low_pos_target = torch.argmax(position, dim=-1)
                low_dir_target = torch.argmax(direction, dim=-1)
                low_value_target = reward + gamma * next_low_value.detach()
                
                pos_correct = (low_pred_pos == low_pos_target).float()
                dir_correct = (low_pred_dir == low_dir_target).float()
                dist_correct = torch.sum(torch.sqrt(low_pred_obj_dist * obj_distances), dim=-1)
                dist_correct = dist_correct.sum(dim=-1)
                low_state_pred_correct = (pos_correct + dir_correct + dist_correct) / 3.0

                # Get the information gain target
                if prev_low_conf is not None:
                    low_info_gain = torch.abs(low_conf - prev_low_conf)
                    if low_info_gain.dim() == 0:
                        low_info_gain = low_info_gain.unsqueeze(0)
                else:
                    low_info_gain = torch.tensor([0.0])

                # Calculate low-level losses.
                # policy loss
                low_advantage = reward + gamma * next_low_value - low_value
                low_log_prob = F.log_softmax(low_logits, dim=-1).squeeze(0)[chosen_action]
                low_policy_loss = -low_log_prob * low_advantage.detach()
                low_value_loss = F.mse_loss(low_value, low_value_target)
                low_pos_loss = F.cross_entropy(low_pos_logits, position)
                low_dir_loss = F.cross_entropy(low_dir_logits, direction)
                low_obj_dist_loss = F.cross_entropy(low_obj_dist_logits, obj_distances)
                low_calibration_loss = F.binary_cross_entropy(low_conf, low_state_pred_correct)
                low_info_loss = F.mse_loss(low_voi, low_info_gain)

                # Total state loss
                low_total_state_loss = low_pos_loss + low_dir_loss + low_obj_dist_loss + low_calibration_loss

                # Total low level loss
                low_loss = policy_loss_weight * low_policy_loss + low_value_loss + pos_loss_weight * low_total_state_loss + info_loss_weight * low_info_loss
                low_optimizer.zero_grad()
                low_loss.backward()
                low_optimizer.step()

                prev_low_conf = low_conf.detach()

                if low_pos_loss.item() > error_threshold and low_conf.item() > high_conf.item():
                    task_switch = True
                if env_terminate:
                    low_done = True
                    episode_done = True

            # ----- High-Level Update (per segment) -----
            # Here we now use the next high value if available.
            # We form new high-level input from the updated external state and current belief.

            # Get the next high-level value.
            _, next_high_value, _, _, _, _, _ = high_agent(position=position, direction=direction, obj_distances=obj_distances, prev_option=belief, belief=None)
            

            # Find high level targets
            high_pred_pos = torch.argmax(high_pos_logits, dim=-1)
            high_pred_dir = torch.argmax(high_dir_logits, dim=-1)
            high_pred_obj_dist = F.softmax(high_obj_dist_logits, dim=-1)

            high_pos_target = torch.argmax(position, dim=-1)
            high_dir_target = torch.argmax(direction, dim=-1)
            high_value_target = reward + gamma * next_high_value.detach()

            high_pos_correct = (high_pred_pos == high_pos_target).float()
            high_dir_correct = (high_pred_dir == high_dir_target).float()
            high_dist_correct = torch.sum(torch.sqrt(high_pred_obj_dist * obj_distances), dim=-1)
            high_state_pred_correct = (high_pos_correct + high_dir_correct + high_dist_correct) / 3.0

            # Get the information gain target
            if prev_high_conf is not None:
                high_info_gain = torch.abs(high_conf - prev_high_conf)
                if high_info_gain.dim() == 0:
                    high_info_gain = high_info_gain.unsqueeze(0)
            else:
                high_info_gain = torch.tensor([0.0])
            high_info_loss = F.mse_loss(high_voi, high_info_gain)


            # Calculate high-level losses.
            # policy loss
            high_advantage = reward + gamma * next_high_value - high_value
            high_log_prob = F.log_softmax(high_logits, dim=-1).squeeze(0)[chosen_high_action]
            high_policy_loss = -high_log_prob * high_advantage.detach()
            high_value_loss = F.mse_loss(high_value, high_value_target)
            high_pos_loss = F.cross_entropy(high_pos_logits, position)
            high_dir_loss = F.cross_entropy(high_dir_logits, direction)
            high_obj_dist_loss = F.cross_entropy(high_obj_dist_logits, obj_distances)
            high_calibration_loss = F.binary_cross_entropy(high_conf, high_state_pred_correct)
            high_info_loss = F.mse_loss(high_voi, high_info_gain)

            # # Total state loss
            high_total_state_loss = high_pos_loss + high_dir_loss + high_obj_dist_loss + high_calibration_loss

            # Total high level loss
            high_loss = policy_loss_weight * high_policy_loss + high_value_loss + pos_loss_weight * high_total_state_loss + info_loss_weight * high_info_loss
            high_optimizer.zero_grad()
            high_loss.backward()
            high_optimizer.step()

            prev_high_conf = high_conf.detach()
            # After a task switch, the high-level agent re-predicts a new belief.


        episode_positions.append(pos_log)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode}, Cumulative Reward: {cumulative_reward_episode:.2f}, Final Env Pos: {env.pos}, High Conf: {high_conf.item():.3f}, Low Conf: {low_conf.item():.3f}, VTE Freq: {vte_freq} / {episode+1}")

if __name__ == "__main__":
    train_hierarchical()
