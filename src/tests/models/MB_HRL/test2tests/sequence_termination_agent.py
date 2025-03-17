import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Multi-sequence environment.
class MultiSeqEnv:
    def __init__(self, reward_seqs, effort_seqs):
        # reward_seqs and effort_seqs are lists of sequences.
        assert len(reward_seqs) == len(effort_seqs), "Mismatch in number of sequences"
        self.reward_seqs = reward_seqs  # e.g., two different sequences.
        self.effort_seqs = effort_seqs
        # Assume all sequences have the same length.
        self.seq_len = len(reward_seqs[0])
        self.reset(np.array([1, 0], dtype=np.float32))  # default context.
    
    def reset(self, context):
        # context: a one-hot vector of length 2 determining which sequence to use.
        self.context = context  # store context.
        self.seq_idx = int(np.argmax(context))  # choose sequence 0 or 1.
        self.reward_seq = self.reward_seqs[self.seq_idx]
        self.effort_seq = self.effort_seqs[self.seq_idx]
        self.pos = 0 if self.seq_idx == 0 else self.seq_len - 1  # Start at opposite ends.
        self.done = False

        # Initial previous action: zero vector (for two possible actions).
        self.prev_action = np.zeros(2, dtype=np.float32)
        return self._get_obs()
    
    def _get_obs(self):
        # One-hot encoding of current position (length = seq_len).
        index_onehot = np.zeros(self.seq_len, dtype=np.float32)
        index_onehot[self.pos] = 1.0
        # Observation: [position one-hot, previous action (2), context (2)].
        obs = np.concatenate([index_onehot, self.prev_action, self.context])
        return obs
    
    def step(self, action):
        if self.done:
            raise Exception("Episode terminated. Call reset() to restart.")
        
        # Compute net reward at current index.
        reward = self.reward_seq[self.pos] - self.effort_seq[self.pos]
        
        if action == 1:  # Terminate.
            self.done = True
        else:
            if self.seq_idx == 0 and self.pos < self.seq_len - 1:  
                self.pos += 1  # Move forward in sequence 0.
            elif self.seq_idx == 1 and self.pos > 0:
                self.pos -= 1  # Move backward in sequence 1.
            else:
                self.done = True
        
        # Update previous action one-hot.
        prev_action_onehot = np.zeros(2, dtype=np.float32)
        prev_action_onehot[action] = 1.0
        self.prev_action = prev_action_onehot
        
        return self._get_obs(), reward, self.done, {}

# Actor-Critic network with an additional next-position prediction head and temperature scaling.
class ActorCritic(nn.Module):
    def __init__(self, state_dim, seq_len, hidden_dim=128, action_dim=2):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        # Head for next position prediction.
        self.fc_position = nn.Linear(hidden_dim, seq_len)
        # Head for information value prediction.
        self.fc_info_value = nn.Linear(hidden_dim, 1)  # Information gain prediction.
        # Learnable temperature parameter (log scale for positivity).
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc_actor(x)
        value = self.fc_critic(x)
        next_pos_logits = self.fc_position(x)
        info_value = self.fc_info_value(x).squeeze(-1)  # Scalar output.

        # Temperature scaling for calibration.
        temperature = torch.exp(self.log_temperature)  # Ensure temperature > 0.
        calibrated_logits = next_pos_logits / temperature
        # Confidence is computed as the maximum probability from the calibrated logits.
        confidence = F.softmax(calibrated_logits, dim=-1).max(dim=-1)[0]
        return logits, value, next_pos_logits, calibrated_logits, confidence, info_value

def train_online_multi_sequence():
    # Define two sequences.
    # Sequence 0:
    reward_seq0 = [1.0, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.0]
    effort_seq0 = [0.5, 0.5, 0.5, 1.0, 11.0, 1.0, 10.0, 1.0]
    # Sequence 1: Slightly different reward/effort profiles.
    reward_seq1 = [0.5, 1.5, 2.5, 3.5, 3.0, 2.5, 2.0, 1.5]
    effort_seq1 = [0.2, 0.3, 5.5, 1.1, 3.9, 20, 1.0, 1.0]
    
    env = MultiSeqEnv([reward_seq0, reward_seq1], [effort_seq0, effort_seq1])
    seq_len = env.seq_len
    # State consists of: one-hot position (seq_len) + previous action (2) + context (2).
    state_dim = seq_len + 2 + 2  
    action_dim = 2  # 0: move forward, 1: terminate.
    
    model = ActorCritic(state_dim, seq_len)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    num_episodes = 1000
    gamma = 0.99
    pos_loss_weight = 1.0  # Weight for next-position prediction loss.
    info_loss_weight = 0.5  # Weight for information value loss.
    
    # Alternate between the two contexts.
    contexts = [np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)]
    
    for episode in range(num_episodes):
        context = contexts[episode % 2]
        state = env.reset(context)
        total_reward = 0.0
        total_pos_loss = 0.0
        done = False
        prev_confidence = None  # Track previous confidence for info gain calculation.

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_dim)
            logits, value, next_pos_logits, calibrated_logits, confidence, info_value = model(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            
            # Bootstrapping for value of next state.
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value, _, _, next_confidence, _ = model(next_state_tensor)
            else:
                next_value = torch.zeros(1)
                next_confidence = torch.tensor(0.0)  # No confidence on termination.
            
            # One-step TD error (advantage).
            advantage = reward + gamma * next_value - value
            
            # Actor (policy) and critic (value) losses.
            policy_loss = -log_prob * advantage.detach()
            value_loss = advantage.pow(2)

            # Next position prediction loss: using calibrated logits.
            if not done:
                # # Target is the index corresponding to the next position.
                # target_index = np.argmax(next_state[:seq_len])
                # target_index = torch.tensor([target_index])
                # pos_loss = F.cross_entropy(calibrated_logits, target_index)

                # Target is the index corresponding to the next position
                target_index = np.argmax(next_state[:seq_len])
                target_index = torch.tensor([target_index])
                
                # Basic prediction loss (using uncalibrated logits)
                pos_loss = F.cross_entropy(next_pos_logits, target_index)
                
                # Additional calibration loss
                # This optimizes the temperature parameter to match confidence with accuracy
                pred_index = torch.argmax(next_pos_logits, dim=1)
                # We could alternatively use the difference in distrobutions
                correct = (pred_index == target_index).float()
                calibration_loss = F.binary_cross_entropy(confidence, correct)
                
                pos_loss_total = pos_loss + calibration_loss
            else:
                pos_loss_total = 0.0

            # Information gain prediction loss.
            if prev_confidence is not None:
                info_gain_actual = next_confidence - prev_confidence
            else:
                info_gain_actual = torch.tensor(0.0)

            info_gain_loss = F.mse_loss(info_value, info_gain_actual)

            prev_confidence = confidence.detach()  # Update confidence for next step.
            
            total_pos_loss += pos_loss_total
            loss = policy_loss + value_loss + pos_loss_weight * pos_loss_total + info_loss_weight * info_gain_loss
            # loss = policy_loss + value_loss + pos_loss + calibration_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # (Optional) Print the confidence for this step if desired.
            # input(f"Step confidence: {confidence.item()}, Info gain loss: {info_gain_loss}, Info gain actual: {info_gain_actual.item()}")

            
            state = next_state
        
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}, Context 1 Total Reward: {total_reward:.2f},  Terminated at: {env.pos}, Prediction Loss: {total_pos_loss:.4f}, Confidence: {confidence.item()}")

        elif episode % 100 == 0:
            print(f"Episode {episode+1}, Context 2 Total Reward: {total_reward:.2f},  Terminated at: {env.pos}, Prediction Loss: {total_pos_loss:.4f}, Confidence: {confidence.item()}")
      
if __name__ == "__main__":
    train_online_multi_sequence()
