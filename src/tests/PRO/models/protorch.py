
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PROControl(nn.Module):
    def __init__(self,
                 n_stimuli,
                 n_responses,
                 n_outcomes,
                 n_delay_units,
                 dt=0.1,
                 theta=1.5,  # Aversion sensitivity
                 alpha_ro=0.1,  # Base learning rate for R-O conjunctions
                 alpha_td=0.1,  # TD learning rate
                 beta=0.1,  # Response gain
                 gamma=0.95,  # Temporal discount
                 lambda_decay=0.95,  # Eligibility trace decay
                 psi=0.1,  # Inhibition scaling
                 phi=0.1,  # Control scaling
                 rho=0.1,  # Excitation scaling
                 response_threshold=0.5,  # Response threshold
                 sigma=0.1, # Noise standard deviation
                 device='cpu'):
        super().__init__()
        self.device = device

        
        # Model dimensions
        self.n_stimuli = n_stimuli
        self.n_responses = n_responses
        self.n_outcomes = n_outcomes
        self.n_ro_conjunctions = n_responses * n_outcomes
        self.n_delay_units = n_delay_units

        # Parameters
        self.dt = dt
        self.theta = theta
        self.alpha_ro = alpha_ro
        self.alpha_td = alpha_td
        self.beta = beta
        self.gamma = gamma
        self.lambda_decay = lambda_decay
        self.response_threshold = response_threshold
        self.psi = psi
        self.phi = phi
        self.rho = rho
        self.sigma = sigma

        # Initialize learnable weights
        # R-O conjunction prediction weights
        self.W_S = nn.Parameter(torch.abs(
            torch.normal(0.1, 0.05, (self.n_ro_conjunctions, n_stimuli))))
        
        # Fixed stimulus-response weights
        self.register_buffer('W_C', torch.ones((n_responses, n_stimuli)))
        
        # Proactive control weights (equation 12 in paper)
        self.W_F = nn.Parameter(torch.normal(0, 0.1, (self.n_ro_conjunctions, n_responses)))
        
        # Reactive control weights
        self.W_R = nn.Parameter(torch.zeros((n_responses, self.n_ro_conjunctions)))
        
        # Mutual inhibition weights
        self.register_buffer('W_I', torch.zeros((n_responses, n_responses)))
        self.W_I.fill_diagonal_(-1)
        
        # Temporal prediction weights
        self.U = nn.Parameter(torch.zeros(
            (self.n_ro_conjunctions, n_delay_units, n_stimuli)))
        # State buffers
        self.register_buffer('delay_chain', 
                           torch.zeros((n_delay_units, n_stimuli)))
        self.register_buffer('eligibility_trace', 
                           torch.zeros((n_delay_units, n_stimuli)))
        self.register_buffer('C', torch.zeros(n_responses))
        
        # Normalize W_F weights
        with torch.no_grad():
            norm_factor = torch.sum(torch.abs(self.W_F)) / (n_responses * n_outcomes)
            if norm_factor > 1:
                self.W_F.data /= norm_factor

    def compute_losses(self, *args): # No valence loss now
        return {'total_loss': torch.tensor(0.0, requires_grad=True)} # Dummy loss

    # PRO Eq. (3)
    def compute_ro_prediction(self, stimuli):
        """Predict response-outcome conjunctions
        returns RO conjunctions shape: n_ro_conjunctions
        """
        ro_conj = torch.matmul(self.W_S, stimuli)
        return ro_conj  # Return RO conjunction
    
    def update_temporal_components(self, stimuli):
        """Update delay chain and eligibility trace"""
        # Roll delay chain. X in PRO Eq. (9)
        self.delay_chain = torch.roll(self.delay_chain, 1, dims=0)
        self.delay_chain[0] = stimuli
        print(f"delay chain: {self.delay_chain}")
        
        # Update eligibility trace PRO Eq. (10)
        self.eligibility_trace = (self.delay_chain + 
                                self.lambda_decay * self.eligibility_trace)
        print(f"eligibility trace: {self.eligibility_trace}")

    # PRO Eq. (8)
    def compute_temporal_prediction(self):
        """Compute temporal predictions using eligibility trace
        returns value vector shape: n_ro_conjunctions"""
        return torch.sum(self.U * self.delay_chain, dim=(1, 2))
    
    # PRO Eq. (15-16)
    def compute_surprise(self, predicted_ro, actual_ro):
        """Compute positive and negative surprise (equation 2)
        returns positive and negative surprize tuple with shapes n_ro_conjuctions"""
        omega_p = F.relu(actual_ro - predicted_ro)
        omega_n = F.relu(predicted_ro - actual_ro)
        return omega_p, omega_n
    
    # PRO Eq. (5)
    def compute_effective_learning_rate(self, omega_p, omega_n):
        """Compute surprise-modulated learning rate
        returns vector of learning rates shape: n_ro_conjunctions"""
        return self.alpha_ro / (1 + omega_p + omega_n)


    # PRO-control Eq. (2)
    def compute_excitation(self, stimuli, ro_predictions):
        """Compute excitation term of activation"""
        direct_term = torch.matmul(stimuli, self.W_C)
        proactive_term = torch.clamp(torch.matmul(-ro_predictions, self.W_F), min=0)
        reactive_term = torch.sum(torch.clamp(-self.W_R, min=0), dim=1)
        excitation = self.rho * (direct_term + proactive_term + reactive_term)
        return excitation

    # PRO-control Eq. (3)
    def compute_inhibition(self, stimuli, ro_predictions):
        """Compute inhibition term of activation"""

        # Direct inhibition
        direct_inhib = self.psi * torch.matmul(stimuli, self.W_I)

        # Control Inhibition
        proactive_inhib = torch.clamp(torch.matmul(ro_predictions, self.W_F), min=0)
        reactive_inhib = torch.sum(torch.clamp(self.W_R, min=0), dim=1)
        control_inhib = self.phi * (proactive_inhib + reactive_inhib)

        inhibition = direct_inhib + control_inhib
        return inhibition

    # Pro Eq. (11) and PRO-Control Eq. (2-3)
    # May want to change this later to use different activation function
    def compute_response_activation(self, stimuli, ro_predictions):
        """Compute response activation (equations 5-7)"""

        excitation = self.compute_excitation(stimuli, ro_predictions)
        
        inhibition = self.compute_inhibition(stimuli, ro_predictions)

        # Compute activation change PRO Eq. (11)
        noise = torch.normal(0, self.sigma, self.C.shape, device=self.device)
        delta_C = self.beta * self.dt * (
            excitation * (1 - self.C) - 
            (self.C + 0.05) * (inhibition + 1) + noise
        )
        
        # Update activation
        self.C = torch.clamp(self.C + delta_C, min=0, max=1)
        return self.C

    def forward(self, stimuli):
        # Predict R-O conjunctions
        ro_predictions = self.compute_ro_prediction(stimuli)

        # Update temporal components
        self.update_temporal_components(stimuli)

        # Compute temporal prediction
        temporal_prediction = self.compute_temporal_prediction()
        
        # Generate response
        response = self.compute_response_activation(stimuli, ro_predictions)

        response_discrete = (response > self.response_threshold).float() # Thresholding
        
        return response, ro_predictions, temporal_prediction, response_discrete

    # Update R-O conjunction weights PRO-Control Eq. (1)
    def update_prediction_weights(self, stimuli, learning_rate, actual_ro, ro_predictions, gating_signal=True):
        """Update prediction weights based on experience"""

        print(f"effective lr: {learning_rate}")
        print(f"stimuli: {stimuli}")
        print(f"self.W_S: {self.W_S}")
        print(f"actual_ro: {actual_ro}")
        print(f"ro_predictions: {ro_predictions}")
        ro_error = (self.theta * actual_ro - ro_predictions)
        print(f"ro_error: {ro_error}")
        self.W_S.data += gating_signal * torch.outer( learning_rate * ro_error, stimuli)

    def update_weights(self, stimuli, response, actual_ro, reward, valence, gating_signal):  # Added target_valence and gating_signal
        """Update all weights based on experience"""
        # Get current predictions
        ro_predictions = self.compute_ro_prediction(stimuli) # Get expected valence
        temporal_prediction = self.compute_temporal_prediction()
        
        # Compute surprise
        omega_p, omega_n = self.compute_surprise(ro_predictions, actual_ro)

        # Compute effective learning rate
        effective_lr = self.compute_effective_learning_rate(omega_p, omega_n)
        
        self.update_prediction_weights(stimuli, effective_lr, actual_ro, ro_predictions, gating_signal)

        # Update temporal prediction weights (Equation 9)
        td_error = (reward + self.gamma * self.compute_temporal_prediction() - temporal_prediction) # Equation 7
        td_error = td_error.reshape(-1, 1, 1)
        self.U.data += (self.alpha_td * td_error * self.eligibility_trace)  # Equation 9
        self.U.data.clamp_(-1, 1)
        self.U.data.clamp_(-1, 1)

        # Update proactive control weights (equation 14)
        response_mask = (response > self.response_threshold).float()
        delta_w_f = 0.01 * torch.outer(actual_ro, response_mask) * valence * gating_signal # Include gating signal
        self.W_F.data += delta_w_f

        # Update reactive control weights (Equation 4 - with gating signal)
        delta_w_r = valence * torch.ger(response, omega_n) * gating_signal # Include valence and gating signal
        self.W_R.data = 0.25 * (self.W_R + delta_w_r)
        self.W_R.data.clamp_(-1, 1)
        
        return {
            'omega_p': omega_p,
            'omega_n': omega_n,
            'td_error': td_error,
        }

class GoNoGoTask:
    def __init__(self, p_go=0.7):
        """
        Initialize Go/No-Go task
        p_go: probability of Go trial (default 0.7 for standard Go bias)
        """
        self.p_go = p_go
        self.n_stimuli = 2  # [Go stimulus, NoGo stimulus]
        self.n_responses = 2  # [Go response, NoGo response]
        self.n_outcomes = 2  # [Correct, Incorrect]
    
    def generate_trial(self):
        """Generate a single trial"""
        is_go_trial = np.random.random() < self.p_go
        
        # Create stimulus vector [Go, NoGo]
        stimulus = torch.zeros(self.n_stimuli)
        stimulus[0 if is_go_trial else 1] = 1.0
        
        # Create correct response vector [Go, NoGo]
        correct_response = torch.zeros(self.n_responses)
        correct_response[0 if is_go_trial else 1] = 1.0
        
        return stimulus, correct_response, is_go_trial

class ChangeSignalTask:
    def __init__(self, change_prob=0.3, change_delay_range=(130, 330)):
        self.change_prob = change_prob
        self.change_delay_range = change_delay_range
        self.n_stimuli = 4  # [Go Left, Go Right, Change Left, Change Right]
        self.n_responses = 2  # [Left, Right]
        self.n_outcomes = 2  # [Correct, Error]

    def generate_trial(self):
        is_change_trial = np.random.rand() < self.change_prob
        initial_direction = np.random.choice([0, 1])  # 0 for Left, 1 for Right

        stimulus = torch.zeros(self.n_stimuli)
        stimulus[initial_direction] = 1.0

        if is_change_trial:
            change_delay = np.random.randint(*self.change_delay_range)
            change_direction = 1 - initial_direction
            stimulus[2 + change_direction] = 1.0  # Activate change stimulus

            correct_response = torch.zeros(self.n_responses)
            correct_response[change_direction] = 1.0
        else:
            correct_response = torch.zeros(self.n_responses)
            correct_response[initial_direction] = 1.0

        return stimulus, correct_response, is_change_trial

class ForagingTask:
    def __init__(self, n_forage_options=6, forage_cost=0.1):
        self.n_forage_options = n_forage_options
        self.forage_cost = forage_cost
        self.n_stimuli = 10  # 10 bins for relative foraging value
        self.n_responses = 2  # [Engage, Forage]
        self.n_outcomes = 2  # [Correct, Error]

    def generate_trial(self):
        # Generate engage option values
        engage_values = np.random.rand(2)

        # Generate forage option values
        forage_values = np.random.rand(self.n_forage_options)

        # Calculate relative foraging value
        relative_foraging_value = np.mean(forage_values) - np.mean(engage_values) - self.forage_cost

        # Discretize relative foraging value into stimulus bins
        bin_index = int(np.clip((relative_foraging_value + 1) * 5, 0, 9))  # Scale and clip to [0, 9]
        stimulus = torch.zeros(self.n_stimuli)
        stimulus[bin_index] = 1.0

        return stimulus, engage_values, forage_values

class RiskAvoidanceTask:
    def __init__(self, risky_win_prob=0.5):
        self.risky_win_prob = risky_win_prob
        self.n_stimuli = 1  # Single stimulus indicating trial onset
        self.n_responses = 2  # [Safe, Risky]
        self.n_outcomes = 2  # [Correct, Error]

    def generate_trial(self):
        stimulus = torch.ones(self.n_stimuli) # Always 1
        return stimulus

class CorrelatedOutcomesTask:
    def __init__(self, p_switch_values):
        self.p_switch_values = p_switch_values  # List of p(switch) values
        self.n_stimuli = 1  # Not used in this task, but kept for consistency
        self.n_responses = 2  # Two options
        self.n_outcomes = 2  # [Win, Loss] (simplified)

    def generate_trial(self, current_correct_option, p_switch):
        stimulus = torch.ones(self.n_stimuli)  # Dummy stimulus

        # Determine if the correct option switches
        if np.random.rand() < p_switch:
            current_correct_option = 1 - current_correct_option

        return stimulus, current_correct_option

class GoNoGoTrainer:
    def __init__(self, model, task, learning_rate=0.001, or_learning_rate=0.001):
        self.model = model
        self.task = task
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Metrics tracking
        self.metrics = defaultdict(list)
        
    def evaluate_response(self, response, correct_response):
        """Evaluate the model's response"""
        # Get the actual response (Go or NoGo) based on threshold
        response_made = (response > self.model.response_threshold).float()
        
        # Check if response matches correct response
        is_correct = torch.all(response_made == correct_response)
        
        # Determine specific trial outcomes
        hit = correct_response[0] == 1 and response_made[0] == 1  # Correct Go
        miss = correct_response[0] == 1 and response_made[0] == 0  # Missed Go
        correct_reject = correct_response[1] == 1 and response_made[1] == 1  # Correct NoGo
        false_alarm = correct_response[1] == 1 and response_made[0] == 1  # Incorrect Go
        
        return {
            'correct': is_correct,
            'hit': hit,
            'miss': miss,
            'correct_reject': correct_reject,
            'false_alarm': false_alarm,
            'response_made': response_made
        }
    
    def run_trial(self, stimulus, correct_response, is_go_trial=None):
        """Run a single trial"""
        self.optimizer.zero_grad()

        # Forward pass
        response, pred_outcomes, pred_temporal, response_discrete = self.model(stimulus) # Call forward ONLY ONCE

        # Evaluate response
        eval_results = self.evaluate_response(response, correct_response)

        # Create outcome vector
        actual_outcomes = torch.zeros(self.task.n_responses * self.task.n_outcomes, device=device)
        response_idx = int(torch.argmax(eval_results['response_made']).item())
        outcome_idx = 0 if eval_results['correct'] else 1
        actual_outcomes[response_idx * self.task.n_outcomes + outcome_idx] = 1.0

        # Reward and Valence (task-specific - needs to be implemented for each task)
        reward, valence = self.get_reward_valence(eval_results, is_go_trial)
        # Important: Detach valence to avoid double gradients through OutcomeRepresentation

        # Gating signal
        gating_signal = 1.0

        # Calculate expected_valence (task-specific - depends on timing of outcomes)
        actual_values = reward + self.model.gamma * self.model.compute_temporal_prediction() # Placeholder - replace with task-specific target
        actual_values = actual_values.detach() # Detach to avoid gradients through temporal prediction pathway

        # Update weights (get losses from here)
        update_results = self.model.update_weights(stimulus, response, actual_outcomes, reward, valence, gating_signal)

        # Compute losses (using update_results)
        losses = self.model.compute_losses(response, pred_outcomes, pred_temporal, actual_outcomes)
        losses = {k: v.item() for k, v in losses.items()} # Convert to items
        
        # Backward pass would go here if we weren't manually updating...

        return losses, eval_results

    def get_reward_valence(self, eval_results, is_go_trial):
        """Get reward and valence - Task-Specific implementation"""
        if isinstance(self.task, GoNoGoTask):
            reward = 1.0 if eval_results['correct'] else -0.5
            valence = torch.tensor([reward], device=device).float()
            return reward, valence
        # ... (Implement for other tasks)
        return 0.0, torch.tensor([0.0], device=device).float()  # Default

    def train_epoch(self, n_trials):
        """Train for one epoch"""
        epoch_metrics = defaultdict(list)

        if isinstance(self.task, ForagingTask):
            # Foraging task specific logic
            for _ in range(n_trials):
                stimulus, engage_values, forage_values = self.task.generate_trial()
                stimulus = stimulus.to(device).float()

                correct_response = torch.tensor([0.0, 1.0], device=device) if np.mean(forage_values) > np.mean(engage_values) else torch.tensor([1.0, 0.0], device=device)
                losses, eval_results = self.run_trial(stimulus, correct_response)

        elif isinstance(self.task, RiskAvoidanceTask):
            # Risk avoidance task specific logic
            for _ in range(n_trials):
                stimulus = self.task.generate_trial().to(device)

                # ... (Risk avoidance task specific training logic)
        elif isinstance(self.task, ChangeSignalTask):
            # Change signal task specific logic
            for _ in range(n_trials):
                stimulus, correct_response, is_change_trial = self.task.generate_trial()
                stimulus, correct_response = stimulus.to(device), correct_response.to(device)

                # ... (Change signal task specific training logic)
        elif isinstance(self.task, CorrelatedOutcomesTask):
            # Correlated Outcomes Task specific logic
            current_correct_option = 0  # Initialize the correct option
            for p_switch in self.task.p_switch_values:
                for _ in range(n_trials):
                    stimulus, current_correct_option = self.task.generate_trial(current_correct_option, p_switch)
                    stimulus = stimulus.to(device)
                    # ... (Correlated Outcomes Task specific training logic)
        else:  # Go/No-Go task (default)
            for _ in range(n_trials):
                stimulus, correct_response, is_go_trial = self.task.generate_trial()
                stimulus, correct_response = stimulus.to(device), correct_response.to(device)

                # Run trial ONLY ONCE
                losses, eval_results = self.run_trial(stimulus, correct_response, is_go_trial) # Call run_trial ONLY ONCE

                # Record metrics
                for key in losses:
                    epoch_metrics[key].append(losses[key])
                epoch_metrics['loss'].append(losses['total_loss'])
                epoch_metrics['correct'].append(float(eval_results['correct']))
                epoch_metrics['hit'].append(float(eval_results['hit']))
                epoch_metrics['miss'].append(float(eval_results['miss']))
                epoch_metrics['correct_reject'].append(float(eval_results['correct_reject']))
                epoch_metrics['false_alarm'].append(float(eval_results['false_alarm']))
                
            # Average metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def train(self, n_epochs, trials_per_epoch):
        """Full training loop"""
        for epoch in range(n_epochs):
            self.model.train()
            metrics = self.train_epoch(trials_per_epoch)
            
            # Store metrics
            for k, v in metrics.items():
                self.metrics[k].append(v)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}")
                print(f"Loss: {metrics['loss']:.4f}")
                print(f"Accuracy: {metrics['correct']:.4f}")
                print(f"Hit Rate: {metrics['hit']:.4f}")
                print(f"False Alarm Rate: {metrics['false_alarm']:.4f}")
                print("---")
    
    def plot_learning_curves(self):
        """Plot learning curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot loss
        axes[0, 0].plot(self.metrics['loss'])
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        
        # Plot accuracy
        axes[0, 1].plot(self.metrics['correct'])
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        
        # Plot hit rate and false alarm rate
        axes[1, 0].plot(self.metrics['hit'], label='Hit Rate')
        axes[1, 0].plot(self.metrics['false_alarm'], label='False Alarm Rate')
        axes[1, 0].set_title('Hit Rate vs False Alarm Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        
        # Plot miss rate and correct reject rate
        axes[1, 1].plot(self.metrics['miss'], label='Miss Rate')
        axes[1, 1].plot(self.metrics['correct_reject'], label='Correct Reject Rate')
        axes[1, 1].set_title('Miss Rate vs Correct Reject Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig

# Example usage
def run_experiment(task, n_epochs=100, trials_per_epoch=100, n_delay_units=20):
    
    # Initialize model
    model = PROControl(
        n_stimuli=task.n_stimuli,
        n_responses=task.n_responses,
        n_outcomes=task.n_outcomes,
        n_delay_units=n_delay_units,
        device=device
    )
    
    # Initialize trainer
    trainer = GoNoGoTrainer(model, task)
    
    # Train model
    trainer.train(n_epochs, trials_per_epoch)
    
    # Plot results
    fig = trainer.plot_learning_curves()
    plt.show()
    
    return model, trainer

if __name__ == "__main__":
    # Run Go/No-Go task
    go_nogo_task = GoNoGoTask(p_go=0.7)
    run_experiment(go_nogo_task)

    # Run Change Signal Task
    change_signal_task = ChangeSignalTask()
    run_experiment(change_signal_task)

    # Run Foraging Task
    foraging_task = ForagingTask()
    run_experiment(foraging_task)

    # Run Risk Avoidance Task
    risk_avoidance_task = RiskAvoidanceTask()
    run_experiment(risk_avoidance_task)

    # Run Correlated Outcomes Task
    p_switch_values = np.linspace(0.1, 0.9, 9)  # Example p(switch) values
    correlated_outcomes_task = CorrelatedOutcomesTask(p_switch_values)
    run_experiment(correlated_outcomes_task)