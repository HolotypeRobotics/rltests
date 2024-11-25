
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
        self.optimizer_or = optim.Adam(self.outcome_rep.parameters(), lr=0.001) # Optimizer for outcome representation

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
        
        # Temporal prediction weights
        self.outcome_rep = OutcomeRepresentation(n_stimuli, self.n_ro_conjunctions).to(device)
         
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