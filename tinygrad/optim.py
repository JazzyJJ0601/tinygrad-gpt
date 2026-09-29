"""Optimizers for training."""
import numpy as np


class AdamW:
    """AdamW optimizer with decoupled weight decay."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Initialize state
        self.state = {}
        for p in self.params:
            self.state[p] = {
                'exp_avg': np.zeros_like(p.data),
                'exp_avg_sq': np.zeros_like(p.data),
            }
    
    def step(self):
        self.t += 1
        beta1, beta2 = self.betas
        
        for p in self.params:
            if p._grad is None:
                continue
            state = self.state[p]
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            
            # Update biased first moment estimate
            exp_avg = beta1 * exp_avg + (1 - beta1) * p._grad
            state['exp_avg'] = exp_avg
            
            # Update biased second raw moment estimate
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (p._grad ** 2)
            state['exp_avg_sq'] = exp_avg_sq
            
            # Bias correction
            bias_corr1 = 1 - beta1 ** self.t
            bias_corr2 = 1 - beta2 ** self.t
            exp_avg_hat = exp_avg / bias_corr1
            exp_avg_sq_hat = exp_avg_sq / bias_corr2
            
            # Compute update
            denom = np.sqrt(exp_avg_sq_hat) + self.eps
            step_size = self.lr * np.sqrt(bias_corr2) / bias_corr1
            
            # Apply weight decay (decoupled) and update
            p.data = p.data - step_size * exp_avg_hat / denom - self.weight_decay * self.lr * p.data
            
            # Reset gradient
            p._grad = None
    
    def zero_grad(self):
        for p in self.params:
            p._grad = None
