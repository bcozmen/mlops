import numpy as np
from sklearn.neighbors import KernelDensity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import os

class KDEModel:
    def __init__(self, bandwidth=0.03, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = None

    def __call__(self, X):
        """Receives the FULL updated dataset from the API and fits completely."""
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(X)

    def visualize(self, grid_size=100):
        if self.kde is None:
            raise ValueError("Model has not been fitted with any data yet.")
        
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
        Z = np.exp(self.kde.score_samples(grid_points)).reshape(X_grid.shape)
        Z /= Z.sum()  # Normalize to get a proper density
        return X_grid, Y_grid, Z



class SpatialDensityNet(nn.Module):
    def __init__(self):
        super(SpatialDensityNet, self).__init__()
        # A deep MLP that maps 2D coordinates -> 1D unnormalized scalar density
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),          # Smooth activations work much better for density contours than ReLU
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)    # Outputs an raw unnormalized score
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

    
class NNModel():
    def __init__(self, bandwidth=0.03, grid_res=100, batch_size=32):
        self.model = SpatialDensityNet()
        
        # --- FIX 1: Add weight_decay to penalize sharp spikes and steep gradients ---
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        
        # --- FIX 2: Define a smoothing 'bandwidth' (noise standard deviation) ---
        self.bandwidth = bandwidth
        
        # Create a static grid tensor used to numerically integrate (normalize) the network outputs

        x_g = torch.linspace(0, 1, grid_res)
        y_g = torch.linspace(0, 1, grid_res)
        X_g, Y_g = torch.meshgrid(x_g, y_g, indexing='xy')
        self.integration_grid = torch.stack([X_g.ravel(), Y_g.ravel()], dim=-1)
        self.batch_size = batch_size

        if os.path.exists("nn_density_model.pth"):
            self.load_model("nn_density_model.pth")
        
        
    def train(self, X, epochs=3): 
        "receives the FULL updated dataset from the API and fits completely."
        self.model.train()

        for epoch in range(epochs):

            
            self.optimizer.zero_grad()
            batch_X = X[torch.randperm(X.size(0))][:self.batch_size]  # Randomly sample a batch of target points
            # --- FIX 3: Inject Gaussian noise to blur points into smooth regions ---
            # This replicates the continuous kernel effect from KDE
            noise = torch.randn_like(batch_X) * self.bandwidth
            jittered_X = torch.clamp(batch_X + noise, 0.0, 1.0) # Keep within [0, 1] bounds
            
            # 1. Forward pass on jittered target coordinates
            target_scores = self.model(jittered_X)
            
            # 2. Enforce total probability conservation via numerical integration
            grid_scores = self.model(self.integration_grid)
            log_partition_function = torch.logsumexp(grid_scores, dim=0)
            
            # 3. Log-likelihood calculation
            log_prob = target_scores - log_partition_function
            
            loss = -log_prob.mean()
            loss.backward()
            self.optimizer.step()

        self.save_model("nn_density_model.pth")

    def visualize(self, grid_size=100):
        self.model.eval()
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1
        x_grid = torch.linspace(x_min, x_max, grid_size)
        y_grid = torch.linspace(y_min, y_max, grid_size)
        X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='xy')
        grid_points = torch.stack([X_grid.ravel(), Y_grid.ravel()], dim=-1)
        
        with torch.no_grad():
            scores = self.model(grid_points)
            probabilities = F.softmax(scores, dim=0)
            Z = probabilities.reshape(X_grid.shape)
        return X_grid.numpy(), Y_grid.numpy(), Z.numpy()

    def save_model(self, path):
        parameters = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(parameters, path)
    def load_model(self, path):
        parameters = torch.load(path)
        self.model.load_state_dict(parameters['model_state_dict'])
        self.optimizer.load_state_dict(parameters['optimizer_state_dict'])