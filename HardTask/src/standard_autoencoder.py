"""
Standard Autoencoder (Non-Variational)
Baseline comparison for VAE methods
Author: Moin Mostakim
Hard Task - Neural Networks Project
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import warnings
warnings.filterwarnings('ignore')


class StandardAutoencoder(nn.Module):
    """
    Standard Autoencoder without variational component.
    Deterministic encoding for baseline comparison.
    """
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64]):
        super(StandardAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        # Bottleneck (latent layer)
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z
    
    def loss_function(self, recon_x, x):
        """Simple reconstruction loss."""
        return nn.MSELoss()(recon_x, x)


class AutoencoderWrapper:
    """Wrapper for Standard Autoencoder."""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64],
                 max_iter=500, batch_size=64, learning_rate=0.001,
                 random_state=42, verbose=True):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = StandardAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
    
    def fit(self, X, y=None):
        """Train the autoencoder."""
        if self.verbose:
            print("\n" + "="*70)
            print("Training Standard Autoencoder (AE)")
            print("="*70)
            print(f"Input dimensions: {X.shape[1]}")
            print(f"Latent dimensions: {self.latent_dim}")
            print(f"Hidden layers: {self.hidden_dims}")
            print(f"Training samples: {X.shape[0]}")
            print(f"Device: {self.device}")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        
        for epoch in range(self.max_iter):
            epoch_loss = 0
            
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                recon_batch, _ = self.model(data)
                loss = self.model.loss_function(recon_batch, data)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{self.max_iter} - Loss: {avg_loss:.4f}")
        
        if self.verbose:
            print("\n" + "="*70)
            print("✓ Autoencoder training complete!")
            print("="*70)
        
        return self
    
    def encode(self, X):
        """Encode to latent space."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            z = self.model.encode(X_tensor)
            return z.cpu().numpy()
    
    def decode(self, z):
        """Decode from latent space."""
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z).to(self.device)
            recon = self.model.decode(z_tensor)
            return recon.cpu().numpy()
    
    def reconstruct(self, X):
        """Reconstruct input."""
        z = self.encode(X)
        return self.decode(z)
    
    def compute_reconstruction_error(self, X):
        """Compute reconstruction MSE."""
        X_recon = self.reconstruct(X)
        return np.mean((X - X_recon) ** 2)
    
    def save(self, filepath):
        """Save model."""
        model_data = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'model_state_dict': self.model.state_dict(),
            'random_state': self.random_state
        }
        joblib.dump(model_data, filepath)
        if self.verbose:
            print(f"✓ Autoencoder saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, verbose=True):
        """Load model."""
        model_data = joblib.load(filepath)
        wrapper = cls(
            input_dim=model_data['input_dim'],
            latent_dim=model_data['latent_dim'],
            hidden_dims=model_data['hidden_dims'],
            random_state=model_data['random_state'],
            verbose=verbose
        )
        wrapper.model.load_state_dict(model_data['model_state_dict'])
        if verbose:
            print(f"✓ Autoencoder loaded from {filepath}")
        return wrapper


def train_standard_autoencoder(features, latent_dim=32, hidden_dims=[128, 64],
                                max_iter=500, batch_size=64,
                                save_model=True, model_path='models/autoencoder.pkl',
                                verbose=True):
    """
    Train standard autoencoder.
    
    Args:
        features: Input features (n_samples, n_features)
        latent_dim: Latent space dimensionality
        hidden_dims: Hidden layer sizes
        max_iter: Training epochs
        batch_size: Batch size
        save_model: Whether to save
        model_path: Save path
        verbose: Print progress
    
    Returns:
        model: Trained autoencoder
        latent_features: Encoded representations
    """
    if verbose:
        print("\n" + "="*70)
        print("STANDARD AUTOENCODER TRAINING PIPELINE")
        print("="*70)
    
    model = AutoencoderWrapper(
        input_dim=features.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=42,
        verbose=verbose
    )
    
    model.fit(features)
    
    if verbose:
        print("\nEncoding features to latent space...")
    
    latent_features = model.encode(features)
    
    if verbose:
        print(f"✓ Latent features shape: {latent_features.shape}")
        print(f"  Mean: {latent_features.mean():.4f}")
        print(f"  Std: {latent_features.std():.4f}")
        
        recon_error = model.compute_reconstruction_error(features)
        print(f"\n✓ Reconstruction error (MSE): {recon_error:.6f}")
    
    if save_model:
        model.save(model_path)
    
    if verbose:
        print("\n" + "="*70)
        print("✓ Autoencoder training complete!")
        print("="*70)
    
    return model, latent_features
