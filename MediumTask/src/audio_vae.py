"""
True Variational Autoencoder for Audio Features
Proper VAE with KL divergence - NO PCA initialization
Author: Moin Mostakim
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import warnings
warnings.filterwarnings('ignore')


class TrueAudioVAE(nn.Module):
    """
    Proper Variational Autoencoder with KL divergence loss.
    NO PCA initialization - learns from scratch.
    """
    
    def __init__(self, input_dim=57, latent_dim=32, hidden_dims=[128, 64]):
        super(TrueAudioVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder
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
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
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
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=0.5):
        """
        VAE loss = Reconstruction loss + KL divergence.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL term (beta-VAE)
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss(reduction='sum')(recon_x, x) / x.size(0)
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class AudioVAEWrapper:
    """
    Wrapper to match sklearn-like API for compatibility.
    """
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64],
                 max_iter=500, batch_size=64, learning_rate=0.001,
                 beta=0.5, random_state=42, verbose=True):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = TrueAudioVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
    
    def fit(self, X, y=None):
        """Train the VAE."""
        if self.verbose:
            print("\n" + "="*70)
            print("Training True Variational Autoencoder (Audio)")
            print("="*70)
            print(f"Input dimensions: {X.shape[1]}")
            print(f"Latent dimensions: {self.latent_dim}")
            print(f"Hidden layers: {self.hidden_dims}")
            print(f"Training samples: {X.shape[0]}")
            print(f"Device: {self.device}")
            print(f"Beta (KL weight): {self.beta}")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.max_iter):
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar = self.model(data)
                
                # Compute loss
                loss, recon_loss, kl_loss = self.model.loss_function(
                    recon_batch, data, mu, logvar, beta=self.beta
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
            
            # Print progress
            if self.verbose and (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / len(dataloader)
                avg_recon = epoch_recon / len(dataloader)
                avg_kl = epoch_kl / len(dataloader)
                print(f"Epoch {epoch+1}/{self.max_iter} - "
                      f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
        
        if self.verbose:
            print("\n" + "="*70)
            print("✓ VAE training complete!")
            print("="*70)
        
        return self
    
    def encode(self, X):
        """Encode data to latent space (returns mean of distribution)."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            mu, _ = self.model.encode(X_tensor)
            return mu.cpu().numpy()
    
    def decode(self, z):
        """Decode latent vectors."""
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z).to(self.device)
            recon = self.model.decode(z_tensor)
            return recon.cpu().numpy()
    
    def reconstruct(self, X):
        """Reconstruct input data."""
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
            'beta': self.beta,
            'random_state': self.random_state
        }
        joblib.dump(model_data, filepath)
        if self.verbose:
            print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, verbose=True):
        """Load model."""
        model_data = joblib.load(filepath)
        
        wrapper = cls(
            input_dim=model_data['input_dim'],
            latent_dim=model_data['latent_dim'],
            hidden_dims=model_data['hidden_dims'],
            beta=model_data.get('beta', 0.5),
            random_state=model_data['random_state'],
            verbose=verbose
        )
        
        wrapper.model.load_state_dict(model_data['model_state_dict'])
        
        if verbose:
            print(f"✓ Model loaded from {filepath}")
        
        return wrapper


def train_audio_vae(audio_features, latent_dim=32, hidden_dims=[128, 64],
                    max_iter=500, batch_size=64, beta=0.5,
                    save_model=True, model_path='models/audio_conv_vae.pkl',
                    verbose=True):
    """
    Train audio VAE (TRUE VAE with KL divergence).
    
    Args:
        audio_features: Preprocessed audio features (n_samples, n_features)
        latent_dim: Latent space dimensionality
        hidden_dims: Hidden layer sizes
        max_iter: Maximum training epochs
        batch_size: Batch size for training
        beta: Weight for KL divergence (beta-VAE parameter)
        save_model: Whether to save trained model
        model_path: Path to save model
        verbose: Print training progress
    
    Returns:
        model: Trained VAE wrapper
        latent_features: Encoded latent representations
    """
    if verbose:
        print("\n" + "="*70)
        print("AUDIO VAE TRAINING PIPELINE (True VAE)")
        print("="*70)
    
    model = AudioVAEWrapper(
        input_dim=audio_features.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        max_iter=max_iter,
        batch_size=batch_size,
        beta=beta,
        random_state=42,
        verbose=verbose
    )
    
    model.fit(audio_features)
    
    if verbose:
        print("\nEncoding audio to latent space...")
    
    latent_features = model.encode(audio_features)
    
    if verbose:
        print(f"✓ Latent features shape: {latent_features.shape}")
        print(f"  Mean: {latent_features.mean():.4f}")
        print(f"  Std: {latent_features.std():.4f}")
        
        recon_error = model.compute_reconstruction_error(audio_features)
        print(f"\n✓ Reconstruction error (MSE): {recon_error:.6f}")
    
    if save_model:
        model.save(model_path)
    
    if verbose:
        print("\n" + "="*70)
        print("✓ Audio VAE training complete!")
        print("="*70)
    
    return model, latent_features
