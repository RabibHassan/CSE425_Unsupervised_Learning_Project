"""
Conditional Variational Autoencoder (CVAE)
Conditions on genre labels for controlled generation
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


class ConditionalVAE(nn.Module):
    """
    Conditional VAE that takes both features and labels as input.
    Enables controlled generation and better disentanglement.
    """
    
    def __init__(self, input_dim, num_classes, latent_dim=32, hidden_dims=[128, 64]):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder: [input + one-hot labels] -> latent
        encoder_input_dim = input_dim + num_classes
        encoder_layers = []
        prev_dim = encoder_input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder: [latent + one-hot labels] -> reconstruction
        decoder_input_dim = latent_dim + num_classes
        decoder_layers = []
        prev_dim = decoder_input_dim
        
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
    
    def encode(self, x, labels_onehot):
        """Encode input with label conditioning."""
        x_cond = torch.cat([x, labels_onehot], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels_onehot):
        """Decode with label conditioning."""
        z_cond = torch.cat([z, labels_onehot], dim=1)
        return self.decoder(z_cond)
    
    def forward(self, x, labels_onehot):
        """Full forward pass with conditioning."""
        mu, logvar = self.encode(x, labels_onehot)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, labels_onehot)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=0.5):
        """CVAE loss = Reconstruction + KL divergence."""
        recon_loss = nn.MSELoss(reduction='sum')(recon_x, x) / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


class CVAEWrapper:
    """Wrapper for Conditional VAE with sklearn-like API."""
    
    def __init__(self, input_dim, num_classes, latent_dim=32, hidden_dims=[128, 64],
                 max_iter=500, batch_size=64, learning_rate=0.001,
                 beta=0.5, random_state=42, verbose=True):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ConditionalVAE(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
    
    def _labels_to_onehot(self, labels):
        """Convert integer labels to one-hot encoding."""
        n_samples = len(labels)
        onehot = np.zeros((n_samples, self.num_classes))
        onehot[np.arange(n_samples), labels] = 1
        return onehot
    
    def fit(self, X, labels, y=None):
        """Train the CVAE with label conditioning."""
        if self.verbose:
            print("\n" + "="*70)
            print("Training Conditional VAE (CVAE)")
            print("="*70)
            print(f"Input dimensions: {X.shape[1]}")
            print(f"Number of classes: {self.num_classes}")
            print(f"Latent dimensions: {self.latent_dim}")
            print(f"Hidden layers: {self.hidden_dims}")
            print(f"Training samples: {X.shape[0]}")
            print(f"Device: {self.device}")
            print(f"Beta (KL weight): {self.beta}")
        
        # Convert labels to one-hot
        labels_onehot = self._labels_to_onehot(labels)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        labels_tensor = torch.FloatTensor(labels_onehot).to(self.device)
        
        dataset = TensorDataset(X_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        
        for epoch in range(self.max_iter):
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            
            for batch_idx, (data, labels_batch) in enumerate(dataloader):
                data = data.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data, labels_batch)
                loss, recon_loss, kl_loss = self.model.loss_function(
                    recon_batch, data, mu, logvar, beta=self.beta
                )
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
            
            if self.verbose and (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / len(dataloader)
                avg_recon = epoch_recon / len(dataloader)
                avg_kl = epoch_kl / len(dataloader)
                print(f"Epoch {epoch+1}/{self.max_iter} - "
                      f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
        
        if self.verbose:
            print("\n" + "="*70)
            print("✓ CVAE training complete!")
            print("="*70)
        
        return self
    
    def encode(self, X, labels):
        """Encode data with label conditioning."""
        self.model.eval()
        labels_onehot = self._labels_to_onehot(labels)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            labels_tensor = torch.FloatTensor(labels_onehot).to(self.device)
            mu, _ = self.model.encode(X_tensor, labels_tensor)
            return mu.cpu().numpy()
    
    def decode(self, z, labels):
        """Decode with label conditioning."""
        self.model.eval()
        labels_onehot = self._labels_to_onehot(labels)
        
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z).to(self.device)
            labels_tensor = torch.FloatTensor(labels_onehot).to(self.device)
            recon = self.model.decode(z_tensor, labels_tensor)
            return recon.cpu().numpy()
    
    def reconstruct(self, X, labels):
        """Reconstruct input with label conditioning."""
        z = self.encode(X, labels)
        return self.decode(z, labels)
    
    def compute_reconstruction_error(self, X, labels):
        """Compute reconstruction MSE."""
        X_recon = self.reconstruct(X, labels)
        return np.mean((X - X_recon) ** 2)
    
    def save(self, filepath):
        """Save model."""
        model_data = {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'model_state_dict': self.model.state_dict(),
            'beta': self.beta,
            'random_state': self.random_state
        }
        joblib.dump(model_data, filepath)
        if self.verbose:
            print(f"✓ CVAE saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, verbose=True):
        """Load model."""
        model_data = joblib.load(filepath)
        wrapper = cls(
            input_dim=model_data['input_dim'],
            num_classes=model_data['num_classes'],
            latent_dim=model_data['latent_dim'],
            hidden_dims=model_data['hidden_dims'],
            beta=model_data.get('beta', 0.5),
            random_state=model_data['random_state'],
            verbose=verbose
        )
        wrapper.model.load_state_dict(model_data['model_state_dict'])
        if verbose:
            print(f"✓ CVAE loaded from {filepath}")
        return wrapper


def train_conditional_vae(features, labels, latent_dim=32, hidden_dims=[128, 64],
                          max_iter=500, batch_size=64, beta=0.5,
                          save_model=True, model_path='models/cvae.pkl',
                          verbose=True):
    """
    Train Conditional VAE.
    
    Args:
        features: Input features (n_samples, n_features)
        labels: Genre labels (n_samples,) - integers 0 to num_classes-1
        latent_dim: Latent space dimensionality
        hidden_dims: Hidden layer sizes
        max_iter: Training epochs
        batch_size: Batch size
        beta: KL divergence weight
        save_model: Whether to save model
        model_path: Save path
        verbose: Print progress
    
    Returns:
        model: Trained CVAE
        latent_features: Encoded representations
    """
    if verbose:
        print("\n" + "="*70)
        print("CONDITIONAL VAE (CVAE) TRAINING PIPELINE")
        print("="*70)
    
    num_classes = len(np.unique(labels))
    
    model = CVAEWrapper(
        input_dim=features.shape[1],
        num_classes=num_classes,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        max_iter=max_iter,
        batch_size=batch_size,
        beta=beta,
        random_state=42,
        verbose=verbose
    )
    
    model.fit(features, labels)
    
    if verbose:
        print("\nEncoding features to latent space...")
    
    latent_features = model.encode(features, labels)
    
    if verbose:
        print(f"✓ Latent features shape: {latent_features.shape}")
        print(f"  Mean: {latent_features.mean():.4f}")
        print(f"  Std: {latent_features.std():.4f}")
        
        recon_error = model.compute_reconstruction_error(features, labels)
        print(f"\n✓ Reconstruction error (MSE): {recon_error:.6f}")
    
    if save_model:
        model.save(model_path)
    
    if verbose:
        print("\n" + "="*70)
        print("✓ CVAE training complete!")
        print("="*70)
    
    return model, latent_features
