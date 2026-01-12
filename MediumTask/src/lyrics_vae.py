"""
True Variational Autoencoder for Lyrics Features
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


class TrueLyricsVAE(nn.Module):
    """
    Proper Variational Autoencoder for lyrics features.
    """
    
    def __init__(self, input_dim=100, latent_dim=16, hidden_dims=[512, 256, 128]):
        super(TrueLyricsVAE, self).__init__()
        
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
                nn.Dropout(0.3)
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
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=0.5):
        recon_loss = nn.MSELoss(reduction='sum')(recon_x, x) / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


class LyricsVAEWrapper:
    """Wrapper for lyrics VAE."""
    
    def __init__(self, input_dim, latent_dim=16, hidden_dims=[512, 256, 128],
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
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TrueLyricsVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
    
    def fit(self, X, y=None):
        if self.verbose:
            print("\n" + "="*70)
            print("Training True Variational Autoencoder (Lyrics)")
            print("="*70)
            print(f"Input dimensions: {X.shape[1]}")
            print(f"Latent dimensions: {self.latent_dim}")
            print(f"Hidden layers: {self.hidden_dims}")
            print(f"Training samples: {X.shape[0]}")
            print(f"Device: {self.device}")
            print(f"Beta (KL weight): {self.beta}")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        
        for epoch in range(self.max_iter):
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
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
            print("✓ VAE training complete!")
            print("="*70)
        
        return self
    
    def encode(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            mu, _ = self.model.encode(X_tensor)
            return mu.cpu().numpy()
    
    def decode(self, z):
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z).to(self.device)
            recon = self.model.decode(z_tensor)
            return recon.cpu().numpy()
    
    def reconstruct(self, X):
        z = self.encode(X)
        return self.decode(z)
    
    def compute_reconstruction_error(self, X):
        X_recon = self.reconstruct(X)
        return np.mean((X - X_recon) ** 2)
    
    def save(self, filepath):
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


def train_lyrics_vae(lyrics_features, latent_dim=16, hidden_dims=[512, 256, 128],
                     max_iter=500, batch_size=64, beta=0.5,
                     save_model=True, model_path='models/lyrics_dense_vae.pkl',
                     verbose=True):
    """Train lyrics VAE (TRUE VAE with KL divergence)."""
    if verbose:
        print("\n" + "="*70)
        print("LYRICS VAE TRAINING PIPELINE (True VAE)")
        print("="*70)
    
    model = LyricsVAEWrapper(
        input_dim=lyrics_features.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        max_iter=max_iter,
        batch_size=batch_size,
        beta=beta,
        random_state=42,
        verbose=verbose
    )
    
    model.fit(lyrics_features)
    
    if verbose:
        print("\nEncoding lyrics to latent space...")
    
    latent_features = model.encode(lyrics_features)
    
    if verbose:
        print(f"✓ Latent features shape: {latent_features.shape}")
        print(f"  Mean: {latent_features.mean():.4f}")
        print(f"  Std: {latent_features.std():.4f}")
        
        recon_error = model.compute_reconstruction_error(lyrics_features)
        print(f"\n✓ Reconstruction error (MSE): {recon_error:.6f}")
    
    if save_model:
        model.save(model_path)
    
    if verbose:
        print("\n" + "="*70)
        print("✓ Lyrics VAE training complete!")
        print("="*70)
    
    return model, latent_features
