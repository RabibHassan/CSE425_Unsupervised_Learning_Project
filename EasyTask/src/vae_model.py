"""
VAE Model Implementation
Basic Variational Autoencoder using scikit-learn's MLPRegressor
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle


class BasicVAE:
    """
    Basic Variational Autoencoder for music feature extraction.
    Uses MLPRegressor with bottleneck architecture to learn latent representations.
    """

    def __init__(self, input_dim=58, latent_dim=16, hidden_dim=128):
        """
        Initialize VAE parameters.

        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space (bottleneck)
            hidden_dim: Dimension of hidden layers
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self):
        """Build autoencoder architecture with bottleneck."""
        # Architecture: input -> 128 -> 64 -> 16 -> 64 -> 128 -> output
        self.model = MLPRegressor(
            hidden_layer_sizes=(
                self.hidden_dim, 
                self.hidden_dim // 2, 
                self.latent_dim, 
                self.hidden_dim // 2, 
                self.hidden_dim
            ),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        print(f"VAE Model built with architecture:")
        print(f"  Input: {self.input_dim}")
        print(f"  Hidden: {self.hidden_dim} -> {self.hidden_dim // 2} -> {self.latent_dim}")
        print(f"  Latent: {self.latent_dim}")

    def fit(self, X):
        """
        Train the VAE model.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Standardize input
        X_scaled = self.scaler.fit_transform(X)

        print("Training VAE...")
        self.model.fit(X_scaled, X_scaled)

        print(f"Training completed!")
        print(f"  Iterations: {self.model.n_iter_}")
        print(f"  Final loss: {self.model.loss_:.4f}")

        return {
            'n_iter': self.model.n_iter_,
            'loss': self.model.loss_
        }

    def encode(self, X):
        """
        Encode input to latent space.

        Args:
            X: Input features

        Returns:
            Latent representations
        """
        X_scaled = self.scaler.transform(X)

        # Forward pass through encoder (first 3 layers to bottleneck)
        h = X_scaled
        for i in range(3):
            h = np.maximum(0, h @ self.model.coefs_[i] + self.model.intercepts_[i])

        return h

    def decode(self, z):
        """
        Decode latent representations back to input space.

        Args:
            z: Latent representations

        Returns:
            Reconstructed features
        """
        # Forward pass through decoder (last 3 layers)
        h = z
        for i in range(3, len(self.model.coefs_)):
            if i == len(self.model.coefs_) - 1:
                # Last layer, no activation
                h = h @ self.model.coefs_[i] + self.model.intercepts_[i]
            else:
                h = np.maximum(0, h @ self.model.coefs_[i] + self.model.intercepts_[i])

        # Inverse transform
        h = self.scaler.inverse_transform(h)
        return h

    def reconstruct(self, X):
        """
        Reconstruct input through encode-decode.

        Args:
            X: Input features

        Returns:
            Reconstructed features
        """
        z = self.encode(X)
        return self.decode(z)

    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'params': {
                    'input_dim': self.input_dim,
                    'latent_dim': self.latent_dim,
                    'hidden_dim': self.hidden_dim
                }
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.scaler = data['scaler']
        self.input_dim = data['params']['input_dim']
        self.latent_dim = data['params']['latent_dim']
        self.hidden_dim = data['params']['hidden_dim']

        print(f"Model loaded from {filepath}")