"""
Medium Task: Multi-Modal VAE for Music Clustering
Audio + Lyrics Feature Fusion

Author: Moin Mostakim
Course: Neural Networks
"""

__version__ = "1.0.0"
__author__ = "Moin Mostakim"

# Import main modules
from . import data_loader
from . import audio_processor
from . import lyrics_processor
from . import audio_vae
from . import lyrics_vae
from . import multimodal_fusion
from . import clustering_advanced
from . import evaluation_advanced
from . import visualization_advanced

__all__ = [
    'data_loader',
    'audio_processor',
    'lyrics_processor',
    'audio_vae',
    'lyrics_vae',
    'multimodal_fusion',
    'clustering_advanced',
    'evaluation_advanced',
    'visualization_advanced'
]
