"""
src/cfg/model.py
"""
from __future__ import annotations

from enum import Enum
from dataclass import dataclass, field
from typing import ClassVar, Dict, Final, Tuple, Type


class ModelType(str, Enum):
    VAE     = "vae"



@dataclass
class ModelConfig:
    """ Base configuration for all models """
    # Model architecture
    n_latent        : int   = 10
    min_variance    : float = 1e-4
    dropout_rate    : float = 0.20

    # Weight initialization
    init_kernel     : float = 10.0
    init_bias       : float = 10.0

    # Training parameters
    learning_rate   : float = 1e-3
    batch_size      : int   = 512
    epochs          : int   = 100

    # Regularization l2 -- perhaps remove later
    l2_regular      : float = 1e-4

    def __post_init__(self):
        """ Validation of configuration parameters """
        if self.n_latent <= 0: 
            raise ValueError("n_latent must be positive")
        
        if self.min_variance < 0:
            raise ValueError("minimum variance cannot be negative")

        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("Dropout rate must be within [0.0, 1.0]")
        
        if self.init_kernel <= 0 or self.init_bias <= 0:
            raise ValueError("Initialization values must be positive")
        
        if self.learning_rate <= 0: 
            raise ValueError("Learning rate must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if self.l2_regular < 0:
            raise ValueError("L2 regularization cannot be negative")
    
    @property
    def model_type(self) -> str:
        return self.__class__.__name__.replace("Config","").lower()
    
    def to_dict(self) -> Dict[str, Any]: return {
            k: v for k, v in self.__dict__.items() if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> ModelConfig:
        """Create configuration from dictionary"""
        valid_fields = { f.name for f in field(cls) if f.init }
        fd = {
            k: v for k, v in config_dict.items() if k in valid_fields
        }
        return cls(**fd)
