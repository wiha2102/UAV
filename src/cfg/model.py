"""
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple, Type


# ============================================================
#       Utility Functions
# ============================================================

def _validate_layers(name: str, layers: Tuple[int, ...]) -> None:
    """Validate layer tuple."""
    if not layers:
        raise ValueError(f"`{name}` layers cannot be empty")

    if any(layer <= 0 for layer in layers):
        raise ValueError(f"All `{name}` layers must be positive integers")


# ============================================================
#       Model Type Enum
# ============================================================

class ModelType(str, Enum):
    VAE = "vae"


# ============================================================
#       Base Model Configuration
# ============================================================

@dataclass
class ModelConfig:
    """
    Base configurations for all models
    """
    # Model Architecture
    n_latent: int = 10
    min_variance: float = 1e-4
    dropout_rate: float = 0.20

    # Weight Initialization
    init_kernel: float = 10.0
    init_bias: float = 10.0

    # Training Parameters
    learning_rate: float = 1e-3
    batch_size: int = 512
    epochs: int = 100

    # Regularization
    l2_regular: float = 1e-4

    def __post_init__(self) -> None:
        """Validate configuration parameters."""

        if self.n_latent <= 0:
            raise ValueError("n_latent must be positive")

        if self.min_variance < 0:
            raise ValueError("minimum variance cannot be negative")

        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be within [0.0, 1.0]")

        if self.init_kernel <= 0 or self.init_bias <= 0:
            raise ValueError("initialization values must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.l2_regular < 0:
            raise ValueError("l2_regular cannot be negative")
    

    @property
    def model_type(self) -> str:
        """Infer model type from class name."""
        return self.__class__.__name__.replace("Config", "").lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""

        valid_fields = {f.name for f in fields(cls)}

        filtered = {
            k: v for k, v in config_dict.items()
            if k in valid_fields
        }

        return cls(**filtered)


# ============================================================
#       VAE Configuration
# ============================================================


@dataclass
class VaeConfig(ModelConfig):
    """
    Configuration for Variational Autoencoder.
    """
    # Architecture
    encoder_layers: Tuple[int, ...] = (200, 80)
    decoder_layers: Tuple[int, ...] = (80, 200)

     # VAE-specific parameters
    beta: float = 0.50
    beta_annealing_step: int = 100_000
    kl_warmup_steps: int = 20

    reconstruction_weight: float = 1.0


    def __post_init__(self) -> None:

        super().__post_init__()

        _validate_layers("encoder", self.encoder_layers)
        _validate_layers("decoder", self.decoder_layers)

        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie within [0.0, 1.0]")

        if self.beta_annealing_step <= 0:
            raise ValueError("beta_annealing_step must be positive")

        if self.kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps cannot be negative")

        if self.reconstruction_weight <= 0:
            raise ValueError("reconstruction_weight must be positive")
    
    @property
    def total_layers(self) -> int:
        """Total network layers including latent projections."""
        return len(self.encoder_layers) + len(self.decoder_layers) + 2


# ============================================================
#       Model Registry
# ============================================================

MODEL_CONFIGS: Dict[ModelType, Type[ModelConfig]] = {
    ModelType.VAE: VaeConfig,
}


# ============================================================
#       Factory Function
# ============================================================

def get_config(model_type: Optional[str | ModelType]) -> ModelConfig:
    """
    Retrieve default configuration for a model type.
    """
    if model_type is None:
        raise ValueError("model_type must be specified")

    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower().strip())

    try:
        config_cls = MODEL_CONFIGS[model_type]
        return config_cls()

    except KeyError:

        supported = ", ".join(m.value for m in MODEL_CONFIGS)
        raise ValueError(
            f"Unknown model '{model_type}'. Supported: {supported}"
        ) from None
