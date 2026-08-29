"""Bolt8: consumer-GPU runtime for 200B-class FP8 GGUF MoE models."""

from bolt8.config import ModelConfig, PROFILES
from bolt8.engine import Bolt8Engine

__all__ = ["Bolt8Engine", "ModelConfig", "PROFILES", "__version__"]
__version__ = "0.1.0"
