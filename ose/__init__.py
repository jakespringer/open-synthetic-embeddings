"""
Open Synthetic Data Engine (OSE)

A library for generating synthetic data using large language models.
"""

from .modeling import (
    Model, 
    ModelConfig,
    VLLMModel, 
    VLLMModelConfig,
    VLLMInstructModel,
    VLLMInstructModelConfig,
    VLLMBaseModel,
    VLLMBaseModelConfig,
    EmbeddingModel,
    EmbeddingModelConfig,
    HFEmbeddingModel,
    HFEmbeddingModelConfig,
    RayEmbeddingModel
)

from .sampling import PromptSchema

__version__ = "0.1.0"
__author__ = "OSE Team"
__email__ = ""
__description__ = "Open Synthetic Data Engine - A library for generating synthetic data using large language models"

__all__ = [
    "Model", 
    "ModelConfig",
    "VLLMModel", 
    "VLLMModelConfig",
    "VLLMInstructModel",
    "VLLMInstructModelConfig", 
    "VLLMBaseModel",
    "VLLMBaseModelConfig",
    "EmbeddingModel",
    "EmbeddingModelConfig", 
    "HFEmbeddingModel",
    "HFEmbeddingModelConfig",
    "RayEmbeddingModel",
    "PromptSchema"
] 