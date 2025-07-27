from jsonargparse import auto_cli
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

import random
import json
import os

import torch
import numpy as np
import ray

from ose.modeling import Model, ModelConfig, VLLMModelConfig, VLLMInstructModelConfig, VLLMBaseModelConfig, EmbeddingModel, HFEmbeddingModelConfig
from ose.sampling import PromptSchema

@dataclass
class SchemaConfig:
    schema_file: str
    extra_args: Dict[str, Any] = None
    extra_args_p: Dict[str, Any] = None

def main(
        model_config: Union[VLLMInstructModelConfig, VLLMBaseModelConfig] = None,
        schema_config: SchemaConfig = None,
        n_samples: int = None,
        output_path: str = None,
        seed: Optional[int] = None,
        overwrite_output: bool = False):
    
    if output_path is not None and os.path.exists(output_path):
        if not overwrite_output:
            print(f"Output file {output_path} already exists. Skipping generation.")
            return
        else:
            print(f"Output file {output_path} already exists. Overwriting.")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    schema = PromptSchema.from_json(schema_config.schema_file, schema_config.extra_args, schema_config.extra_args_p)
    samples = schema.sample(n_samples)

    model = Model.from_config(model_config)
    prompts = [sample['formatted_prompt'] for sample in samples]
    outputs = model.generate(prompts)

    for i in range(len(samples)):
        samples[i]['output'] = outputs[i]

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')


def cli():
    auto_cli(main)

if __name__ == "__main__":
    cli()