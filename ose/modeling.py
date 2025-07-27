from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Type, Optional, Literal
from dataclasses import dataclass

import torch
import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig


@dataclass
class ModelConfig:
    pass

@dataclass
class VLLMModelConfig(ModelConfig):
    model_path: str = None
    engine_kwargs: Dict[str, Any] = None
    concurrency: int = None
    batch_size: int = None
    default_sampling_params: Dict[str, Any] = None


@dataclass 
class VLLMInstructModelConfig(VLLMModelConfig):
    model_type: Literal["vllm_instruct"] = "vllm_instruct"
    default_system_prompt: str = None


@dataclass
class VLLMBaseModelConfig(VLLMModelConfig):
    model_type: Literal["vllm_base"] = "vllm_base"


class Model(ABC):
    _models = {}

    @abstractmethod
    def generate(self, prompts, **kwargs):
        pass

    @staticmethod
    def from_config(model_config: ModelConfig):
        if model_config.model_type not in Model._models:
            raise ValueError(f"Model type {model_config.model_type} not found")
        model_cls = Model._models[model_config.model_type]
        return model_cls.from_model_config(model_config)

    @staticmethod
    def register_model(model_type: str, model_class: Type):
        Model._models[model_type] = model_class

    @classmethod
    @abstractmethod
    def from_model_config(cls, model_config: ModelConfig):
        pass


class VLLMModel(Model):
    def __init__(self, model_path: str, engine_kwargs: Dict[str, Any], concurrency: int, batch_size: int, default_sampling_params: Dict[str, Any]):
        self.model_path = model_path
        self.engine_kwargs = engine_kwargs
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.default_sampling_params = default_sampling_params

        self.engine_processor_config = vLLMEngineProcessorConfig(
            model_source=model_path,
            engine_kwargs=engine_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )

        self.processor = build_llm_processor(
            self.engine_processor_config,
            preprocess=self.preprocess,
            postprocess=self.postprocess)

    @abstractmethod
    def preprocess(self, row: Any):
        pass

    @abstractmethod
    def postprocess(self, row: Any):
        pass

    def generate(self, prompts: List[str], **kwargs):
        if len(prompts) == 0:
            return []

        def _update_sampling_params(p: Dict[str, Any]):
            sampling_params = self.default_sampling_params.copy()
            if 'sampling_params' in p:
                sampling_params.update(p['sampling_params'])
            return sampling_params

        if isinstance(prompts, List) and isinstance(prompts[0], str):
            ds = ray.data.from_items([{
                'text': p,
                'sampling_params': self.default_sampling_params,
                'id': i,
            } for i, p in enumerate(prompts)])
        elif isinstance(prompts, List) and isinstance(prompts[0], Dict):
            entries = []
            for i, p in enumerate(prompts):
                entries.append({
                    'text': p['text'],
                    'sampling_params': _update_sampling_params(p),
                    'id': i,
                })
            ds = ray.data.from_items(entries)
        elif isinstance(prompts, ray.data.Dataset):
            ds = prompts.map(lambda row: {
                'text': row['text'],
                'sampling_params': _update_sampling_params(row),
                'id': row['id'],
            })
        else:
            raise ValueError(f"Invalid prompts type: {type(prompts)}")

        results = list(self.processor(ds).take(len(prompts)))
        results = sorted(results, key=lambda x: x['id'])
        return [r['generated_text'] for r in results]

    @classmethod
    def from_model_config(cls, model_config: VLLMModelConfig):
        return cls(
            model_path=model_config.model_path,
            engine_kwargs=model_config.engine_kwargs,
            concurrency=model_config.concurrency,
            batch_size=model_config.batch_size,
            default_sampling_params=model_config.default_sampling_params,
        )

Model.register_model("vllm", VLLMModel)


class VLLMInstructModel(VLLMModel):
    def __init__(self, model_path: str, engine_kwargs: Dict[str, Any], concurrency: int, batch_size: int, default_sampling_params: Dict[str, Any], default_system_prompt: str):
        super().__init__(model_path, engine_kwargs, concurrency, batch_size, default_sampling_params)
        self.default_system_prompt = default_system_prompt

    def preprocess(self, row: Any):
        messages = []
        if self.default_system_prompt is not None:
            messages.append({"role": "system", "content": self.default_system_prompt})
        messages.append({"role": "user", "content": row['text']})
        return {
            'messages': messages,
            'sampling_params': row['sampling_params'],
        }

    def postprocess(self, row: Any):
        return row

    @classmethod
    def from_model_config(cls, model_config: ModelConfig):
        return cls(
            model_path=model_config.model_path,
            engine_kwargs=model_config.engine_kwargs,
            concurrency=model_config.concurrency,
            batch_size=model_config.batch_size,
            default_sampling_params=model_config.default_sampling_params,
            default_system_prompt=model_config.default_system_prompt,
        )

Model.register_model("vllm_instruct", VLLMInstructModel)


class VLLMBaseModel(VLLMModel):
    def __init__(self, model_path: str, engine_kwargs: Dict[str, Any], concurrency: int, batch_size: int, default_sampling_params: Dict[str, Any]):
        super().__init__(model_path, engine_kwargs, concurrency, batch_size, default_sampling_params)

    def preprocess(self, row: Any):
        return {
            'text': row['text'],
            'sampling_params': row['sampling_params'],
        }

    def postprocess(self, row: Any):
        return row

    @classmethod
    def from_model_config(cls, model_config: ModelConfig):
        return cls(
            model_path=model_config.model_path,
            engine_kwargs=model_config.engine_kwargs,
            concurrency=model_config.concurrency,
            batch_size=model_config.batch_size,
            default_sampling_params=model_config.default_sampling_params,
        )

Model.register_model("vllm_base", VLLMBaseModel)


@dataclass
class EmbeddingModelConfig:
    pass


@dataclass
class HFEmbeddingModelConfig(EmbeddingModelConfig):
    model_type: Literal["hf_embedding"]
    model_path: str
    batch_size: int
    device: str
    max_length: int
    concurrency: int
    pooling_strategy: Literal["mean", "cls"]


class EmbeddingModel(ABC):
    _embedding_models = {}

    @abstractmethod
    def embed(self, texts: List[str]) -> torch.Tensor:
        pass

    @staticmethod
    def from_config(model_config: EmbeddingModelConfig):
        if model_config.model_type not in EmbeddingModel._embedding_models:
            raise ValueError(f"Embedding model type {model_config.model_type} not found")
        model_cls = EmbeddingModel._embedding_models[model_config.model_type]
        return model_cls.from_embedding_config(model_config)

    @staticmethod
    def register_model(model_type: str, model_class: Type):
        EmbeddingModel._embedding_models[model_type] = model_class

    @classmethod
    @abstractmethod
    def from_embedding_config(cls, model_config: EmbeddingModelConfig):
        pass


class RayEmbeddingModel:
    def __init__(self, model_path=None, max_length=None, pooling_strategy=None):
        self.model_path = model_path
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.model = None
        self.tokenizer = None

    @torch.no_grad()
    def __call__(self, batch: Dict[str, List[str]]) -> torch.Tensor:
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_path).cuda().eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        encoded_inputs = self.tokenizer(
            list(batch['text']),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        encoded_inputs = {k: v.cuda() for k, v in encoded_inputs.items()}
        outputs = self.model(**encoded_inputs)

        attention_mask = encoded_inputs['attention_mask']
        if self.pooling_strategy == "mean":
            last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0)
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling_strategy == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")
        
        return {"embeddings": embeddings.cpu().numpy()}


class HFEmbeddingModel(EmbeddingModel):
    def __init__(self, model_path: str, batch_size: int, device: str, max_length: int, concurrency: int, pooling_strategy: Literal["mean", "cls"], normalize: bool = True):
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.concurrency = concurrency
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize

    @torch.no_grad()
    def embed(self, texts: List[str]) -> torch.Tensor:
        ds = ray.data.from_items([{'text': t} for t in texts])

        embeddings_results = ds.map_batches(
            RayEmbeddingModel,
            fn_constructor_kwargs={
                'model_path': self.model_path,
                'max_length': self.max_length,
                'pooling_strategy': self.pooling_strategy,
            },
            batch_size=self.batch_size,
            num_gpus=1,
            num_cpus=1,
            concurrency=self.concurrency,
        )

        all_embeddings = []
        for batch in embeddings_results.iter_batches(batch_format="numpy"):
            all_embeddings.append(batch['embeddings'])
        
        embeddings = torch.from_numpy(np.concatenate(all_embeddings, axis=0))
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

    @classmethod
    def from_embedding_config(cls, model_config: HFEmbeddingModelConfig):
        return cls(
            model_path=model_config.model_path,
            batch_size=model_config.batch_size,
            device=model_config.device,
            max_length=model_config.max_length,
            concurrency=model_config.concurrency,
            pooling_strategy=model_config.pooling_strategy,
        )

EmbeddingModel.register_model("hf_embedding", HFEmbeddingModel)