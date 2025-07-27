from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Literal
from dataclasses import dataclass
import json
import ast
import re
import random
import numpy as np
import torch

from ose.modeling import EmbeddingModel, HFEmbeddingModelConfig


@dataclass
class PostprocessorConfig:
    pass


@dataclass
class BrainstormingPostprocessorConfig(PostprocessorConfig):
    postprocessor_type: Literal["brainstorming"] = "brainstorming"


@dataclass
class JsonPostprocessorConfig(PostprocessorConfig):
    postprocessor_type: Literal["json"] = "json"


@dataclass
class SyntheticEmbeddingsDataPostprocessorConfig(PostprocessorConfig):
    postprocessor_type: Literal["synthetic_embeddings_data"] = "synthetic_embeddings_data"
    data_type: str = None


@dataclass
class HardNegativeRetrievalPostprocessorConfig(PostprocessorConfig):
    postprocessor_type: Literal["hard_negative_retrieval"] = "hard_negative_retrieval"
    embedding_config: HFEmbeddingModelConfig = None
    batch_size: int = 32
    top_k: int = 10
    embedder_query_template: str = "{query}"
    embedder_document_template: str = "{positive}"


class ResponsePostprocessor(ABC):
    _postprocessors = {}

    @abstractmethod
    def postprocess(self, entries: List[Dict[str, Any]], **kwargs) -> List[str]:
        pass

    @staticmethod
    def from_config(postprocessor_config: PostprocessorConfig):
        if postprocessor_config.postprocessor_type not in ResponsePostprocessor._postprocessors:
            raise ValueError(f"Postprocessor type {postprocessor_config.postprocessor_type} not found")
        postprocessor_cls = ResponsePostprocessor._postprocessors[postprocessor_config.postprocessor_type]
        return postprocessor_cls.from_postprocessor_config(postprocessor_config)

    @staticmethod
    def register_postprocessor(postprocessor_type: str, postprocessor_class: Type):
        ResponsePostprocessor._postprocessors[postprocessor_type] = postprocessor_class

    @classmethod
    @abstractmethod
    def from_postprocessor_config(cls, postprocessor_config: PostprocessorConfig):
        pass


class BrainstormingResponsePostprocessor(ResponsePostprocessor):
    
    def __init__(self):
        pass

    def postprocess(self, entries: List[Dict[str, Any]], **kwargs) -> List[str]:
        all_ideas = []
        
        for entry in entries:
            if 'output' not in entry:
                continue
                
            response = entry['output']
            ideas = self._extract_ideas_from_response(response)
            all_ideas.extend(ideas)
        
        unique_ideas = list(set(all_ideas))
        return unique_ideas

    def _extract_ideas_from_response(self, response: str) -> List[str]:
        ideas = []
        
        # Parse Python list literals: ['item1', 'item2']
        try:
            list_pattern = r'\[.*?\]'
            matches = re.findall(list_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    parsed_list = ast.literal_eval(match)
                    if isinstance(parsed_list, list):
                        ideas.extend([str(item).strip() for item in parsed_list if item])
                        break
                except (ValueError, SyntaxError):
                    continue
                    
        except Exception:
            pass
        
        # Parse JSON list format: ["item1", "item2"]
        if not ideas:
            try:
                list_pattern = r'\[.*?\]'
                matches = re.findall(list_pattern, response, re.DOTALL)
                
                for match in matches:
                    try:
                        parsed_list = json.loads(match)
                        if isinstance(parsed_list, list):
                            ideas.extend([str(item).strip() for item in parsed_list if item])
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
                        
            except Exception:
                pass
        
        # Extract quoted strings: "item1" "item2"
        if not ideas:
            quoted_pattern = r'"([^"]+)"'
            quoted_matches = re.findall(quoted_pattern, response)
            if quoted_matches:
                ideas.extend([match.strip() for match in quoted_matches if match.strip()])
        
        # Extract bullet points/numbered lists: - item1, 1. item2
        if not ideas:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^[-*•]\s+', line) or re.match(r'^\d+\.\s+', line):
                    cleaned_line = re.sub(r'^[-*•]\s+', '', line)
                    cleaned_line = re.sub(r'^\d+\.\s+', '', cleaned_line)
                    if cleaned_line:
                        ideas.append(cleaned_line.strip())
        
        # Split by sentences if only one result found
        if len(ideas) == 1:
            sentences = ideas[0].split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) > 1:
                ideas = sentences
        
        return ideas

    @classmethod
    def from_postprocessor_config(cls, postprocessor_config: PostprocessorConfig):
        return cls()


class JsonResponsePostprocessor(ResponsePostprocessor):
    
    def __init__(self):
        pass

    def postprocess(self, entries: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        results = []
        
        for entry in entries:
            if 'output' not in entry:
                continue
                
            response = entry['output']
            parsed_json = self._parse_first_json_from_response(response)
            if parsed_json is not None:
                results.append(parsed_json)
        
        return results

    def _parse_first_json_from_response(self, response: str, retry=True):
        pattern = r'```(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        for i, match in enumerate(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                continue

        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        for i, match in enumerate(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                continue

        pattern = r'```python(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        for i, match in enumerate(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                continue
                
        pattern = r'\{(.*?)\}'
        matches = re.findall(pattern, response, re.DOTALL)
        for i, match in enumerate(matches):
            try:
                match = '{'+match+'}'
                return json.loads(match)
            except json.JSONDecodeError as e:
                continue

        if retry:
            pattern = r',\s*}'
            response = re.sub(pattern, '}', response)
            return self._parse_first_json_from_response(response, retry=False)

        pattern = r'(?<!\\)"(.*?)(?<!\\)"'
        matches = re.findall(pattern, response, re.DOTALL)
        if len(matches) % 2 == 0 and len(matches) > 0:
            return dict(zip(matches[::2], matches[1::2]))

        return None

    def _parse_first_list_from_response(self, response: str, retry=True):
        response = response.replace('\t', ' ')

        pattern = r'```(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        for i, match in enumerate(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                continue
                
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, response, re.DOTALL)
        for i, match in enumerate(matches):
            try:
                match = '['+match+']'
                return json.loads(match)
            except json.JSONDecodeError as e:
                continue

        if retry:
            pattern = r',\s*\]'
            response = re.sub(pattern, r']', response)
            return self._parse_first_list_from_response(response, retry=False)

        return None

    @classmethod
    def from_postprocessor_config(cls, postprocessor_config: PostprocessorConfig):
        return cls()


class SyntheticEmbeddingsDataPostprocessor(JsonResponsePostprocessor):
    
    def __init__(self, data_type: str):
        super().__init__()
        self.data_type = data_type

    def postprocess(self, entries: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        results = []
        
        for entry in entries:
            if 'output' not in entry:
                continue
                
            response = entry['output']
            parsed_json = self._parse_first_json_from_response(response)
            
            if parsed_json is not None:
                elem = {
                    'prompt': entry.get('formatted_prompt', entry.get('input', '')),
                    'response': parsed_json,
                    'sample': entry.get('sample', {})
                }
                parsed_elem = self._parse_by_id(self.data_type, elem)
                if parsed_elem is not None:
                    results.append(parsed_elem)
        
        return results

    def _parse_long_long(self, elem):
        if elem is None:
            return None
        try:
            response = elem['response']
            if response is None: return None
            if not isinstance(elem['prompt'], str):
                return None
            if not isinstance(response['input'], str):
                return None
            if not isinstance(response['positive_document'], str):
                return None
            
            if 'sample' in elem and 'task' in elem['sample']:
                instruction = elem['sample']['task']
            else:
                prompt_parts = elem['prompt'].split(': ')
                if len(prompt_parts) > 1:
                    instruction = prompt_parts[1].split('\n')[0]
                else:
                    instruction = elem['prompt'].split('\n')[0] if '\n' in elem['prompt'] else elem['prompt']
            
            return {
                'instruction': instruction,
                'query': response['input'],
                'positive': response['positive_document'],
                'negative': ''
            }
        except (KeyError, IndexError) as e:
            return None

    def _parse_long_short(self, elem):
        if elem is None:
            return None
        try:
            response = elem['response']
            if response is None: return None
            if not isinstance(elem['prompt'], str):
                return None
            if not isinstance(response['input_text'], str):
                return None
            if not isinstance(response['label'], str):
                return None
            if not isinstance(response['misleading_label'], str):
                return None
            
            if 'sample' in elem and 'task' in elem['sample']:
                instruction = elem['sample']['task']
            else:
                prompt_parts = elem['prompt'].split(': ')
                if len(prompt_parts) > 1:
                    instruction = prompt_parts[1].split('\n')[0]
                else:
                    instruction = elem['prompt'].split('\n')[0] if '\n' in elem['prompt'] else elem['prompt']
            
            return {
                'instruction': instruction,
                'query': response['input_text'],
                'positive': response['label'],
                'negative': response['misleading_label']
            }
        except (KeyError, IndexError) as e:
            return None

    def _parse_short_long(self, elem):
        if elem is None:
            return None
        try:
            response = elem['response']
            if response is None: return None
            if not isinstance(elem['prompt'], str):
                return None
            if not isinstance(response['user_query'], str):
                return None
            if not isinstance(response['positive_document'], str):
                return None
            
            if 'sample' in elem and 'task' in elem['sample']:
                instruction = elem['sample']['task']
            else:
                prompt_parts = elem['prompt'].split(': ')
                if len(prompt_parts) > 1:
                    instruction = prompt_parts[1].split('\n')[0]
                else:
                    instruction = elem['prompt'].split('\n')[0] if '\n' in elem['prompt'] else elem['prompt']
            
            return {
                'instruction': instruction,
                'query': response['user_query'],
                'positive': response['positive_document'],
                'negative': ''
            }
        except (KeyError, IndexError) as e:
            return None

    def _parse_short_short(self, elem):
        if elem is None:
            return None
        try:
            response = elem['response']
            if response is None: return None
            if not isinstance(elem['prompt'], str):
                return None
            if not isinstance(response['input'], str):
                return None
            if not isinstance(response['positive_document'], str):
                return None
            
            if 'sample' in elem and 'task' in elem['sample']:
                instruction = elem['sample']['task']
            else:
                prompt_parts = elem['prompt'].split(': ')
                if len(prompt_parts) > 1:
                    instruction = prompt_parts[1].split('\n')[0]
                else:
                    instruction = elem['prompt'].split('\n')[0] if '\n' in elem['prompt'] else elem['prompt']
            
            return {
                'instruction': instruction,
                'query': response['input'],
                'positive': response['positive_document'],
                'negative': ''
            }
        except (KeyError, IndexError) as e:
            return None

    def _parse_sts(self, elem):
        if elem is None:
            return None
        try:
            response = elem['response']
            if response is None: return None
            if not isinstance(response['S1'], str):
                return None
            if not isinstance(response['S2'], str):
                return None
            if not isinstance(response['S3'], str):
                return None
            return {
                'instruction': 'Retrieve semantically similar text.',
                'query': response['S1'],
                'positive': response['S2'],
                'negative': response['S3']
            }
        except KeyError as e:
            return None

    def _parse_bitext(self, elem):
        if elem is None:
            return None
        try:
            response = elem['response']
            if response is None: return None
            if not isinstance(response['S1'], str):
                return None
            if not isinstance(response['S2'], str):
                return None
            if not isinstance(response['S3'], str):
                return None
            return {
                'instruction': 'Retrieve parallel sentences.',
                'query': response['S1'],
                'positive': response['S2'],
                'negative': response['S3']
            }
        except KeyError as e:
            return None

    def _parse_by_id(self, id: str, elem):
        if id.startswith('long_long'):
            return self._parse_long_long(elem)
        elif id.startswith('long_short'):
            return self._parse_long_short(elem)
        elif id.startswith('short_long'):
            return self._parse_short_long(elem)
        elif id.startswith('short_short'):
            return self._parse_short_short(elem)
        elif id.startswith('sts'):
            return self._parse_sts(elem)
        elif id.startswith('bitext'):
            return self._parse_bitext(elem)
        else:
            raise ValueError(f'Unknown id: {id}')

    @classmethod
    def from_postprocessor_config(cls, postprocessor_config: SyntheticEmbeddingsDataPostprocessorConfig):
        return cls(data_type=postprocessor_config.data_type)


class HardNegativeRetrievalPostprocessor(ResponsePostprocessor):
    
    def __init__(self, embedding_config: HFEmbeddingModelConfig, batch_size: int, top_k: int, embedder_query_template: str, embedder_document_template: str):
        self.embedding_config = embedding_config
        self.batch_size = batch_size
        self.top_k = top_k
        self.embedder_query_template = embedder_query_template
        self.embedder_document_template = embedder_document_template
        self.embedding_model = EmbeddingModel.from_config(embedding_config)

    def postprocess(self, entries: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        query_texts = [self.embedder_query_template.format(**entry) for entry in entries]
        document_texts = [self.embedder_document_template.format(**entry) for entry in entries]
        
        query_embeddings = self.embedding_model.embed(query_texts)
        document_embeddings = self.embedding_model.embed(document_texts)
        
        results = []
        for i in range(0, len(entries), self.batch_size):
            batch_query_embs = query_embeddings[i:i+self.batch_size]
            similarities = torch.mm(batch_query_embs, document_embeddings.T)
            
            for j, sim_scores in enumerate(similarities):
                original_idx = i + j
                original_positive = entries[original_idx]['positive']
                
                top_indices = torch.topk(sim_scores, self.top_k).indices
                candidates = [entries[idx]['positive'] for idx in top_indices 
                             if entries[idx]['positive'] != original_positive and entries[idx]['positive'].strip() != ""]
                
                if not candidates:
                    candidates = [entries[idx]['positive'] for idx in top_indices[1:] 
                                 if entries[idx]['positive'].strip() != ""]
                
                if not candidates:
                    fallback_idx = (original_idx + 1) % len(entries)
                    while entries[fallback_idx]['positive'].strip() == "" or entries[fallback_idx]['positive'] == original_positive:
                        fallback_idx = (fallback_idx + 1) % len(entries)
                    negative = entries[fallback_idx]['positive']
                else:
                    negative = random.choice(candidates)
                
                result = entries[original_idx].copy()
                result['negative'] = negative
                results.append(result)
        
        return results

    @classmethod
    def from_postprocessor_config(cls, postprocessor_config: HardNegativeRetrievalPostprocessorConfig):
        return cls(
            embedding_config=postprocessor_config.embedding_config,
            batch_size=postprocessor_config.batch_size,
            top_k=postprocessor_config.top_k,
            embedder_query_template=postprocessor_config.embedder_query_template,
            embedder_document_template=postprocessor_config.embedder_document_template
        )


ResponsePostprocessor.register_postprocessor("brainstorming", BrainstormingResponsePostprocessor)
ResponsePostprocessor.register_postprocessor("json", JsonResponsePostprocessor)
ResponsePostprocessor.register_postprocessor("synthetic_embeddings_data", SyntheticEmbeddingsDataPostprocessor)
ResponsePostprocessor.register_postprocessor("hard_negative_retrieval", HardNegativeRetrievalPostprocessor)
