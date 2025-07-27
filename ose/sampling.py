import json
import numpy as np
from typing import List, Dict, Any


class PromptSchema:
    
    def __init__(self, prompt_template: str, args: Dict[str, Any], args_p: Dict[str, Any], constraints: List[Dict[str, Any]] = None):
        self.prompt_template = prompt_template
        self.args = args
        self.args_p = args_p
        self.constraints = constraints
        if not set(args_p.keys()).issubset(set(args.keys())):
            raise ValueError("The keys in args_p must be a subset of the keys in args")

        for k, v in list(args.items()):
            if isinstance(v, str):
                with open(v, 'r') as f:
                    self.args[k] = json.load(f)
                    v = self.args[k]
            
            if not isinstance(v, list):
                raise ValueError(f"The value of {k} must be a list")

            if k not in self.args_p:
                self.args_p[k] = [1.0 / len(v)] * len(v)
            else:
                if abs(sum(self.args_p[k]) - 1.0) > 1e-6:
                    raise ValueError(f"The sum of the values in args_p[{k}] must be 1.0")

        if self.constraints is not None:
            for c in self.constraints:
                if not set(c.keys()) == {'type', 'value1', 'value2'}:
                    raise ValueError("The constraints must be a list of dictionaries with the keys 'type', 'value1', and 'value2'")
                if c['type'] not in {'neq'}:
                    raise ValueError(f"The type of constraint {c} must be 'neq'")
                if c['value1'] not in self.args.keys() or c['value2'] not in self.args.keys():
                    raise ValueError(f"The values of constraint {c} must be in the keys of args")

    @staticmethod
    def _apply_constraints(indices: Dict[str, np.ndarray], constraints: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        if constraints is None:
            return indices

        canonical_key = next(iter(indices.keys()))
        meta_indices = np.ones_like(indices[canonical_key], dtype=bool)
        
        for c in constraints:
            if c['type'] == 'neq':
                meta_indices &= indices[c['value1']] != indices[c['value2']]
            else:
                raise ValueError(f"The constraint {c} is not supported")
        
        return {
            k: v[meta_indices]
            for k, v in indices.items()
        }

    def sample(self, n: int) -> List[Dict[str, Any]]:
        indices = {
            k: np.random.choice(np.arange(len(v)), size=n, p=self.args_p[k])
            for k, v in self.args.items()
        }

        if self.constraints is not None:
            indices = self._apply_constraints(indices, self.constraints)
            canonical_key = next(iter(indices.keys()))
            if len(indices[canonical_key]) == 0:
                raise ValueError("No samples found after applying constraints, likely due to inconsistent constraints")
            
            while len(indices[canonical_key]) < n:
                est_inv_ratio = 1.5 * n / len(indices[canonical_key])  ## with 50% margin
                new_n = int(est_inv_ratio * (n - len(indices[canonical_key])))
                new_indices = {
                    k: np.random.choice(np.arange(len(v)), size=new_n, p=self.args_p[k])
                    for k, v in self.args.items()
                }
                new_indices = self._apply_constraints(new_indices, self.constraints)
                indices = {
                    k: np.concatenate([v, new_indices[k]])[:n]
                    for k, v in indices.items()
                }

        samples = []
        for i in range(n):
            sample = {}
            for k, v in self.args.items():
                sample[k] = v[indices[k][i]]
            samples.append({
                'sample': sample,
                'formatted_prompt': self.prompt_template.format(**sample)
            })

        return samples

    @classmethod
    def from_json(cls, path: str, extra_args: Dict[str, Any] = None, extra_args_p: Dict[str, Any] = None):
        with open(path, 'r') as f:
            data = json.load(f)

        args = data.get('args', {})
        args_p = data.get('args_p', {})
        if extra_args is not None:
            args.update(extra_args)
        if extra_args_p is not None:
            args_p.update(extra_args_p)

        return cls(data['prompt_template'], args, args_p, data.get('constraints', None))