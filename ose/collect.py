from jsonargparse import auto_cli
from dataclasses import dataclass
from typing import Union, Literal
import json
import os
import csv

from ose.postprocessing import ResponsePostprocessor, BrainstormingPostprocessorConfig, JsonPostprocessorConfig, SyntheticEmbeddingsDataPostprocessorConfig, HardNegativeRetrievalPostprocessorConfig


def main(
        postprocessor_config: Union[BrainstormingPostprocessorConfig, JsonPostprocessorConfig, SyntheticEmbeddingsDataPostprocessorConfig, HardNegativeRetrievalPostprocessorConfig] = None,
        input_path: str = None,
        output_path: str = None,
        output_format: Literal["csv", "json", "jsonl"] = None,
        overwrite_output: bool = False):
    
    if output_path is not None and os.path.exists(output_path):
        if not overwrite_output:
            print(f"Output file {output_path} already exists. Skipping collection.")
            return
        else:
            print(f"Output file {output_path} already exists. Overwriting.")
    
    entries = []
    with open(input_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    postprocessor = ResponsePostprocessor.from_config(postprocessor_config)
    results = postprocessor.postprocess(entries)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            if output_format == "csv":
                writer = csv.writer(f)
                writer.writerow(results[0].keys())
                for result in results:
                    writer.writerow(result.values())
            elif output_format == "json":
                f.write(json.dumps(results))
            elif output_format == "jsonl":
                for result in results:
                    f.write(json.dumps(result) + "\n")
            else:
                raise ValueError(f"Invalid output format: {output_format}")


def cli():
    auto_cli(main)

if __name__ == "__main__":
    cli()
