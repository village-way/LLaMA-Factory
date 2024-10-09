import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import pandas as pd

def prepare_full(
    source_path: Path,
    destination_path: Path,
    split: str = "train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset
    
    if not filenames:
        raise RuntimeError(
            f"No files matching found at {source_path}. \n"
            "Make sure you download the data..."
        )

    json_data = []  # Collect all processed text data in memory

    for filepath in filenames:
        print(f"Processing {filepath}")
        try:
            # Read parquet file and extract 'content' column
            contents = pd.read_parquet(filepath, engine='pyarrow')['content']
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # Prepare JSON data format
        json_data.extend([{"text": text} for text in contents])

    # Write the collected JSON data to a single file
    output_filepath = destination_path / f"{split}_data_{process_id}.json"
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Process {process_id} finished processing {len(filenames)} files and wrote {len(json_data)} documents.")


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    destination_path: Path = Path("data/red_pajama_sample_json"),
    split: str = "train",
    percentage: float = 1.0,
    filenames_subset: List[str] = None,
) -> None:
    import time

    filenames = glob.glob(os.path.join(source_path, "*/*.parquet"), recursive=True)

    # Filter filenames by subset if provided
    if filenames_subset:
        filenames = [f for f in filenames if any([prefix in f for prefix in filenames_subset])]
    
    # Adjust the number of files to process based on the percentage
    filenames = filenames[:int(len(filenames) * percentage)]

    # Split files across processes
    num_processes = cpu_count()  # Use available CPU cores
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    # Spawn processes to handle different file chunks
    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, destination_path, split, list(subset), i))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
