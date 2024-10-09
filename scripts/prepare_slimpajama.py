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

# Filename for SlimPajama
slimpajama_sets = {
    "train": "train/chunk*/*",
    "validation": "validation/chunk*/*",
    "test": "test/chunk*/*",
}

def prepare_full(
    source_path: Path,
    destination_path: Path,
    split: str = "train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 
    
    if not filenames:
        raise RuntimeError(
            f"No files matching {slimpajama_sets[split]} found at {source_path}. \n"
            "Make sure you download the data..."
        )

    json_data = []  # Collect all processed text data

    for filepath in filenames:
        print(f"Processing {filepath}")
        with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
            for row in tqdm(f):
                row_data = json.loads(row)
                # Skip RedPajamaGithub data
                if row_data["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                    continue 
                # Collect text in the desired JSON format
                json_data.append({"text": row_data["text"]})

    # Write collected data to a JSON file
    output_filepath = destination_path / f"{split}_data_{process_id}.json"
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Process {process_id} finished processing {len(filenames)} files.")


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    destination_path: Path = Path("data/red_pajama_sample_json"),
    split: str = "train",
    percentage: float = 1.0,
) -> None:
    import time

    filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
    filenames = filenames[:int(len(filenames) * percentage)]
    
    num_processes = cpu_count()  # Use available CPU cores
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, destination_path, split, list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
