import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from gptopt.data_utils import tokenize, write_datafile, process_and_save_docs


data_settings = {
    "fineweb10B": {
        "load_args": ["HuggingFaceFW/fineweb"],
        "load_kwargs": {"name": "sample-10BT", "split": "train", "streaming": True}
    },
    "fineweb_edu10B": {
        "load_args": ["HuggingFaceFW/fineweb-edu"],
        "load_kwargs": {"name": "sample-10BT", "split": "train", "streaming": True}
    },
    "tiny_shakespeare": {
        "load_args": ["tiny_shakespeare"],
        "load_kwargs": {"name": ""}
    },
    "wikitext": {
        "load_args": ["wikitext"],
        "load_kwargs": {"name": "wikitext-103-v1"},
    },
    "slim_pajama": {
        "load_args": ["cerebras/SlimPajama-627B"],
        "load_kwargs": {"split": "train", "streaming": True},
    }
}
DATA_DIR = "/mnt/ceph/users/mcrawshaw/huggingface/"


# parse command line arguments
parser = argparse.ArgumentParser(description="Preprocessing hugging face datasets")
parser.add_argument("--name", type=str, help="Name of the dataset")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-t", "--tokenizer", type=str, default="gpt2", help="tokenizer to use")
parser.add_argument("-n", "--nprocs", type=int, default=0, help="number of processes, default N-2")
args = parser.parse_args()

# download dataset
name = args.name
settings = data_settings[name]
load_args = settings["load_args"]
load_kwargs = settings["load_kwargs"]
dataset = load_dataset(*load_args, trust_remote_code=True, **load_kwargs)

# make directory to store dataset
dataset_path = os.path.join(DATA_DIR, f'{name}-{args.tokenizer}/')
os.makedirs(dataset_path, exist_ok=True)
print("Data will be saved in the path : ", dataset_path)

# Process and save it
enc = tiktoken.get_encoding(args.tokenizer)
if name == "tiny_shakespeare":
    dataset['val'] = dataset['test'][0]
    dataset['train'] = dataset['train'][0]
    for split, shard_index in ['val', 0], ['train', 1]:
        filename = os.path.join(dataset_path, f"{split}_{shard_index:06d}.bin")
        tokens = tokenize(dataset[split], enc)
        write_datafile(filename, tokens)

elif name == "wikitext":
    dataset['val'] = {'text' : ''.join(t for t in dataset['test']['text'])}
    dataset['train'] =  {'text' : ''.join(t for t in dataset['train']['text'])}
    print(dataset['val'].keys())
    print(len(dataset['val']))
    for split, shard_index in ['val', 0], ['train', 1]:
        filename = os.path.join(dataset_path, f"{split}_{shard_index:06d}.bin")
        tokens = tokenize(dataset[split], enc)
        write_datafile(filename, tokens)

elif any([dset in name for dset in ['fineweb', 'slim_pajama']]):
    process_and_save_docs(dataset, dataset_path, encoding=enc, shard_size=args.shard_size, nprocs=args.nprocs)

print(f"{name} data processed and saved in {dataset_path}")
