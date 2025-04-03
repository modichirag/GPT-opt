import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from gptopt.data_utils import tokenize, write_datafile, process_and_save_docs

## Dict to map local data name to huggingface path and name
datadict = {
    "fineweb10B" : ["HuggingFaceFW/fineweb", "sample-10BT"],
    "fineweb_edu10B" : ["HuggingFaceFW/fineweb-edu", "sample-10BT"],
    "tiny_shakespeare" : ["tiny_shakespeare", ""],
    "wikitext" : ["wikitext", "wikitext-103-v1"]    
}
DATA_DIR = "/mnt/ceph/users/cmodi/huggingface/"

# parse command line arguments
parser = argparse.ArgumentParser(description="Preprocessing hugging face datasets")
parser.add_argument("--name", type=str, help="Name of the dataset")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-t", "--tokenizer", type=str, default="gpt2", help="tokenizer to use")
parser.add_argument("-n", "--nprocs", type=int, default=0, help="number of processes, default N-2")
args = parser.parse_args()

name = args.name
hf_path, remote_name = datadict[name]
enc = tiktoken.get_encoding(args.tokenizer)
dataset_path = DATA_DIR + f'/{name}-{args.tokenizer}/'
os.makedirs(dataset_path, exist_ok=True)
print("Data will be saved in the path : ", dataset_path)

# download dataset
dataset = load_dataset(hf_path, name=remote_name, trust_remote_code=True)

# Process and save it
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

elif 'fineweb' in name:
    process_and_save_docs(dataset['train'], dataset_path, encoding=enc, shard_size=args.shard_size, nprocs=args.nprocs)
    
print(f"{name} data processed and saved in {dataset_path}")
