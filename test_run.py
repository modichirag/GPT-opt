import yaml
import argparse
import torch
import matplotlib.pyplot as plt
from gptopt.utils import get_default_config, load_config
from gptopt.data import load_data
from gptopt.train_distributed import train
from gptopt.optim.utils import get_scheduler, get_optimizer
from gptopt.utils import set_seed, get_worker_info
from gptopt.model import load_model
from gptopt.dataloader import DATA_DIR, ShardedDataLoader
import copy 
import json
from gptopt.utils import hash_config
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

parser = argparse.ArgumentParser(description='Train GPT-2 with optional config file.')
parser.add_argument('--config', type=str, help='Path to config file', default=None)
config_file = parser.parse_args().config
print(f"Loading configuration from {config_file}")
config = load_config(get_default_config(), config_file)
print(f"Training on dataset {config['dataset']['name']}")
outputname = config_file.replace("configs/","").replace('.yaml','')
output_dir = f"gptopt/outputs/{outputname}"
os.makedirs(output_dir, exist_ok=True)  

set_seed(42)

# set up DDP (distributed data parallel).
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    dist.init_process_group(backend='nccl')
world_size, rank, local_rank, device = get_worker_info()
master_process = rank == 0 # this process will do logging, checkpointing etc.
torch.cuda.set_device(device)
device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f"Using device: {device}")


# Load model
model = load_model(config, device)

# Set the training parameters
training_params = config['training_params'] 
list_optimizer_params = config["optimizer_params"]
if 'matmul_precision' in training_params:
    torch.set_float32_matmul_precision(training_params['matmul_precision'])

# Load data
dataset_path = DATA_DIR + f"/{config['dataset']['name']}-{config['gpt_model']['tokenizer']}/"
print(f"Load data from {dataset_path}")
B, T = training_params['batch_size'], training_params['context_length']
train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device)
val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
print(f"Length of train/val dataset : {len(train_dataloader)}/{len(val_dataloader)} tokens")

# Loop over optimizers
for optimizer_config in list_optimizer_params:
    for lr in optimizer_config['lr']:
        print()
        print(f"Training with optimizer {optimizer_config['name']} and learning rate {lr}")
        model_copy = copy.deepcopy(model).to(device)  # The model remains the same
        if training_params['compile']:
            print("Compiling model")
            model_copy = torch.compile(model_copy)
        if ddp:
            model_copy = DDP(model_copy, device_ids=[local_rank])

        optimizer_obj, hyperp = get_optimizer(optimizer_config, lr=lr)
        p = model_copy.named_parameters() if 'muon' in optimizer_config['name'] else model_copy.parameters()
        optimizer = optimizer_obj(p, **hyperp)
        total_iterations = training_params['num_epochs'] * len(train_dataloader)
        scheduler = get_scheduler(optimizer_config, optimizer, total_iterations=total_iterations)
        # Train
        output = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params, device=device, scheduler=scheduler)

        # Save
        output['name'] = optimizer_config['name'] + '-lr-' + str(lr)
        # Generate hash for the current optimizer configuration
        config_hash = hash_config(optimizer_config, training_params, config['gpt_model'])
        file_name = f"{optimizer_config['name']}-lr-{lr}-{optimizer_config['lr_schedule']}-{config_hash}.json"
        output_path = os.path.join(output_dir, file_name)
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Overwriting")
        with open(output_path, 'w') as file:
            json.dump(output, file)
        print(f"Saved output to {output_path}")


if ddp:
    dist.destroy_process_group()

