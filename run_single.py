import yaml
import argparse
import torch
import matplotlib.pyplot as plt
from gptopt.train_distributed import train
from gptopt.optim.utils import get_scheduler, get_optimizer
from gptopt.utils import hash_config, set_seed, get_worker_info
#from gptopt.utils import get_default_config, load_config
from gptopt.model import load_model
from gptopt.dataloader import DATA_DIR, ShardedDataLoader
import torch.distributed as dist
import copy 
import json
import os


parser = argparse.ArgumentParser(description='Train GPT-2 with optional config file.')
parser.add_argument('--config', type=str, help='Path to config file', default=None)
parser.add_argument('--suffix', type=str, help='Path to config file', default='')
args = parser.parse_args()
set_seed(42)

# First set up DDP
world_size, rank, local_rank, device = get_worker_info()
master_process = (rank == 0) # this process will do logging, checkpointing etc.
device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f"Using device: {device}")

# Parse config
config_file = args.config
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
#config = load_config(get_default_config(), config_file)
outputname = config_file.replace("configs/","").replace('.yaml','')
output_dir = f"gptopt/outputs/{outputname}"
CKPT_DIR = config['logging_params']['ckpt_dir']
ckpt_dir_base = CKPT_DIR + f"/{outputname}/" if CKPT_DIR != "" else ""

if master_process:
    print(f"Loading configuration from {config_file}")
    print(f"Training on dataset {config['dataset']['name']}")
    os.makedirs(output_dir, exist_ok=True)  
    if CKPT_DIR != "": os.makedirs(ckpt_dir_base, exist_ok=True)  

# Load model
model = load_model(config['gpt_model'], device)
    
# Set the training parameters
training_params = config['training_params'] 
list_optimizer_params = config["optimizer_params"]
torch.set_float32_matmul_precision(training_params['tensorcore_precision'])

# Load data
dataset_path = DATA_DIR + f"/{config['dataset']['name']}-gpt2/"
if master_process: print(f"Load data from {dataset_path}")
B, T = training_params['batch_size'], training_params['context_length']
assert training_params['tokens_processed'] % (world_size * B * T) == 0 
train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device)
val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
if master_process:
    print(f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens")
    print(f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens")
    print(f"Total number of iterations : {total_iterations}")

# Loop over optimizers
for opt_config in list_optimizer_params:
    for lr in opt_config['lr']:
        print()
        if master_process:
            print(f"Training with optimizer {opt_config['name']} and learning rate {lr}")
            
        # Generate hash for the current optimizer configuration
        opt_config_copy = copy.deepcopy(opt_config)
        opt_config_copy['lr'] = lr
        config_hash = hash_config(opt_config_copy, training_params, config['gpt_model'])
        file_name = f"{opt_config['name']}-lr-{lr}-{opt_config['lr_schedule']}-{config_hash}-world{world_size}"
        if args.suffix != '': file_name += f"-{args.suffix}"
        output_path = os.path.join(output_dir, file_name + '.json')
        ckpt_dir = os.path.join(ckpt_dir_base, file_name) + '/' if CKPT_DIR != "" else ""
        
        # copy model to ensure consistency
        model_copy = copy.deepcopy(model).to(device)
        if training_params['compile']:
            if master_process: print("Compiling model")
            model_copy = torch.compile(model_copy)

        # Setup optimizer
        optimizer_obj, hyperp = get_optimizer(opt_config, lr=lr)
        p = model_copy.named_parameters() if 'muon' in opt_config['name'] else model_copy.parameters()
        optimizer = optimizer_obj(p, **hyperp)
        scheduler = get_scheduler(opt_config, optimizer, total_iterations=total_iterations)
        
        # Train
        logger = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params,
                       scheduler=scheduler, ckpt_dir=ckpt_dir,
                       logging_params=config['logging_params'])

        # Save
        if master_process:
            logger.name = opt_config['name'] + '-lr-' + str(lr)
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Overwriting")
            with open(output_path, 'w') as file:
                json.dump(logger.__dict__, file)
            print(f"Saved output to {output_path}")
