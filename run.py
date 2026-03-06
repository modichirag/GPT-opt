import yaml
import argparse
import torch
import matplotlib.pyplot as plt
from gptopt.train_distributed import train
from gptopt.optim.utils import get_scheduler, get_optimizer
from gptopt.utils import hash_config, set_seed, get_worker_info, get_data_dir, swap_linears_for_xtx
#from gptopt.utils import get_default_config, load_config
from gptopt.model import load_model
from gptopt.dataloader import ShardedDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy 
import json
import os
import wandb

parser = argparse.ArgumentParser(description='Train GPT-2 with optional config file.')
parser.add_argument('--config', type=str, help='Path to config file', default=None)
parser.add_argument('--suffix', type=str, help='Path to config file', default='')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
if args.seed != 42:
    seed_suffix = f"seed{args.seed}"
    args.suffix = seed_suffix if not args.suffix else f"{seed_suffix}-{args.suffix}"
set_seed(args.seed)

# First set up DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    dist.init_process_group(backend='nccl')
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
data_dir = get_data_dir(config['dataset']['name'])
dataset_path = data_dir + f"/{config['dataset']['name']}-gpt2/"
if master_process: print(f"Load data from {dataset_path}")
B, T = training_params['batch_size'], training_params['context_length']
assert training_params['tokens_processed'] % (world_size * B * T) == 0 
num_microbatches = int(training_params['tokens_processed'] / (world_size * B * T))

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
        opt_name = opt_config['name']
        # Setup optimizer
        optimizer_obj, hyperp = get_optimizer(opt_config, lr=lr)

        if opt_config.get('mb_subsampling', False) and opt_config.get('xtx_subsample', None):
            xtx_rate = opt_config['xtx_subsample']
            training_params['mb_subsampling'] = xtx_rate

            if num_microbatches < 1 / xtx_rate:
                opt_config['xtx_subsample'] = num_microbatches * xtx_rate
            else:
                opt_config['xtx_subsample'] = None
        else:
            training_params['mb_subsampling'] = None

        if 'dap' in opt_name:
            swap_linears_for_xtx(model_copy)

        if training_params['compile']:
            if master_process: print("Compiling model")
            model_copy = torch.compile(model_copy)

        if ddp:
            model_copy = DDP(model_copy, device_ids=[local_rank])
        
        p = model_copy.named_parameters() if ('muon' in opt_name or 'dap' in opt_name) else model_copy.parameters()

        if 'dap' in opt_name:
            # Provide micro-batch count for DAP (matches grad_accum_steps in training loop)
            hyperp['num_microbatches'] = num_microbatches
            optimizer = optimizer_obj(model_copy, p, **hyperp)
        else:
            optimizer = optimizer_obj(p, **hyperp)

        scheduler = get_scheduler(opt_config, optimizer, total_iterations=total_iterations)

        # Initialize wandb
        if master_process and config['logging_params'].get('wandb', None) is not None:
            config_no_optimizer = copy.deepcopy(config)
            del config_no_optimizer['optimizer_params']
            wandb_config = dict(one_optimizer_params=opt_config_copy, **config_no_optimizer, world_size=world_size)
            if "dir" not in config['logging_params']['wandb']:
                config['logging_params']['wandb']['dir'] = f"{config['logging_params']['ckpt_dir']}/../wandb"
            wandb_run = wandb.init(
                **config['logging_params']['wandb'],
                config=wandb_config,
                reinit='create_new',
            )
        else:
            wandb_run = None

        # Train
        try:
            logger = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params,
                        scheduler=scheduler, ckpt_dir=ckpt_dir,
                        logging_params=config['logging_params'], wandb_run=wandb_run)
        finally:
            if master_process and wandb_run is not None:
                wandb_run.finish()

        # Save
        if master_process:
            logger.name = opt_config['name'] + '-lr-' + str(lr)
            logger.hyperparams = opt_config_copy
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Overwriting")
            with open(output_path, 'w') as file:
                json.dump(logger.__dict__, file)
            print(f"Saved output to {output_path}")

if ddp:
    dist.destroy_process_group()
