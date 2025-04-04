import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
from gptopt.utils import get_default_config, load_config
from gptopt.data import load_data
from gptopt.train import train
from gptopt.optim.utils import get_scheduler, get_optimizer
from gptopt.utils import set_seed
from gptopt.model import load_model
from gptopt.dataloader import DATA_DIR, ShardedDataLoader
import copy 
import json
from gptopt.utils import hash_config
import os

def main(config_file=None):

    set_seed(42)
    default_config = get_default_config()      # Default parameters if no config file is provided
    if config_file:
        config = load_config(default_config, config_file)
    outputname = config_file.replace("configs/","").replace('.yaml','')
    output_dir = f"gptopt/outputs/{outputname}"
    os.makedirs(output_dir, exist_ok=True)  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if 'matmul_precision' in config['training_params']:
        torch.set_float32_matmul_precision(config['training_params']['matmul_precision'])

    # Load model
    model = load_model(config, device)
                                                   
    # Set the training parameters
    training_params = config['training_params'] 
    print(f"Training on dataset {config['dataset']['name']}")
    # Access the optimizer parameters
    list_optimizer_params = config["optimizer_params"]

    # Load data
    dataset_path = DATA_DIR + f"/{config['dataset']['name']}-{config['gpt_model']['tokenizer']}/"
    print(f"Load data from {dataset_path}")
    B, T = training_params['batch_size'], training_params['context_length']
    train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device)
    val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
    print(f"Length of train/val dataset : {len(train_dataloader)}/{len(val_dataloader)} tokens")
    
    for optimizer_config in list_optimizer_params:
        for lr in optimizer_config['lr']:
            print()
            print(f"Training with optimizer {optimizer_config['name']} and learning rate {lr}")
            model_copy = copy.deepcopy(model).to(device)  # The model remains the same
            if training_params['compile']:
                print("Compiling model")
                model_copy = torch.compile(model_copy)
                
            optimizer_obj, hyperp = get_optimizer(optimizer_config, lr=lr)
            if 'muon' in optimizer_config['name']: # muon needs named params to split b/w muon and adamW
                optimizer = optimizer_obj(named_params=model_copy.named_parameters(), **hyperp)
            else:
                optimizer = optimizer_obj(params=model_copy.parameters(), **hyperp)
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

            
if __name__ == "__main__":
    # Argument parser to optionally provide a config file
    parser = argparse.ArgumentParser(description='Train GPT-2 with optional config file.')
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    
    args = parser.parse_args()
    if args.config:
        print(f"Loading configuration from {args.config}")
    else:
        print("No config file provided, using default settings.")
    main(args.config)



