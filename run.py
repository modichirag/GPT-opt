import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
from gptopt.utils import compute_cross_entropy_loss, get_default_config, merge_configs, load_config, get_outputfile_from_configfile
from gptopt.data import load_data
from gptopt.train import train
from gptopt.optim.utils import get_scheduler, get_optimizer
from gptopt.utils import set_seed
import copy 
import json



def main(config_file=None):
    set_seed(42)
    default_config = get_default_config()      # Default parameters if no config file is provided
    if config_file:
        config = load_config(default_config, config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataloader, test_dataloader = load_data(config['dataset']['name'], batch_size=config['training_params']['batch_size'])
    # Load model using "model_name"
    if 'model_name' in config['gpt_model']:
        model = AutoModelForCausalLM.from_pretrained(config['gpt_model']['model_name'], device_map="auto").to(device)
        # model = GPT2LMHeadModel.from_pretrained(config['gpt_model']['model_name']).to(device)
    else:
        gpt_config = config['gpt_model']
        model_config = GPT2Config(
            n_embd=gpt_config['n_embd'],   # Hidden size used in distilgpt2
            n_layer=gpt_config['n_layer'],    # Number of layers in distilgpt2
            n_head=gpt_config['n_head'],    # Number of attention heads in distilgpt2
            vocab_size=gpt_config['vocab_size'],  # Standard GPT-2 vocabulary size
        )
        model = GPT2LMHeadModel(model_config).to(device)   # Initialize a new model with random weights using this configuration
    tokenizer = AutoTokenizer.from_pretrained(config['gpt_model']['tokenizer_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Set the training parameters
    training_params = config['training_params'] 
    print(f"Training on dataset {config['dataset']['name']}")
    # Access the optimizer parameters
    list_optimizer_params = config["optimizer_params"]

    import pdb; pdb.set_trace()
    outputs = []
    for optimizer_config in list_optimizer_params:
        # print(f"Optimizer: {optimizer['name']}")
        # print(f"Learning Rates: {optimizer['lr']}")
        # print(f"Weight Decay: {optimizer.get('weight_decay', 0)}")  # Use .get() for optional keys
        # print(f"LR Schedule: {optimizer['lr_schedule']}") 
        for lr in optimizer_config['lr']:
            model_copy = copy.deepcopy(model).to(device)  # The model remains the same
            total_iterations = training_params['num_epochs'] * len(train_dataloader)
            optimizer_obj, hyperp = get_optimizer(optimizer_config, lr = lr)
            optimizer = optimizer_obj(params=model_copy.parameters(), **hyperp)
            scheduler = get_scheduler(optimizer_config, optimizer, total_iterations = total_iterations)
            output = train(tokenizer, train_dataloader, model_copy, optimizer, training_params, device=device, scheduler=scheduler)
            output['name'] = optimizer_config['name'] + '-lr-'+str(lr)
            outputs.append(output)

    # adam_lr = float(config['optimizer_params']['adam_learning_rate'])  # Adjust the key if necessary
    # model_copy = copy.deepcopy(model).to(device)  # The model remains the same
    # adam_optimizer = torch.optim.Adam(model_copy.parameters(), lr=adam_lr)  # Replace SGD with Adam
    # adam_output = train(tokenizer, train_dataloader, model_copy, adam_optimizer, training_params, device=device)
    # sgd_output['name'] = 'sgd'
    # sgd_sch_output['name'] = 'sgd-sch' 
    # adam_output['name'] = 'adam'
    # adam_sch_output['name'] = 'adam-sch' 

    # outputs = [sgd_output, sgd_sch_output, adam_output, adam_sch_output ]

    outputfile = get_outputfile_from_configfile(config_file) 
    with open(outputfile, 'w') as file: json.dump(outputs, file)
   
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
    


