import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
from gptopt.utils import compute_cross_entropy_loss, get_default_config, merge_configs, load_config, get_outputfile_from_configfile
from gptopt.data import load_data
from gptopt.train import train, get_scheduler
from gptopt.utils import smoothen_dict, set_seed
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
        model = AutoModelForCausalLM.from_pretrained(config['gpt_model']['model_name'], torch_dtype=torch.float16, device_map="auto").to(device)
        # teacher_model = GPT2LMHeadModel.from_pretrained(config['gpt_model']['teacher_model']).to(device)
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
    
# # Adam
    adam_lr = float(config['optimizer_params']['adam_learning_rate'])  # Adjust the key if necessary
    sgd_model = copy.deepcopy(model).to(device)  # The model remains the same
    adam_optimizer = torch.optim.Adam(sgd_model.parameters(), lr=adam_lr)  # Replace SGD with Adam
    adam_output = train(tokenizer, train_dataloader, sgd_model, adam_optimizer, training_params, device=device)
    # Adam scheduler warm-up *2 then cosine decay
    adam_lr = float(config['optimizer_params']['adam_learning_rate'])  # Adjust the key if necessary
    sgd_model = copy.deepcopy(model).to(device)  # The model remains the same
    adam_optimizer = torch.optim.Adam(sgd_model.parameters(), lr=adam_lr*config['training_params']['adam_warm_up_peak_mult'])  # Replace SGD with Adam
    adam_sch_output = train(tokenizer, train_dataloader, sgd_model, adam_optimizer, training_params, device=device, get_scheduler_fn=get_cosine_schedule_with_warmup)
     # SGD
    sgd_lr = float(config['optimizer_params']['sgd_learning_rate'])
    sgd_model = copy.deepcopy(model).to(device)
    sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=sgd_lr,  momentum=0.9, dampening=0.9)
    sgd_output = train(tokenizer, train_dataloader, sgd_model, sgd_optimizer, training_params, device = device)
    # SGD scheduler warm-up *2 then cosine decay
    sgd_model_sch = copy.deepcopy(model).to(device)
    sgd_optimizer_sch = torch.optim.SGD(sgd_model_sch.parameters(), lr=sgd_lr*config['training_params']['warm_up_peak_mult'],  momentum=0.9, dampening=0.9)
    sgd_sch_output = train(tokenizer, train_dataloader, sgd_model_sch, sgd_optimizer_sch,  training_params, device = device, get_scheduler_fn=get_cosine_schedule_with_warmup)
    
    sgd_output['name'] = 'sgd'
    sgd_sch_output['name'] = 'sgd-sch' 
    adam_output['name'] = 'adam'
    adam_sch_output['name'] = 'adam-sch' 

    outputs = [sgd_output, sgd_sch_output, adam_output, adam_sch_output ]

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
    


