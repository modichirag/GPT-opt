import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
from iams.iams_opt import IAMS
from iams.iamsAdam_opt import IAMSAdam
from iams.llms.utils import compute_cross_entropy_loss, get_default_config, merge_configs, load_config, get_outputfile_from_configfile
from iams.llms.data import load_data
from iams.llms.train import train, get_scheduler
from iams.utils import smoothen_dict, set_seed
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
    # Load the GPT-2 teacher model
    if config['gpt_model']['teacher_model'] == "EleutherAI/gpt-j-6B":
        teacher_model = AutoModelForCausalLM.from_pretrained(config['gpt_model']['teacher_model'], torch_dtype=torch.float16, device_map="auto").to(device)
    else:
        teacher_model = GPT2LMHeadModel.from_pretrained(config['gpt_model']['teacher_model']).to(device)
    gpt_config = config['gpt_model']
    student_config = GPT2Config(
        n_embd=gpt_config['n_embd'],   # Hidden size used in distilgpt2
        n_layer=gpt_config['n_layer'],    # Number of layers in distilgpt2
        n_head=gpt_config['n_head'],    # Number of attention heads in distilgpt2
        vocab_size=gpt_config['vocab_size'],  # Standard GPT-2 vocabulary size
    )
    student_model = GPT2LMHeadModel(student_config).to(device)   # Initialize a new model with random weights using this configuration
    tokenizer = AutoTokenizer.from_pretrained(config['gpt_model']['teacher_model'])
    tokenizer.pad_token = tokenizer.eos_token

    # Set the training parameters
    training_params = config['training_params'] 
    print(f"Training with teacher model {config['gpt_model']['teacher_model']} on dataset {config['dataset']['name']}")

    # Freeze teacher model's parameters for distillation
    for param in teacher_model.parameters():
        param.requires_grad = False
    
# # Adam
    adam_lr = float(config['optimizer_params']['adam_learning_rate'])  # Adjust the key if necessary
    sgd_model = copy.deepcopy(student_model).to(device)  # The model remains the same
    adam_optimizer = torch.optim.Adam(sgd_model.parameters(), lr=adam_lr)  # Replace SGD with Adam
    adam_output = train(tokenizer, train_dataloader, sgd_model, adam_optimizer, training_params, device=device)
    # Adam scheduler warm-up *2 then cosine decay
    adam_lr = float(config['optimizer_params']['adam_learning_rate'])  # Adjust the key if necessary
    sgd_model = copy.deepcopy(student_model).to(device)  # The model remains the same
    adam_optimizer = torch.optim.Adam(sgd_model.parameters(), lr=adam_lr*config['training_params']['adam_warm_up_peak_mult'])  # Replace SGD with Adam
    adam_sch_output = train(tokenizer, train_dataloader, sgd_model, adam_optimizer, training_params, device=device, get_scheduler_fn=get_cosine_schedule_with_warmup)
    # IAMS-Adam
    iamsadam_learning_rate = float(config['optimizer_params']['iamsadam_learning_rate'])
    model = copy.deepcopy(student_model).to(device)
    optimizer = IAMSAdam(model.parameters(), lr=iamsadam_learning_rate)
    iamsadam_output  = train(tokenizer, train_dataloader, model,  optimizer,  training_params, teacher_model=teacher_model,  device=device)
    iamsadam_output['learning_rates'] = optimizer.state['step_size_list']
     # SGD
    sgd_lr = float(config['optimizer_params']['sgd_learning_rate'])
    sgd_model = copy.deepcopy(student_model).to(device)
    sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=sgd_lr,  momentum=0.9, dampening=0.9)
    sgd_output = train(tokenizer, train_dataloader, sgd_model, sgd_optimizer, training_params, device = device)
    # SGD scheduler warm-up *2 then cosine decay
    sgd_model_sch = copy.deepcopy(student_model).to(device)
    sgd_optimizer_sch = torch.optim.SGD(sgd_model_sch.parameters(), lr=sgd_lr*config['training_params']['warm_up_peak_mult'],  momentum=0.9, dampening=0.9)
    sgd_sch_output = train(tokenizer, train_dataloader, sgd_model_sch, sgd_optimizer_sch,  training_params, device = device, get_scheduler_fn=get_cosine_schedule_with_warmup)
    # IAMS
    iams_learning_rate = float(config['optimizer_params']['iams_learning_rate'])
    model = copy.deepcopy(student_model).to(device)
    optimizer = IAMS(model.parameters(), lr=iams_learning_rate)
    iams_output  = train(tokenizer, train_dataloader, model,  optimizer,  training_params, teacher_model=teacher_model,  device=device)
    iams_output['learning_rates'] = optimizer.state['step_size_list']
    
    sgd_output['name'] = 'sgd'
    sgd_sch_output['name'] = 'sgd-sch' 
    iams_output['name'] = 'iams'
    adam_output['name'] = 'adam'
    adam_sch_output['name'] = 'adam-sch' 
    iamsadam_output['name'] = 'iams-adam'
    outputs = [sgd_output, sgd_sch_output, iams_output, adam_output, adam_sch_output, iamsadam_output ]

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
    


