import numpy as np
import torch
import random
import yaml

def compute_cross_entropy_loss(model, input_ids, attention_mask, labels):
    # Get model outputs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
    # Shift the logits and labels for language modeling
    shift_logits = logits[:, :-1, :].contiguous()  # Remove the last token's logits
    shift_labels = labels[:, 1:].contiguous()      # Remove the first token in the labels
    
    # Flatten the logits and labels for cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding token (assumes -100 for padding)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss

def get_outputfile_from_configfile(config_file):
    return 'gpt_distill/outputs/' + config_file.replace('configs/', '', 1).replace('.yaml', '', 1) + '.json'

def get_default_config():
    default_config = {
        'optimizer_params': {
            'aimsadam_learning_rate': 9.0,
            'adam_learning_rate': 1e-4,
            'sgd_learning_rate': 1e-4,
            'iams_learning_rate': 9.0
        },
        'training_params': {
            'batch_size': 8,
            'num_epochs': 1,
            'max_length': 512,
            'warm_up_percent': 0.1,
            'warm_up_peak_mult': 3.0,
            'adam_warm_up_peak_mult': 1.5
        },
        'gpt_model': {
            'teacher_model': 'gpt2-medium',
            'n_embd': 768,    # Hidden size used in distilgpt2
            'n_layer': 5,    # Number of layers in distilgpt2
            'n_head': 8,    # Number of attention heads in distilgpt2
            'vocab_size': 50257 
        },
        'dataset': {
            'name': 'wikitext-2-raw-v1',
            'problem_name': 'default'
        }
    }
    return default_config

    # Function to recursively merge dictionaries
def merge_configs(default_config, user_config):
    for key, value in default_config.items():
        if key not in user_config:
            user_config[key] = value
        elif isinstance(value, dict) and isinstance(user_config[key], dict):
            merge_configs(value, user_config[key])
    return user_config

def load_config(default_config, config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return merge_configs(default_config, config)

def smoothen_curve_batch(data, num_points):
    smooth_data =[data[0]]
    t =0
    data_av = 0.0
    total_iterations = len(data)
    av_interval = max(1, total_iterations // num_points)

    for count, item in enumerate(data, start=0): 
        data_av = data_av*t/(t+1) + item*(1/(t+1))
        t = t+1
        if count % av_interval == 0:
            smooth_data.append(data_av)
            data_av =0.0
            t=0.0
    return smooth_data

def smoothen_curve_exp(data, num_points):
    smooth_data =[data[0]]
    beta = 0.05
    data_av = data[0]
    total_iterations = len(data)
    av_interval = max(1, total_iterations // num_points)
    for count, item in enumerate(data, start=0): 
        if np.isnan(item):
            continue
        data_av = (1-beta)*data_av + beta*item
        if count % av_interval == 0:
            smooth_data.append(data_av)
    return smooth_data

def smoothen_dict(dict, num_points):
    smooth_dict = {}
    name = dict['name']
    for key in dict.keys():
        if key == 'name':
            continue
        if key == 'learning_rates' and 'sch' in name:
            continue
        if key == 'learning_rates':
            dict[key] = smoothen_curve_exp(dict[key], 400)
        else:
            dict[key] = smoothen_curve_exp(dict[key], num_points)
        # dict[key] = smoothen_curve(dict[key], num_points)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility

