import numpy as np
import torch
import random
import yaml
import hashlib
import json


def hash_config(optimizer_config, training_params, gpt_model):
    """
    Generate a hash from the relevant fields of the current optimizer configuration,
    training parameters, and GPT model configuration.

    Parameters
    ----------
    optimizer_config : dict
        The configuration dictionary for the current optimizer.
    training_params : dict
        The training parameters dictionary.
    gpt_model : dict
        The GPT model configuration dictionary.

    Returns
    -------
    str
        A compressed hash string.
    """
    # Combine relevant fields
    relevant_fields = {
        "optimizer_config": optimizer_config,
        "training_params": training_params,
        "gpt_model": gpt_model
    }
    # Convert to a JSON string and hash it
    config_str = json.dumps(relevant_fields, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility

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
    return 'gptopt/outputs/' + config_file.replace('configs/', '', 1).replace('.yaml', '', 1) + '.json'

def get_default_config():
    default_config = {
        "optimizer_params": [
        {
            "name": "adam",
            "lr": [ 0.0001],
            "weight_decay": 0,
            "lr_schedule": "constant"
        },
        {
            "name": "momo-adam",
            "lr": [ 0.1],
            "weight_decay": 0,
            "lr_schedule": "constant"
        },
        {
            "name": "sgd-m",
            "lr": [0.001],
            "weight_decay": 0,
            "momentum": 0.9,
            "dampening": 0.9,
            "lr_schedule": "warm-up-cosine",
            "warm_up_percent": 0.2
        }
    ],
    "training_params": {
        "batch_size": 8,
        "num_epochs": 1,
        "max_length": 512
    },
    "gpt_model": {
        "model_name": "gpt2-medium",  # You can use one of the pre-defined models of transformers, or you can specify the exact dimension below
        "n_embd": 768,  # Hidden size used in distilgpt2
        "n_layer": 2,  # Number of layers in distilgpt2
        "n_head": 4,  # Number of attention heads in distilgpt2
        "vocab_size": 50304,
        "tokenizer_name": "gpt2-large"
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

## Plotting related functions
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
    for key in dict.keys():
        if key == 'losses':
            dict[key] = smoothen_curve_exp(dict[key], num_points)
        elif key == 'step_size_list':
            dict[key] = smoothen_curve_exp(dict[key], len(dict[key]))
        # dict[key] = smoothen_curve(dict[key], num_points)



