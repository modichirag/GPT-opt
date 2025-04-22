from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM
from .gpt_model import GPT, GPTConfig

def load_model_and_tokenizer(config, device):

    if 'model_name' in config['gpt_model']:
        print(f"Loading model {config['gpt_model']['model_name']} and ignoring other configurations if specified")
        model = AutoModelForCausalLM.from_pretrained(config['gpt_model']['model_name'], device_map="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(config['gpt_model']['model_name'])
        if not config['gpt_model']['pretrained']:
            model_config = model.config
            del model
            model = GPT2LMHeadModel(model_config).to(device)
        else:
            print("Using pre-trained version")
            
    else:
        gpt_config = config['gpt_model']
        model_config = GPT2Config(
            n_embd=gpt_config['n_embd'],   
            n_layer=gpt_config['n_layer'],  
            n_head=gpt_config['n_head'],    
            vocab_size=gpt_config['vocab_size'], 
        )
        model = GPT2LMHeadModel(model_config).to(device)   # Initialize a new model with random weights using this configuration
        print("Loading gpt2 tokenizer as default tokenizer\n")
        tokenizer = AutoTokenizer.from_pretrained(config['gpt_model']['gpt2'])
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_huggingface(config, device):

    if 'model_name' in config['gpt_model']:
        print(f"Loading model {config['gpt_model']['model_name']} and ignoring other configurations if specified")
        model = AutoModelForCausalLM.from_pretrained(config['gpt_model']['model_name'], device_map="auto")#.to(device)
        if not config['gpt_model']['pretrained']:
            model_config = model.config
            del model
            model = GPT2LMHeadModel(model_config)#.to(device)
        else:
            print("Using pre-trained version")
            
    else:
        gpt_config = config['gpt_model']
        model_config = GPT2Config(
            n_embd=gpt_config['n_embd'],   
            n_layer=gpt_config['n_layer'],  
            n_head=gpt_config['n_head'],    
            vocab_size=gpt_config['vocab_size'], 
        )
        model = GPT2LMHeadModel(model_config)#.to(device)   # Initialize a new model with random weights using this configuration
    return model


def load_model(config, device):
    gptconfig = GPTConfig()
    gptconfig.n_embd = config['n_embd']
    gptconfig.n_layer = config['n_layer']
    gptconfig.n_head = config['n_head']
    gptconfig.vocab_size = config['vocab_size']
    model = GPT(gptconfig, device, flash_attention=config['flash_attention'])
    return model

