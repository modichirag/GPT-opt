import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings
from typing import Tuple
from transformers import get_cosine_schedule_with_warmup
from .momo import Momo
from .momo_adam import MomoAdam
from .muon import Muon


def get_optimizer_factory(name: str):
    """
    Returns the optimizer class corresponding to the given name.
    """
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD
    elif name == 'adam':
        return torch.optim.Adam
    elif name == 'adamw':
        return torch.optim.AdamW
    elif name == 'momo':
        return Momo
    elif name == 'momo-adam':
        return MomoAdam
    elif name == 'muon':
        return Muon
    else:
        raise ValueError(f"Unknown optimizer name: {name}")


def get_optimizer(opt_config: dict, lr = 1e-3) -> Tuple[torch.optim.Optimizer, dict]:
    """
    Main function mapping opt configs to an instance of torch.optim.Optimizer and a dict of hyperparameter arguments (lr, weight_decay,..).  
    For all hyperparameters which are not specified, we use PyTorch default.
    """
    
    name = opt_config['name']
    
    if opt_config.get('lr') is None:
        warnings.warn("You have not specified a learning rate. A default value of 1e-3 will be used.")
    
    if name == 'sgd':
        opt_obj = torch.optim.SGD
        
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0)
                  }
        
    elif name == 'sgd-m':
        opt_obj = torch.optim.SGD
        # sgd-m with exp. weighted average should have dampening = momentum
        if opt_config.get('dampening') == 'momentum':
            dampening = opt_config.get('momentum', 0.9)
        else:
            dampening = opt_config.get('dampening', 0)
            
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': False,
                  'dampening': dampening
                  }

    elif name == 'sgd-nesterov':
        opt_obj = torch.optim.SGD
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': True,
                  'dampening': opt_config.get('dampening', 0)
                  }
               
    elif name == 'adam':
        opt_obj = torch.optim.Adam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'fused': True
                  }
    
    elif name == 'adamw':
        opt_obj = torch.optim.AdamW
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'fused': True
                  }
    
    elif name == 'momo':
        opt_obj = Momo
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': False
                  }
    
    elif name == 'momo-adam':
        opt_obj = MomoAdam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': False
                  }
        
    elif name == 'momo-star':
        opt_obj = Momo
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': True
                  }
        
    elif name == 'momo-adam-star':
        opt_obj = MomoAdam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': True
                  }
  
    elif name == 'muon-you':
        opt_obj = Muon
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'rms_scaling': opt_config.get('rms_scaling', False),
                  'nuclear_scaling': opt_config.get('nuclear_scaling', False),
                  'polar_method': 'Jiacheng',
                  'polar_params': opt_config.get('polar_params', {})
                  }
    elif name == 'muon-jordan':
        opt_obj = Muon
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'rms_scaling': opt_config.get('rms_scaling', False),
                  'nuclear_scaling': opt_config.get('nuclear_scaling', False),
                  'polar_method': 'Keller',
                  'polar_params': opt_config.get('polar_params', {})
                  }
    elif name == 'muon-polarexpress':
        opt_obj = Muon
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'rms_scaling': opt_config.get('rms_scaling', True),
                  'polar_method': 'polarexpress',
                  'polar_params': opt_config.get('polar_params', {})
                  }        

    else:
        raise KeyError(f"Unknown optimizer name {name}.")
        
    return opt_obj, hyperp

def get_scheduler(config: dict, opt: torch.optim.Optimizer, total_iterations = None) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.
    """
    # if not specified, use constant step sizes
    name = config.get('name', 'constant')
    
    if name == 'constant':
        lr_fun = lambda epoch: 1 # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
    
    elif name == 'linear':
        lr_fun = lambda epoch: 1/(epoch+1) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'sqrt':
        lr_fun = lambda epoch: (epoch+1)**(-1/2) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif 'exponential' in name:
        # use sth like 'exponential_60_0.5': decay by factor 0.5 every 60 epochs
        step_size = int(name.split('_')[1])
        gamma = float(name.split('_')[2])
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

    elif 'warm-up-cosine' in name:
        num_warmup_steps = int(config['warm_up_fraction'] * total_iterations) 
        scheduler = get_cosine_schedule_with_warmup(
                    opt,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_iterations
                    )
    elif 'constant-linear' in name:  # New scheduler
        num_warmup_steps = int(config['warm_up_fraction'] * total_iterations)

        # def get_lr(step: int):
        #     x = step / total_iterations  # progress in training
        #     assert 0 <= x < 1
        #     if x < config['warm_up_fraction']:
        #         return 1.0  # Stable learning rate
        #     else:
        #         w = (1 - x) / (1-config['warm_up_fraction'])
        #         return w * 1.0 + (1 - w) * 0.1  # Linear decay during cooldown


        def get_lr(step):
            if step < num_warmup_steps:
                return 1.0  # Constant learning rate during warm-up
            else:
                # Linearly decay after warm-up
                return max(0.1, 1.0 - (step - num_warmup_steps) / (total_iterations - num_warmup_steps))

        scheduler = LambdaLR(opt, lr_lambda=get_lr)

        
    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")
    
    return scheduler
