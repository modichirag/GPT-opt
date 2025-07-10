import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings
from typing import Tuple
from transformers import get_cosine_schedule_with_warmup
from .momo import Momo
from .momo_adam import MomoAdam
from .muon import Muon
from .sign_gd import SignGD
from .iams_adam import IAMSAdam
from .iams import IAMS
from .schedulefree import SGDScheduleFree
from .schedulep import SGDScheduleP
from .adamw_schedulefree import AdamWScheduleFree   
from .adamw_schedulep import AdamWScheduleP
# from .sps import SPS
# from .adabound import AdaBoundW
# from .adabelief import AdaBelief
# from .lion import Lion

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
    elif name == 'sgd-schedulefree':
        opt_obj = SGDScheduleFree
        hyperp = {'lr': lr,
                  'momentum': opt_config.get('momentum', 0.9),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'warmup_steps': opt_config.get('warmup_steps', 0),
                  'r': opt_config.get('r', 0.0),
                  'weight_lr_power': opt_config.get('weight_lr_power', 2.0),
                  'foreach': opt_config.get('foreach', True)
                  }
    elif name == 'sgd-schedulep': 
        opt_obj = SGDScheduleP
        hyperp = {'lr': lr,
                  'momentum': opt_config.get('momentum', 0.9),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'warmup_steps': opt_config.get('warmup_steps', 0),
                  'r': opt_config.get('r', 0.0),
                  'weight_lr_power': opt_config.get('weight_lr_power', 2.0),
                  'foreach': opt_config.get('foreach', True),
                  'lb': opt_config.get('lb', 0.0)
                  }
    elif name == 'adamw-schedulefree':
        opt_obj = AdamWScheduleFree
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'foreach': opt_config.get('foreach', True)
                  }
    elif name == 'adamw-schedulep':   
        opt_obj = AdamWScheduleP
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'foreach': opt_config.get('foreach', True),
                  'lb': opt_config.get('lb', 0.0)
                  }
    elif name == 'iams':
        opt_obj = IAMS
        hyperp = {'lr': lr,
                  'lmbda': opt_config.get('lmbda', 9.0),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'lb': opt_config.get('lb', 0.0)
                  }
    elif name == 'iams-adam':
        opt_obj = IAMSAdam
        hyperp = {'lr': lr,
                  'lmbda': opt_config.get('lmbda', 9.0),
                  'beta2': opt_config.get('betas', (0.9, 0.999))[1],
                  'eps': opt_config.get('eps', 1e-8),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'lb': opt_config.get('lb', 0.0)
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

    elif 'muon' in name:
        opt_obj = Muon
        lmo = 'nonlmo' not in name
        l2_prod_norm = 'l2_prod' in name
        rms_layer_norm = 'rms' in name
        if "nuc_fro" in name:
            nuc_approx = "fro"
        elif "nuc_past" in name:
            nuc_approx = "past"
        else:
            nuc_approx = None
        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'l2_prod_norm': l2_prod_norm,
                  'nuc_approx': nuc_approx,
                  'rms_layer_norm': rms_layer_norm,
                  }

    elif name == 'sign-gd':
        opt_obj = SignGD
        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': opt_config.get('nesterov', False),
                  'lmo': True
                  }

    elif name == 'sign-gd-nonlmo':
        opt_obj = SignGD
        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': opt_config.get('nesterov', False),
                  'lmo': False
                  }

    # elif name == 'iam':
    #     opt_obj = IAM
    #     hyperp = {'lr': lr,
    #               'weight_decay': opt_config.get('weight_decay', 0),
    #               'betas': opt_config.get('betas', (0.9, 0.999)),
    #               'eps': opt_config.get('eps', 1e-8),
    #               'lb': opt_config.get('lb', 0.),
    #               'divide': opt_config.get('divide', True),
    #               'use_fstar': True
    #               }          
    # elif name == 'prox-sps':
    #     opt_obj = SPS
    #     hyperp = {'lr': lr,
    #               'weight_decay': opt_config.get('weight_decay', 0),
    #               'lb': opt_config.get('lb', 0.),
    #               'prox': True
    #               }
    
    # elif name == 'adabound':
    #     opt_obj = AdaBoundW
        
    #     hyperp = {'lr': lr,
    #               'weight_decay': opt_config.get('weight_decay', 0),
    #               'betas': opt_config.get('betas', (0.9, 0.999)),
    #               'eps': opt_config.get('eps', 1e-8),
    #               'final_lr': opt_config.get('final_lr', 0.1)
    #               }

    # elif name == 'adabelief':
    #     opt_obj = AdaBelief
    #     hyperp = {'lr': lr,
    #               'weight_decay': opt_config.get('weight_decay', 0),
    #               'betas': opt_config.get('betas', (0.9, 0.999)),
    #               'eps': opt_config.get('eps', 1e-16),
    #               }
        
    # elif name == 'lion':
    #     opt_obj = Lion
    #     hyperp = {'lr': lr,
    #               'weight_decay': opt_config.get('weight_decay', 0),
    #               'betas': opt_config.get('betas', (0.9, 0.99)),
    #               }
    else:
        raise KeyError(f"Unknown optimizer name {name}.")
        
    return opt_obj, hyperp

def get_scheduler(config: dict, opt: torch.optim.Optimizer, total_iterations = None) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.
    """
    # if not specified, use constant step sizes
    name = config.get('lr_schedule', 'constant')
    
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

        def get_lr(step):
            if step < num_warmup_steps:
                return 1.0  # Constant learning rate during warm-up
            else:
                # Linearly decay after warm-up
                return max(0.1, 1.0 - (step - num_warmup_steps) / (total_iterations - num_warmup_steps))

        scheduler = LambdaLR(opt, lr_lambda=get_lr)

    elif 'warm-up-constant-linear' in name:  # New scheduler
        num_warmup_steps = int(config['warm_up_fraction'] * total_iterations)
        num_cooldown_steps = int(config['cool_down_fraction'] * total_iterations)
        num_constant_steps = total_iterations - num_warmup_steps - num_cooldown_steps
        min_lr = config.get('min_lr', 1e-7)  # Default minimal learning rate

        def get_lr(step):
            if step < num_warmup_steps:
                return step / num_warmup_steps  # Linear warm-up
            elif step < num_warmup_steps + num_constant_steps:
                return 1.0  # Constant learning rate
            elif num_cooldown_steps > 0:
                # Linear decay during cool-down
                return max(min_lr, 1.0 - (step - num_warmup_steps - num_constant_steps) / num_cooldown_steps)
            else:
                return min_lr  # Minimum learning rate if no cool-down phase

        scheduler = LambdaLR(opt, lr_lambda=get_lr)
        
    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")
    
    return scheduler
