import torch
from gptopt.train_distributed import train
from gptopt.optim.utils import get_scheduler, get_optimizer
from gptopt.utils import hash_config, set_seed, get_worker_info
from gptopt.model import load_model
from gptopt.dataloader import DATA_DIR, ShardedDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy 
import json
import os
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="hydra_conf")
def main(config : DictConfig):
    set_seed(42)

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

    # Logging
    outputname = HydraConfig.get().job.config_name
    output_dir = config['logging_params'].get('results_dir', f"dap_outputs/hydra-results/{outputname}")
    CKPT_DIR = config['logging_params']['ckpt_dir']
    ckpt_dir_base = CKPT_DIR + f"/{outputname}/" if CKPT_DIR != "" else ""
    if master_process:
        # print(f"Loading configuration from {config_file}")
        print(f"Training on dataset {config['training_data']['dataset']['name']}")
        os.makedirs(output_dir, exist_ok=True)  
        if CKPT_DIR != "": os.makedirs(ckpt_dir_base, exist_ok=True)

    # Load model
    model = load_model(config['gpt_model'], device)
        
    # Set the training parameters
    training_params = config['training_data']['training_params']
    opt_config = config["optimizer_params"]
    torch.set_float32_matmul_precision(training_params['tensorcore_precision'])

    # Load data
    dataset_path = DATA_DIR + f"/{config['training_data']['dataset']['name']}-gpt2/"
    if master_process: print(f"Load data from {dataset_path}")
    B, T = training_params['batch_size'], training_params['context_length']
    assert training_params['tokens_processed'] % (world_size * B * T) == 0 
    train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device)
    val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    if master_process:
        print(f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens")
        print(f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens")
        print(f"Total number of iterations : {total_iterations}")

    print()
    if master_process:
        print(f"Training with optimizer {opt_config['name']} and learning rate {opt_config['lr']}")
        
    # Generate hash for the current optimizer configuration
    config_hash = hash_config(OmegaConf.to_container(opt_config), OmegaConf.to_container(training_params), OmegaConf.to_container(config['gpt_model']))
    file_name = f"{opt_config['name']}-lr-{opt_config['lr']}-{opt_config['lr_schedule']}-{config_hash}-world{world_size}"
    output_path = os.path.join(output_dir, file_name + '.json')
    ckpt_dir = os.path.join(ckpt_dir_base, file_name) + '/' if CKPT_DIR != "" else ""
    
    # copy model to ensure consistency
    model_copy = copy.deepcopy(model).to(device)
    
    # Setup optimizer
    optimizer_obj, hyperp = get_optimizer(opt_config, lr=opt_config['lr'])

    if training_params['compile']:
        if master_process: print("Compiling model")
        model_copy = torch.compile(model_copy)

    if ddp:
        model_copy = DDP(model_copy, device_ids=[local_rank])
    
    opt_name = opt_config['name']
    p = model_copy.named_parameters() if ('muon' in opt_name or 'dap' in opt_name) else model_copy.parameters()

    if 'dap' in opt_name:
        optimizer = optimizer_obj(model_copy, p, **hyperp)
    else:
        optimizer = optimizer_obj(p, **hyperp)

    scheduler = get_scheduler(opt_config, optimizer, total_iterations=total_iterations)

    # Initialize wandb
    if master_process and config['logging_params'].get('wandb', None) is not None:
        config_no_optimizer = OmegaConf.to_container(config, resolve=True)
        del config_no_optimizer['optimizer_params']
        config_for_wandb_logging = dict(one_optimizer_params=opt_config, **config_no_optimizer, world_size=world_size)
        wandb_config = OmegaConf.to_container(config['logging_params']['wandb'], resolve=True)
        if "dir" not in wandb_config:
            wandb_config['dir'] = f"{config['logging_params']['results_dir']}/../wandb"
        wandb_run = wandb.init(
            **wandb_config,
            config=config_for_wandb_logging,
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
        logger.name = opt_config['name'] + '-lr-' + str(opt_config['lr'])
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Overwriting")
        with open(output_path, 'w') as file:
            json.dump(logger.__dict__, file)
        print(f"Saved output to {output_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
