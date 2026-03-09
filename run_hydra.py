import torch
from gptopt.train_distributed import train
from gptopt.optim.utils import get_scheduler, get_optimizer
from gptopt.utils import hash_config, set_seed, get_worker_info, get_data_dir, swap_linears_for_xtx, get_git_info
from gptopt.model import load_model
from gptopt.dataloader import ShardedDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy 
import json
import os
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("div", lambda x, y: x // y)

@hydra.main(version_base=None, config_path="hydra_conf", config_name="config")
def main(config : DictConfig):
    git_info = get_git_info()
    if git_info.startswith("dirty-") and not os.environ.get("ALLOW_DIRTY"):
        print(f"ERROR: Uncommitted changes detected ({git_info}).")
        print("Commit your code first for reproducibility, or set ALLOW_DIRTY=1 to override.")
        raise SystemExit(1)

    set_seed(config.get('seed', 42))

    # Establish Hydra run directory for saving outputs
    hydra_run_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(hydra_run_dir, exist_ok=True)

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

    # Set the training parameters
    training_params = config['training']['training_params']
    opt_config = config["optimizer"]["optimizer_params"]
    logging_config = config["logging"]["logging_params"]
    model_config = config["model"]["config"]

    # Logging
    outputname = HydraConfig.get().job.config_name
    # Save results into Hydra's run directory for this job
    logging_config['results_dir'] = hydra_run_dir
    output_dir = hydra_run_dir

    CKPT_DIR = logging_config['ckpt_dir']
    ckpt_dir_base = CKPT_DIR + f"/{outputname}/" if CKPT_DIR != "" else ""
    if master_process:
        # print(f"Loading configuration from {config_file}")
        print(f"Training on dataset {config['data']['dataset']['name']}")
        os.makedirs(output_dir, exist_ok=True)  
        if CKPT_DIR != "": os.makedirs(ckpt_dir_base, exist_ok=True)

    # Load model
    model = load_model(model_config, device)
    torch.set_float32_matmul_precision(training_params['tensorcore_precision'])

    # Load data
    data_dir = get_data_dir(config['data']['dataset']['name'])
    dataset_path = data_dir + f"/{config['data']['dataset']['name']}-gpt2/"
    if master_process: print(f"Load data from {dataset_path}")
    B, T = training_params['batch_size'], training_params['context_length']
    assert training_params['tokens_processed'] % (world_size * B * T) == 0 
    num_microbatches = int(training_params['tokens_processed'] / (world_size * B * T))

    train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device)
    val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    schedule_steps = training_params.get('schedule_steps', 0)
    max_steps = training_params.get('max_steps', 0)
    if schedule_steps:
        total_iterations = schedule_steps
    elif max_steps:
        total_iterations = max_steps
    if master_process:
        print(f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens")
        print(f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens")
        print(f"Total number of iterations : {total_iterations}")

    print()
    if master_process:
        print(f"Training with optimizer {opt_config['name']} and learning rate {opt_config['lr']}")
        
    # Generate hash for the current optimizer configuration
    opt_config_for_hash = {**OmegaConf.to_container(opt_config), 'seed': config.get('seed', 42)}
    config_hash = hash_config(opt_config_for_hash, OmegaConf.to_container(training_params), OmegaConf.to_container(model_config))
    file_name = f"{opt_config['name']}-lr-{opt_config['lr']}-{opt_config['lr_schedule']}-{config_hash}-world{world_size}"
    output_path = os.path.join(output_dir, file_name + '.json')
    ckpt_dir = os.path.join(ckpt_dir_base, file_name) + '/' if CKPT_DIR != "" else ""
    
    # copy model to ensure consistency
    model_copy = copy.deepcopy(model).to(device)
    opt_name = opt_config['name']
    # Setup optimizer
    optimizer_obj, hyperp = get_optimizer(opt_config, lr=opt_config['lr'])

    if opt_config.get('mb_subsampling', False) and opt_config.get('xtx_subsample', None):
        xtx_rate = opt_config['xtx_subsample']
        training_params['mb_subsampling'] = xtx_rate

        if num_microbatches < 1 / xtx_rate:
            opt_config['xtx_subsample'] = num_microbatches * xtx_rate
        else:
            opt_config['xtx_subsample'] = None
    else:
        training_params['mb_subsampling'] = None

    if 'dap' in opt_name:
        swap_linears_for_xtx(model_copy)

    if ddp:
        model_copy = DDP(model_copy, device_ids=[local_rank])


    p = model_copy.named_parameters() if ('muon' in opt_name or 'dap' in opt_name) else model_copy.parameters()

    if 'dap' in opt_name:
        hyperp['num_microbatches'] = num_microbatches
        optimizer = optimizer_obj(model_copy, p, **hyperp)
    else:
        optimizer = optimizer_obj(p, **hyperp)

    if training_params['compile']:
        if master_process: print("Compiling model")
        model_copy = torch.compile(model_copy)

    scheduler = get_scheduler(opt_config, optimizer, total_iterations=total_iterations)

    # Initialize wandb
    if master_process and logging_config.get('wandb', None) is not None:
        config_no_optimizer = copy.deepcopy(config)
        config_no_optimizer = OmegaConf.to_container(config_no_optimizer, resolve=True)
        config_no_optimizer.pop('optimizer_params', None)

        wandb_config = dict(one_optimizer_params=opt_config, **config_no_optimizer, world_size=world_size)

        wandb_args = dict(logging_config['wandb'])
        wandb_args.setdefault('dir', f"{logging_config['results_dir']}/../wandb")

        wandb_run = wandb.init(
            **wandb_args,
            config=wandb_config,
            reinit='create_new',
        )
    else:
        wandb_run = None

    # Train
    try:
        logger = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params,
                    scheduler=scheduler, ckpt_dir=ckpt_dir,
                    logging_params=logging_config, wandb_run=wandb_run,
                    output_path=output_path if master_process else None)
    finally:
        if master_process and wandb_run is not None:
            wandb_run.finish()

    # Save
    if master_process:
        logger.name = opt_config['name'] + '-lr-' + str(opt_config['lr'])
        logger.hyperparams = {**OmegaConf.to_container(opt_config), 'seed': config.get('seed', 42)}
        logger.git_info = get_git_info()
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Overwriting")
        with open(output_path, 'w') as file:
            json.dump(logger.__dict__, file)
        print(f"Saved output to {output_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
