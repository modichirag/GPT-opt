import torch
import time 
import contextlib
import torch.distributed as dist
from gptopt.utils import get_worker_info, save_checkpoint, load_checkpoint
import json

typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}

class Logging():

    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []


def train(train_dataloader, val_dataloader, model, optimizer, training_params, logging_params, scheduler=None, hface_model=False, ckpt_dir=None):
    
    world_size, rank, local_rank, device  = get_worker_info()
    master_process = (rank == 0)
    logger = Logging()
    optimizer_name = optimizer.__class__.__name__
    if 'Momo' in optimizer_name:
        pass_loss = True
    else:
        pass_loss = False
    if master_process: print(f"Set pass_loss to {pass_loss} for optimizer {optimizer_name}")

    autocast_ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        autocast_ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']])     
    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (world_size*B*T))
    val_accum_steps = int(logging_params['val_tokens_processed'] / (world_size*B*T))
    if master_process: print(f"Accumulate gradient for {grad_accum_steps} steps")
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    max_grad_norm = training_params['gradnorm'] if training_params['gradnorm'] != 0. else float('inf')

    load_ckpt_step = logging_params['load_ckpt_step']
    if load_ckpt_step != 0:
        model, optimizer, train_dataloader, scheduler = load_checkpoint(ckpt_dir, load_ckpt_step, model, \
                                                        optimizer, train_dataloader, scheduler=None)
        
        
    # Training loop
    for epoch in range(training_params['num_epochs']):
        if master_process:
            print(f"Epoch {epoch+1} of {training_params['num_epochs']}")

        model.train()
        start_epoch = time.time()
        start_time = time.time() 
        loss_accum = 0.
        step = 1 if load_ckpt_step == 0 else int(load_ckpt_step)
        optimizer.zero_grad()
        if step != 1: print(train_dataloader.get_state())
        for batch in train_dataloader:            
            with autocast_ctxt:
                if hface_model:
                     loss = model(batch[0], labels=batch[0]).loss
                else:
                     loss = model(batch[0], batch[1], return_logits=False)[1]
                loss /= grad_accum_steps
            loss_accum += loss.detach()
                
            # Check if accummulated enough gradients to take a step
            if step % grad_accum_steps != 0:
                with (model.no_sync() if world_size > 1 else contextlib.nullcontext()):
                    loss.backward()
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if world_size > 1: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                if pass_loss:
                    optimizer.step(closure=None, loss=loss_accum)
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                    
                #bookkeeping
                torch.cuda.synchronize()
                step_time = time.time() - start_time
                logger.step_times.append(step_time)  # Are these different across ranks?
                logger.grad_norms.append(norm.item())
                for param_group in optimizer.param_groups:
                    logger.learning_rates.append(param_group['lr'])
                logger.losses.append(loss_accum.item())
                if hasattr(optimizer, 'step_size_list'):  
                    logger.step_size_list = optimizer.step_size_list  
                
                if (step % logging_params['log_step'] == 0) & master_process:
                    tps = training_params["tokens_processed"] / step_time
                    print(f"Step {step} of {total_iterations*grad_accum_steps}.")
                    print(f"Time taken : {step_time*1000:0.1f}ms | Tokens/s : {tps/1000:0.1f}k | Loss : {loss_accum.item():0.3f}")
                    
                if (step % logging_params['val_step'] == 0):
                    model.eval()
                    val_loss, counter = 0., 0
                    with torch.no_grad():
                        for batch in val_dataloader:
                            with autocast_ctxt:
                                if hface_model:
                                    val_loss += model(batch[0], labels=batch[0]).loss.item()
                                else:
                                    val_loss += model(batch[0], batch[1], return_logits=False)[1]
                            counter += 1
                            if counter >= val_accum_steps: break
                    val_loss = torch.tensor(val_loss, device=device)/val_accum_steps
                    if world_size > 1: dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                    logger.val_losses.append(val_loss.item())
                    if master_process:
                        print(f"Validation Loss: {val_loss.item()}")
                    model.train()
                    
                if (step % logging_params['save_ckpt_step'] == 0):
                    save_checkpoint(ckpt_dir, step, model, optimizer, loss_accum.item(),
                                    train_dataloader, scheduler)
                    if master_process:
                        with open(ckpt_dir + '/log.json', 'w') as file:
                            json.dump(logger.__dict__, file)
                loss_accum = 0.
                start_time = time.time() 
            step += 1
            
            
        print(f"In rank: {rank}, epoch {epoch+1}, Train Loss: {logger.losses[-1]}")
        print(f"In rank: {rank}, time taken for epoch {epoch+1} : ", time.time() - start_epoch)
        
        # Evaluate on val set
        model.eval()
        val_loss, counter = 0., 0
        with torch.no_grad():
            for _, batch in enumerate(val_dataloader):
                with autocast_ctxt:
                    if hface_model:
                        val_loss += model(batch[0], labels=batch[0]).loss.item()
                    else:
                        val_loss += model(batch[0], batch[1], return_logits=False)[1]
                counter += 1
        if counter: val_loss /= counter
        val_loss = torch.tensor(val_loss, device=device)
        if world_size > 1: dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        logger.val_losses.append(val_loss.item())
        print(f"In rank: {rank}, epoch {epoch+1}, Validation Loss: {val_loss.item()}")

    if hasattr(optimizer, 'step_size_list'):      # Check if optimizer has a step_size_list attribute
        logger.step_size_list = optimizer.step_size_list  
    return logger
