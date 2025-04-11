from gptopt.utils import get_worker_info
import torch
import time 
import contextlib
import torch.distributed as dist

typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}

class Logging():

    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []

        
def train(train_dataloader, val_dataloader, model, optimizer, training_params, scheduler=None):
    world_size, rank, local_rank, device  = get_worker_info()
    master_process = (rank == 0)
    logger = Logging()
    optimizer_name = optimizer.__class__.__name__
    if 'Momo' in optimizer_name:
        pass_loss = True
    else:
        pass_loss = False
    if master_process: print(f"Set pass_loss to {pass_loss} for optimizer {optimizer_name}")
    
    ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']]) 

    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (world_size*B*T))
    if master_process: print(f"Accumulate gradient for {grad_accum_steps} steps")
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])

    if 'gradnorm' in training_params: max_grad_norm = training_params['gradnorm']    
    else: max_grad_norm = float('inf')

    # Training loop
    for epoch in range(training_params['num_epochs']):
        if master_process:
            print(f"Epoch {epoch+1} of {training_params['num_epochs']}")

        model.train()
        start_epoch = time.time()
        start_time = time.time() 
        loss_accum = 0.
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            
            with ctxt:
                loss = model(batch[0], labels=batch[0]).loss
                loss /= grad_accum_steps
            loss_accum += loss.detach()
                
            # Check if accummulated enough gradients to take a step
            if (step + 1) % grad_accum_steps != 0:
                with model.no_sync():
                    loss.backward() 
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
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
                if ((step + 1) % 10 == 0) & master_process :
                    tps = training_params["tokens_processed"] / step_time
                    print(f"In rank: {rank}, step {step+1} of {total_iterations*grad_accum_steps}.")
                    print(f"Time taken : {step_time*1000:0.1f}ms | Tokens/s : {tps/1000:0.1f} thousand | Loss : {loss_accum.item():0.3f}")
                loss_accum = 0.
                start_time = time.time() 
        
        print(f"In rank: {rank}, epoch {epoch+1}, Train Loss: {logger.losses[-1]}")
        print(f"In rank: {rank}, time taken for epoch {epoch+1} : ", time.time() - start_epoch)
        
        # Evaluate on val set
        model.eval()
        val_loss, counter = 0., 0
        with torch.no_grad():
            for _, batch in enumerate(val_dataloader):
                with ctxt:
                    val_loss += model(batch[0], labels=batch[0]).loss.item()
                counter += 1
        if counter: val_loss /= counter
        val_loss = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        logger.val_losses.append(val_loss.item())
        print(f"In rank: {rank}, epoch {epoch+1}, Validation Loss: {val_loss.item()}")

    if hasattr(optimizer, 'step_size_list'):      # Check if optimizer has a step_size_list attribute
        logger.step_size_list = optimizer.step_size_list  
    return logger
