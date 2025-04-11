from gptopt.utils import compute_cross_entropy_loss
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import time  # Import time module to measure execution time
import contextlib

typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}


def train(train_dataloader, val_dataloader, model, optimizer, training_params, device="cuda", scheduler=None):
    losses = []
    val_losses = []
    learning_rates = []
    step_times = []  # List to record time spent on optimizer.step()
    optimizer_name = optimizer.__class__.__name__
    if 'Momo' in optimizer_name:
        pass_loss = True
    else:
        pass_loss = False
    print(f"Set pass_loss to {pass_loss} for optimizer {optimizer_name}")
    
    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (B*T))
    print(f"Accumulate gradient for {grad_accum_steps} steps")
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']]) 

    # Training loop
    for epoch in range(training_params['num_epochs']):
        print(f"Epoch {epoch+1} of {training_params['num_epochs']}")

        model.train()
        start_epoch = time.time()
        loss_accum = 0.0
        start_time = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):

            # Compute training loss
            with ctxt:
                loss = model(batch[0], labels=batch[0]).loss
                loss /= grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            
            # Optimization step
            if (step + 1) % grad_accum_steps == 0:
                if 'gradnorm' in training_params:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_params['gradnorm'])
                if pass_loss:
                    optimizer.step(closure=None, loss=loss)
                else:
                    optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                step_time = time.time() - start_time  # Calculate elapsed time
                step_times.append(step_time)  # Record the time
                losses.append(loss_accum.item())
                if ((step + 1) % 10 == 0):
                    tps = training_params["tokens_processed"] / step_time
                    print(f"Step {step+1} of {total_iterations*grad_accum_steps}.")
                    print(f"Time taken : {step_time*1000:0.1f}ms | Tokens/s : {tps/1000:0.1f} thousand | Loss : {loss_accum.item():0.3f}")
                if scheduler is not None:
                    scheduler.step()
                for param_group in optimizer.param_groups:
                    learning_rates.append(param_group['lr'])
                loss_accum = 0.
                start_time = time.time()
                
        print(f"Epoch {epoch+1}, Train Loss: {losses[-1]}")
        print(f"Time for epoch {epoch+1} : ", time.time() - start_epoch)

        # Evaluate on validation set
        model.eval()
        val_loss, counter = 0., 0
        with torch.no_grad():
            for _, batch in enumerate(val_dataloader):
                with ctxt:
                    val_loss += model(batch[0], labels=batch[0]).loss.item()
                counter += 1
        val_loss /= counter
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        
    output = {'losses': losses, 'val_losses': val_losses, 'learning_rates': learning_rates, 'step_times': step_times}
    # Check if optimizer has a step_size_list attribute
    step_size_list = None
    if hasattr(optimizer, 'step_size_list'):  # Correctly check for the attribute
        step_size_list = optimizer.step_size_list  # Access the attribute using dot notation
        output['step_size_list'] = step_size_list
    return output
