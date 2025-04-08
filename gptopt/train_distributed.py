from gptopt.utils import compute_cross_entropy_loss
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import time  # Import time module to measure execution time
import contextlib
import os

typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}


def train(train_dataloader, val_dataloader, model, optimizer, training_params, device="cuda", scheduler=None):
    losses = []
    test_losses = []
    learning_rates = []
    step_times = []  # List to record time spent on optimizer.step()
    optimizer_name = optimizer.__class__.__name__
    if 'Momo' in optimizer_name:
        pass_loss = True
    else:
        pass_loss = False
    print(f"Set pass_loss to {pass_loss} for optimizer {optimizer_name}")
    ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']]) 

    # Training loop
    for epoch in range(training_params['num_epochs']):

        model.train()
        start_epoch = time.time()
        for ib, batch in enumerate(train_dataloader):
            
            start_time = time.time() 
            # Compute training loss
            with ctxt:
                loss = model(batch[0], labels=batch[0]).loss
            losses.append(loss.item())
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            if 'gradnorm' in training_params:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_params['gradnorm'])
            if pass_loss:
                optimizer.step(closure=None, loss=loss)
            else:
                optimizer.step()
            torch.cuda.synchronize()
            step_time = time.time() - start_time  # Calculate elapsed time
            step_times.append(step_time)  # Record the time
            #if ib % 10 == 0 :
            #    print(f"Time taken for step : {step_time*1000:0.2f}ms | Loss : {loss.item():0.3f}")
            if scheduler is not None:
                scheduler.step()
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
        print(f"Epoch {epoch+1}, Train Loss: {losses[-1]}")
        print(f"Time taken for epoch {epoch+1} : ", time.time() - start_epoch)
        # # Evaluate on test set
        # model.eval()
        # test_loss = 0
        # with torch.no_grad():
        #     for batch in val_dataloader:
        #         with ctxt:
        #             test_loss += model(batch[0], labels=batch[0]).loss.item()
        # test_loss /= len(val_dataloader)
        # test_losses.append(test_loss)
        # print(f"Epoch {epoch+1}, Test Loss: {test_loss}")

        
    output = {'losses': losses, 'test_losses': test_losses, 'learning_rates': learning_rates, 'step_times': step_times}
    # Check if optimizer has a step_size_list attribute
    step_size_list = None
    if hasattr(optimizer, 'step_size_list'):  # Correctly check for the attribute
        step_size_list = optimizer.step_size_list  # Access the attribute using dot notation
        output['step_size_list'] = step_size_list
    return output
