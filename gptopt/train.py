from gptopt.utils import compute_cross_entropy_loss
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def compute_loss_on_batch(tokenizer, model, batch, device, max_length):
    """Simplified helper function to compute loss on a batch."""
    inputs = tokenizer(batch['text'], return_tensors="pt", max_length=max_length, padding=True, truncation=True).to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss

def train(tokenizer, train_dataloader, test_dataloader, model, optimizer, training_params, device="cuda", scheduler=None, pass_loss=False):
    losses = []
    test_losses = []
    learning_rates = []
    for epoch in range(training_params['num_epochs']):
        model.train()
        for batch in train_dataloader:
            # Compute training loss
            loss = compute_loss_on_batch(tokenizer, model, batch, device, training_params['max_length'])
            losses.append(loss.item())
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            if pass_loss:
                optimizer.step(closure=None, loss=loss)
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
        print(f"Epoch {epoch+1}, Train Loss: {losses[-1]}")

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                test_loss += compute_loss_on_batch(tokenizer, model, batch, device, training_params['max_length']).item()
        test_loss /= len(test_dataloader)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss}")

    output = {'losses': losses, 'test_losses': test_losses, 'learning_rates': learning_rates}
    # Check if optimizer has a step_size_list attribute
    step_size_list = None
    if hasattr(optimizer, 'step_size_list'):  # Correctly check for the attribute
        step_size_list = optimizer.step_size_list  # Access the attribute using dot notation
        output['step_size_list'] = step_size_list
    return output



