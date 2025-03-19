from gptopt.utils import compute_cross_entropy_loss
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def train(tokenizer, train_dataloader, model,  optimizer, training_params,  device = "cuda", scheduler=None, pass_loss=False):
    losses = [];  learning_rates = []
    for epoch in range(training_params['num_epochs']):
        model.train() 
        for batch in train_dataloader:
            input_text = batch['text']
            inputs = tokenizer(input_text, return_tensors='pt', max_length=training_params['max_length'] , truncation=True, padding='max_length')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)  # Move attention_mask to the GPU
            # Forward pass 
            labels = input_ids.clone().to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            outputs  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss     = outputs.loss  # Cross-entropy loss is computed internally when labels are provided
            losses.append(loss.item())
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            if pass_loss:
                optimizer.step(closure = None, loss=loss)
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
        print(f"Epoch {epoch+1},  Loss: {losses[-1]}") 

    # Check if optimizer has a state called 'step_size_list'
    step_size_list = None
    output = {'losses': losses, 'learning_rates': learning_rates}
    if hasattr(optimizer.state, 'step_size_list'):
        step_size_list = optimizer.state['step_size_list']
        output['step_size_list'] = step_size_list
    return output



