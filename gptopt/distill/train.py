from gptopt.utils import compute_cross_entropy_loss
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
# from transformers.optimization import get_cosine_schedule_with_warmup

# Scheduler creation function with default behavior for warm-up and cosine decay
def get_scheduler(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    # Define the warm-up function
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    # Warm-up scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
    # Cosine decay scheduler
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps)
    return warmup_scheduler, cosine_scheduler

def train(tokenizer, train_dataloader, student_model,  optimizer, training_params, teacher_model=None, device = "cuda", get_scheduler_fn=None):
    if teacher_model is not None:
        teacher_model.eval() # try this eventually
    num_epochs = training_params['num_epochs']
    student_losses = []; teacher_losses = []; learning_rates = []
    total_iterations = num_epochs * len(train_dataloader)
    # Initialize scheduler if provided
    if get_scheduler_fn is not None:
        num_warmup_steps = int(training_params['warm_up_percent'] * total_iterations) 
        scheduler = get_scheduler_fn(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_iterations
                    )
    else:
        scheduler = None

    for epoch in range(num_epochs):
        student_model.train() 
        for batch in train_dataloader:
            input_text = batch['text']
            inputs = tokenizer(input_text, return_tensors='pt', max_length=training_params['max_length'] , truncation=True, padding='max_length')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)  # Move attention_mask to the GPU
            # Forward pass for the student model
            labels = input_ids.clone().to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            outputs  = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss     = outputs.loss  # Cross-entropy loss is computed internally when labels are provided
            student_losses.append(loss.item())
            if teacher_model is not None: #get teacher loss
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    teacher_loss = teacher_outputs.loss   
                    teacher_losses.append(teacher_loss.item())
            optimizer.zero_grad()
            loss.backward()
            if teacher_model is not None:
                optimizer.step(loss =loss, lb = teacher_loss)
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
        if teacher_model is None:
            print(f"Epoch {epoch+1}, Student Loss: {student_losses[-1]}") 
        else:
            print(f"Epoch {epoch+1}, Student Loss: {student_losses[-1]}, Teacher Loss: {teacher_losses[-1]}")

    if teacher_model is None:
        return {'student_losses' : student_losses, 'learning_rates': learning_rates }
    else:
        return {'student_losses' : student_losses, 'teacher_losses' : teacher_losses, 'learning_rates': learning_rates }

