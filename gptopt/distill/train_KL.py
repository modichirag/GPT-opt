import torch

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    # Apply temperature to logits
    student_probs = torch.nn.functional.softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    # Compute Kullback-Leibler divergence
    loss = torch.nn.functional.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    return loss

# Function to evaluate the model on the test set
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_text = batch['text'][0]
            inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
            teacher_outputs = teacher_model(**inputs)
            student_outputs = model(**inputs)
            loss = distillation_loss(student_outputs.logits, teacher_outputs.logits)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
num_epochs = 3
temperature = 2.0
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    student_model.train()
    for batch in train_dataloader:
        # Prepare input text
        input_text = batch['text'][0]
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        
        # Forward pass through teacher and student
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
        student_outputs = student_model(**inputs)
        
        # Compute the distillation loss
        loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, temperature)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Collect training loss
        train_losses.append(loss.item())
    
    # Evaluate on the test set
    test_loss = evaluate(student_model, test_dataloader)
    test_losses.append(test_loss)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Test Loss: {test_loss}")
