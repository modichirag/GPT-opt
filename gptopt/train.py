import torch 
import copy

def evaluate_model(model, loss_fun, loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch).squeeze()
            loss = loss_fun(y_batch, y_pred)
            test_loss += loss.item()
    # Average test loss for the epoch
    test_loss /= len(loader)
    return test_loss


def train_iams(num_epochs, model, model_star, IAMS_optimizer, loss_fun, train_loader, test_loader):
    print("training with ", IAMS_optimizer)
    train_losses = []
    test_losses = []
    train_losses.append(evaluate_model(model, loss_fun, train_loader))
    test_losses.append(evaluate_model(model, loss_fun, test_loader))
    # Training loop
    for epoch in range(num_epochs):
    # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch).squeeze()
            loss = loss_fun(y_batch, y_pred)
            with torch.no_grad():
                y_pred_star = model_star(X_batch).squeeze()
                loss_star = loss_fun(y_pred_star, y_batch)
            # Backward pass and optimization
            IAMS_optimizer.zero_grad()
            loss.backward()
            # IAMS_optimizer.step(loss =loss, lb = 0.0)
            IAMS_optimizer.step(loss =loss, lb = loss_star)
            train_loss += loss.item()

        # Average train loss for the epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluation phase
        test_losses.append(evaluate_model(model, loss_fun, test_loader))

        # Print losses every 10 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    return model, train_losses, test_losses


def evaluate_full_batch(model, loss_fun, X, y):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        y_train_pred = model(X).squeeze()
        loss = loss_fun(y, y_train_pred).item()
    return loss

def train_full_batch(num_epochs, model, optimizer, loss_fun, X_train, y_train, X_test, y_test):
    print("training with ", optimizer)
    train_losses = []
    test_losses = []
    train_losses.append(evaluate_full_batch(model, loss_fun, X_train, y_train))
    test_losses.append(evaluate_full_batch(model, loss_fun, X_test, y_test))
    best_model = None
    best_loss = float('inf')

    for epoch in range(num_epochs):
        def closure():
            optimizer.zero_grad()
            y_pred = model(X_train).squeeze()
            loss = loss_fun(y_train, y_pred)
            loss.backward()
            return loss
        # Perform an optimization step
        optimizer.step(closure)
        # Evaluation phase
        train_losses.append(evaluate_full_batch(model, loss_fun, X_train, y_train))
        test_losses.append(evaluate_full_batch(model, loss_fun, X_test, y_test))
        if train_losses[-1] < best_loss:
            best_loss = train_losses[-1]
            best_model = copy.deepcopy(model)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

    return best_model, train_losses, test_losses

def train(num_epochs, model, optimizer, loss_fun, train_loader, test_loader):
    print("training with ", optimizer)
    train_losses = []
    test_losses = []
    train_losses.append(evaluate_model(model, loss_fun, train_loader))
    test_losses.append(evaluate_model(model, loss_fun, test_loader))
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch).squeeze()
            loss = loss_fun(y_batch, y_pred)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # train_loss += loss.item()
        
        # Average train loss for the epoch
        # train_loss /= len(train_loader)
        # train_losses.append(train_loss)
        train_losses.append(evaluate_model(model, loss_fun, train_loader))
        # Evaluation phase
        test_losses.append(evaluate_model(model, loss_fun, test_loader))
        # Print losses every epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

    return model, train_losses, test_losses
