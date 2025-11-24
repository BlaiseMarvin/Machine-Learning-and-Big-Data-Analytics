import torch 

def train_oned(epochs, model, train_data, val_data, device, criterion, optimizer, early_stopper, performance_scheduler=None,other_scheduler=None, l1_norm=False, l2_norm=False, l1_lambda=1e-5, l2_lambda=1e-4):
    n_epochs = epochs

    train_loss = [0]*n_epochs
    val_loss = [0]*n_epochs

    for epoch in range(n_epochs):
        model.train()
        for x_batch,y_batch in train_data:
            # zero the gradients
            optimizer.zero_grad()
            # move the data to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # compute the forward pass
            out = model(x_batch)
            # compute the loss
            if l1_norm:
                norm = sum(p.abs().sum() for p in model.parameters())
                loss = criterion(out.squeeze(1), y_batch) + l1_lambda*norm
            elif l2_norm:
                norm = sum((p ** 2).sum() for p in model.parameters())
                loss = criterion(out.squeeze(1), y_batch) + l2_lambda*norm
            else:
                loss = criterion(out.squeeze(1), y_batch)
            # Compute the gradients
            loss.backward()
            # Backpropagation
            optimizer.step()

            if other_scheduler:
                other_scheduler.step()
            # training loss computation
            train_loss[epoch] += loss.item()
        train_loss[epoch] /= len(train_data.dataset)

        # Validation step
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_data:
                # move the data to GPU
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # forward pass
                out = model(x_batch)
                loss = criterion(out.squeeze(1), y_batch)
                val_loss[epoch]+= loss.item()
            val_loss[epoch]/=len(val_data.dataset)

            if performance_scheduler:
                performance_scheduler.step(val_loss[epoch])
            print(f'Epoch: {epoch+1}| Train loss: {train_loss[epoch]:.4f}| Val loss: {val_loss[epoch]:.4f}')

            # early stopping
            early_stopper(val_loss[epoch], model, optimizer, epoch)
            if early_stopper.should_stop:
                print("Stopping at epoch ",epoch)
                break   
    
    return train_loss, val_loss


def train_multi_d(epochs, model, train_data, val_data, device, criterion, optimizer, early_stopper, performance_scheduler=None,other_scheduler=None, l1_norm=False, l2_norm=False, l1_lambda=1e-5, l2_lambda=1e-4):
    n_epochs = epochs

    train_loss = [0]*n_epochs
    val_loss = [0]*n_epochs

    for epoch in range(n_epochs):
        model.train()
        for x_batch,y_batch in train_data:
            # zero the gradients
            optimizer.zero_grad()
            # move the data to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # compute the forward pass
            out = model(x_batch)
            # compute the loss
            if l1_norm:
                norm = sum(p.abs().sum() for p in model.parameters())
                loss = criterion(out, y_batch) + l1_lambda*norm
            elif l2_norm:
                norm = sum((p ** 2).sum() for p in model.parameters())
                loss = criterion(out, y_batch) + l2_lambda*norm
            else:
                loss = criterion(out, y_batch)
            # Compute the gradients
            loss.backward()
            # Backpropagation
            optimizer.step()

            if other_scheduler:
                other_scheduler.step()
            # training loss computation
            train_loss[epoch] += loss.item()
        train_loss[epoch] /= len(train_data.dataset)

        # Validation step
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_data:
                # move the data to GPU
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # forward pass
                out = model(x_batch)
                loss = criterion(out, y_batch)
                val_loss[epoch]+= loss.item()
            val_loss[epoch]/=len(val_data.dataset)

            if performance_scheduler:
                performance_scheduler.step(val_loss[epoch])
            print(f'Epoch: {epoch+1}| Train loss: {train_loss[epoch]:.4f}| Val loss: {val_loss[epoch]:.4f}')

            # early stopping
            early_stopper(val_loss[epoch], model, optimizer, epoch)
            if early_stopper.should_stop:
                print("Stopping at epoch ",epoch)
                break   
    
    return train_loss, val_loss

def train_oned2(epochs, model, train_data, val_data, device, criterion, optimizer, early_stopper, performance_scheduler=None,other_scheduler=None, l1_norm=False, l2_norm=False, l1_lambda=1e-5, l2_lambda=1e-4):
    n_epochs = epochs

    train_loss = [0]*n_epochs
    val_loss = [0]*n_epochs

    for epoch in range(n_epochs):
        model.train()
        for x_batch,y_batch in train_data:
            # zero the gradients
            optimizer.zero_grad()
            # move the data to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # compute the forward pass
            out = model(x_batch)
            # compute the loss
            if l1_norm:
                norm = sum(p.abs().sum() for p in model.parameters())
                loss = criterion(out, y_batch) + l1_lambda*norm
            elif l2_norm:
                norm = sum((p ** 2).sum() for p in model.parameters())
                loss = criterion(out, y_batch) + l2_lambda*norm
            else:
                loss = criterion(out, y_batch)
            # Compute the gradients
            loss.backward()
            # Backpropagation
            optimizer.step()

            if other_scheduler:
                other_scheduler.step()
            # training loss computation
            train_loss[epoch] += loss.item()
        train_loss[epoch] /= len(train_data.dataset)

        # Validation step
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_data:
                # move the data to GPU
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # forward pass
                out = model(x_batch)
                loss = criterion(out, y_batch)
                val_loss[epoch]+= loss.item()
            val_loss[epoch]/=len(val_data.dataset)

            if performance_scheduler:
                performance_scheduler.step(val_loss[epoch])
            print(f'Epoch: {epoch+1}| Train loss: {train_loss[epoch]:.4f}| Val loss: {val_loss[epoch]:.4f}')

            # early stopping
            early_stopper(val_loss[epoch], model, optimizer, epoch)
            if early_stopper.should_stop:
                print("Stopping at epoch ",epoch)
                break   
    
    return train_loss, val_loss