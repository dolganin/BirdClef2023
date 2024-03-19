def train(model, data_loader, optimizer, scheduler):
    model.train()
    
    cnt = 0
    for mels, labels in data_loader:
        optimizer.zero_grad()
        
        mels = mels.to(device)
        labels = labels.to(device)
        
        outputs = model(mels)
        _, preds = torch.max(outputs, 1)
       
        loss = loss_fn(outputs, labels)
        if(cnt%10==0):
            print(loss.item())
        cnt+=1
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()