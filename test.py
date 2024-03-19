#test
def test(model, data_loader):
    model.eval()
    
    pred = []
    label = []
    
    for mels, labels in data_loader:
        mels = mels.to(device)
        labels = labels.to(device)
        
        outputs = model(mels)
        _, preds = torch.max(outputs, 1)
        
        label.append(labels.cpu().detach())
        pred.append(outputs.sigmoid().cpu().detach())
        
        
    labels_df = torch.cat([x for x in label], dim=0)
    pred_df = torch.cat([x for x in pred], dim=0)
    label_df = pd.DataFrame(labels_df)  
    pred_df = pd.DataFrame(pred_df)  
    current_score = padded_cmap(label_df, pred_df)
    
    return current_score