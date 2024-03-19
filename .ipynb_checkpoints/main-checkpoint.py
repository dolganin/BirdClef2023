#model, optim, scheduler, loss, transform 
birds_classifier = Model()
birds_classifier = birds_classifier.to(device)
optim = AdamW(birds_classifier.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, eta_min=1e-5, T_max=10)
loss_fn = nn.CrossEntropyLoss()
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((120, 224))])

data = pd.read_csv('/kaggle/input/birdclef-2023/train_metadata.csv')
data = pd.concat([ pd.Series(data['primary_label']), pd.Series(data['type']), pd.Series(data['filename'], name='path')], axis=1, names=['primary_label', 'type', 'path'])

data = pd.concat([data, pd.get_dummies(data['primary_label'])], axis=1)
birds = list(pd.get_dummies(data['primary_label']).columns)

train_data, test_data = train_test_split(data, train_size=0.8)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

train_dataset = BirdDataset(train_data, transform=transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs)

test_dataset = BirdDataset(test_data)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=bs)


temp_score = 0
for i in range(epochs):
    print(i)
    train(birds_classifier, train_dataloader, optim, scheduler)
    cur_score = test(birds_classifier, test_dataloader)
    print(cur_score)
    if(cur_score)>temp_score:
        temp_score = cur_score
        torch.save(birds_classifier.state_dict(), '/kaggle/working/best.pth')