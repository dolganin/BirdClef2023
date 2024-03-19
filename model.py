#model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b1', pretrained=False)
        self.in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(self.in_features, classnum))
    
    def forward(self, images):
        features = self.model(images)
        return features