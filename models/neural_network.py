import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        #self.net #old version
        self.features_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Linear(256, num_classes)   #old version
        )

        self.classifier = nn.Linear(256, num_classes) #new version for ICaRL

    def forward(self, x):
        #return self.net(x) #old version
        features = self.features_extractor(x) #new version for ICaRL
        logits = self.classifier(features) #new version for ICaRL
        return logits #new version for ICaRL
