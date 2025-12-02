import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, use_sigmoid=False):
        super(NeuralNetwork, self).__init__()
        
        # Architettura Base
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # SE USI ICARL: Serve la Sigmoid
        if use_sigmoid:
            self.classifier = nn.Sequential(
                nn.Linear(256, num_classes),
                nn.Sigmoid() 
            )
        # SE USI REPLAY/DER: Basta il Linear
        else:
            self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x