import torch.nn as nn
import torch.nn.functional as F 

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        h1, h2, h3, h4, dropout       
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.LayerNorm(h1),       
            nn.LeakyReLU(0.01),     
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.LayerNorm(h2),        
            nn.LeakyReLU(0.01),   
            nn.Dropout(dropout),

            nn.Linear(h2, h3),
            nn.LayerNorm(h3),        
            nn.LeakyReLU(0.01),   
            nn.Dropout(dropout),

            nn.Linear(h3, h4),
            nn.LayerNorm(h4),       
            nn.LeakyReLU(0.01),    
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(h4, num_classes, bias=False)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        x = self.feature_extractor(x)
        x = F.normalize(x, p=2, dim=1)
        return x