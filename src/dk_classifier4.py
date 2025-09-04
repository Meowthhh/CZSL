import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Encoder(nn.Module):
    def __init__(self, input_dim=2048, output_dim=85, hidden_dims=[1024, 512]):
        super(Encoder, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            in_dim = h_dim
        
        # Final layer to 85D output
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
    

class Decoder(nn.Module):
    def __init__(self, input_dim=85, output_dim=2048, hidden_dims=[512, 1024]):
        super(Decoder, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            in_dim = h_dim
        
        # Final layer to 85D output
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class DKClassifier4(nn.Module):
    def __init__(self, in_dim=2048, out_dim=85, hidden_dims=[1024, 512]):
        super(DKClassifier4, self).__init__()
        self.encoder = Encoder(input_dim=in_dim, output_dim=out_dim, hidden_dims=hidden_dims)
        self.decoder = Decoder(input_dim=out_dim, output_dim=in_dim, hidden_dims=hidden_dims[::-1])
        
        
    def forward(self, x):
        logits = self.encoder(x)

        if self.training:
            reconstructed_x = self.decoder(logits)
            return logits, reconstructed_x

        return logits 
    

if __name__ == "__main__":
    model = DKClassifier4()
    x = torch.randn(64, 2048) 
    output = model(x)
    print(output.shape) # Should be (64, 85)