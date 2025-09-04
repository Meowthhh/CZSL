import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class KernelGenerator(nn.Module):
    def __init__(self, in_dim=2048, num_kernels=85):
        super(KernelGenerator, self).__init__()
        self.in_dim = in_dim
        self.num_kernels = num_kernels
        
        self.fc  = nn.Linear(in_dim, num_kernels)
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc(x)

        kernels = self.fc.weight.clone().detach()
        kernels = kernels.unsqueeze(0).expand(batch_size, -1, -1)

        return x, kernels
    

class KernelUpdateHead(nn.Module):
    def __init__(self, alpha=0.5, in_dim=2048, num_heads=8):
        super(KernelUpdateHead, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.in_dim = in_dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(embed_dim=in_dim,
                                         num_heads=num_heads,
                                         batch_first=True)
        self.norm1 = nn.LayerNorm(in_dim)
        
        ff_dim = in_dim * 2
        self.ff = nn.Sequential(
            nn.Linear(in_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, in_dim)
        )
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(0.1)

        self.init_weights()

    def init_weights(self):
        # Initialize feedforward layers
        for m in self.ff:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
        

    def forward(self, feats, kernels, x):
        # Kernel update
        update = feats.unsqueeze(-1) * x.unsqueeze(1)
        k_bar = self.alpha * kernels + self.beta * update

        # Use MHA + fc on the kernels
        # attn_out = self.mha(k_bar, k_bar, k_bar)[0]
        # residual = self.norm1(k_bar + self.dropout(attn_out))
        # ff_out = self.ff(residual)
        # new_kernels = self.norm2(residual + self.dropout(ff_out))

        new_kernels = k_bar

        # Generate the new features 
        new_feats = torch.einsum("bij,bj->bi", new_kernels, x)
        new_feats = new_feats + feats  # Residual connection

        return new_feats, new_kernels


class DKClassifier3(nn.Module):
    def __init__(self, in_dim=2048, num_kernels=85, num_stages=3, alpha=0.5, num_heads=8):
        super(DKClassifier3, self).__init__()
        self.num_stages = num_stages
        self.kernel_generator = KernelGenerator(in_dim=in_dim,
                                                num_kernels=num_kernels)
        self.kernel_update_head = nn.ModuleList([KernelUpdateHead(in_dim=in_dim, 
                                                                  num_heads=num_heads,
                                                                  alpha=alpha) for _ in range(num_stages)])
        
    def forward(self, x):
        feats, kernels = self.kernel_generator(x)
        for i in range(self.num_stages):
            feats, kernels = self.kernel_update_head[i](feats, kernels, x)

        return feats 
    

if __name__ == "__main__":
    model = DKClassifier3()
    x = torch.randn(64, 2048) 
    output = model(x)
    print(output.shape) # Should be (64, 85)