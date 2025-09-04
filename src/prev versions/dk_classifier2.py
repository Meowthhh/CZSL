import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelGenerator(nn.Module):
    def __init__(self, in_dim=2048, kernel_dim=3, num_kernels=85):
        super(KernelGenerator, self).__init__()
        self.in_dim = in_dim
        self.kernel_dim = kernel_dim
        self.num_kernels = num_kernels
        self.kernel_generator = nn.Linear(in_dim, kernel_dim * num_kernels)
        self.feature_expander = nn.Linear(kernel_dim, in_dim)

    def forward(self, features):
        batch_size = features.size(0)
        
        # Generate kernels of shape (batch_size, num_kernels * kernel_dim)
        kernels = self.kernel_generator(features)
        kernels = kernels.view(batch_size, self.num_kernels, self.kernel_dim)
        
        expanded_features = features.unsqueeze(1).repeat(1, self.num_kernels, 1)  # Shape: (batch_size, num_kernels, in_dim)
        reduced_features = torch.einsum('bni,bnk->bnk', expanded_features, kernels)  # Shape: (batch_size, num_kernels, kernel_dim)
        convolved_features = self.feature_expander(reduced_features)  # Shape: (batch_size, num_kernels, in_dim)
        
        return convolved_features, kernels


class StaticKernelUpdater(nn.Module):
    def __init__(self, alpha=0.5):
        super(StaticKernelUpdater, self).__init__()
        # Learnable parameters for weighting
        self.alpha = alpha  # Weight for fk
        self.beta = 1-self.alpha  # Weight for kernels
        self.fc = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU()
        )

    def forward(self, fk, kernels):
        # Expand fk to match the shape of kernels
        fk_expanded = fk.unsqueeze(-1)  # [B, N, 1]
        fk_expanded = self.fc(fk_expanded)  # [B, N, 3]
        fk_expanded = fk_expanded.sigmoid()

        # Perform the weighted sum
        # k_bar = alpha * fk_expanded + beta * kernels  # [B, N, K]
        k_bar = self.alpha * fk_expanded + self.beta * kernels

        return k_bar


class AdaptiveKernelUpdater(nn.Module):
    def __init__(self, feat_channels=3, kernel_size=3):
        super(AdaptiveKernelUpdater, self).__init__()
        self.feat_channels = feat_channels

        # Dynamic Layer for parameter generation
        self.dynamic_layer = nn.Linear(1, feat_channels*2)

        # Layers for input transformation and gating
        self.input_layer = nn.Linear(kernel_size, feat_channels*2)
        self.input_gate = nn.Linear(feat_channels, feat_channels)
        self.update_gate = nn.Linear(feat_channels, feat_channels)

        # Normalization layers
        self.input_norm_in = nn.BatchNorm1d(feat_channels)
        self.input_norm_out = nn.BatchNorm1d(feat_channels)
        self.norm_in = nn.BatchNorm1d(feat_channels)
        self.norm_out = nn.BatchNorm1d(feat_channels)

        # Activation function
        self.activation = nn.ReLU()

        # Final transformation layer
        self.fc_layer = nn.Linear(feat_channels, feat_channels)
        self.fc_norm = nn.BatchNorm1d(feat_channels)


    def forward(self, fk, kernels):
        batch_size, num_kernels, kernel_size = kernels.shape

        # Reshape fk for dynamic layer (adding singleton dimension)
        fk = fk.unsqueeze(-1)  # [B, N, 1]
        fk = fk.reshape(-1, 1)  # [B*N, 1]

        # Generate dynamic parameters
        parameters = self.dynamic_layer(fk)  # [B*N, 2*feat_channels]
        param_in = parameters[:, :self.feat_channels].view(-1, self.feat_channels)  # [B*N, feat_channels]
        param_out = parameters[:, self.feat_channels:].view(-1, self.feat_channels)  # [B*N, feat_channels]

        # Input transformation for kernels
        kernels_reshaped = kernels.reshape(-1, kernel_size)  # [B*N, K]
        input_feats = self.input_layer(kernels_reshaped)  # [B*N, 2*feat_channels]
        input_in = input_feats[:, :self.feat_channels]  # [B*N, feat_channels]
        input_out = input_feats[:, self.feat_channels:]  # [B*N, feat_channels]

        # Reshape for BatchNorm1d
        input_in = input_in.unsqueeze(-1)  # [B*N, feat_channels, 1]
        param_in = param_in.unsqueeze(-1)  # [B*N, feat_channels, 1]

        # Normalize with BatchNorm1d
        input_in = self.input_norm_in(input_in).squeeze(-1)  # [B*N, feat_channels]
        param_in = self.norm_in(param_in).squeeze(-1)  # [B*N, feat_channels]

        # Gating mechanisms
        gate_feats = input_in * param_in  # Element-wise interaction [B*N, feat_channels]
        input_gate = self.input_norm_in(self.input_gate(gate_feats).unsqueeze(-1)).squeeze(-1).sigmoid()
        update_gate = self.norm_in(self.update_gate(gate_feats).unsqueeze(-1)).squeeze(-1).sigmoid()

        # Reshape param_out and input_out for BatchNorm1d
        param_out = param_out.unsqueeze(-1)  # [B*N, feat_channels, 1]
        input_out = input_out.unsqueeze(-1)  # [B*N, feat_channels, 1]

        # Normalize with BatchNorm1d
        param_out = self.norm_out(param_out).squeeze(-1)  # [B*N, feat_channels]
        input_out = self.input_norm_out(input_out).squeeze(-1)  # [B*N, feat_channels]

        # Update kernels
        param_out = self.activation(param_out)  # [B*N, feat_channels]
        input_out = self.activation(input_out)  # [B*N, feat_channels]

        k_bar = update_gate * param_out + input_gate * input_out  # [B*N, feat_channels]
        k_bar = self.fc_layer(k_bar)  # [B*N, feat_channels]
        k_bar = self.fc_norm(k_bar.unsqueeze(-1)).squeeze(-1)  # [B*N, feat_channels]
        k_bar = self.activation(k_bar)  # [B*N, feat_channels]

        # Reshape back to kernel shape
        k_bar = k_bar.view(batch_size, num_kernels, kernel_size)  # [B, N, K]

        return k_bar


class KernelUpdateHead1D(nn.Module):
    def __init__(self, feat_channels=2048, kernel_size=3):
        super(KernelUpdateHead1D, self).__init__()
        self.feat_channels = feat_channels
        self.kernel_size = kernel_size

        # Adaptive kernel updater
        self.adaptive_kernel_updater = AdaptiveKernelUpdater(
            feat_channels=kernel_size, kernel_size=kernel_size
        )
        self.static_kernel_updater = StaticKernelUpdater(alpha=0.5) 

        # Multi-head attention for kernel interaction
        self.multihead_attn = nn.MultiheadAttention(embed_dim=kernel_size, num_heads=3)
        self.attn_norm = nn.LayerNorm(kernel_size)

        # Mask prediction (optional, used for additional feature transformation)
        self.fc_mask = nn.Linear(kernel_size, kernel_size)

        # Regularization layer
        self.reg_layer = nn.Sequential(
            nn.Linear(kernel_size, kernel_size),
            nn.LayerNorm(kernel_size),
            nn.ReLU()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.multihead_attn.in_proj_weight)
        nn.init.constant_(self.multihead_attn.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.multihead_attn.out_proj.weight)
        nn.init.constant_(self.multihead_attn.out_proj.bias, 0.)
        for layer in self.reg_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)

    def forward(self, features, kernels, initial_feats):
        batch_size, num_kernels, C = kernels.shape

        # Feature accumulation: aggregate features with initial maps
        fk = torch.einsum('bnl,bl->bn', initial_feats, features) # Might try thresholding initial_feats 

        # Adaptive feature update
        k_bar = self.adaptive_kernel_updater(fk, kernels)
        # k_bar = self.static_kernel_updater(fk, kernels)
        # k_bar = kernels

        # Kernel interaction using multi-head attention
        k_bar = k_bar.permute(1, 0, 2)  # Shape: [num_kernels, batch_size, feat_channels]
        k_bar = self.multihead_attn(k_bar, k_bar, k_bar)[0]
        k_bar = self.attn_norm(k_bar)
        k_bar = k_bar.permute(1, 0, 2)  # Shape: [batch_size, num_kernels, feat_channels]

        # Kernel update
        k_new = k_bar

        # New feature map generation
        k_bar = self.reg_layer(k_bar)
        k_bar = self.fc_mask(k_bar)

        new_feats = []
        for i in range(batch_size):
            # Apply 1D convolution using kernels to features
            conv_features = F.conv1d(
                features[i:i+1].unsqueeze(1),  # [1, 1, feat_channels]
                k_bar[i].unsqueeze(1),        # [num_kernels, 1, feat_channels]
                padding='same',
            )
            new_feats.append(conv_features)
        new_feats = torch.cat(new_feats, dim=0)  # [batch_size, num_kernels, feat_channels]

        # new_feats = F.softmax(new_feats, dim=1)

        return new_feats, k_new


class DKClassifier2(nn.Module):
    def __init__(self, in_dim=2048, kernel_dim=3, num_kernels=85, num_attributes=85, num_stages=3):
        super(DKClassifier2, self).__init__()
        self.num_stages = num_stages
        self.kernel_generator = KernelGenerator(in_dim=in_dim,
                                                kernel_dim=kernel_dim,
                                                num_kernels=num_kernels)
        self.kernel_update_head = KernelUpdateHead1D(feat_channels=in_dim, kernel_size=kernel_dim)
        self.dim_reduction_fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.logit_fc = nn.Linear(num_kernels, num_attributes)

    def forward(self, features):
        # Generate kernels and initial feats
        feats, kernels = self.kernel_generator(features) # (B, N, C) (B, N, K)

        for i in range(self.num_stages):
            feats, kernels = self.kernel_update_head(features, kernels, feats)

        feats = self.dim_reduction_fc(feats).squeeze(-1)
        logits = self.logit_fc(feats)

        return logits
    


if __name__ == '__main__':
    model = DKClassifier2()
    input_tensor = torch.randn(64, 2048)
    output = model(input_tensor)
    print(output.shape)