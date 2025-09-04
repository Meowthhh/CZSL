import torch
import torch.nn as nn
import torch.nn.functional as F
    

class AdaptiveKernelUpdater(nn.Module):
    def __init__(self, feat_channels=32):
        super(AdaptiveKernelUpdater, self).__init__()
        self.in_channels = feat_channels
        self.feat_channels = feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.feat_channels + self.feat_channels
        )
        self.input_layer = nn.Linear(
            self.in_channels, self.feat_channels + self.feat_channels, 1
        )
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.input_norm_in = nn.LayerNorm(self.feat_channels)
        self.input_norm_out = nn.LayerNorm(self.feat_channels)
        self.norm_in = nn.LayerNorm(self.feat_channels)
        self.norm_out = nn.LayerNorm(self.feat_channels)
        self.activation = nn.ReLU()
        self.fc_layer = nn.Linear(self.feat_channels, self.feat_channels, 1)
        self.fc_norm = nn.LayerNorm(self.feat_channels)

    def forward(self, fk, kernels):
        fk = fk.reshape(-1, self.feat_channels)
        num_proposals = fk.size(0)

        # Generate dynamic parameters
        parameters = self.dynamic_layer(fk)
        param_in = parameters[:, :self.feat_channels].view(
            -1, self.feat_channels)
        param_out = parameters[:, -self.feat_channels:].view(
            -1, self.feat_channels)
        
        # Input transformation for kernels
        input_feats = self.input_layer(
            kernels.reshape(num_proposals, -1, self.feat_channels))
        input_in = input_feats[..., :self.feat_channels]
        input_out = input_feats[..., -self.feat_channels:]
        
        # Gating mechanisms
        gate_feats = input_in * param_in.unsqueeze(-2)
        input_gate = self.input_norm_in(self.input_gate(gate_feats)).sigmoid()
        update_gate = self.norm_in(self.update_gate(gate_feats)).sigmoid()
        
        # Update kernels
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)
        param_out = self.activation(param_out)
        input_out = self.activation(input_out)

        k_bar = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out
        k_bar = self.fc_layer(k_bar)
        k_bar = self.fc_norm(k_bar)
        k_bar = self.activation(k_bar)

        return k_bar


class KernelUpdateHead(nn.Module):
    def __init__(self, feat_channels=32, kernel_size=3):
        super(KernelUpdateHead, self).__init__()
        self.feat_channels = feat_channels
        self.kernel_size = kernel_size
        self.adaptive_kernel_updater = AdaptiveKernelUpdater(
            feat_channels=feat_channels
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feat_channels*kernel_size**2, num_heads=8
        )
        self.attn_norm = nn.LayerNorm(feat_channels*kernel_size**2)
        self.fc_mask = nn.Linear(feat_channels, feat_channels)
        self.reg_layer = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.LayerNorm(feat_channels),
            nn.ReLU()
        )

    def forward(self, features, kernels, initial_maps):
        batch_size, num_kernels, C = kernels.shape[:3]
        
        # Feature accumulation
        fk = torch.einsum('bnhw,bchw->bnc', initial_maps, features)
        
        # Adaptive feature update
        kernels = kernels.reshape(batch_size, num_kernels,
        C, -1).permute(0, 1, 3, 2) # [B, N, C, K, K] -> [B, N, K*K, C]
        k_bar = self.adaptive_kernel_updater(fk, kernels)

        # Kernel interaction
        k_bar = k_bar.reshape(batch_size, num_kernels,
                             -1).permute(1, 0, 2) # [B, N, K*K, C] -> [N, B, K*K*C]
        k_bar = self.multihead_attn(k_bar, k_bar, k_bar)[0]
        k_bar = self.attn_norm(k_bar)
        k_bar = k_bar.permute(1, 0, 2) # [N, B, K*K*C] -> [B, N, K*K*C]

        # [B, N, K*K*C] -> [B, N, K*K, C]
        k_bar = k_bar.reshape(batch_size, num_kernels, -1, C)
        k_new = k_bar 

        # New feature map generation
        k_bar = self.reg_layer(k_bar)
        k_bar = self.fc_mask(k_bar).permute(0, 1, 3, 2)
        k_bar = k_bar.reshape(batch_size, num_kernels, C, 
                                self.kernel_size, self.kernel_size)
        new_feature_maps = []
        for i in range(batch_size):
            new_feature_maps.append(
                F.conv2d(
                    features[i:i+1],
                    k_bar[i],
                    padding=self.kernel_size // 2
                )
            )
        new_feature_maps = torch.cat(new_feature_maps, dim=0)
        new_feature_maps = F.softmax(new_feature_maps, dim=1)

        k_new = k_new.permute(0, 1, 3, 2).reshape(
            batch_size, num_kernels, C, self.kernel_size,
            self.kernel_size
        )
        return new_feature_maps, k_new


class DKClassifier(nn.Module):
    def __init__(self, num_attributes=85, num_stages=3, feat_in_dim=2048, feat_out_dim=32, 
                 img_size=128, threshold=0.5, kernel_size=3, kernel_inter_dim=128,
                 num_kernels=85):
        super(DKClassifier, self).__init__()
        self.num_attributes = num_attributes
        self.num_stages = num_stages
        self.img_size = img_size
        self.threshold = threshold
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.feat_out_dim = feat_out_dim
        self.kernel_generator = nn.Sequential(
            nn.Conv2d(feat_out_dim, kernel_inter_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(kernel_inter_dim, num_kernels * feat_out_dim * kernel_size**2,
                      kernel_size=1),
        )
        self.feat_projector = nn.Linear(feat_in_dim, (img_size//2)**2)
        self.feat_expander = nn.Conv2d(1, feat_out_dim, kernel_size=1)
        self.feat_upsampler = nn.ConvTranspose2d(in_channels=feat_out_dim,
                                                 out_channels=feat_out_dim,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)
        self.kernel_update_head = KernelUpdateHead(
            feat_channels=feat_out_dim,
            kernel_size=kernel_size
        )
        self.logit_fc = nn.Linear(num_kernels, num_attributes)
        

    def forward(self, features):
        # Convert raw features to feature maps
        batch_size = features.shape[0]
        projected_features = self.feat_projector(features)
        projected_features = projected_features.view(-1, 1,
                                                     (self.img_size//2), 
                                                     (self.img_size//2))
        projected_features = self.feat_expander(projected_features)
        projected_features = self.feat_upsampler(projected_features)
        
        # Generate the kernels
        kernels = self.kernel_generator(projected_features)
        kernels = kernels.view(batch_size, self.num_kernels, -1,
                            self.kernel_size, self.kernel_size)

        # Get initial feature maps for the kernels
        feature_maps = []
        for i in range(batch_size):
            feature_map = F.conv2d(
                projected_features[i:i+1],
                kernels[i],
                padding=self.kernel_size // 2
            )
            feature_maps.append(feature_map)
        feature_maps = torch.cat(feature_maps, dim=0)
        feature_maps = F.softmax(feature_maps, dim=1)
        # feature_maps = (feature_maps >= self.threshold).float()

        # Iterative kernel update
        for i in range(self.num_stages):
            feature_maps, kernels = self.kernel_update_head(
                projected_features, kernels, feature_maps)

        # Global average pooling
        feature_maps = torch.mean(feature_maps, dim=(2, 3))

        # Generate the logits (B, n_attributes)
        logits = self.logit_fc(feature_maps)
        
        return logits
    



if __name__ == '__main__':
    model = DKClassifier()
    input_tensor = torch.randn(128, 2048)
    output = model(input_tensor)
    print(output.shape)
        