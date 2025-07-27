import torchvision
import torch
from torch import nn
import torch.nn.functional as F
class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale,
        dropout = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, 4*dim, bias=True)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4*dim, dim, bias=True)
        
        # self.block = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
        #     Permute([0, 2, 3, 1]),
        #     nn.LayerNorm(dim),
        #     nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
        #     nn.GELU(),
        #     nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
        #     Permute([0, 3, 1, 2]),
        # )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        identity = x
        output = self.conv(x)
        if self.dropout is not None:
            output = self.dropout(output)
        output = output.permute(0, 2, 3, 1)
        output = self.norm(output)
        output = self.linear1(output)
        output = self.gelu(output)
        output = self.linear2(output)
        output = output.permute(0, 3, 1, 2)
        
        output = self.layer_scale * output
        output += identity
        return output
    
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class ConvNextNet(nn.Module):
    def __init__(self, num_classes=7001):
        super().__init__()
        channels = [64, 96, 192, 384, 768]
        num_layers = [1, 1, 2, 3, 1]
        self.features = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=4, stride=4, padding=0),
            LayerNorm2d(channels[0]))
        ])
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            for l in range(num_layers[i]-1):
                self.features.append(nn.Sequential(
                    CNBlock(in_channels, 1.0, dropout=True),
                    LayerNorm2d(in_channels),
                ))
            # last layer in the block does downsampling
            downsample = nn.Conv2d(in_channels, out_channels, 2, stride=2)
            if i == len(channels) - 2:
                downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2)
            self.features.append(nn.Sequential(
                CNBlock(in_channels, 1.0, dropout=True),
                LayerNorm2d(in_channels),
                downsample
            ))
        self.features.append(
            nn.AvgPool2d((3, 3))
        )
        self.cls_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, num_classes)
        )
        # weight initializations
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
    def forward(self, x, return_feats=False):
        feats = x
        for layer in self.features:
            feats = layer(feats)
        feats = feats.flatten(1)
        out_logits = self.cls_layer(feats)
        
        if return_feats:
            return feats, out_logits
        else:
            return out_logits    