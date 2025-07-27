import torch
import torchvision
from torch import nn
from modeling_utils import *

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_channels,
        squeeze_channels,
        activation = nn.ReLU,
        scale_activation = nn.Sigmoid
    ):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()
        
    def _scale(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)
    
    def forward(self, x):
        scale = self._scale(x)
        return scale * x
    
class MBConv(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        expand_ratio = 1,
        kernel_size = 3,
        stride = 1
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and input_channels == output_channels
        
        hidden_dim = int(round(input_channels * expand_ratio))
        layers = []
        
        norm_layer = nn.BatchNorm2d
        activation_layer = nn.SiLU
        if expand_ratio != 1:
            # expanding conv
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            )
        
        # dw
        if kernel_size == 5:
            padding = 2
        else:
            padding = 1
            
        layers.append(
            Conv2dNormActivation(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        )
        
        # squeeze excite
        squeeze_channels = max(1, input_channels // 4)
        layers.append(
            SqueezeExcitation(
                hidden_dim,
                squeeze_channels,
                activation=nn.SiLU        
            )
        )
        
        # projection
        layers.extend([
            nn.Dropout(0.2), #after squeeze excite, we have large number of channels,..
            nn.Conv2d(hidden_dim, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x += identity
        return x
    
class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes=7001,
    ):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        
        config = [
            # e k s i o n => e o n k s
            [1, 16, 1, 3, 1],
            [6, 24, 2, 3, 2],
            [6, 40, 2, 5, 2],
            [6, 80, 3, 3, 2],
            [6, 112,3, 5, 1],
            [6, 192,4, 5, 2],
            [6, 320,1, 3, 1],
        ]
        
        in_channels = 32
        last_channel = 512
        
        layers = []
        layers.append(
            Conv2dNormActivation(
                3, in_channels, 3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )
        
        for expand, channels, count, kernel, stride in config:
            stage = []
            out_channels = channels
            for n in range(count):
                stage.append(
                    MBConv(
                        in_channels, 
                        out_channels, 
                        expand_ratio=expand, 
                        kernel_size=kernel, 
                        stride=stride,
                    )
                )
                stride = 1
                in_channels = out_channels
            layers.append(nn.Sequential(*stage))
            
        layers.extend([
            nn.Conv2d(in_channels, last_channel, 1),
            nn.BatchNorm2d(last_channel)
        ])
        
        self.features = nn.Sequential(*layers)
        
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, return_feats=False):
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        feats = torch.flatten(x, 1)
        out_logits = self.classifier(feats)
        if return_feats:
            return feats, out_logits
        else:
            return out_logits    