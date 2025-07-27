import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_features, features):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_features, features, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(features, features,  3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(features)
        self.downsample = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, features, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(features)
        )
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
class TripletFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # self.convnext = torchvision.models.convnext_base(pretrained=True).features.eval()
        # freeze the backbone network
        #for param in self.convnext.parameters():
        #   param.requires_grad_(False)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU()
        )
        self.backbone = nn.Sequential(
            BasicBlock(32, 64),
            BasicBlock(64, 128),
            BasicBlock(128, 256),
            BasicBlock(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.head = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
        )

    def forward_once(self, x):
        # x = self.convnext(x)
        x = self.init_conv(x)
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

    def forward(self, a, p, n):
        a = self.forward_once(a)
        p = self.forward_once(p)
        n = self.forward_once(n)
        return a, p, n

        
class ContrastiveFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnext = torchvision.models.convnext_base(pretrained=True).features.eval()
        # freeze the backbone network
        for param in self.convnext.parameters():
            param.requires_grad_(False)
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.head = nn.Sequential(
            nn.Linear(50176, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
        )

    def forward_once(self, x):
        x = self.convnext(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

    def forward(self, a, b):
        a = self.forward_once(a)
        b = self.forward_once(b)
        return a, b
    
    
# class CenterLoss(nn.Module):
#     def __init__(self, num_classes=7001, embed_dim=512):
#         super().__init__()
#         self.num_classes = num_classes        
#         self.embedding = nn.Embedding(num_classes, embed_dim)
#         self.embedding.weight.data.uniform_(-1.0/self.num_classes, 1.0/self.num_classes)
#         self.beta = 0.5
        
#     def forward(self, features, labels):
#         embeddings = self.embedding(labels)
#         # center_loss = (features - embeddings.detach()).square().mean() + self.beta * (features.detach() - embeddings).square().mean()
#         center_loss = (features - embeddings).square().mean()
#         return center_loss
        
    
class CenterLoss(nn.Module):
    """
    The original code is from the bootcamp (allowed according to course policy).
    I also used mask_select in place of loops -> performance improvement
    """

    def __init__(self, num_classes=7001, embed_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = embed_dim
        self.classes = torch.arange(self.num_classes).long().cuda()

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distances = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) \
                    + torch.pow(self.centers, 2).sum(
                dim=1, keepdim=True).expand(self.num_classes, batch_size).transpose(-1, -2)

        distances -= 2 * x @ self.centers.t()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.expand(batch_size, self.num_classes))

        return torch.mean(torch.masked_select(distances, mask))