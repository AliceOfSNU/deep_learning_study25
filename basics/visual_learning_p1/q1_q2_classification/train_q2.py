import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
import wandb

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
            
class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        #self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
        # self.linear = nn.Linear(1000, num_classes)
        
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0 = BasicBlock(32, 64)
        self.layer1 = BasicBlock(64, 128)
        self.pool = torch.nn.AvgPool2d(2, 2)
        
        self.fc1 = torch.nn.Linear(2048, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        # output = self.resnet(x)
        # output = self.linear(output)
        N = x.size(0)
        output = self.conv(x)
        output = self.layer0(output)
        output = self.layer1(output)
        output = self.pool(output)
        
        output = output.view(N, -1)
        output = self.fc1(output)
        output = self.fc2(output)
        
        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 50 in 50 epochs
    ##################################################################
    args = ARGS(
        epochs=40,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=1e-3,# TODO,
        batch_size=64,#TODO,
        step_size=4,#TODO,
        gamma=0.7#TODO
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)
    wandb.login(key="1d70da927fe2f85af2f2aa8e4861950d6f63844e")
    run = wandb.init(
        name = "initial", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        project = "recitation1_classification", ### Project should be created in your wandb account
    )
    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
