import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import random
import wandb

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # Use image size of 64x64 in Q1. We will use a default size of
    # 224x224 for the rest of the questions.
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 22 in 5 epochs
    ##################################################################
    args = ARGS(
        epochs=10,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=1e-3,# TODO,
        batch_size=32,#TODO,
        step_size=5,#TODO,
        gamma=0.7#TODO
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    print(args)
    # initialize wandb - fill in your api key here
    wandb.login(key="1d70da927fe2f85af2f2aa8e4861950d6f63844e")
    run = wandb.init(
        name = "initial", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        project = "recitation1_classification", ### Project should be created in your wandb account
    )
    # initializes the model
    model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3)
    # 1. initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
