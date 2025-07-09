from __future__ import print_function
import torch
import numpy as np
import utils
import wandb
from voc_dataset import VOCDataset
from tqdm import tqdm

def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):

    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)
    
    # TODO: what is the correct loss function for multi-class classification?
    criterion = ...

    cnt = 0

    for epoch in range(args.epochs): # epoch loop
        batch_bar = tqdm(len(train_loader), desc='train', dynamic_ncols=True, leave=True)
        train_loss = 0.0
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            output = output * wgt# zeros out 'difficult' labels
            
            loss = ...
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_bar.update()
            cnt += 1

        # evaluate model every epoch and log
        model.eval()
        ap, map = utils.eval_dataset_map(model, args.device, test_loader)
        #TODO: log train training loss, learning rate, and map from evaluation
        model.train()
        
        if scheduler is not None:
            scheduler.step()

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)
            
        batch_bar.close()
        
    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
