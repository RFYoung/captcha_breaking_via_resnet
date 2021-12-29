import io
import os
import random
import torch
import torch.nn.functional as Func
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np



def calc_acc(target, output):
    output_argmax = output.detach().argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([np.array_equal(real,pred) for real, pred in zip(target, output_argmax)])
    # print(a.all())
    return a.mean()


def train(model:torch.nn.Module, optimizer:torch.optim.Optimizer, epoch:int, dataloader:DataLoader):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    with tqdm(dataloader) as pybar:
        for _, (data, target) in enumerate(pybar):
            data, target = data.cuda(), target.cuda()

            random.seed(os.urandom(32))

            output = model(data)
            optimizer.zero_grad()
            fl_output = torch.flatten(output,start_dim=0,end_dim=1)
            fl_target = torch.flatten(target,start_dim=0,end_dim=1)
            loss = Func.cross_entropy(fl_output, fl_target)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            current_acc = calc_acc(target, output)
            
            running_loss += current_loss
            running_acc += current_acc 

            
            pybar.set_description(f'Train: {epoch} Loss: {current_loss:.4f} Acc: {current_acc:.4f} ')


def valid(model:torch.nn.Module, epoch:int, dataloader:DataLoader):
    model.eval()
    with tqdm(dataloader) as pybar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target) in enumerate(pybar):
            data, target = data.cuda(), target.cuda()

            random.seed(os.urandom(32))

            output = model(data)
            fl_output = torch.flatten(output,start_dim=0,end_dim=1)
            fl_target = torch.flatten(target,start_dim=0,end_dim=1)
            loss = Func.cross_entropy(fl_output, fl_target)

            loss = loss.item()
            acc = calc_acc(target, output)

            loss_sum += loss
            acc_sum += acc

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)

            pybar.set_description(f'Valid : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')

