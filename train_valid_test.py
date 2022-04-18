import io
import os
import random
import torch
import torch.nn.functional as Func
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np



def calc_acc(target, output):
    '''
    Calculate the accracy
    
    target : the true result of character index
    output : the output of the model
    '''
    output_argmax = output.detach().argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([np.array_equal(real,pred) for real, pred in zip(target, output_argmax)])
    return a.mean()



def train(model:torch.nn.Module, optimizer:torch.optim.Optimizer, epoch:int, dataloader:DataLoader, boardwriter:SummaryWriter):
    '''
    Training the model
    
    model: the model of neural network
    optimizer : the parameter optimizer
    epoch : index of epaches
    dataloader : the dataloader of PyTorch
    boardwriter : tensorboard data writer
    '''
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    passed_time = 0.0
    with tqdm(dataloader) as pybar:
        for batch_index, (data, target) in enumerate(pybar):
            data, target = data.cuda(), target.cuda()

            # set seed of pseudorandom generator
            random.seed(os.urandom(32))

            output = model(data)
            optimizer.zero_grad()
            fl_output = torch.flatten(output,start_dim=0,end_dim=1)
            fl_target = torch.flatten(target,start_dim=0,end_dim=1)
            loss = Func.cross_entropy(fl_output, fl_target)

            # do backpropagation and optimise parameters
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            current_acc = calc_acc(target, output)
            
            running_loss += current_loss
            running_acc += current_acc 

            # write the results per 20 training to tensorboard
            if batch_index % 20 == 19:
                x_value = epoch * len(dataloader) + batch_index
                boardwriter.add_scalar('training loss', running_loss / 20, x_value)
                boardwriter.add_scalar('training acc', running_acc / 20,x_value)
                time = pybar.format_dict["elapsed"]
                boardwriter.add_scalar('training speed', 20 / (time-passed_time),x_value)
                passed_time = time

                running_loss = 0.0
                running_acc = 0.0

            
            pybar.set_description(f'Train: {epoch} Loss: {current_loss:.4f} Acc: {current_acc:.4f} ')


def valid(model:torch.nn.Module, epoch:int, dataloader:DataLoader, boardwriter:SummaryWriter):
    '''
    Validating the model

    model: the model of neural network
    epoch : index of epaches
    dataloader : the dataloader of PyTorch
    boardwriter : tensorboard data writer
    '''
    model.eval()
    with tqdm(dataloader) as pybar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target) in enumerate(pybar):
            data, target = data.cuda(), target.cuda()

            # set seed of pseudorandom generator
            random.seed(os.urandom(32))

            output = model(data)
            fl_output = torch.flatten(output,start_dim=0,end_dim=1)
            fl_target = torch.flatten(target,start_dim=0,end_dim=1)
            loss = Func.cross_entropy(fl_output, fl_target)

            # do backpropagation and optimise parameters
            loss = loss.item()
            acc = calc_acc(target, output)

            loss_sum += loss
            acc_sum += acc

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)

            pybar.set_description(f'Valid : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')

            # write validation result to tensorboard
        boardwriter.add_scalar('valid loss', loss_mean, epoch)
        boardwriter.add_scalar('valid acc', acc_mean, epoch)
        
        
        
def test(model:torch.nn.Module, total_test_num:int, dataloader:DataLoader, boardwriter:SummaryWriter, file:io.TextIOWrapper):
    '''
    Validating the model

    model: the model of neural network
    total_test_num : the times of tests
    dataloader : the dataloader of PyTorch
    boardwriter : tensorboard data writer
    file : the file to write recognision results
    '''
    model.eval()
    total_lost_mean = 0.0
    total_acc_mean = 0.0
    for test_index in range(total_test_num):
        with tqdm(dataloader) as pybar, torch.no_grad():
            loss_sum = 0
            acc_sum = 0
            for batch_index, (data, target) in enumerate(pybar):
                data, target = data.cuda(), target.cuda()

                # set seed of pseudorandom generator
                random.seed(os.urandom(32))

                output = model(data)
                            
                fl_output = torch.flatten(output,start_dim=0,end_dim=1)
                fl_target = torch.flatten(target,start_dim=0,end_dim=1)
                loss = Func.cross_entropy(fl_output, fl_target)

                loss = loss.item()
                acc = calc_acc(target, output)

                # do backpropagation and optimise parameters
                loss_sum += loss
                acc_sum += acc

                loss_mean = loss_sum / (batch_index + 1)
                acc_mean = acc_sum / (batch_index + 1)

                pybar.set_description(f'Test : {test_index} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')

            # write test result to tensorboard
            boardwriter.add_scalar('test loss', loss_sum / len(dataloader), test_index)
            total_lost_mean += loss_mean
            boardwriter.add_scalar('test acc', acc_sum / len(dataloader), test_index)
            total_acc_mean += acc_mean
            
    total_lost_mean = total_lost_mean / float(total_test_num)
    total_acc_mean = total_acc_mean / float(total_test_num)
    print("total_lost : ", total_lost_mean,"\ntotal_acc : ", total_acc_mean, file=file)
    
    
        


