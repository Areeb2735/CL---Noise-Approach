#!/usr/bin/env python

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import utils
import models.builer as builder
import torch.nn.functional as F
from my_dataloader import NoisyDataset, NoisyDataset_test
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)

from tqdm import tqdm
from torchinfo import summary

import wandb

# set seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--name', type=str, 
                    help='wandb name')
    parser.add_argument('--replay', type=int,
                        help='Is replay')
    parser.add_argument('--mean', type=float,
                        help='Noise Mean Value to be added')
    parser.add_argument('--std', type=float, default=0.1,
                        help='Noise Std Value to be added')
    parser.add_argument('--initclass', type=int,
                        help='Class to start with')
    parser.add_argument('--increment', type=int, default=10,
                        help='Increment added to the initial class')  
    parser.add_argument('--prev_task_weights', type=str, default=None,
                        help='Weights of the previous task')   
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--pth-save-fold', default='results/tmp', type=str,
                        help='The folder to save pths')
    parser.add_argument('--pth-save-epoch', default=1, type=int,
                        help='The epoch to save pth')
    parser.add_argument('--parallel', type=int, default=1, 
                        help='1 for parallel, 0 for non-parallel')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')                                            

    args = parser.parse_args()

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    ngpus_per_node = torch.cuda.device_count()
    print('=> ngpus : {}'.format(ngpus_per_node))

    wandb.init(project="CL", entity = 'moareeb', name=args.name, config=args)

    if args.parallel == 1: 
        # single machine multi card       
        args.gpus = ngpus_per_node
        args.nodes = 1
        args.nr = 0
        args.world_size = args.gpus * args.nodes

        args.workers = int(args.workers / args.world_size)
        args.batch_size = int(args.batch_size / args.world_size)
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        args.world_size = 1
        main_worker(ngpus_per_node, args)
    
def main_worker(gpu, args):
    
    utils.init_seeds(1 + gpu, cuda_deterministic=False)
    if args.parallel == 1:
        args.gpu = gpu
        args.rank = args.nr * args.gpus + args.gpu

        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  
           
    else:
        # two dummy variable, not real
        args.rank = 0
        args.gpus = 1 
    if args.rank == 0:
        print('=> modeling the network {} ...'.format(args.arch))

    task_id = ((args.initclass)//10)+1
    mean_diff = 1

    model = builder.BuildAutoEncoder(args)
    print('Autoencoder')

    # if task_id == 0:       # warm up
    #     for param in model.module.encoder.parameters():
    #         param.requires_grad = False

    if task_id > 1:
        checkpoint = torch.load(args.prev_task_weights)['state_dict']
        model.load_state_dict(checkpoint)

        # for param in model.module.encoder.parameters():
        #     param.requires_grad = False

    summary(model.cuda(), [(args.batch_size,3,224,224),(args.batch_size,512)], col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
    # summary(model.cuda(), (args.batch_size,3,224,224), col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 

    if args.rank == 0:       
        total_params = sum(p.numel() for p in model.parameters())
        print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    
    if args.rank == 0:
        print('=> building the oprimizer ...')
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,)
    optimizer = torch.optim.SGD(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)    
    if args.rank == 0:
        print('=> building the dataloader ...')

    trsf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
    
    dataset = NoisyDataset(root = './data', mean=args.mean, std = args.std, 
                           classes_subset=list(np.arange(args.initclass, args.initclass + args.increment)), 
                           max_samples = None, transform=trsf, args=args)
    
    if args.replay:
        dataset_replay = []
        for num in range(len(range(task_id-1))):
            datasets = NoisyDataset(root = './data', mean=args.mean - ((num * mean_diff) + 1), std = args.std,
                                classes_subset=list(np.arange(((task_id-2-num)*10), (((task_id-2-num)*10) + args.increment))), 
                                max_samples = 2000//(task_id-1), transform=trsf, args=args)
            dataset_replay.append(datasets)
            print(num, args.mean - ((num * mean_diff) + 1), (task_id-2-num)*10, 2000//(task_id-1))
        
        # breakpoint()
        dataset_replay.append(dataset)

        con_datatset = ConcatDataset(dataset_replay)
        train_loader = DataLoader(con_datatset,
                                shuffle=True,
                                batch_size=args.batch_size, 
                                num_workers=args.workers, 
                                pin_memory=True,
                                drop_last=True)

    else:
        train_loader = DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=True
                    )
        
    if args.rank == 0:
        print('=> building the criterion ...')
    
    criterion = nn.L1Loss(reduction='none')
    print('L1 Loss')

    global iters
    iters = 0

    model.train()
    if args.rank == 0:
        print('=> starting training engine ...')
    for epoch in range(args.start_epoch, args.epochs):
        
        global current_lr
        current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args)
        
        # if epoch == 0:        #### Just to check whether the weights of the previous task is loaded properly
        #     do_validate(model, epoch, args, task_id, mean_diff)

        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch+1) % args.print_freq == 0 and args.rank == 0:
            do_validate(model, epoch, args, task_id, mean_diff)

        # save pth
        if (epoch+1) % args.pth_save_epoch == 0 and args.rank == 0:
            state_dict = model.state_dict()

            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'optimizer' : optimizer.state_dict(),
                },
                os.path.join(args.pth_save_fold, '{}.pth'.format(str(epoch+1).zfill(3)))
            )
            
            print(' : save pth for epoch {}'.format(epoch + 1))

def do_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.2f')
    data_time = utils.AverageMeter('Data', ':2.2f')
    losses = utils.AverageMeter('Loss', ':.4f')
    learning_rate = utils.AverageMeter('LR', ':.4f')
    
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate],
        prefix="Epoch: [{}]".format(epoch+1))
    end = time.time()

    # model.train()
    # update lr
    learning_rate.update(current_lr)
    total_steps = len(train_loader)
    for i, (input, target, added_noise, gt_noise) in enumerate(train_loader):
        data_time.update(time.time() - end)
        global iters
        iters += 1
        
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        added_noise = added_noise.cuda(non_blocking=True)
        gt_noise = gt_noise.cuda(non_blocking=True)

        output = model(input, added_noise)
        # output = model(input)

        # weights_value = utils.task_weight(((args.initclass)//10))
        # weights_value = { 1: 1, 0: 100}

        # adjusted_weights_list = []
        # for value in (target // args.increment):
        #     adjusted_weights_list.append(weights_value[int(value.item())])

        loss1 = criterion(output - (target // args.increment).view(args.batch_size, 1),
                            gt_noise - (target // args.increment).view(args.batch_size, 1))
        # loss1_up = torch.mul(torch.mean(loss1, axis=(1)),  torch.tensor(adjusted_weights_list).cuda())
        loss1_mean = torch.mean(loss1)

        loss2 = criterion(torch.floor(torch.mean(output, dim=(1))), torch.floor(torch.mean(gt_noise, dim=(1))))
        # loss2 = criterion(torch.mean(output, dim=(1)), torch.floor(torch.mean(gt_noise, dim=(1))))
        # loss2_up = torch.mul(loss2,  torch.tensor(adjusted_weights_list).cuda())
        
        loss2_mean = torch.mean(loss2)

        # loss1 = criterion(output - (target // args.increment).view(args.batch_size, 1,1,1),
        #                     gt_noise - (target // args.increment).view(args.batch_size, 1,1,1))
        # loss1_mean = torch.mean(loss1)

        # loss2 = criterion(torch.floor(torch.mean(output, dim=(1,2,3))), torch.floor(torch.mean(gt_noise, dim=(1,2,3))))
        # loss2_mean = torch.mean(loss2)

        print("------------")
        print("GT Noise: ", gt_noise[0].mean())
        print("Added Noise: ", added_noise[0].mean())
        print("Model Mean: ", output[0].mean())
        print("Model STD: ", output[0].std())
        print("Loss 1: ", loss1[0].mean())
        print("Loss 2: ", loss2[0])
        print('------------')

        loss = loss1_mean + loss2_mean

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        torch.cuda.synchronize()
        losses.update(loss.item(), input.size(0))          

        if args.rank == 0:
            batch_time.update(time.time() - end)        
            end = time.time()   

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)
            wandb.log({
                "loss/loss": loss,
                "loss/loss1": loss1_mean,
                "loss/loss2": loss2_mean,
                'model/model_mean': output.mean(),
                'model/model_std': output.std(),
                'data/Added noise mean': added_noise.mean(),
                'data/GT noise mean': gt_noise.mean(),
                'data/Added noise std': added_noise.std(),
                'data/GT noise std': gt_noise.std(),
                "experiment/LR": current_lr,
                "experiment/Epoch": epoch+1
            }, step=epoch*total_steps +i)

def do_validate(model, epoch, args, task_id, mean_diff):

    current_mean, current_std = validate(model, args.mean)
    print(f"Validation - current mean = ({current_mean},{args.mean})")

    accuracy, precision_recall_fscore_support = validate_classification(model, args.mean, task_id)
    wandb.log({
        "Validation/F1 Score/Accuracy": accuracy,
        "Validation/Precision/Precision class 0": precision_recall_fscore_support[0][0],
        "Validation/Precision/Precision class 1": precision_recall_fscore_support[0][1],
        "Validation/Recall/Recall class 0": precision_recall_fscore_support[1][0],
        "Validation/Recall/Recall class 1": precision_recall_fscore_support[1][1],
        "Validation/F1 Score/F1 Score class 0": precision_recall_fscore_support[2][0],
        "Validation/F1 Score/F1 Score class 1": precision_recall_fscore_support[2][1],
    })

    if task_id > 1:                                                           
        for num in range(len(range(task_id-1))):
            prev_mean, prev_stf = validate(model, args.mean-((num * mean_diff) + 1))                      
            print(f"Validation - prev_mean = ({prev_mean},{args.mean-((num * mean_diff) + 1)})")          
            wandb.log({
            f"Validation/Noise/Mean after adding {args.mean - ((num * mean_diff) + 1)}": prev_mean,                         
            })

    wandb.log({
        f"Validation/Noise/Mean after adding {args.mean}": current_mean,
    })

def validate(model, test_noise_mean):

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    val_dataset = NoisyDataset_test(root = './data', mean=test_noise_mean, std = args.std, 
                           classes_subset=list(np.arange(args.initclass, args.initclass + args.increment)), 
                           max_samples = None, train = False, transform=test_transform)
    val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            # drop_last=True
            )  

    model.eval()
    pred_noises = []
    with torch.no_grad():
        for _, (input, target, noise) in tqdm(enumerate(val_loader)):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            noise = noise.cuda(non_blocking=True)
            
            output = model(input, noise)
            # output = model(input)
            pred_noises.append(torch.mean(output, dim=(1)).detach().cpu())        
    
    concat_pred_noises = torch.cat(pred_noises)
    # print(concat_pred_noises.shape)
    return torch.mean(concat_pred_noises), torch.std(concat_pred_noises)

def validate_classification(model, test_noise_mean, task_id):

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    val_dataset = NoisyDataset_test(root = './data', mean=test_noise_mean, std = args.std, 
                           classes_subset=list(np.arange(0, task_id*10)), 
                           max_samples = None, train = False, transform=test_transform)
    val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.workers,
            pin_memory=True,
            # drop_last=True
            )  

    noises = []
    if task_id > 1:
        for num in range(len(range(task_id-1))):
            noises.append(torch.normal(test_noise_mean-(num + 1), 0.1, size=(512,1)).cuda(non_blocking=True))
    
    noises.append(torch.normal(test_noise_mean, 0.1, size=(512,1)).cuda(non_blocking=True))

    criterion = nn.L1Loss(reduction='none')
    model.eval()

    pred = []
    gt = []

    with torch.no_grad():
        for _, (input, target, _) in tqdm(enumerate(val_loader)):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            dataset_number = (target // args.increment)

            losses = []
            for idx, noise in enumerate(noises):
                output = model(input, noise.transpose(0, 1))
                # output = model(input)

                loss = criterion(torch.mean(output, dim=(1)), torch.mean(noise.transpose(0, 1), dim=(1)))
                losses.append(loss)
            
            pred.append(losses.index(min(losses)))

            gt.append(dataset_number.item())
            
    cm = confusion_matrix(gt, pred)
    report = classification_report(gt, pred)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return accuracy_score(gt, pred), precision_recall_fscore_support(gt, pred)

if __name__ == '__main__':

    args = get_args()

    main(args)


