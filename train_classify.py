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
from my_dataloader import ClassifyDataset, Classify_bloodMNIST, Classify_organAMNIST, Classify_pathMNIST, Classify_tissueMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import PIL
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from torchinfo import summary

# set seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)

import wandb

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--name', type=str, 
                    help='wandb name')
    parser.add_argument('--dataset', type=str, 
                    help='name of the dataset')
    parser.add_argument('--medmnist_task_id', type=int,
                        help='ID number of the task (Only for MEDMNIST Dataset)', default=0)
    parser.add_argument('--initclass', type=int,
                        help='Class to start with (Only for CIFAR Dataset)')
    parser.add_argument('--increment', type=int, default=10,
                        help='Increment added to the initial class (Only for CIFAR Dataset)')    
    # parser.add_argument('--train_list', type=str)
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

    if args.dataset == 'cifar100':
        task_id = ((args.initclass)//10)+1
    elif args.dataset == 'medmnist':
        task_id = args.medmnist_task_id
    else:
        NameError('Dataset not found')
    
    model = builder.BuildClassfy(args, utils.medmnist_dataset_to_target(args.dataset, args.medmnist_task_id))
    # checkpoint = torch.load('results/unfrozen_encoder_0-resnet18/100.pth')['state_dict']
    # encoder_state_dict = {key: value for key, value in checkpoint.items() if key.startswith('module.encoder')}
    # new_state_dict = {key.replace('module.encoder.', ''): value for key, value in encoder_state_dict.items()}

    # model.module.encoder.load_state_dict(new_state_dict)
    # for param in model.module.encoder.parameters():
    #     param.requires_grad = False
    
    print('Classifier')

    summary(model, input_size=(args.batch_size,3,224,224),
                         col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
    

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

    if args.dataset == 'cifar100':
        trsf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        
        dataset = ClassifyDataset(root = './data', 
                            classes_subset=list(np.arange(args.initclass, args.initclass + args.increment)), 
                            transform=trsf)
            
        train_loader = DataLoader(
                        dataset,
                        shuffle=True,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        pin_memory=True,
                        drop_last=False 
                        )
    
    elif args.dataset == 'medmnist':

        trsf = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
        
        if args.medmnist_task_id == 1:
            dataset = Classify_bloodMNIST(split='train', transform = trsf, download = True, as_rgb = True)
            train_loader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False
                ) 
        elif args.medmnist_task_id == 2:
            dataset = Classify_organAMNIST(split='train', transform = trsf, download = True, as_rgb = True)
            train_loader = DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=  False
                    )   
        elif args.medmnist_task_id == 3:
            dataset = Classify_pathMNIST(split='train', transform = trsf, download = True, as_rgb = True)
            train_loader = DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=False
                    ) 
        elif args.medmnist_task_id == 4:
            dataset = Classify_tissueMNIST(split='train', transform = trsf, download = True, as_rgb = True)
            train_loader = DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=False
                    ) 

    if args.rank == 0:
        print('=> building the criterion ...')
    
    criterion = nn.CrossEntropyLoss()
    print('CrossEntropy Loss')

    global iters
    iters = 0

    model.train()
    if args.rank == 0:
        print('=> starting training engine ...')
    for epoch in range(args.start_epoch, args.epochs):
        
        global current_lr
        current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args)
        
        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args)

        val_accurcay = validate(model, args)
        wandb.log({
                "Validation Accuracy": val_accurcay,
            })
        print(val_accurcay)

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

    model.train()
    # update lr
    learning_rate.update(current_lr)
    total_steps = len(train_loader)
    for i, (input, target, dataset_number) in enumerate(train_loader):
        data_time.update(time.time() - end)
        global iters
        iters += 1
         
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        dataset_number = dataset_number.cuda(non_blocking=True)

        output = model(input)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        losses.update(loss.item(), input.size(0))          

        if args.rank == 0:
            batch_time.update(time.time() - end)        
            end = time.time()   

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)
            wandb.log({
                "Loss": loss,
                "LR": current_lr,
                "Epoch": epoch+1
            }, step=epoch*total_steps +i)


def validate(model, args):

    if args.dataset == 'cifar100':
        test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        val_dataset = ClassifyDataset(root = './data',
                            classes_subset=list(np.arange(args.initclass, args.initclass + args.increment)), 
                            train = False, transform=test_transform)
    
    elif args.dataset == 'medmnist':
        test_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])

        if args.medmnist_task_id == 1:
            val_dataset = Classify_bloodMNIST(split='test', transform = test_transform, as_rgb = True)
        elif args.medmnist_task_id == 2:
            val_dataset = Classify_organAMNIST(split='test', transform = test_transform, as_rgb = True)
        elif args.medmnist_task_id == 3:
            val_dataset = Classify_pathMNIST(split='test', transform = test_transform, as_rgb = True )
        elif args.medmnist_task_id == 4:
            val_dataset = Classify_tissueMNIST(split='test', transform = test_transform, as_rgb = True)
    
    val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.workers,
            pin_memory=True,
            # drop_last=True
            )  

    model.eval()
    all_gts, all_preds = [], []
    with torch.no_grad():
        for _, (input, target, dataset_number) in tqdm(enumerate(val_loader)):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            dataset_number = dataset_number.cuda(non_blocking=True)
            
            output = model(input)

            all_gts.append(target.cpu().item())
            all_preds.append(torch.argmax(output.cpu()).item())

    return accuracy_score(all_preds, all_gts)

if __name__ == '__main__':

    args = get_args()

    main(args)


