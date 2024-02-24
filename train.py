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
import random

random.seed(0)

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
    parser.add_argument('--replay', type=int, default=0,
                        help='Is replay')
    parser.add_argument('--mean', type=float,
                        help='Noise Mean Value to be added')
    parser.add_argument('--std', type=float, default=0.1,
                        help='Noise Std Value to be added')
    parser.add_argument('--initclass', type=int, default=0,
                        help='Class to start with')
    parser.add_argument('--increment', type=int, default=10,
                        help='Increment added to the initial class')  
    # parser.add_argument('--prev_task_weights', type=str, default=None,
    #                     help='Weights of the previous task')   
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=31.5, type=int, metavar='N',
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

    # wandb.init(project="CL", entity = 'moareeb', name=args.name, config=args)

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


    mean_diff = 1
    args.replay = False

    for k in range(10):
        print(f"Start of Task number: {((k * 10)//10)+1}")
        print(f"Start of Class number: {k*10}")
        print(f"Mean: {args.mean}")
        print(f"Std: {args.std}")
        print(f"Replay: {args.replay}")

        args.initclass = k*10
        task_id = ((args.initclass)//10)+1
        
        model = builder.BuildAutoEncoder(args)
        print('Autoencoder')

        if task_id > 1:
            args.replay = True
            checkpoint = torch.load(f"{args.pth_save_fold}{str(task_id-1).zfill(2)}/200.pth")['state_dict']
            model.load_state_dict(checkpoint)

        # summary(model.cuda(), [(args.batch_size,3,224,224),(args.batch_size,512)], col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
        summary(model.cuda(), (args.batch_size,3,224,224), col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 

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
                datasets = NoisyDataset(root = './data', mean=args.mean - ((num * mean_diff) + mean_diff), std = args.std,
                                    classes_subset=list(np.arange(((task_id-2-num)*10), (((task_id-2-num)*10) + args.increment))), 
                                    max_samples = 2000//(task_id-1), transform=trsf, args=args)
                dataset_replay.append(datasets)
                print(num, args.mean - ((num * mean_diff) + mean_diff), (task_id-2-num)*10, 2000//(task_id-1))
            
            # breakpoint()
            dataset_replay.append(dataset)

            con_datatset = ConcatDataset(dataset_replay)

            # custom_sampler = CustomSampler(con_datatset)
            # train_loader = torch.utils.data.DataLoader(con_datatset, batch_size=args.batch_size, sampler=custom_sampler)
            # print(next(iter(train_loader)))

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
            do_train(train_loader, model, criterion,  optimizer, epoch, args, task_id)

            # if (epoch+1) % args.print_freq == 0 and args.rank == 0:
            #     do_validate(model, epoch, args, task_id, mean_diff)

            # save pth
            save_path = os.path.join(args.pth_save_fold, str(task_id).zfill(2), '{}.pth'.format(str(epoch+1).zfill(3)))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if (epoch+1) % args.pth_save_epoch == 0 and args.rank == 0:
                state_dict = model.state_dict()
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': state_dict,
                        'optimizer' : optimizer.state_dict(),
                    },  save_path  
                    # os.path.join(args.pth_save_fold, str(task_id).zfill(2), '{}.pth'.format(str(epoch+1).zfill(3)))
                )
                
                print(' : save pth for epoch {}'.format(epoch + 1))
        
        args.mean += 1
        

def do_train(train_loader, model, criterion,  optimizer, epoch, args, task_id):
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

    for i, (image, target, added_noise, gt_noise) in enumerate(train_loader):
        
        breakpoint()
        optimizer.zero_grad()
        data_time.update(time.time() - end)
        global iters
        iters += 1
        
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        added_noise = added_noise.cuda(non_blocking=True)
        gt_noise = gt_noise.cuda(non_blocking=True)

        # added_noise = random.choice(noise_list[:task_id - 1])
        # gt_noise = noise_list[task_id-1]

        # added_noise_pad = utils.pad_noise(added_noise)
        # gt_noise_pad = utils.pad_noise(gt_noise)

        # output = model(image, added_noise)
        output = model(image)

        # print("Images from Different Datasets", np.unique((target // args.increment).cpu(), return_counts=True)[1])

        # if len(np.unique((target // args.increment).cpu(), return_counts=True)[1]) == 1:
            # breakpoint()

        # if len(np.unique((target // args.increment).cpu(), return_counts=True)[1]) == 1:
        #     weight_2 = 1
        #     weight_1 = 0
        #     weight_0 = 0
        # elif len(np.unique((target // args.increment).cpu(), return_counts=True)[1]) == 2:
        #     weight_2 = args.batch_size/(np.unique((target // args.increment).cpu(), return_counts=True)[1][1])
        #     weight_1 = args.batch_size/(np.unique((target // args.increment).cpu(), return_counts=True)[1][0])
        #     weight_0 = 1
        # else:
        #     weight_1 = args.batch_size/(np.unique((target // args.increment).cpu(), return_counts=True)[1][1])
        #     weight_0 = args.batch_size/(np.unique((target // args.increment).cpu(), return_counts=True)[1][0])
        
        # keys_array, values_array =np.unique((target // args.increment).cpu(), return_counts=True)
        # original_dict = dict(zip(keys_array, values_array))
        # weights_value = {key: args.batch_size / value for key, value in original_dict.items()}

        # weights_value = utils.task_weight(((args.initclass)//10))
        # weights_value = { 2: weight_2, 1: weight_1, 0: weight_0}

        # weights_value = utils.weight_dictionary(task_id, exponential_factor=1.3)

        # print("Weight Value of datasets: ", weights_value)

        # adjusted_weights_list = []
        # for value in (target // args.increment):
        #     adjusted_weights_list.append(weights_value[int(value.item())])
        
        # loss1 = criterion(output - (target // args.increment).view(args.batch_size, 1),
        #                     gt_noise - (target // args.increment).view(args.batch_size, 1))
        
        # loss1 = criterion(output, gt_noise) 
        
        # loss1_up = torch.mul(torch.mean(loss1, axis=(1)),  torch.tensor(adjusted_weights_list).cuda())
        # loss1_mean = torch.mean(loss1)
        # loss1 = criterion(output, gt_noise)

        loss1 = criterion(output, gt_noise)
        # loss1_up = torch.mul(torch.mean(loss1, axis=(1)),  torch.tensor(adjusted_weights_list).cuda())
        
        # This is when we calculate loss between the means of the output and the gt noise
        # loss1 = criterion(torch.mean(output, dim=(1)), torch.mean(gt_noise, dim=(1)))
        # loss1_up = torch.mul(loss1,  torch.tensor(adjusted_weights_list).cuda())

        loss1_mean = torch.mean(loss1)


        # loss2 = criterion(torch.floor(torch.mean(output, dim=(1))), torch.floor(torch.mean(gt_noise, dim=(1))))
        # loss2_mean = torch.mean(loss2)
        # loss2 = criterion(torch.mean(output, dim=(1)), torch.mean(gt_noise, dim=(1)))

        # loss2 = torch.abs(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0', requires_grad=True))

        # loss2 = torch.tensor([0])
        # loss2 = criterion(torch.floor((torch.mean(output, dim=(1))/10)), torch.floor((torch.mean(gt_noise, dim=(1)))/10))

        # loss2_up = torch.mul(loss2,  torch.tensor(adjusted_weights_list).cuda())
        
        # loss2_mean = torch.mean(loss2)
        
        # print(np.unique((target // args.increment).cpu(), return_counts=True)[1][0])
        # print(np.unique((target // args.increment).cpu(), return_counts=True)[1][1])
        # print(np.unique((target // args.increment).cpu(), return_counts=True)[1][0] / np.unique((target // args.increment).cpu(), return_counts=True)[1][1])

        # loss1 = criterion(output - (target // args.increment).view(args.batch_size, 1,1,1),
        #                     gt_noise - (target // args.increment).view(args.batch_size, 1,1,1))
        # loss1_mean = torch.mean(loss1)

        # loss2 = criterion(torch.floor(torch.mean(output, dim=(1,2,3))), torch.floor(torch.mean(gt_noise, dim=(1,2,3))))
        # loss2_mean = torch.mean(loss2)

        # print("------------")
        # print("GT Noise: ", gt_noise[0].mean())
        # print("Added Noise: ", added_noise[0].mean())
        # print("Model Mean: ", output[0].mean())
        # print("Model STD: ", output[0].std())
        # print("Loss 1: ", loss1[0].mean())
        # print("Loss 2: ", loss2[0])
        # print('------------')

        # loss = loss1_mean + loss2_mean
        # loss = loss1_mean

        loss1_mean.backward()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        torch.cuda.synchronize()
        losses.update(loss1_mean.item(), image.size(0))          

        if args.rank == 0:
            batch_time.update(time.time() - end)        
            end = time.time()   

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)
            # wandb.log({
            #     # f"{task_id}/loss/loss": loss,
            #     f"{task_id}/loss/loss1": loss1_mean,
            #     # f"{task_id}/loss/loss2": loss2_mean,
            #     f"{task_id}/model/model_mean": output.mean(),
            #     f"{task_id}/model/model_std": output.std(),
            #     f"{task_id}/data/Added noise mean": added_noise.mean(),
            #     f"{task_id}/data/GT noise mean": gt_noise.mean(),
            #     f"{task_id}/data/Added noise std": added_noise.std(),
            #     f"{task_id}/data/GT noise std": gt_noise.std(),
            #     f"{task_id}/experiment/LR": current_lr,
            #     f"{task_id}/experiment/Epoch": epoch+1
            # }, 
            # # step=(epoch*total_steps +i) + g_step
            # )
        
        # return (epoch*total_steps +i) + g_step

def do_validate(model, epoch, args, task_id, mean_diff):

    current_mean, current_std = validate(model, args.mean)
    print(f"Validation - current mean = ({current_mean},{args.mean})")

    if task_id > 1:
        accuracy, precision_recall_fscore_support = validate_classification(model, args.mean, task_id)
        wandb.log({f"{task_id}/Validation/F1 Score/Accuracy": accuracy})
        for num in reversed(range(len(range(task_id)))):
            wandb.log({
                f"{task_id}/Validation/Precision/Precision class {num}": precision_recall_fscore_support[0][task_id-1-num],
                f"{task_id}/Validation/Recall/Recall class {num}": precision_recall_fscore_support[1][num],
                f"{task_id}/Validation/F1 Score/F1 Score class {num}": precision_recall_fscore_support[2][num],
            })

    if task_id > 1:                                                           
        for num in range(len(range(task_id-1))):
            prev_mean, prev_stf = validate(model, args.mean-((num * mean_diff) + 1))                      
            print(f"{task_id}/Validation - prev_mean = ({prev_mean},{args.mean-((num * mean_diff) + 1)})")          
            wandb.log({
            f"{task_id}/Validation/Noise/Mean after adding {args.mean - ((num * mean_diff) + 1)}": prev_mean,                         
            })

    wandb.log({
        f"{task_id}/Validation/Noise/Mean after adding {args.mean}": current_mean,
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
        for _, (image, target, noise) in tqdm(enumerate(val_loader)):

            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            noise = noise.cuda(non_blocking=True)
            # noise = noise_list[task_id-1]

            # noise_pad = utils.pad_noise(noise)
            
            # output = model(image, noise)
            output = model(image)
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
        for num in reversed(range(len(range(task_id-1)))):
            noises.append(torch.normal(test_noise_mean-(num + 1), 0.1, size=(512,1)).cuda(non_blocking=True))
    
    noises.append(torch.normal(test_noise_mean, 0.1, size=(512,1)).cuda(non_blocking=True))

    criterion = nn.L1Loss(reduction='none')
    model.eval()

    pred = []
    gt = []

    with torch.no_grad():
        for _, (image, target, _) in tqdm(enumerate(val_loader)):

            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            dataset_number = (target // args.increment)

            losses = []
            for idx, noise in enumerate(noises):

                # noise = utils.pad_noise(noise)

                # noise_pad = utils.pad_noise(noise.transpose(0, 1))

                # output = model(image, noise.transpose(0, 1))
                # output = model(image, noise_pad)
                output = model(image)

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


