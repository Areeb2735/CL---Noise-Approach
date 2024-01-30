#!/usr/bin/env python

import os
import time
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

import utils
import models.builer as builder
import model_dataloader
from sklearn.metrics import accuracy_score

# torch.manual_seed(0)

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Evaluate for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--istrain', action='store_false',
                        help='Training')
    parser.add_argument('--isautoencoder', type=int,
                        help='If training the Autoenocder')
    parser.add_argument('--noisevalue', type=float,
                        help='Noise Value to be added')
    parser.add_argument('--initclass', type=int,
                        help='Class to start with')
    parser.add_argument('--increment', type=int, default=10,
                        help='Increment added to the initial class')  
    # parser.add_argument('--val_list', type=str)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--folder', type=str)   
    parser.add_argument('--start_epoch', default=0, type=int)                                 
    parser.add_argument('--epochs', default=100, type=int) 

    args = parser.parse_args()

    args.parallel = 0

    return args

def main(args):

    # args = argparse.Namespace(arch='resnet18', batch_size=1, epochs=100, folder=None, isautoencoder=1, istrain=False, parallel=0, print_freq=10, resume='caltech256-resnet18.pth', start_epoch=0, workers=8)
    
    print('=> torch version : {}'.format(torch.__version__))
    ngpus_per_node = torch.cuda.device_count()
    print('=> ngpus : {}'.format(ngpus_per_node))

    utils.init_seeds(1, cuda_deterministic=False)
    
    print('=> modeling the network ...')

    if args.isautoencoder == 2:
        model = builder.BuildAutoClassfy(args)
        print('AUtoencoder and Classifier')

    elif args.isautoencoder == 1:
        model = builder.BuildAutoEncoder(args) 

        print('Autoencoder')

        # checkpoint = torch.load('caltech256-resnet18.pth')['state_dict']
        # model.load_state_dict(checkpoint)

        # model = builder.my_model_auto(model)
        
    else:
        model = builder.BuildClassfy(args) 

        print('Classifier')

        # checkpoint = torch.load('caltech256-resnet18.pth')['state_dict']
        # model.load_state_dict(checkpoint)

        # model = builder.my_model_class(model)  

    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    
    print('=> building the dataloader ...')
    val_loader = model_dataloader.train_loader(args)

    print('=> building the criterion ...')
    if args.isautoencoder == 2:
        criterion_1 = nn.L1Loss()
        criterion_2 = nn.CrossEntropyLoss()
        print("L1 and CrossEntropy Loss")
    
    elif args.isautoencoder == 1:
        criterion = nn.L1Loss()
        print('L1 Loss')
    else:
        criterion = nn.CrossEntropyLoss()
        print('CrossEntropy Loss')

    print('=> starting evaluating engine ...')
    if args.folder:
        best_loss = None
        best_epoch = 1
        losses = []
        for epoch in range(args.start_epoch, args.epochs):
            print()
            print("Epoch {}".format(epoch+1))
            resume_path = os.path.join(args.folder, "%03d.pth" % epoch)
            print('=> loading pth from {} ...'.format(resume_path))
            utils.load_dict(resume_path, model)
            loss = do_evaluate(val_loader, model, criterion, args)
            print("Evaluate loss : {:.4f}".format(loss))

            losses.append(loss)
            if best_loss:
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch + 1
            else:
                best_loss = loss
        print()
        print("Best loss : {:.4f} Appears in {}".format(best_loss, best_epoch))

        max_loss = max(losses)

        plt.figure(figsize=(7,7))

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim((0,args.epochs+1)) 
        plt.ylim([0, float('%.1g' % (1.22*max_loss))])

        plt.scatter(range(1, args.epochs+1), losses, s=9)

        plt.savefig("figs/evalall.jpg")

    else:
        print('=> loading pth from {} ...'.format(args.resume))
        # breakpoint()
        utils.load_dict(args.resume, model)
        if args.isautoencoder == 2:
            loss, accuarcy, mean, std = do_evaluate(val_loader, model, [criterion_1,criterion_2], args)
            print("Evaluate loss : {:.4f}".format(loss))
            print("Accuracy : {:.4f}".format(accuarcy))
            print("Mean : {:.4f}".format(mean))
            print("STD : {:.4f}".format(std))                
        elif args.isautoencoder == 0:
            loss, accuarcy = do_evaluate(val_loader, model, criterion, args)
            print("Evaluate loss : {:.4f}".format(loss))
            print("Accuracy : {:.4f}".format(accuarcy))
        else:
            loss  = do_evaluate(val_loader, model, criterion, args)
            print("Evaluate loss : {:.4f}".format(loss))

def do_evaluate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.2f')
    data_time = utils.AverageMeter('Data', ':2.2f')
    losses = utils.AverageMeter('Loss', ':.4f')
    
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses],
        prefix="Evaluate ")
    end = time.time()

    model.eval()

    if args.isautoencoder == 0:
        all_gts, all_preds = [], []
    elif args.isautoencoder == 2:
        all_gts, all_preds, noise_all = [], [], []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)


            target_ = target - 10
            
            if args.isautoencoder == 2:
                output, noise, logits = model(input)
                loss_1 = criterion[0](output, noise)
                # breakpoint()
                loss_2 = criterion[1](logits.unsqueeze(0), target)
                loss = loss_1 + loss_2
                all_gts.append(target.cpu().item())
                all_preds.append(torch.argmax(logits.cpu()).item())
                noise_all.append(output)
    
            elif args.isautoencoder == 1:
                output, noise = model(input, args)
                loss = criterion(output, noise)
                
            elif args.isautoencoder == 0:
                output = model(input)
                all_gts.append(target.cpu().item())
                all_preds.append(torch.argmax(output.cpu()).item())
                loss = criterion(output, target)

            # record loss
            losses.update(loss.item(), input.size(0))          
            batch_time.update(time.time() - end)        
            end = time.time()   

            if i % args.print_freq == 0:
                progress.display(i)

    if args.isautoencoder == 0:
        # correct_class = 0
        # for i in range(len(all_gts)):
        #     assert len(all_gts) == len(all_preds)
        #     if all_preds[i]==all_gts[i]:
        #         correct_class = correct_class + 1

        # return losses.avg, correct_class/len(all_preds)
        return losses.avg, accuracy_score(all_preds, all_gts)
    elif args.isautoencoder == 2:
        mean=[]
        std=[]
        for noise in noise_all:
            mean.append(output.mean().item())
            std.append(output.std().item())
        return losses.avg, accuracy_score(all_preds, all_gts), np.mean(mean), np.mean(std)


    return losses.avg

if __name__ == '__main__':

    args = get_args()
    # breakpoint()

    main(args)


