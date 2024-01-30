import numpy as np
import argparse
import models.builer as builder 
from tqdm import tqdm
import os
import utils
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from torchvision import datasets
from typing import Callable, Any, Tuple, Union
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from torchinfo import summary
import seaborn as sns
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--initclass', type=int,
                        help='Class to start with')
    parser.add_argument('--increment', type=int,
                        help='Increment added to the initial class')
    parser.add_argument('--latent', type=int, default=0, 
                        help='Adding Noise in the Latent')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    # parser.add_argument('--pth-save-fold', default='results/tmp', type=str,
    #                     help='The folder to save pths')
    parser.add_argument('--parallel', type=int, default=0, 
                        help='1 for parallel, 0 for non-parallel')   
    parser.add_argument('--resume', type=str)                                      
    args = parser.parse_args()

    return args

class dataset_all(datasets.cifar.CIFAR100):
    def __init__(self, 
                 root: str, 
                 classes_subset: None,
                 train: bool = False, 
                 transform: Union[Callable[..., Any], None] = None,  
                 target_transform: Union[Callable[..., Any], None] = None, 
                 download: bool = False) -> None:
        super().__init__(root, 
                         train, 
                         transform, 
                         target_transform, 
                         download)

        if classes_subset is not None:
            self.filter_dt_classes(classes_subset)

    def filter_dt_classes(self, classes_subset):
        new_data = []
        new_targets = []
        # dataset_number = []
        for i, target in enumerate(self.targets):
            if target in classes_subset:
                new_data.append(self.data[i])
                new_targets.append(target)

        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        dataset_number = (label // 10)
        return img, label, dataset_number

def main(args):

    trsf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])
    
    val_dataset = dataset_all(root = './data', 
                           classes_subset=list(np.arange(args.initclass, args.initclass + args.increment)), 
                            train = False, transform=trsf)
    
    val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            )  
     
    recons_model = builder.BuildAutoEncoder(args)
    recons_checkpoint = torch.load(args.resume)['state_dict']
    recons_model.load_state_dict(recons_checkpoint)
    summary(recons_model, input_size=[(args.batch_size,3,224,224),(args.batch_size,512)],
                         col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
    # summary(recons_model, input_size=(args.batch_size,3,224,224),
                        #  col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
    
    criterion = nn.L1Loss()
    recons_model.eval()

    class_model = builder.BuildClassfy(args, utils.medmnist_dataset_to_target("cifar100", 0))
    summary(class_model, input_size=(args.batch_size,3,224,224),
                         col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
    
    dataset_pred = []
    dataset_gt = []
    class_pred = []
    class_gt = []
    mean_1 = []
    mean_2 = []
    mean_3 = []
    mean_4 = []
    mean_5 = []
    mean_6 = []
    mean_7 = []

    with torch.no_grad():
        for i, (input, target, dataset_number) in tqdm(enumerate(val_loader)):

            input = input.cuda(non_blocking=True)

            noise_1 = torch.normal(0.5, 0.1, size=(512,1)).cuda(non_blocking=True)
            noise_2 = torch.normal(1.5, 0.1, size=(512,1)).cuda(non_blocking=True)
            noise_3 = torch.normal(2.5, 0.1, size=(512,1)).cuda(non_blocking=True)
            noise_4 = torch.normal(3.5, 0.1, size=(512,1)).cuda(non_blocking=True)
            noise_5 = torch.normal(4.5, 0.1, size=(512,1)).cuda(non_blocking=True)
            noise_6 = torch.normal(5.5, 0.1, size=(512,1)).cuda(non_blocking=True)
            noise_7 = torch.normal(6.5, 0.1, size=(512,1)).cuda(non_blocking=True)

            output_1 = recons_model(input, noise_1.transpose(0, 1))
            output_2 = recons_model(input, noise_2.transpose(0, 1))
            output_3 = recons_model(input, noise_3.transpose(0, 1))
            output_4 = recons_model(input, noise_4.transpose(0, 1))
            output_5 = recons_model(input, noise_5.transpose(0, 1))
            output_6 = recons_model(input, noise_6.transpose(0, 1))
            output_7 = recons_model(input, noise_7.transpose(0, 1))

            # output_1 = recons_model(input)
            # output_2 = recons_model(input)
            # output_3 = recons_model(input)
            # output_4 = recons_model(input)
            # output_5 = recons_model(input)
            # output_6 = recons_model(input)
            # output_7 = recons_model(input)
            
            mean_1.append(output_1.mean().item())
            mean_2.append(output_2.mean().item())
            mean_3.append(output_3.mean().item())
            mean_4.append(output_4.mean().item())
            mean_5.append(output_5.mean().item())
            mean_6.append(output_6.mean().item())
            mean_7.append(output_7.mean().item())

            loss1 = criterion(torch.mean(output_1, dim=(1)), torch.mean(noise_1.transpose(0, 1), dim=(1)))      
            loss2 = criterion(torch.mean(output_2, dim=(1)), torch.mean(noise_2.transpose(0, 1), dim=(1)))
            loss3 = criterion(torch.mean(output_3, dim=(1)), torch.mean(noise_3.transpose(0, 1), dim=(1)))
            loss4 = criterion(torch.mean(output_4, dim=(1)), torch.mean(noise_4.transpose(0, 1), dim=(1)))
            loss5 = criterion(torch.mean(output_5, dim=(1)), torch.mean(noise_5.transpose(0, 1), dim=(1)))
            loss6 = criterion(torch.mean(output_6, dim=(1)), torch.mean(noise_6.transpose(0, 1), dim=(1)))
            loss7 = criterion(torch.mean(output_7, dim=(1)), torch.mean(noise_7.transpose(0, 1), dim=(1)))

            # if loss1 < loss2 and loss1 < loss3 and loss1 < loss4 and loss1 < loss5 and loss1 < loss6 and loss1 < loss7:
            #     dataset_pd = 0
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_1_newAA-resnet18/100.pth")['state_dict']
            # elif loss2 < loss1 and loss2 < loss3 and loss2 < loss4 and loss2 < loss5 and loss2 < loss6 and loss2 < loss7:
            #     dataset_pd = 1
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_2_newAA-resnet18/100.pth")['state_dict']
            # elif loss3 < loss1 and loss3 < loss2 and loss3 < loss4 and loss3 < loss5 and loss3 < loss6 and loss3 < loss7:
            #     dataset_pd = 2
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_3_newAA-resnet18/100.pth")['state_dict']
            # elif loss4 < loss1 and loss4 < loss2 and loss4 < loss3 and loss4 < loss5 and loss4 < loss6 and loss4 < loss7:
            #     dataset_pd = 3
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_4_newAA-resnet18/100.pth")['state_dict']
            # elif loss5 < loss1 and loss5 < loss2 and loss5 < loss3 and loss5 < loss4 and loss5 < loss6 and loss5 < loss7:
            #     dataset_pd = 4
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_5_newAA-resnet18/100.pth")['state_dict']
            # elif loss6 < loss1 and loss6 < loss2 and loss6 < loss3 and loss6 < loss4 and loss6 < loss5 and loss6 < loss7:
            #     dataset_pd = 5
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_6_newAA-resnet18/100.pth")['state_dict']
            # elif loss7 < loss1 and loss7 < loss2 and loss7 < loss3 and loss7 < loss4 and loss7 < loss5 and loss7 < loss6:
            #     dataset_pd = 6
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_7_newAA-resnet18/100.pth")['state_dict']

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            if loss1 < loss2 and loss1 < loss3 and loss1 < loss4 and loss1 < loss5 and loss1 < loss6:
                dataset_pd = 0
                dataset_pred.append(dataset_pd)
                class_checkpoint = torch.load("results/cifar_mlp_1_newAA-resnet18/100.pth")['state_dict']
            elif loss2 < loss1 and loss2 < loss3 and loss2 < loss4 and loss2 < loss5 and loss2 < loss6:
                dataset_pd = 1
                dataset_pred.append(dataset_pd)  
                class_checkpoint = torch.load("results/cifar_mlp_2_newAA-resnet18/100.pth")['state_dict']
            elif loss3 < loss1 and loss3 < loss2 and loss3 < loss4 and loss3 < loss5 and loss3 < loss6:
                dataset_pd = 2
                dataset_pred.append(dataset_pd)  
                class_checkpoint = torch.load("results/cifar_mlp_3_newAA-resnet18/100.pth")['state_dict']
            elif loss4 < loss1 and loss4 < loss2 and loss4 < loss3 and loss4 < loss5 and loss4 < loss6:
                dataset_pd = 3
                dataset_pred.append(dataset_pd)  
                class_checkpoint = torch.load("results/cifar_mlp_4_newAA-resnet18/100.pth")['state_dict']
            elif loss5 < loss1 and loss5 < loss2 and loss5 < loss3 and loss5 < loss4 and loss5 < loss6:
                dataset_pd = 4
                dataset_pred.append(dataset_pd)
                class_checkpoint = torch.load("results/cifar_mlp_5_newAA-resnet18/100.pth")['state_dict']
            elif loss6 < loss1 and loss6 < loss2 and loss6 < loss3 and loss6 < loss4 and loss6 < loss5:
                dataset_pd = 5
                dataset_pred.append(dataset_pd)
                class_checkpoint = torch.load("results/cifar_mlp_6_newAA-resnet18/100.pth")['state_dict']

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # if loss1 < loss2 and loss1 < loss3 and loss1 < loss4 and loss1 < loss5:
            #     dataset_pd = 0
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_1_newAA-resnet18/100.pth")['state_dict']
            # elif loss2 < loss1 and loss2 < loss3 and loss2 < loss4 and loss2 < loss5:
            #     dataset_pd = 1
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_2_newAA-resnet18/100.pth")['state_dict']
            # elif loss3 < loss1 and loss3 < loss2 and loss3 < loss4 and loss3 < loss5:
            #     dataset_pd = 2
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_3_newAA-resnet18/100.pth")['state_dict']
            # elif loss4 < loss1 and loss4 < loss2 and loss4 < loss3 and loss4 < loss5:
            #     dataset_pd = 3
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_4_newAA-resnet18/100.pth")['state_dict']
            # elif loss5 < loss1 and loss5 < loss2 and loss5 < loss3 and loss5 < loss4:
            #     dataset_pd = 4
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_5_newAA-resnet18/100.pth")['state_dict']

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # if loss1 < loss2 and loss1 < loss3 and loss1 < loss4:
            #     dataset_pd = 0
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_1_newAA-resnet18/100.pth")['state_dict']
            # elif loss2 < loss1 and loss2 < loss3 and loss2 < loss4:
            #     dataset_pd = 1
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_2_newAA-resnet18/100.pth")['state_dict']
            # elif loss3 < loss1 and loss3 < loss2 and loss3 < loss4:
            #     dataset_pd = 2
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_3_newAA-resnet18/100.pth")['state_dict']
            # elif loss4 < loss1 and loss4 < loss2 and loss4 < loss3:
            #     dataset_pd = 3
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_4_newAA-resnet18/100.pth")['state_dict']

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # if loss1 < loss2 and loss1 < loss3:
            #     dataset_pd = 0
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_1_newAA-resnet18/100.pth")['state_dict']
            # elif loss2 < loss1 and loss2 < loss3:
            #     dataset_pd = 1
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_2_newAA-resnet18/100.pth")['state_dict']
            # elif loss3 < loss1 and loss3 < loss2:
            #     dataset_pd = 2
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_3_newAA-resnet18/100.pth")['state_dict']
            
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # if loss1 < loss2:
            #     dataset_pd = 0
            #     dataset_pred.append(dataset_pd)
            #     class_checkpoint = torch.load("results/cifar_mlp_1_newAA-resnet18/100.pth")['state_dict']
            # elif loss2 < loss1:
            #     dataset_pd = 1
            #     dataset_pred.append(dataset_pd)  
            #     class_checkpoint = torch.load("results/cifar_mlp_2_newAA-resnet18/100.pth")['state_dict']

            dataset_gt.append(dataset_number.item())

            class_model.load_state_dict(class_checkpoint)
            class_model.eval()

            output = class_model(input)
            class_gt.append(target.cpu().item())
            class_pred.append(torch.argmax(output.cpu()).item() + (dataset_pd*10))
        
    cm_1 = confusion_matrix(dataset_gt, dataset_pred)
    cm_2 = confusion_matrix(class_gt, class_pred)

    report_1 = classification_report(dataset_gt, dataset_pred)
    report_2 = classification_report(class_gt, class_pred)

    print("Confusion Matrix:")
    print(cm_1)
    print("\nClassification Report:")
    print(report_1)

    print("Confusion Matrix:")
    print(cm_2)
    print("\nClassification Report:")
    print(report_2)

    print(accuracy_score(dataset_gt, dataset_pred))
    print(accuracy_score(class_pred, class_gt))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_1, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Dataset')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(os.path.dirname(args.resume), 'confusion_matrix_dataset.png'))
    plt.show()

    # Plot and save confusion matrix for classes
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_2, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix for Classes')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(os.path.dirname(args.resume), 'confusion_matrix_classes.png'))
    plt.show()

    breakpoint()


if __name__ == '__main__':

    args = get_args()
    main(args)

# CUDA_VISIBLE_DEVICES=0 python pipeline.py --arch resnet18 --resume results/incremnet_2_0.5-resnet18/100.pth --initclass 0 --increment 20