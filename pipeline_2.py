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
    parser.add_argument('--initclass', type=int, default=0,
                        help='Class to start with')
    parser.add_argument('--increment', type=int, default=10,
                        help='Increment added to the initial class')
    parser.add_argument('--std', type=float, default=0.1,
                        help='Noise Std Value to be added')
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
    parser.add_argument('--checkpoint', type=str)                                      
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
    #  We need to change the incremnet for each task
    for k in range(1, 10):  # We don't need to do classification for th first tasks as there are only 1 classes.

        print(f"Start of Evaluation till Task number {((k * 10)//10)+1}")
        print(f"Number of classes (from starting): {args.increment + (k * 10)}")

        task_id = int((args.initclass + args.increment + (k * 10))/10)
        args.mean = task_id - 0.5

        print(f"Mean: {args.mean}")
        print(f"Std: {args.std}")

        trsf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])
    
        val_dataset = dataset_all(root = './data', 
                            classes_subset=list(np.arange(args.initclass, args.initclass + args.increment + (k * 10))), 
                            train = False, transform=trsf)
    
        val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                )  
     
        recons_model = builder.BuildAutoEncoder(args)
        recons_checkpoint = torch.load(os. path.join(args.checkpoint, str(task_id).zfill(2),'100.pth'))['state_dict']
        recons_model.load_state_dict(recons_checkpoint)

        summary(recons_model, input_size=[(args.batch_size,3,224,224),(args.batch_size,512)],
                            col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
        # summary(recons_model, input_size=(args.batch_size,3,224,224),
        #                      col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
    
        criterion = nn.L1Loss(reduction='none')
        recons_model.eval()

        class_model = builder.BuildClassfy(args, utils.medmnist_dataset_to_target("cifar100", 0))
        summary(class_model, input_size=(args.batch_size,3,224,224),
                            col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
    
        noises = []
        for num in reversed(range(len(range(task_id-1)))):  # We need to add noise for all the previous tasks
            noises.append(torch.normal(args.mean-(num + 1), 0.1, size=(32,1)).cuda(non_blocking=True))
            print(f"Mean: {args.mean-(num + 1)}")
        
        noises.append(torch.normal(args.mean, 0.1, size=(32,1)).cuda(non_blocking=True))

        dataset_pred = []
        dataset_gt = []
        class_pred = []
        class_gt = []

        with torch.no_grad():
            for i, (image, target, dataset_number) in tqdm(enumerate(val_loader)):

                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                dataset_number = dataset_number.cuda(non_blocking=True)

                losses = []
                for noise in noises:
                    noise_pad = utils.pad_noise(noise.transpose(0, 1))

                    # output = recons_model(image, noise_pad.transpose(0, 1))
                    output = recons_model(image, noise_pad)
                    # output = recons_model(image)
                    losses.append(criterion(torch.mean(output, dim=(1)), torch.mean(noise.transpose(0, 1), dim=(1))))
                
                dataset_pd = torch.argmin(torch.stack(losses)).item()
                dataset_pred.append(dataset_pd)
                dataset_gt.append(dataset_number.item())

                if dataset_pd == 0:
                    class_checkpoint = torch.load("results/mlp_1-resnet18/030.pth")['state_dict']
                elif dataset_pd == 1:
                    class_checkpoint = torch.load("results/mlp_2-resnet18/030.pth")['state_dict']
                elif dataset_pd == 2:
                    class_checkpoint = torch.load("results/mlp_3-resnet18/030.pth")['state_dict']
                elif dataset_pd == 3:
                    class_checkpoint = torch.load("results/mlp_4-resnet18/030.pth")['state_dict']
                elif dataset_pd == 4:
                    class_checkpoint = torch.load("results/mlp_5-resnet18/030.pth")['state_dict']
                elif dataset_pd == 5:
                    class_checkpoint = torch.load("results/mlp_6-resnet18/030.pth")['state_dict']
                elif dataset_pd == 6:
                    class_checkpoint = torch.load("results/mlp_7-resnet18/030.pth")['state_dict']
                elif dataset_pd == 7:
                    class_checkpoint = torch.load("results/mlp_8-resnet18/030.pth")['state_dict']
                elif dataset_pd == 8:
                    class_checkpoint = torch.load("results/mlp_9-resnet18/030.pth")['state_dict']
                elif dataset_pd == 9:
                    class_checkpoint = torch.load("results/mlp_10-resnet18/030.pth")['state_dict']

                class_model.load_state_dict(class_checkpoint)
                class_model.eval()

                output = class_model(image)
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
        plt.savefig(os.path.join(os.path.dirname(os. path.join(args.checkpoint, str(task_id).zfill(2), '')), 'confusion_matrix_dataset.png'))
        plt.show()

        # Plot and save confusion matrix for classes
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_2, annot=True, fmt='d', cmap='Greens')
        plt.title('Confusion Matrix for Classes')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.savefig(os.path.join(os.path.dirname(os. path.join(args.checkpoint, str(task_id).zfill(2), '')), 'confusion_matrix_classes.png'))
        plt.show()


if __name__ == '__main__':

    args = get_args()
    main(args)

# CUDA_VISIBLE_DEVICES=0 python pipeline_2.py --arch resnet18 --checkpoint 