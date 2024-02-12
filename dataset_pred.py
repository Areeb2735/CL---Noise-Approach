import numpy as np
import argparse
import models.builer as builder 
from tqdm import tqdm
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from torchvision import datasets
from typing import Callable, Any, Tuple, Union
from torchinfo import summary
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


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
                # if target in classes_subset[i:i+10]:
                # dataset_number.append((i // 10) * 10)
        self.data = new_data
        self.targets = new_targets
        # self.dataset_number = dataset_number

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        # gt_noise = torch.normal(self.mean, self.std, size=img.shape)
        dataset_number = (label // 10)
        return img, dataset_number

def main(args):
    # breakpoint()

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
     
    model = builder.BuildAutoEncoder(args)
    # summary(model.cuda(), (args.batch_size,3,224,224), col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 
    summary(model.cuda(), [(args.batch_size,3,224,224), (args.batch_size ,512)], col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"]) 

    checkpoint = torch.load(args.resume)['state_dict']
    model.load_state_dict(checkpoint)
    criterion = nn.L1Loss(reduction='none')

    model.eval()
    pred = []
    gt = []
    mean_1 = []
    mean_2 = []
    mean_3 = []

    with torch.no_grad():
        for i, (input, dataset_number) in tqdm(enumerate(val_loader)):

            input = input.cuda(non_blocking=True)

            size = (512,1)

            noise_1 = torch.normal(0.5, 0.1, size=size).cuda(non_blocking=True)
            noise_2 = torch.normal(1.5, 0.1, size=size).cuda(non_blocking=True)
            noise_3 = torch.normal(2.5, 0.1, size=size).cuda(non_blocking=True)
            # noise_4 = torch.normal(3.5, 0.1, size=size).cuda(non_blocking=True)
            
            # output_1 = model(input, noise_1)
            # output_2 = model(input, noise_2)
            # output_3 = model(input, noise_3)

            output_1 = model(input, noise_1.transpose(0, 1))
            output_2 = model(input, noise_2.transpose(0, 1))
            output_3 = model(input, noise_3.transpose(0, 1))
            # output_4 = model(input, noise_4.transpose(0, 1))

            # output_1 = model(input)
            # output_2 = model(input)
            # output_3 = model(input)

            mean_1.append(output_1.mean().item())
            mean_2.append(output_2.mean().item())
            mean_3.append(output_3.mean().item())
            # mean_4.append(output_4.mean().item())

            # loss1 = criterion(output_1, noise_1.transpose(0, 1))
            # loss2 = criterion(output_2, noise_2.transpose(0, 1))
            # loss3 = criterion(output_3, noise_3.transpose(0, 1))

            loss1 = criterion(torch.mean(output_1, dim=(1)), torch.mean(noise_1.transpose(0, 1), dim=(1)))       ######
            loss2 = criterion(torch.mean(output_2, dim=(1)), torch.mean(noise_2.transpose(0, 1), dim=(1)))         ######
            loss3 = criterion(torch.mean(output_3, dim=(1)), torch.mean(noise_3.transpose(0, 1), dim=(1)))         ######
            # loss4 = criterion(torch.mean(output_4, dim=(1)), torch.mean(noise_4.transpose(0, 1), dim=(1)))         ######


            # loss1 = criterion(torch.mean(output_1, dim=(1,2,3)), torch.mean(noise_1, dim=(1,2,3)))     
            # loss2 = criterion(torch.mean(output_2, dim=(1,2,3)), torch.mean(noise_2, dim=(1,2,3)))
            # loss3 = criterion(torch.mean(output_3, dim=(1,2,3)), torch.mean(noise_3, dim=(1,2,3)))  

            # if torch.mean(loss1) < torch.mean(loss2):
            #     pred.append(0)
            # elif torch.mean(loss1) > torch.mean(loss2):
            #     pred.append(1)  

            if loss1 < loss2:
                pred.append(0)
            elif loss1 > loss2:
                pred.append(1)  

            # if loss1 < loss2 and loss1 < loss3:
            #     pred.append(0)
            # elif loss1 > loss2 and loss2 < loss3:
            #     pred.append(1)  
            # elif loss1 > loss3 and loss2 > loss3:
            #     pred.append(2)
            # elif loss1 > loss4 and loss2 > loss4 and loss3 > loss4:
            #     pred.append(3)

            gt.append(dataset_number.item())

    print(np.mean(mean_1), np.mean(mean_2))

    cm = confusion_matrix(gt, pred)
    report = classification_report(gt, pred)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    breakpoint()

if __name__ == '__main__':

    args = get_args()
    main(args)

# CUDA_VISIBLE_DEVICES=1 python dataset_pred.py --arch resnet18  --resume results/NN_512_3-resnet18/100.pth --initclass 0 --increment 30