import os
import sys
import math
import random
import numpy as np

from loguru import logger

import torch
from torch.backends import cudnn

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    decay = args.lr_drop_ratio if epoch in args.lr_drop_epoch else 1.0
    lr = args.lr * decay
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.lr = current_lr
    return current_lr

def adjust_learning_rate_cosine(optimizer, epoch, args):
    """cosine learning rate annealing without restart"""
    lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return current_lr

def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model

def medmnist_target_to_dataset(key):
    if 0 <= key <= 7:
        return 0
    elif 8 <= key <= 18:
        return 1
    elif 19 <= key <= 27:
        return 2
    elif 28 <= key <= 35:
        return 3
    else:
        return None

def  medmnist_dataset_to_target(dataset, task_id):
    if dataset == 'cifar100':
        return 10
    elif dataset == 'medmnist':
        if task_id == 0:
            return 0
        elif task_id == 1:
            return 8
        elif task_id == 2:
            return 11
        elif task_id == 3:
            return 9
        elif task_id == 4:
            return 8

# def task_weight(n):
#     values = list(range(0, n + 1))
#     epsilon = 1e-10
#     adjusted_values = [value + epsilon for value in values]
#     weights = [1/value for value in adjusted_values]
#     mean_weight = sum(weights) / len(weights)
#     adjusted_weights = [weight / mean_weight for weight in weights]
#     result_dict = {value: adjusted_weight for value, adjusted_weight in zip(values, adjusted_weights)}
#     for value, weight in zip(values, adjusted_weights):
#         print(f"Value: {value}, Weight: {weight}")
#     print("Mean of Adjusted Weights:", sum(adjusted_weights) / len(adjusted_weights))
#     return result_dict

def task_weight(n):
    values = list(range(n, -1, -1))
    print(values)
    numbers = np.linspace(0.5, 1.5, n+1)
    print(numbers)
    mean_difference = 1 - np.mean(numbers)
    numbers += mean_difference
    result_dict = {values: numbers for values, numbers in zip(values, numbers)}
    for value, weight in zip(values, numbers):
        print(f"Value: {value}, Weight: {weight}")
    print("Mean of Adjusted Weights:", sum(numbers) / len(numbers))
    return result_dict

def weight_dictionary(n, exponential_factor=1.6):
    if n <= 0:
        raise ValueError("Input 'n' must be a positive integer.")

    # result_dict = {i: 2.5 * (n-1) if i != n - 1 else 1 for i in range(n)}
    result_dict = {i: 2.5 * (n-1) * (exponential_factor ** (n-1-i)) if i != n - 1 else 1 for i in range(n)}
    return result_dict

def pad_noise(original_tensor):
    desired_size = 512
    num_zeros_to_pad = desired_size - original_tensor.size(1)
    return torch.nn.functional.pad(original_tensor, (num_zeros_to_pad, 0), mode='constant', value=0)