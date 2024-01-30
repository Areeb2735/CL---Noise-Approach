from torchvision import datasets
from medmnist.info import DEFAULT_ROOT
from typing import Callable, Any, Tuple, Union
import torch
import random
from medmnist.dataset import PathMNIST, BloodMNIST, TissueMNIST, OrganAMNIST
    
class NoisyDataset(datasets.cifar.CIFAR100):
    def __init__(self, 
                 root: str, 
                 mean: float,
                 std: float,
                 args: None,
                 classes_subset: None,
                 max_samples: None,
                 train: bool = True, 
                 transform: Union[Callable[..., Any], None] = None,  
                 target_transform: Union[Callable[..., Any], None] = None, 
                 download: bool = False) -> None:
        super().__init__(root, 
                         train, 
                         transform, 
                         target_transform, 
                         download)

        self.replay = args.replay
        self.mean = mean
        self.std = std
        if classes_subset is not None:
            self.filter_dt_classes(classes_subset)
        
        if max_samples is not None:
            self.data = self.data[:max_samples]
            self.targets = self.targets[:max_samples]

    def filter_dt_classes(self, classes_subset):
        new_data = []
        new_targets = []
        for i, target in enumerate(self.targets):
            if target in classes_subset:
                new_data.append(self.data[i])
                new_targets.append(target)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)

        size = (512,1)
        # size = (512,7,7,1)

        gt_noise = torch.normal(self.mean, self.std, size=size)

        if self.mean == 1.5:
            other_noise = torch.normal(random.choice([0.5]), self.std, size=size)
        elif self.mean == 0.5 and self.replay:
            other_noise = torch.normal(random.choice([1.5]), self.std, size=size)
        else:
            other_noise = torch.normal(random.choice([0.5]), self.std, size=size)

# ----------------------------------------------------------------------------------------

        # if self.mean == 2.5:                                                            
        #     other_noise = torch.normal(random.choice([0.5, 1.5]), self.std, size=size)  
        
        # elif self.mean == 0.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([1.5, 2.5]), self.std, size=size) 

        # elif self.mean == 1.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 2.5]), self.std, size=size) 
        # else:
        #     other_noise = torch.normal(random.choice([0.5]), self.std, size=size) 

# ----------------------------------------------------------------------------------------

        # if self.mean == 3.5:                                                            
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5]), self.std, size=size)  
        
        # elif self.mean == 0.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([1.5, 2.5, 3.5]), self.std, size=size) 

        # elif self.mean == 1.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 2.5, 3.5]), self.std, size=size) 

        # elif self.mean == 2.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 3.5]), self.std, size=size) 
        
        # else:
        #     other_noise = torch.normal(random.choice([0.5]), self.std, size=size) 

# ----------------------------------------------------------------------------------------

        # if self.mean == 4.5:                                                            
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 3.5]), self.std, size=size)  
        
        # elif self.mean == 0.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([1.5, 2.5, 3.5, 4.5]), self.std, size=size) 

        # elif self.mean == 1.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 2.5, 3.5, 4.5]), self.std, size=size) 

        # elif self.mean == 2.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 3.5, 4.5]), self.std, size=size) 

        # elif self.mean == 3.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 4.5]), self.std, size=size) 
        
        # else:
        #     other_noise = torch.normal(random.choice([0.5]), self.std, size=size) 

# ----------------------------------------------------------------------------------------

        # if self.mean == 5.5:                                                            
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 3.5, 4.5]), self.std, size=size)  
        
        # elif self.mean == 0.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([1.5, 2.5, 3.5, 4.5, 5.5]), self.std, size=size) 

        # elif self.mean == 1.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 2.5, 3.5, 4.5, 5.5]), self.std, size=size) 

        # elif self.mean == 2.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 3.5, 4.5, 5.5]), self.std, size=size) 

        # elif self.mean == 3.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 4.5, 5.5]), self.std, size=size) 
        
        # elif self.mean == 4.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 3.5, 5.5]), self.std, size=size) 
        
        # else:
        #     other_noise = torch.normal(random.choice([0.5]), self.std, size=size) 

# ----------------------------------------------------------------------------------------

        # if self.mean == 6.5:                                                            
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), self.std, size=size)  
        
        # elif self.mean == 0.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), self.std, size=size) 

        # elif self.mean == 1.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 2.5, 3.5, 4.5, 5.5, 6.5]), self.std, size=size) 

        # elif self.mean == 2.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 3.5, 4.5, 5.5, 6.5]), self.std, size=size) 

        # elif self.mean == 3.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 4.5, 5.5, 6.5]), self.std, size=size) 
        
        # elif self.mean == 4.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 3.5, 5.5, 6.5]), self.std, size=size) 
        
        # elif self.mean == 5.5 and self.replay:                                        
        #     other_noise = torch.normal(random.choice([0.5, 1.5, 2.5, 3.5, 4.5, 6.5]), self.std, size=size) 
        
        # else:
        #     other_noise = torch.normal(random.choice([0.5]), self.std, size=size) 
            
# ----------------------------------------------------------------------------------------
            
        if random.uniform(0,1) > 0.5:
            added_noise = other_noise
        else:
            added_noise = gt_noise
        
        return img, label, added_noise.squeeze(-1), gt_noise.squeeze(-1)
        # return img, label, added_noise, gt_noise
    
class NoisyDataset_test(datasets.cifar.CIFAR100):
    def __init__(self, 
                 root: str, 
                 mean: float,
                 std: float,
                 classes_subset: None,
                 max_samples: None,
                 train: bool = False, 
                 transform: Union[Callable[..., Any], None] = None,  
                 target_transform: Union[Callable[..., Any], None] = None, 
                 download: bool = False) -> None:
        super().__init__(root, 
                         train, 
                         transform, 
                         target_transform, 
                         download)

        self.mean = mean
        self.std = std
        if classes_subset is not None:
            self.filter_dt_classes(classes_subset)
        
        if max_samples is not None:
            self.data = self.data[:max_samples]
            self.targets = self.targets[:max_samples]

        # self.previous_noises = []
        # while mean > 1:
        #     mean -= 1
        #     self.previous_noises.append(mean)

    def filter_dt_classes(self, classes_subset):
        new_data = []
        new_targets = []
        for i, target in enumerate(self.targets):
            if target in classes_subset:
                new_data.append(self.data[i])
                new_targets.append(target)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        gt_noise = torch.normal(self.mean, self.std, size=(512,1))       ######
        # img = img + gt_noise
        return img, label, gt_noise.squeeze(-1)

class ClassifyDataset(datasets.cifar.CIFAR100):
    def __init__(self, 
                 root: str, 
                 classes_subset: None,
                 train: bool = True, 
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
        for i, target in enumerate(self.targets):
            if target in classes_subset:
                new_data.append(self.data[i])
                new_targets.append(target)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        dataset_number = (label // 10)*10
        label = label - dataset_number
        return img, label, dataset_number

class ClassifyDataset(datasets.cifar.CIFAR100):
    def __init__(self, 
                 root: str, 
                 classes_subset: None,
                 train: bool = True, 
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
        for i, target in enumerate(self.targets):
            if target in classes_subset:
                new_data.append(self.data[i])
                new_targets.append(target)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        dataset_number = (label // 10)
        label = label - (dataset_number*10)
        return img, label, dataset_number

class Classify_bloodMNIST(BloodMNIST):
    def __init__(self, 
                 split,
                 root=DEFAULT_ROOT, 
                 transform: Union[Callable[..., Any], None] = None,  
                 target_transform: Union[Callable[..., Any], None] = None, 
                 as_rgb=False,
                 download: bool = False) -> None:
        super().__init__(split, transform, target_transform, download, as_rgb, root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        label = label.item()
        return img, label, 0
    
class Classify_organAMNIST(OrganAMNIST):
    def __init__(self, 
                 split,
                 root=DEFAULT_ROOT, 
                 transform: Union[Callable[..., Any], None] = None,  
                 target_transform: Union[Callable[..., Any], None] = None, 
                 as_rgb=False,
                 download: bool = False) -> None:
        super().__init__(split, transform, target_transform, download, as_rgb, root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        label = label.item()
        return img, label, 1

class Classify_pathMNIST(PathMNIST):
    def __init__(self, 
                 split,
                 root=DEFAULT_ROOT, 
                 transform: Union[Callable[..., Any], None] = None,  
                 target_transform: Union[Callable[..., Any], None] = None, 
                 as_rgb=False,
                 download: bool = False) -> None:
        super().__init__(split, transform, target_transform, download, as_rgb, root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        label = label.item()
        return img, label, 2
    
class Classify_tissueMNIST(TissueMNIST):
    def __init__(self, 
                 split,
                 root=DEFAULT_ROOT, 
                 transform: Union[Callable[..., Any], None] = None,  
                 target_transform: Union[Callable[..., Any], None] = None, 
                 as_rgb=False,
                 download: bool = False) -> None:
        super().__init__(split, transform, target_transform, download, as_rgb, root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        label = label.item()
        return img, label, 3