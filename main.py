from model import ACL
from config import Config
import copy
from logger import MyLogger
import yaml
from torchinfo import summary
import torch

def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def overwrite_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        setattr(args, key, value)


myconfig = Config()
# self._config.backbone, self._config.pretrained, self._config.pretrain_path, MLP_projector=self._config.MLP_projector

# Load YAML config
yaml_config = load_yaml_config('./configs/acl.yaml')
# breakpoint()
# Overwrite args with YAML config
overwrite_args_with_yaml(myconfig, yaml_config['options']['cifar100']['resnet18'])

myconfig.backbone = "resnet18"
myconfig.pretrained = True
myconfig.pretrain_path  = None
myconfig.method = 'acl'
myconfig.task_id = 0

logger = MyLogger(myconfig)
model = ACL(logger, myconfig)

model.prepare_model()
breakpoint()

checkpoint_1 = torch.load('logs/seed1993_task0_checkpoint.pkl')['state_dict']
keys_to_remove = ['aux_fc.weight', 'aux_fc.bias', 'seperate_fc.0.weight', 'seperate_fc.0.bias']
filtered_checkpoint_1 = {key: value for key, value in checkpoint_1.items() if not any(unwanted_key in key for unwanted_key in keys_to_remove)}
model._network.load_state_dict(filtered_checkpoint_1)


print('all: ',len([i for i in model._network.parameters()]))
print('trainable: ', len([i for i in model._network.parameters() if i.requires_grad]))

model.prepare_model()
breakpoint()

checkpoint_2 = torch.load('logs/seed1993_task1_checkpoint.pkl')['state_dict']
keys_to_remove = ['aux_fc.weight', 'aux_fc.bias', 'seperate_fc.0.weight', 'seperate_fc.0.bias', 'seperate_fc.1.weight', 'seperate_fc.1.bias']
filtered_checkpoint_2 = {key: value for key, value in checkpoint_2.items() if not any(unwanted_key in key for unwanted_key in keys_to_remove)}
model._network.load_state_dict(filtered_checkpoint_2)

print('all: ',len([i for i in model._network.parameters()]))
print('trainable: ', len([i for i in model._network.parameters() if i.requires_grad]))

# model.prepare_model()

# model._network.load_state_dict(torch.load('logs/seed1993_task2_checkpoint.pkl')['state_dict'])

# model.prepare_model()
# model._network.load_state_dict(torch.load('logs/seed1993_task3_checkpoint.pkl')['state_dict'])

# model.prepare_model()

# summary(model._network.cuda(), (1,3,224,224), col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"])

# model1 = ACL(logger, myconfig)
# model1.prepare_model()

# summary(model1._network.cuda(), (1,3,224,224), col_names=['input_size', 'output_size' , "num_params", "kernel_size", "trainable"])
# breakpoint()