import torch.nn as nn
import torch.nn.parallel as parallel
import torch
from torchvision.models.resnet import resnet18
from collections import OrderedDict

from . import vgg, resnet

def BuildAutoEncoder(args):

    if args.arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(args.arch)
        model = vgg.VGGAutoEncoder(configs)

    elif args.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(args.arch)
        # print(configs, bottleneck)
        model = resnet.ResNetAutoEncoder(configs, bottleneck, args)
    
    else:
        return None
    
    if args.parallel == 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = parallel.DistributedDataParallel(
                        model.to(args.gpu),
                        device_ids=[args.gpu],
                        output_device=args.gpu
                    )   
    
    else:
        model = nn.DataParallel(model).cuda()

    return model

def BuildClassfy(args, output_dim):

    model = resnet.ResNetClassify(output_dim)
    
    if args.parallel == 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = parallel.DistributedDataParallel(
                        model.to(args.gpu),
                        device_ids=[args.gpu],
                        output_device=args.gpu
                    )   
    
    else:
        model = nn.DataParallel(model).cuda()

    return model

def BuildAutoClassfy(args):

    if args.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(args.arch)
        model = resnet.ResNetAutoClassify(configs, bottleneck)
    
    else:
        return None
    
    if args.parallel == 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = parallel.DistributedDataParallel(
                        model.to(args.gpu),
                        device_ids=[args.gpu],
                        output_device=args.gpu
                    )   
    
    else:
        model = nn.DataParallel(model).cuda()

    return model

# class my_model_auto(nn.Module):

#     def __init__(self, model):

#         super(my_model_auto, self).__init__()

#         self.encoder = nn.Sequential(OrderedDict([*(list(resnet18(pretrained=False).named_children())[:-2])])).cuda()
#         self.decoder = model.module.decoder.cuda()
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         # self.prompt = nn.Parameter(torch.randn(512, 7, 7))
#         self.forward_counter = 0
    
#     def forward(self, x, args):

#         batch , _, _, _ = x.size()
            
#         noise = torch.normal(args.noisevalue, 0.1, size=(batch, 3, 224, 224)).cuda()
#         x = self.encoder(x + noise)
#         x = self.decoder(x)

#         self.forward_counter += 1

#         return x, noise
    
# class my_model_class(nn.Module):

#     def __init__(self, model):

#         super(my_model_class, self).__init__()

#         self.model = model
#         for param in self.model.parameters():
#             param.requires_grad = False
#         # self.prompt = torch.load('results/cifar_pretrained-resnet18/600.pth')['state_dict']['prompt']
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=1, padding=0,),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU(inplace=True),
#         ).cuda()
#         self.fc = nn.Linear(in_features=256, out_features=10).cuda()
    
#     def forward(self, x):

#         batch_size, _, _, _ = x.size()

#         x = self.model.module.encoder(x)
#         # x = x + self.prompt.expand(batch_size, -1, -1, -1).cuda()
#         x = self.conv(x)
#         x = self.fc(x.squeeze())

#         return x