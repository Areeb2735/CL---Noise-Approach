import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from collections import OrderedDict

def get_configs(arch='resnet50'):

    # True or False means wether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError("Undefined model")
    
class ResNetAutoEncoder(nn.Module):

    """This Runs an Autoencoder Model"""

    def __init__(self):

        super(ResNetAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([*(list(resnet18(pretrained=True).named_children())[:-2])]))

        for param in self.encoder.parameters():
            param.requires_grad = False
        # self.encoder.eval()
        
        # self.decoder = nn.Sequential(
        #         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
        #         nn.BatchNorm2d(num_features=256),
        #         nn.ReLU(inplace=True),

        #         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
        #         nn.BatchNorm2d(num_features=512),
        #         nn.ReLU(inplace=True),

        #         nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(num_features=1024),
        #         nn.ReLU(inplace=True),

        #         nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(num_features=512),
        #         nn.ReLU(inplace=True),

        #         nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(num_features=256),
        #         nn.ReLU(inplace=True),

        #         nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        #         # nn.BatchNorm2d(num_features=128),
        #         # nn.ReLU(inplace=True),

        #         # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        #         # nn.BatchNorm2d(num_features=128),
        #         # nn.ReLU(inplace=True),
        # )
            
        self.decoder_1 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=7, stride=1, padding=0,),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(inplace=True),
        )
        # self.decoder_2 = nn.Sequential(
        #         # nn.BatchNorm1d(num_features=256),
        #         nn.Linear(in_features=128, out_features=256),
        #         nn.BatchNorm1d(num_features=256),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(in_features=256, out_features=512),
        #         nn.BatchNorm1d(num_features=512),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.4),
        #         nn.Linear(in_features=512, out_features=768),
        #         nn.BatchNorm1d(768),  
        #         nn.ReLU(),
        #         nn.Linear(in_features=768, out_features=1024),
        #         nn.BatchNorm1d(1024),
        #         nn.ReLU(),  
        #         nn.Dropout(0.4),
        #         nn.Linear(in_features=1024, out_features=768),
        #         nn.BatchNorm1d(768),  
        #         nn.ReLU(),
        #         nn.Linear(in_features=768, out_features=512),
        #         nn.BatchNorm1d(512),  
        #         nn.ReLU(),
        #         nn.Dropout(0.4),
        #         nn.Linear(in_features=512, out_features=256),
        #         nn.BatchNorm1d(256),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(in_features=256, out_features=128),
        #         nn.BatchNorm1d(128),
        #         nn.ReLU(inplace=True),
        # )
        # self.decoder_3 = nn.Sequential(
        #         nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=7, stride=1, padding=0, output_padding=0, bias=False),
        # )

        self.decoder_2 = nn.Sequential(

                nn.Linear(in_features=512, out_features=768),
                nn.BatchNorm1d(768),  
                nn.ReLU(),
                # nn.Dropout(0.2),

                nn.Linear(in_features=768, out_features=1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),  
                nn.Dropout(0.2),

                nn.Linear(in_features=1024, out_features=768),
                nn.BatchNorm1d(768),  
                nn.ReLU(),
                # nn.Dropout(0.2),

                nn.Linear(in_features=768, out_features=512),
                nn.ReLU()
        )

        # self.decoder_2 = nn.Sequential(

        #         nn.Linear(in_features=512, out_features=768),
        #         nn.BatchNorm1d(768),  
        #         nn.ReLU(),

        #         nn.Linear(in_features=768, out_features=512),
        #         nn.BatchNorm1d(512),
        #         nn.ReLU(),

        #         nn.Linear(in_features=512, out_features=256),
        #         nn.BatchNorm1d(256),
        #         nn.ReLU(),

        #         nn.Linear(in_features=256, out_features=128),
        #         nn.BatchNorm1d(128),
        #         nn.ReLU(),

        #         nn.Linear(in_features=128, out_features=32),
        #         nn.ReLU(),
        # )

        # self.decoder = nn.Sequential(
        #         nn.Linear(in_features=512, out_features=768),
        #         nn.BatchNorm1d(768),  
        #         nn.ReLU(),
        #         nn.Dropout(0.2),

        #         nn.Linear(in_features=768, out_features=1024),
        #         nn.BatchNorm1d(1024),
        #         nn.ReLU(),  
        #         nn.Dropout(0.2),

        #         nn.Linear(in_features=1024, out_features=2048),
        #         nn.BatchNorm1d(2048),  
        #         nn.ReLU(),
        #         nn.Dropout(0.2),

        #         nn.Linear(in_features=2048, out_features=1024),
        #         nn.BatchNorm1d(1024),  
        #         nn.ReLU(),
        #         nn.Dropout(0.2),

        #         nn.Linear(in_features=1024, out_features=768),
        #         nn.BatchNorm1d(768),  
        #         nn.ReLU(),
        #         nn.Dropout(0.2),

        #         nn.Linear(in_features=768, out_features=512),
        # )

    def forward(self, x, noise):
        x = self.encoder(x)
        x = self.decoder_1(x)
        x = self.decoder_2(x.squeeze(dim=-1).squeeze(dim=-1) + noise)
        # x = self.decoder_3(x.unsqueeze(dim=-1).unsqueeze(dim=-1))
        return x

    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = self.decoder_1(x)
    #     x = self.decoder_2(x.squeeze(dim=-1).squeeze(dim=-1))
    #     return x

    ## Adding Noise in the Latent
    # def forward(self, x, noise):
    #     x = self.encoder(x)
    #     x = self.decoder(x.squeeze(dim=-1).squeeze(dim=-1) + noise)
    #     # x = self.decoder(x.squeeze(dim=-1).squeeze(dim=-1))
    #     return x
    
    ## Not Adding Noise in the Latent but outputting the noise
    # def forward(self, x):
    #     x = self.encoder(x)
    #     # x = self.decoder(x + noise)
    #     x = self.decoder(x.squeeze(dim=-1).squeeze(dim=-1))
    #     return x


class ResNetClassify(nn.Module):
    def __init__(self, out_dim):
        super(ResNetClassify, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([*(list(resnet18(pretrained=True).named_children())[:-2])]))
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.decoder = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=7, stride=1, padding=0,),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True),
        )
            
        self.fc = nn.Sequential(
            # nn.Linear(in_features=512, out_features=256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),

            # nn.Linear(in_features=256, out_features=128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=out_dim),
        )

        # nn.Linear(in_features=512, out_features=256)
        # self.fc_2 = nn.Linear(in_features=256, out_features=out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # x = self.fc(x.squeeze())
        x = self.fc(torch.squeeze(x, dim=(2,3)))
        return x
    
class ResNetAutoClassify(nn.Module):

    def __init__(self, configs, bottleneck):

        super(ResNetAutoClassify, self).__init__()

        self.encoder = resnet18(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = ResNetDecoder(configs=configs[::-1], bottleneck=bottleneck)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=1, padding=0,),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):

        batch_size, _, _, _ = x.size()
        noise = torch.normal(0.5, 0.1, size=(batch_size, 3, 224, 224)).cuda()
        x1 = self.encoder(x + noise)
        x4 = self.decoder(x1)
        x2 = self.conv(x1)
        x3 = self.fc(x2.squeeze())

        return x4, noise , x3

class ResNet(nn.Module):

    def __init__(self, configs, bottleneck=False, num_classes=1000):
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(configs, bottleneck)

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        if bottleneck:
            self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.encoder(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetEncoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:

            self.conv2 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=256,  layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=128, up_channels=512,  layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderBottleneckBlock(in_channels=512,  hidden_channels=256, up_channels=1024, layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048, layers=configs[3], downsample_method="conv")

        else:

            self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=configs[3], downsample_method="conv")

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class ResNetDecoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024, layers=configs[0])
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,  layers=configs[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,  layers=configs[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,   layers=configs[3])


        else:

            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=configs[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )

        self.gate = nn.ReLU()

    def forward(self, x, skip_con = None):

        if skip_con is not None:
            # breakpoint()
            x1, x2, x3 = skip_con
            x = self.conv1(x) + x3         # B, 256, 14, 14  +   B, 256, 14 , 14
            x = self.conv2(x) + x2         # B, 128, 28, 28  +   B, 128, 28, 28
            x = self.conv3(x) + x1         # B, 64, 56, 56  +   B, 64, 56, 56
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.gate(x)

        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.gate(x)

        return x

class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, layers, downsample_method="conv"):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=True)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method="conv"):
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=True)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x

class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, downsample):
        super(EncoderBottleneckLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
            )
        else:
            self.upsample = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x

if __name__ == "__main__":

    configs, bottleneck = get_configs("resnet152")

    encoder = ResNetEncoder(configs, bottleneck)

    input = torch.randn((5,3,224,224))

    print(input.shape)

    output = encoder(input)

    print(output.shape)

    decoder = ResNetDecoder(configs[::-1], bottleneck)

    output = decoder(output)

    print(output.shape)