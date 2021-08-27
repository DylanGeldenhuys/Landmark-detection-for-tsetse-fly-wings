
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
class RegressionNet(nn.Module):
    def __init__(self, n_outputs):
        super(RegressionNet, self).__init__()

        # Use alexnet as a pretrained model. We want to adapt this
        # so that we can choose multiple pretrained nets
#        self.model = torchvision.models.alexnet(pretrained=True)
#         self.model = torchvision.models.vgg16_bn(pretrained=True)
        self.model  = torchvision.models.resnet18(pretrained=False)
#         self.model = torchvision.models.densenet121(pretrained=True)
        # Freeze alexnet
#        for param in self.model.parameters():
 #           param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, n_outputs))
#                             nn.ReLU(),
#                             nn.Linear(100,n_outputs)) 
        # Custom network appended to pretrained network. 
#        num_ftrs =self.model.classifier[6].in_features
#        self.model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, n_outputs)) 
 
  

    def forward(self, x):
        return self.model(x)


class LeftRightNet(nn.Module):
    def __init__(self):
        super(LeftRightNet, self).__init__()

        # Use alexnet as a pretrained model. We want to adapt this
        # so that we can choose multiple pretrained nets
#        self.model = torchvision.models.alexnet(pretrained=True)
#         self.model = torchvision.models.vgg16_bn(pretrained=True)
        self.model  = torchvision.models.resnet18(pretrained=True)
#         print("This is resnet", self.model)
    
#         self.model = torchvision.models.densenet121(pretrained=True)
        # Freeze alexnet
#        for param in self.model.parameters():
 #           param.requires_grad = False
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 1))
#                             nn.ReLU(),
#                             nn.Linear(100,n_outputs)) 
        # Custom network appended to pretrained network. 
#        num_ftrs =self.model.classifier[6].in_features
#        self.model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, n_outputs)) 
 
  

    def forward(self, x):
        return self.model(x)