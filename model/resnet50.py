import torch
from torchvision import models
from torch import nn
from torchvision import transforms


class ResNet50EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        
        resnet.blocks[5].proj = torch.nn.Identity() 
        #resnet.blocks[5].output_pool= torch.nn.Identity() 

        for param in resnet.parameters():
            param.requires_grad = True
        self.model = resnet
        
        self.summarize = torch.nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.summarize.requires_grad = True

    def forward(self, x: torch.Tensor):
        out = self.model.forward(x)
        out = self.summarize(out)
        return out

'''
model = ResNet50EncoderModel()
print(model.forward(torch.ones(1, 3, 9, 256, 256)).shape)
'''
