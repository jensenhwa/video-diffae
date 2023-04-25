import torch
from torchvision import models
from torch import nn
from torchvision import transforms

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

class ResNet50EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

        resnet.blocks[5].proj = torch.nn.Identity()
        #resnet.blocks[5].output_pool= torch.nn.Identity()

        for param in resnet.parameters():
            param.requires_grad = True
        self.model = resnet
        
        self.transform = transforms.Compose([transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)), transforms.Resize((256, 256)), transforms.CenterCrop(256)])
        
        #self.summarize = torch.nn.Linear(in_features=2048, out_features=1024, bias=True)
        #self.summarize.requires_grad = True

    def forward(self, x: torch.Tensor):
        #print(x.shape)
        x = (x + 1) / 2
        unflatten_shape = (x.shape[0], x.shape[2])
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()
        x = self.transform(x).unflatten(0, unflatten_shape).permute(0, 2, 1, 3, 4)
        #print(x.shape)
        out = self.model.forward(x)
        #out = self.summarize(out)
        #print(out.shape)
        return out

'''
model = ResNet50EncoderModel()
print(model.forward(torch.ones(8, 3, 9, 64, 64)).shape)
'''
