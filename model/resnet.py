import torch
from torchvision import models
from torch import nn
from torchvision.models.video import R3D_18_Weights


class ResNetEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT
        resnet18 = models.video.r3d_18(weights=weights)
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        for param in resnet18.parameters():
            param.requires_grad = True
        self.model = resnet18

        self.transform_resnet = weights.transforms()
        
        self.flow_model = nn.Sequential(nn.Conv2d(2, 2, 2, stride=2), nn.Conv2d(2, 2, 2, stride=2), nn.Conv2d(2, 2, 2, stride=2), nn.Conv2d(2, 2, 2, stride=2))
        
        for param in self.flow_model.parameters():
            param.requires_grad = True

        self.downsize_flow = nn.Linear(9*2*8*8, 512)
        self.downsize_flow.requires_grad = True


    def forward(self, x: torch.Tensor, flows: torch.Tensor):
        batch_size = x.shape[0]
        x = (x + 1) / 2  # Undo the 0.5,0.5 normalization used by the default UNet diffae. [-1., 1.] -> [0., 1.]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H, W) -> (N, T, C, H, W)
        x = self.transform_resnet(x)  # (N, T, C, H, W) -> (N, C, T, H, W)
        out = self.model(x)
        out = out.reshape(out.shape[0], out.shape[1])
        out_flow = self.flow_model(torch.flatten(flows, start_dim=0, end_dim=1))
        out_flow = torch.flatten(out_flow, start_dim=1)
        out_flow = out_flow.reshape((batch_size, 9, out_flow.shape[1]))
        out_flow = torch.flatten(out_flow, start_dim=1)
        out_flow = self.downsize_flow(out_flow)
        out = torch.cat((out, out_flow), dim=1)
        return out  # (N, 1024)
