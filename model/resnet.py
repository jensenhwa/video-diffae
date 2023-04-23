import torch
from torchvision import models
from torch import nn
from torchvision.models.video import R3D_18_Weights


class ResNetEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
<<<<<<< HEAD
        resnet18 = models.video.r3d_18('DEFAULT')
=======
        weights = R3D_18_Weights.DEFAULT
        resnet18 = models.video.r3d_18(weights=weights)
>>>>>>> 15c8ca676914fb0652d5a21279bb56ddf900fb72
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        for param in resnet18.parameters():
            param.requires_grad = True
        self.model = resnet18

        self.transform_resnet = weights.transforms()

    def forward(self, x: torch.Tensor):
        x = (x + 1) / 2  # Undo the 0.5,0.5 normalization used by the default UNet diffae. [-1., 1.] -> [0., 1.]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H, W) -> (N, T, C, H, W)
        x = self.transform_resnet(x)  # (N, T, C, H, W) -> (N, C, T, H, W)
        out = self.model(x)
        out = out.reshape(out.shape[0], out.shape[1])
        return out  # (N, 512)
