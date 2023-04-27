from templates_video import *
from dataset_video import *
from experiment import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
import cv2
from metrics import *
from metrics import psnr
from ssim import ssim
import lpips

DATASET = "/home/eprakash/shanghaitech/testing/small_obj_test_list.txt"
IDX = 1549

device = 'cuda:4'
conf = video_64_autoenc()
model = LitModel(conf)

state = torch.load(f'/home/eprakash/temp/video-diffae-git/video-diffae/checkpoints/resnet50/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.model.to(device)

data = ShanghaiTechDataset(path=DATASET, image_size=conf.img_size, cf_stride=False)
frame_batch = data[IDX]['img'][None]

print("Encoding...")
cond = model.encode(frame_batch.to(device))

#Use random semantic subcode
#seed = np.random.randint(0, 1000000)
#torch.manual_seed(seed)
#cond = torch.randn(1, 512, device=device)

xT = model.encode_stochastic(x=frame_batch.to(device), cond=cond, T=None)

#Use random stochastic encoding
#xT = torch.randn(1, 3, 9, conf.img_size, conf.img_size, device=device)

print("Decoding...")
pred = model.render(noise=xT, cond=cond, T=None)
print("Plotting...")
F = 9
fig, ax = plt.subplots(1, F, figsize=(F * 5, 5))
frame_batch_ori = (frame_batch + 1)/2
frame_batch_ori = frame_batch_ori.permute(0, 2, 1, 3, 4)

for i in range(F):
    img = frame_batch_ori[0][i]
    ax[i].imshow(img.permute(1, 2, 0).cpu())
plt.savefig("viz/shanghaitech_ori_ex.png")

fig, ax = plt.subplots(1, F, figsize=(F * 5, 5))
pred = pred.permute(0, 2, 1, 3, 4)
for i in range(F):
    img = pred[0][i]
    ax[i].imshow(img.permute(1, 2, 0).cpu())
plt.savefig("viz/shanghaitech_gen_ex.png")

#Metrics
lpips_fn = lpips.LPIPS(net='alex').to(device)
for i in range(F-1):
    print("SSIM: ", ssim(frame_batch_ori[0, i:i+1].to(device), pred[0, i:i+1].to(device), size_average=False))
    print("PSNR: ", psnr(frame_batch_ori[0, i:i+1].to(device), pred[0, i:i+1].to(device)))
    print("LPIPS: ", lpips_fn.forward(frame_batch_ori[0, i:i+1].to(device), pred[0, i:i+1].to(device)).view(-1))
    print("MSE: ", (frame_batch_ori[0, i:i+1].to(device) - pred[0, i:i+1].to(device)).pow(2).mean(dim=[1, 2, 3]))
    print("\n")

print("DONE!")
