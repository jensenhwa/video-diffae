from templates_video import *
from dataset_video import *
from experiment import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
import cv2

DATASET = "/home/eprakash/shanghaitech/testing/small_obj_test_list.txt"
IDX = 1549

device = 'cuda:0'
conf = video_64_autoenc()
model = LitModel(conf)

state = torch.load(f'/home/evaprakash/video-diffae/checkpoints/resnet18_normfix/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.model.to(device)

data = ShanghaiTechDataset(path=DATASET, need_eval=True, image_size=conf.img_size, cf_stride=False)
frame_batch = data[IDX]['img'][None]
frame_batch_ori = data[IDX]['img_resnet'][None]

print("Encoding...")
cond = model.encode(frame_batch_ori.to(device))

#Use random semantic subcode
#seed = np.random.randint(0, 1000000)
#torch.manual_seed(seed)
#cond = torch.randn(1, 512, device=device)

xT = model.encode_stochastic(x=frame_batch.to(device), cond=cond, T=2)

#Use random stochastic encoding
#xT = torch.randn(1, 3, 9, conf.img_size, conf.img_size, device=device)

print("Decoding...")
pred = model.render(noise=xT, cond=cond, T=None)

print("Plotting...")
F = 9
fig, ax = plt.subplots(1, F, figsize=(F * 5, 5))
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

print("DONE!")
