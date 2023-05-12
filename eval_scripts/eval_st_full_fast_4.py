import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
from tqdm import tqdm
import sys

from dataset_video import ShanghaiTechDataset
from experiment import LitModel
from metrics import psnr
from ssim import ssim
from templates_video import video_autoenc

torch.cuda.empty_cache()

NUM_GPUS = 4  # TODO: Number of total GPUs being used, should be set to 1 if you are running python eval_st_anomaly.py with no args
batch_size = 4  # TODO: 64 is fastest?
cuda_device = 0
if len(sys.argv) > 1:
    cuda_device = int(sys.argv[1])
else:
    NUM_GPUS = 1
print(NUM_GPUS)
device = 'cuda:' + str(cuda_device)
'''
if (cuda_device == 2 or cuda_device == 3):
    sys.exit()
'''
conf = video_autoenc()

print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/opticalflow_bbox/last.ckpt', map_location=device)  # TODO: Set model checkpoint path
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

conf.img_size = 128
data = ShanghaiTechDataset(path="/home/eprakash/shanghaitech/testing/test_list.txt",
                           flow_path='/home/eprakash/diffae/test_raw_flows_masked',  # TODO
                           image_size=128,
                           cf_stride=False)
num_batches = int(len(data) / batch_size) + 1
start_batch = (cuda_device * num_batches) // NUM_GPUS  # Floored lower bound
end_batch = ((cuda_device + 1) * num_batches) // NUM_GPUS  # Floored upper bound
print(str(cuda_device) + ": processing {} batches from idx {} to idx {}".format(num_batches, start_batch, end_batch))
# import pdb; pdb.set_trace()
avg_loss = []

F = 9
timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M')

with torch.no_grad():
    for b in tqdm(range(start_batch, end_batch)):
        # print("Building batch...")
        batch_len = batch_size
        if (b == num_batches - 1):
            batch_len = len(data) - batch_size * b
        frame_batch = torch.zeros(size=(batch_len, 3, F, 128, 128), device=device)
        frame_batch_ori = torch.zeros(size=(batch_len, 3, F, conf.img_size, conf.img_size), device=device)
        for i in range(batch_len):
            img_i = data[batch_size * b + i]['img'][None]
            frame_batch[i] = img_i
            frame_batch_ori[i] = (img_i + 1) / 2

        cond = model.encode(frame_batch)
        xT = model.encode_stochastic(frame_batch, cond, T=None)

        # Decoding
        pred = model.render(xT, cond, T=None).permute(0, 2, 1, 3, 4)
        ori = frame_batch_ori.permute(0, 2, 1, 3, 4)

        # Calculating new metrics
        for j in range(batch_len):
            ssim_scores = []
            psnr_scores = []
            lpips_scores = []
            mse_scores = []
            for f in range(F - 1):
                ssim_scores.append(ssim(ori[j, i:i + 1], pred[j, i:i + 1], size_average=False).item())
                psnr_scores.append(psnr(ori[j, i:i + 1], pred[j, i:i + 1]).item())
                lpips_scores.append(0)  # lpips_fn.forward(ori[j, i:i+1], pred[j, i:i+1]).view(-1).item())
                mse_scores.append((ori[j, i:i + 1] - pred[j, i:i + 1]).pow(2).mean(dim=[1, 2, 3]).item())

            methods = {"ssim": np.mean(ssim_scores), "psnr": np.mean(psnr_scores), "lpips": np.mean(lpips_scores),
                       "mse": np.mean(mse_scores)}
            for m in methods:
                with open(f"{timestamp}_{m}_obj_{cuda_device}.log", "a") as fp:
                    if (j == 0):
                        fp.write("Batch results: ")
                    fp.write(str(methods[m]) + "|")
                    if (j == (batch_len - 1)):
                        fp.write("\n")

        ori = ori.permute(0, 2, 1, 3, 4)
        pred = pred.permute(0, 2, 1, 3, 4)

        # Compute one-class semantic encoding differences
        # path = Path(f"/home/eprakash/video-diffae/checkpoints/opticalflow_final/latent.pkl")
        # c = torch.load(path)['conds_mean']
        #
        # dists = torch.sum((cond - c.to(device)) ** 2, dim=1)
        # with open(f"{timestamp}_jh_score_{cuda_device}.log", "a") as fp:
        #     fp.write("Batch diffs: ")
        #     for i in range(len(dists)):
        #         fp.write(str(dists[i].item()) + "|")
        #     fp.write("\n")

        pred = model.render(xT, cond, T=None)
        ori = (frame_batch + 1) / 2

        # print("Calculating losses...")
        abs_diff = torch.abs(pred.to(device) - ori.to(device))
        abs_diff = torch.flatten(abs_diff, start_dim=1)
        top = torch.topk(abs_diff, 10000, dim=1)[0]
        mean_diff = torch.mean(top, dim=1)
        with open(f"{timestamp}_st_test_diffs_optical_flow_bw_exp_multi_gpu_{cuda_device}.log", "a") as fp:
            fp.write("Batch diffs: ")
            for i in range(len(mean_diff)):
                fp.write(str(mean_diff[i].item()) + "|")
            fp.write("\n")
        # print(mean_diff)
        diff = torch.square(pred.to(device) - ori.to(device))
        diff = diff.reshape(batch_len, F, 3, conf.img_size, conf.img_size)
        diff_flat = torch.flatten(diff, start_dim=2)
        # diff = torch.mean(torch.topk(diff_flat, int(3*128*128*0.10), dim=2)[0], dim=2)
        diff = torch.mean(diff_flat, dim=2)
        with open(f"{timestamp}_st_test_mse_optical_flow_bw_exp_multi_gpu_{cuda_device}.log", "a") as fp:
            fp.write("Batch scores: ")
            batch_mean = torch.mean(diff, dim=1)
            for i in range(len(batch_mean)):
                fp.write(str(batch_mean[i].item()) + "|")
            fp.write("\n")
        scores = 10 * torch.log10(torch.div(1, diff))
        min_scores = torch.min(scores, dim=1)[0]
        max_scores = torch.max(scores, dim=1)[0]
        score = torch.div(torch.subtract(scores[:, 4], min_scores), torch.subtract(max_scores, min_scores))
        with open(f"{timestamp}_st_test_scores_optical_flow_bw_exp_multi_gpu_{cuda_device}.log", "a") as fp:
            fp.write("Batch scores: ")
            for i in range(len(score)):
                fp.write(str(score[i].item()) + "|")
            fp.write("\n")
        # print("Done batch " +  str(b))
    # print("Average loss: " +  str(np.mean(avg_loss)))
print("DONE!")
