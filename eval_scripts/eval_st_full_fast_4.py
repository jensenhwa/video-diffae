
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision import models
from tqdm import tqdm
import sys

from dataset_video import ShanghaiTechDataset
from experiment import LitModel
from templates_video import video_autoenc

torch.cuda.empty_cache()

NUM_GPUS = 4 # Number of total GPUs being used, should be set to 1 if you are running python eval_st_anomaly.py with no args
batch_size = 4 # 64 is fastest?
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
state = torch.load(f'checkpoints/opticalflow_bw2/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

conf.img_size = 128
data = ShanghaiTechDataset(path="/home/eprakash/shanghaitech/testing/test_list.txt",
                           flow_path='/home/jy2k16/video-diffae/test_raw_flows_pt',
                           image_size=128,
                           cf_stride=False)
num_batches = int(len(data)/batch_size) + 1
start_batch = (cuda_device * num_batches) // NUM_GPUS # Floored lower bound
end_batch = ((cuda_device + 1) * num_batches) // NUM_GPUS # Floored upper bound
print(str(cuda_device) + ": processing {} batches from idx {} to idx {}".format(num_batches, start_batch, end_batch))
#import pdb; pdb.set_trace()
avg_loss = []

F = 9

with torch.no_grad():
    for b in tqdm(range(start_batch, end_batch)):
        #print("Building batch...")
        batch_len = batch_size
        if (b == num_batches - 1):
            batch_len = len(data) - batch_size * b
        frame_batch = torch.zeros(size=(batch_len, 3, F, 128, 128), device=device)
        for i in range(batch_len):
            frame_batch[i] = data[batch_size * b + i]['img'][None]
            #if (i % 10 == 0):
                #print("Done " + str(i))
        #print("Encoding...")
        cond = model.encode(frame_batch)
        #name = 'st_anomaly_semantic_128/'+ str(b) + '.txt'
        #with open(name, 'wb') as f:
            #    np.save(f, cond.clone().cpu().detach().numpy())
        #seed = np.random.randint(0, 1000000)
        #torch.manual_seed(seed)
        #xT = torch.from_numpy(np.load("avg_st_train_encoding_256_4_full.txt")[None, :]).to(device).repeat(batch_len, 1, 1, 1, 1)
        xT = model.encode_stochastic(frame_batch, cond, T=None)
        #xT = xT.permute((2, 0, 1, 3, 4))
        #xT = torch.mean(xT, dim=0)[None, :]
        #xT = xT.repeat(9, 1, 1, 1, 1) 
        #xT = xT.permute((1, 2, 0, 3, 4)).to(device)
        #print(xT.shape)
        #name = 'st_anomaly_stochastic_256/'+ str(b) + '.txt'
        #with open(name, 'wb') as f:
        #    np.save(f, xT.clone().cpu().detach().numpy())
        #xT_2 = torch.from_numpy(np.load("dummy_mnist_train_encoding.txt")).to(device).repeat(batch_len, 1, 1, 1, 1)#.view(batch_len, 3, 9, conf.img_size, conf.img_size)
        #xT = (xT_1 + xT_2)/2
        #torch.randn(batch_len, 3, 9, conf.img_size, conf.img_size).to(device)#model.encode_stochastic(frame_batch.to(device), cond, T=250)

        #print("Decoding...")
        pred = model.render(xT, cond, T=None)
        ori = (frame_batch + 1) / 2
        ori_tensor = ori.clone().cpu()
        pred_tensor = pred.clone().cpu()

        #print("Calculating losses...")
        abs_diff = torch.abs(pred.to(device) - ori.to(device))
        abs_diff = torch.flatten(abs_diff, start_dim=1)
        top = torch.topk(abs_diff, 10000, dim=1)[0]
        mean_diff = torch.mean(top, dim=1)
        with open("st_test_diffs_optical_flow_bw_exp_multi_gpu_{}.log".format(cuda_device), "a") as fp:
            fp.write("Batch diffs: ")
            for i in range(len(mean_diff)):
                fp.write(str(mean_diff[i].item()) + "|")
            fp.write("\n")
        #print(mean_diff)
        diff = torch.square(pred.to(device) - ori.to(device))
        diff = diff.reshape(batch_len, F, 3, conf.img_size, conf.img_size)
        diff_flat = torch.flatten(diff, start_dim=2)
        #diff = torch.mean(torch.topk(diff_flat, int(3*128*128*0.10), dim=2)[0], dim=2)
        diff = torch.mean(diff_flat, dim=2)
        with open("st_test_mse_optical_flow_bw_exp_multi_gpu_{}.log".format(cuda_device), "a") as fp:
            fp.write("Batch scores: ")
            batch_mean = torch.mean(diff, dim=1)
            for i in range(len(batch_mean)):
                fp.write(str(batch_mean[i].item()) + "|")
            fp.write("\n")
        scores = 10 * torch.log10(torch.div(1, diff))
        min_scores = torch.min(scores, dim=1)[0]
        max_scores = torch.max(scores, dim=1)[0]
        score = torch.div(torch.subtract(scores[:, 4], min_scores), torch.subtract(max_scores, min_scores))
        with open("st_test_scores_optical_flow_bw_exp_multi_gpu_{}.log".format(cuda_device), "a") as fp:
            fp.write("Batch scores: ")
            for i in range(len(score)):
                fp.write(str(score[i].item()) + "|")
            fp.write("\n")
        #print("Done batch " +  str(b))
    #print("Average loss: " +  str(np.mean(avg_loss)))
print("DONE!")
