
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
from templates_video import video_64_autoenc


def encode_stochastic(model, x, cond, flows, T=None):
    if T is None:
        sampler = model.eval_sampler
    else:
        sampler = model.conf._make_diffusion_conf(T).make_sampler()
    out = sampler.ddim_reverse_sample_loop(model.model,
                                           x,
                                           flows,
                                           model_kwargs={'cond': cond})
    return out


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
conf = video_64_autoenc()
print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/semanticof/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.model.to(device)


data = ShanghaiTechDataset(path="/home/eprakash/shanghaitech/testing/test_list.txt", image_size=conf.img_size,
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
        frame_batch = torch.zeros(size=(batch_len, 3, F, conf.img_size, conf.img_size), device=device)
        flows = torch.zeros(size=(batch_len, 9, 2, conf.img_size*2, conf.img_size*2), device=device)
        for i in range(batch_len):
            frame_batch[i] = data[batch_size * b + i]['img'][None]
            flows[i] = data[batch_size * b + i]['flows'][None]
            #if (i % 10 == 0):
                #print("Done " + str(i))
        #print("Encoding...")
        cond = model.encode(frame_batch, flows)
        #name = 'st_anomaly_semantic_128/'+ str(b) + '.txt'
        #with open(name, 'wb') as f:
            #    np.save(f, cond.clone().cpu().detach().numpy())
        #seed = np.random.randint(0, 1000000)
        #torch.manual_seed(seed)
        #xT = torch.from_numpy(np.load("avg_st_train_encoding_256_4_full.txt")[None, :]).to(device).repeat(batch_len, 1, 1, 1, 1)
        xT = encode_stochastic(model, frame_batch, cond, flows, T=None)
        xT_all = xT['sample_t']
        xT = xT['sample']
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

        ori = (frame_batch + 1) / 2
        ori_tensor = ori.clone().cpu()
        for xT_iter, xT in enumerate(xT_all):
            #print("Decoding...")
            pred = model.render(xT, cond, T=None)
            pred_tensor = pred.clone().cpu()

            #print("Calculating losses...")
            abs_diff = torch.abs(pred.to(device) - ori.to(device))
            abs_diff = torch.flatten(abs_diff, start_dim=1)
            top = torch.topk(abs_diff, 10000, dim=1)[0]
            mean_diff = torch.mean(top, dim=1)
            with open(f"semantic_optical_flow_diffs_{cuda_device}_{xT_iter}.log", "a") as fp:
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
            with open(f"semantic_optical_flow_mse_{cuda_device}_{xT_iter}.log", "a") as fp:
                fp.write("Batch scores: ")
                batch_mean = torch.mean(diff, dim=1)
                for i in range(len(batch_mean)):
                    fp.write(str(batch_mean[i].item()) + "|")
                fp.write("\n")
            scores = 10 * torch.log10(torch.div(1, diff))
            min_scores = torch.min(scores, dim=1)[0]
            max_scores = torch.max(scores, dim=1)[0]
            score = torch.div(torch.subtract(scores[:, 4], min_scores), torch.subtract(max_scores, min_scores))
            with open(f"semantic_optical_flow_scores_{cuda_device}_{xT_iter}.log", "a") as fp:
                fp.write("Batch scores: ")
                for i in range(len(score)):
                    fp.write(str(score[i].item()) + "|")
                fp.write("\n")
        #print("Done batch " +  str(b))
    #print("Average loss: " +  str(np.mean(avg_loss)))
print("DONE!")
