from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms
from tqdm import tqdm


class ShanghaiTechDataset(Dataset):
    def __init__(self,
                 path,  # "/home/eprakash/shanghaitech/scripts/full_train_img_to_video.txt"
                 image_size=128,
                 original_resolution=128,
                 stride: int = 16,
                 # Stride between frames (1 = use all frames, 2 = skip every other frame, etc.)
                 cf_stride: bool = True,
                 use_flow: bool = True,):
        image_size = 128
        self.original_resolution = original_resolution
        self.use_flow = use_flow
        self.data, self.idx_to_vid, self.vid_to_idxs = self.load_data(path)
        self.stride = stride
        self.cf_stride = cf_stride
        self.frame_batch_size = 4  # Number of frames to the right and left of center frame
        self.img_size = image_size
        # Account for frames that cannot be used as center frame
        self.length = len(self.data) - len(self.vid_to_idxs) * self.stride * self.frame_batch_size * 2
        print(self.length)


        self.idx_to_centerframe = None
        self.get_center_frames()  # Set center frames to be used in __getitem__()

        self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.transform_flow = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4413), (0.2030)),
            ])

    def load_data(self, path):
        print("LOADING DATA...")
        i = 0
        data = []
        idx_to_vid = {}
        vid_to_idxs = defaultdict(list)
        with open(path, "r") as fp:
            for line in fp:
                img, vid = line.split(",")
                data.append(img)
                idx_to_vid[i] = vid
                vid_to_idxs[vid].append(i)
                i += 1
        print("FINISHED LOADING DATA!")
        """
        flow_data = {}
        i=0
        if self.use_flow:
            for file in tqdm(os.listdir('/home/jy2k16/diffae/train_raw_flows')):
                file_name = '/home/jy2k16/diffae/train_raw_flows/'+file
                flows = torch.from_numpy(np.load(file_name)).permute((2,0,1))
                flow_data[int(file.split("_")[1])] = flows
                i+=1
                if i == len(os.listdir('/home/jy2k16/diffae/train_raw_flows'))//2:
                    break
            print("FINISHED LOADING FLOW DATA!")
        """
            
        return data, idx_to_vid, vid_to_idxs

    def get_center_frames(self):
        # fp = open("idx_to_centerframe_8.txt", "w")
        self.idx_to_centerframe = {}
        i = 0

        for vid, idxs in self.vid_to_idxs.items():
            num_frames = len(idxs)
            start_idx = self.frame_batch_size * self.stride
            end_idx = num_frames - self.frame_batch_size * self.stride
            for idx in range(start_idx, end_idx):
                self.idx_to_centerframe[i] = idxs[idx]
                # fp.write(str(i) + "," + str(idxs[idx]) + "\n")
                i += 1
        print(self.length, i)
        # fp.close()
        if self.cf_stride:
            filtered_cf_idxs = np.load('/home/eprakash/diffae/cfs_16.npy')
            filtered_idx_to_centerframe = {}
            j = 0
            for idx in filtered_cf_idxs:
                filtered_idx_to_centerframe[j] = self.idx_to_centerframe[idx]
                j += 1
            self.idx_to_centerframe = filtered_idx_to_centerframe
            self.length = j
            print("Dataset length: ", str(self.length))
        else:
            self.length = i
            assert i == self.length  # Should fill up all usable indices
        return

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = self.idx_to_centerframe[index]
        vid = self.idx_to_vid[index]
        idxs = self.vid_to_idxs[vid]
        left_lim = index - self.frame_batch_size * self.stride
        right_lim = index + self.frame_batch_size * self.stride + 1  # + 1 since not inclusive

        assert left_lim >= min(idxs)
        assert right_lim <= max(idxs) + 1

        # if (left_lim < min(idxs)):
        #     right_lim += abs(min(idxs) - abs(left_lim))
        #     left_lim = min(idxs)
        #
        # if (right_lim > max(idxs) + 1):
        #     left_lim = left_lim - abs(max(idxs) + 1 - abs(right_lim))
        #     right_lim = max(idxs) + 1

        assert (right_lim - left_lim == self.frame_batch_size * self.stride * 2 + 1)

        frame_batch = []
        flows_batch = []

        for i in range(left_lim, right_lim, self.stride):
            frame_img_path = self.data[i]

            # dot = frame_img_path.rfind('.')
            # frame_img = torch.load(frame_img_path[:dot] + ".pt")
            if not self.use_flow:
                frame_img_orig = Image.open(frame_img_path).convert('RGB')
                frame_img = self.transform(frame_img_orig)
            else:
                frame_img_orig = Image.open(frame_img_path).convert('L')
                frame_img = self.transform_flow(frame_img_orig)
                #print('img shape',frame_img.shape)
                if f'flow_{i}_{i+self.stride}.pt' in os.listdir('/home/jy2k16/video-diffae/train_raw_flows_pt'):
                    file_name = f'/home/jy2k16/video-diffae/train_raw_flows_pt/flow_{i}_{i+self.stride}.pt'
                    flows = torch.load(file_name)
                    flows = transforms.Normalize((0.3,0.2462),(10.7345,6.0563))(flows)
                else:
                    flows = torch.zeros(2,128,128)
                flows_batch.append(flows)

            frame_batch.append(frame_img)

        frame_batch = torch.stack(frame_batch)
        if self.use_flow:
            flows_batch = torch.stack(flows_batch)
            frame_batch = torch.cat((frame_batch, flows_batch), dim=1)

        frame_batch = frame_batch.permute((1, 0, 2, 3))  # (C, T, H, W)
        return {'img': frame_batch, 'index': index}
