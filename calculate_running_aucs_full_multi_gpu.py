import numpy as np
import math
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from torchmetrics import AUROC
from torchmetrics import AveragePrecision
from collections import defaultdict

losses = []
labels = []

def find_closest_center_frame(idx, centerframe_to_idx):
   return centerframe_to_idx[min(centerframe_to_idx.keys(),key=lambda i:abs(i-idx))]
ADD_LIST = [0, 1693, 3387, 5080]
ADD_IDX = 0
ADD = ADD_LIST[ADD_IDX]
NUM_GPUS = 4
for i in range(NUM_GPUS):
    #with open("st_test_scores_256_emb_enc_nc_16_multi_gpu_2.log".format(i), "r") as fp:
    with open("resnet50_scores_obj_{}.log".format(i), "r") as fp:
        for line in fp:
            numbers = line.strip().split(" ")[2]
            numbers = numbers.split("|")[:-1]
            for number in numbers:
                losses.append(float(number))
                #losses.append(10 * np.log10(1/float(number)))
losses = np.array(losses)
print("Num examples: ", str(len(losses)))

#Load test set labels
with open("/home/eprakash/shanghaitech/testing/obj_test_labels_list.txt", "r") as fp:
    for line in fp:
        label = int(line.strip())
        labels.append(label)
test_len = len(labels)

#Load centerframes
idx_to_centerframe = defaultdict(int)
centerframe_to_idx = defaultdict(int)
with open("/home/eprakash/diffae/idx_to_centerframe_obj.txt", "r") as fp:
    for line in fp:
        i, idx = line.strip().split(",")
        i = int(i)
        idx = int(idx)
        idx_to_centerframe[i] = idx
        centerframe_to_idx[idx] = i

#Load SOTA scores
#scores = np.load("scores/final_test_scores.npy")
#scores = np.load("scores/final_deep_features_scores.npy")
#scores = np.load("scores/final_pose_scores.npy")
#scores = np.load("scores/final_velocity_scores.npy")

#Find running labels and SOTA scores for centerframes
curr_labels = []
#curr_scores = []
normal_scores = []
anomaly_scores = []
zeroes = np.where(losses == 0)[0]
ones = np.where(losses == 1)[0]
best_anomaly_idxs = []
worst_anomaly_idxs = []
best_normal_idxs = []
worst_normal_idxs = []
for n in range(len(losses)):
    label = labels[idx_to_centerframe[n + ADD]]
    if (label == 1):
        anomaly_scores.append(losses[n])
        if n in zeroes:
            best_anomaly_idxs.append((n, losses[n], idx_to_centerframe[n]))
        if n in ones:
            worst_anomaly_idxs.append((n, losses[n], idx_to_centerframe[n]))
    else:
        normal_scores.append(losses[n])
        if n in zeroes:
            worst_normal_idxs.append((n, losses[n], idx_to_centerframe[n]))
        if n in ones:
            best_normal_idxs.append((n, losses[n], idx_to_centerframe[n]))
    label = label != 1
    curr_labels.append(label)
    #curr_scores.append(1 - scores[idx_to_centerframe[n]])
#print("Best anomaly idxs: ", best_anomaly_idxs, len(best_anomaly_idxs))
#print("Worst anomaly idxs: ", worst_anomaly_idxs, len(worst_anomaly_idxs))
#print("Best normal idxs: ", best_normal_idxs, len(best_normal_idxs))
#print("Worst normal idxs: ", worst_normal_idxs, len(worst_normal_idxs))
labels = np.array(curr_labels)
#curr_scores = np.array(curr_scores)
print("Anomaly mean score: ", np.mean(anomaly_scores), ", normal mean score: ", np.mean(normal_scores))
print("Percentage of anomalies: ", str(len(anomaly_scores)/len(losses)))
#losses = (losses + curr_scores)/2
#losses = curr_scores
threshold = 0.5
false_negatives = len(np.where(np.array(anomaly_scores) > threshold)[0])/len(losses)
false_positives = len(np.where(np.array(normal_scores) <= threshold)[0])/len(losses)
true_negatives = len(np.where(np.array(normal_scores) > threshold)[0])/len(losses)
true_positives = len(np.where(np.array(anomaly_scores) <= threshold)[0])/len(losses)

print("False negatives: ", false_negatives, " False positives: ", false_positives, " True negatives: ", true_negatives, " True positives: ", true_positives)

#Load edge frames
edge_frame_idxs = list(set(idx_to_centerframe.values()) ^ set(range(test_len)))
edge_centerframe_idxs = [find_closest_center_frame(idx, centerframe_to_idx) for idx in edge_frame_idxs] 
edge_centerframe_losses = np.array([losses[idx] for idx in edge_centerframe_idxs])
edge_centerframe_labels = np.array([labels[idx] for idx in edge_centerframe_idxs])
print("Percentage edge frames: ", str(len(edge_centerframe_losses)/(len(losses) + len(edge_centerframe_losses))))
losses = np.concatenate((losses, edge_centerframe_losses), axis=0)
labels = np.concatenate((labels, edge_centerframe_labels), axis=0)

#Calculate AUCs
print(losses)
print(labels)
auroc = roc_auc_score(labels, losses)
auprc = average_precision_score(labels, losses)
print(len(losses), len(labels))
print("AUROC: " + str(auroc))
print("AUPRC: " + str(auprc))
