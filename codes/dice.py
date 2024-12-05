import numpy as np
import torch
import torch.nn.functional as F


g_cpu = torch.Generator()
g_cpu.manual_seed(12)
predicts = torch.randn(4,10,generator=g_cpu)
labels = torch.LongTensor([2,3,0,1])
num_classes = 10
eps = 1e-7

# Calculate the softmax of logits
output_probs = F.softmax(predicts,dim=1)
preds = torch.argmax(output_probs,dim=1)
print(preds)
preds_flat = preds.view(-1)
dice_score=[]

for i in range(num_classes):
    true_classes = (labels == i).float()
    print(true_classes)
    preds_classes = (preds_flat == i).float()
    print(preds_classes)

    intersection = torch.sum(true_classes*preds_classes)
    union = torch.sum(true_classes) + torch.sum(preds_classes)
    
    dice = (2.0 * intersection) / (union + eps)
    dice_score.append(dice)
    
dice_loss = 1-torch.tensor(dice_score).mean()
print(dice_loss)
