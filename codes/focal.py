import numpy as np
import torch
import torch.nn.functional as F


g_cpu = torch.Generator()
g_cpu.manual_seed(12)
predicts = torch.randn(4,10,generator=g_cpu)
labels = torch.LongTensor([2,3,0,1])
gamma = 2 # focusing factor
alpha = 0.25  # Balancing factor


# Calculate the softmax of logits
output_probs = F.softmax(predicts,dim=1)
print(output_probs)
print(labels)

# Gather the true class probability 
true_class_probs = output_probs.gather(1,labels.view(-1,1))
print(true_class_probs)

# Calculate the focal loss
loss = torch.mean(-torch.log(true_class_probs))
focal_loss = torch.mean(-alpha*torch.pow((1-true_class_probs),gamma)*torch.log(true_class_probs))

print(loss)
print(focal_loss)


