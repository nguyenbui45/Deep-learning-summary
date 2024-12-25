import torch

weight = torch.tensor([0.1234, 0.6990, -0.7890, 0.0012],dtype=torch.float32)
print(weight)
#->tensor([0.1234, 0.6990, 0.7890, 0.0012])

scale = 127/torch.max(torch.abs(weight))
quantized_weight = (weight*scale).round().to(torch.int8)
print(quantized_weight)
#->tensor([  20,  113, -127,    0], dtype=torch.int8)

dequanited_weight = quantized_weight/scale
print(dequanited_weight)
#-tensor([ 0.1243,  0.7020, -0.7890,  0.0000])>