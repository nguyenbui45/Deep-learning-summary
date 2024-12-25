import torch

weight = torch.tensor([0.1234, 0.6990, -0.7890, 0.0012],dtype=torch.float32)
print(weight)
#->tensor([ 0.1234,  0.6990, -0.7890,  0.0012]) 

scale = 255/(torch.max(weight) - torch.min(weight))
zeropoint = -(scale*torch.min(weight)).round() - 128

quantized_weight = (scale*weight + zeropoint).round().to(torch.int8)
print(quantized_weight)
#->tensor([  28,  127, -128,    7], dtype=torch.int8) 

dequantized_weight = (quantized_weight - zeropoint)/ scale
print(dequantized_weight)
#->tensor([ 0.1225,  0.7002, -0.7878,  0.0000]) 