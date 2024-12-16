import torch

file = 'cross_valid_example/fold_1.pkl'
f = open(file,'rb')
data = torch.load(f,map_location='cpu')#可使用cpu或gpu
print(data)
