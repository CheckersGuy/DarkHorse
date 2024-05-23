import torch
import numpy as np

matrix = torch.randn(3,6)
indices = torch.from_numpy(np.array([[False,False,False,True,True,True],
           [False,False,False,True,True,True],
           [False,False,False,True,True,True]]));

matrix.masked_fill_(~indices,-1000)
print(matrix)
