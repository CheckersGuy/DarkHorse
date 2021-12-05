import struct
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import sys
import Helper as h


class TrainData2(Dataset):
    def __init__(self, path, transform=None):
        super(TrainData2, self).__init__()
        self.transform = transform
        temp = np.fromfile(path,
                           dtype=[('color', 'i4'), ('WP', 'i4'), ('BP', 'i4'), ('K', 'i4'), ('key', 'i8'),
                                  ('result', 'i4'), ('move', 'i4')])
        indices = [i for i in range(0, len(temp)) if temp[i]["move"] != -1]

        removed = len(temp) - len(indices)
        print("We removed ", removed, " from ", len(temp), " positions")
        self.data = temp[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        point = self.data[index]
        if self.transform is not None:
            return self.transform(point["WP"], point["BP"], point["K"], point["color"], point["move"])

        return point["WP"], point["BP"], point["K"], point["color"], point["move"]
