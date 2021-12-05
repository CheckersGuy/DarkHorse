import struct
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import sys
import Helper as h


class TrainData(Dataset):
    def __init__(self, path, transform=None):
        super(TrainData, self).__init__()
        self.transform = transform
        temp = np.fromfile(path,
                           dtype=[('color', np.int32), ('WP', np.uint32), ('BP', np.uint32), ('K', np.uint32),
                                  ('key', np.uint64),
                                  ('result', np.int32), ('move', np.int32)])
        indices = [i for i in range(0, len(temp)) if not
        h.has_jumps(temp[i]["WP"], temp[i]["BP"], temp[i]["K"], temp[i]["color"])]

        removed = len(temp) - len(indices)
        print("We removed ", removed, " from ", len(temp), " positions")
        self.data = temp[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        point = self.data[index]
        if self.transform is not None:
            return self.transform(point["WP"], point["BP"], point["K"], point["color"], point["result"])

        return point["WP"], point["BP"], point["K"], point["color"], point["result"]
