
import numpy as np
import torch.nn as nn
import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import struct
import ctypes
import pathlib

class Relu1(nn.Module):
    def __init__(self):
        super(Relu1, self).__init__()

    def forward(self, x):
        return (127.0/128.0)*torch.clamp(x,0.0,1.0)**2




  
class LitDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, buffer_size=1500000, batch_size=1000):
        super(LitDataModule, self).__init__()
        self.train_set = NetBatchDataSet(batch_size, buffer_size, train_data, False)
        self.val_set = NetBatchDataSet(batch_size, 1000000, val_data, True)
        self.train_data = train_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=None, batch_sampler=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=None, batch_sampler=None, shuffle=False)

class BatchDataSet(torch.utils.data.IterableDataset):

    def __init__(self, batch_size, buffer_size, file_path, is_val_set=False):
        super(BatchDataSet, self).__init__()
        self.batch_size = batch_size
        self.is_val_set = is_val_set
        self.buffer_size = buffer_size
        self.file_path = file_path
        libname = pathlib.Path().absolute().__str__() + "/libpyhelper.so"
        self.c_lib = ctypes.cdll.LoadLibrary(libname)
        print("Loaded library")
        if not is_val_set:
            temp = self.c_lib.init_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                            ctypes.c_char_p(self.file_path.encode('utf-8')))
        else:
            temp = self.c_lib.init_val_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                                ctypes.c_char_p(self.file_path.encode('utf-8')))
        self.file_size = temp
        print("Called initialization")

    def __iter__(self):
        return self

    def __len__(self):
        return self.file_size // self.batch_size


class NetBatchDataSet(BatchDataSet):

    def __init__(self, batch_size, buffer_size, file_path, is_val_set=False):
        super(NetBatchDataSet, self).__init__(batch_size, buffer_size, file_path, is_val_set)

    def __next__(self):
        input_size = 120
        results = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        #to be continued adding buckets
        moves = np.zeros(shape=(self.batch_size, 1), dtype=np.int64)
        buckets = np.zeros(shape=(self.batch_size, 1), dtype=np.int64)
        inputs = np.zeros(shape=(self.batch_size, input_size), dtype=np.float32)
        res_p = results.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        inp_p = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        moves_p = moves.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        buckets_p = buckets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        if not self.is_val_set:
            self.c_lib.get_next_batch(res_p, moves_p, buckets_p, inp_p)
        else:
            self.c_lib.get_next_val_batch(res_p, moves_p, buckets_p, inp_p)

        return results, moves,buckets, inputs


class WDLDataSet(BatchDataSet):

    def __init__(self, batch_size, buffer_size, file_path, is_val_set=False):
        super(WDLDataSet, self).__init__(batch_size, buffer_size, file_path, is_val_set)

    def __next__(self):
        input_size = 120
        results = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        #to be continued adding buckets
        moves = np.zeros(shape=(self.batch_size, 1), dtype=np.int64)
        buckets = np.zeros(shape=(self.batch_size, 1), dtype=np.int64)
        inputs = np.zeros(shape=(self.batch_size, input_size), dtype=np.float32)
        res_p = results.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        inp_p = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        moves_p = moves.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        buckets_p = buckets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


        if not self.is_val_set:
            self.c_lib.get_next_batch(res_p, moves_p, buckets_p, inp_p)
        else:
            self.c_lib.get_next_val_batch(res_p, moves_p, buckets_p, inp_p)

        #wdl-transform
        result = torch.Tensor(results)
        #to be continued
        won= torch.ones(size=(self.batch_size,1))
        draw = torch.ones(size=(self.batch_size,1))*0.5
        lost = torch.zeros(size=(self.batch_size,1))
        is_won = torch.eq(result,won).to(torch.int64)
        is_draw =torch.eq(result,draw).to(torch.int64)
        is_lost = torch.eq(result,lost).to(torch.int64)
        wdl_values = is_won*0+is_lost*1+is_draw*2
        return wdl_values, torch.LongTensor(moves), torch.LongTensor(buckets), torch.Tensor(inputs)
