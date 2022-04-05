import torch.nn as nn
import torch
import torchmetrics

import Helper as h
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ranger import Ranger
import struct
import ctypes
import pathlib
import numpy as np

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)


class Relu1(nn.Module):
    def __init__(self):
        super(Relu1, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)




class Network(pl.LightningModule):

    def __init__(self, hidden, output="form_network3.weights"):
        super(Network, self).__init__()
        layers = []
        self.output = output
        for i in range(len(hidden) - 2):
            layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            layers.append(Relu1())

        layers.append(nn.Linear(hidden[len(hidden) - 2], hidden[len(hidden) - 1]))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.criterion = torch.nn.MSELoss()
        self.init_weights()
        print(self.net)

    def forward(self, x):
        return self.net.forward(x)

    def on_epoch_end(self) -> None:
        self.save_parameters(self.output)

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters())
        return optimizer

    def training_step(self, train_batch, batch_idx):
        result, move, x = train_batch
        out = self.forward(x)
        loss = self.criterion(out, result)
        tensorboard_logs = {"avg_val_loss": loss}
        self.log('train_loss', loss)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        result, move, x = val_batch
        out = self.forward(x)
        loss = self.criterion(out, result)
        self.log('val_loss', loss.detach())
        return {"val_loss": loss.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"loss": avg_loss, "log": tensorboard_logs}

    def init_weights(self):
        self.net.apply(init_weights)

    def save_parameters(self, output):
        buffer_weights = bytearray()
        buffer_bias = bytearray()
        num_weights = 0
        num_bias = 0
        file = open(output, "wb")
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().numpy().flatten("F")
                bias = layer.bias.detach().numpy().flatten("F")
                buffer_weights += weights.tobytes()
                buffer_bias += bias.tobytes()
                num_weights += len(weights)
                num_bias += len(bias)

        file.write(struct.pack("I", num_weights))
        file.write(buffer_weights)
        file.write(struct.pack("I", num_bias))
        file.write(buffer_bias)
        file.close()

        return

    def save(self, output):
        torch.save(self.state_dict(), output)


class PolicyNetwork(pl.LightningModule):

    def __init__(self, hidden, output="policy.weights"):
        super(PolicyNetwork, self).__init__()
        layers = []
        self.output = output
        for i in range(len(hidden) - 2):
            layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            layers.append(Relu1())

        layers.append(nn.Linear(hidden[len(hidden) - 2], hidden[len(hidden) - 1]))
        self.net = nn.Sequential(*layers)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.init_weights()
        self.accuracy = torchmetrics.Accuracy()
        print(self.net)

    def forward(self, x):
        return self.net.forward(x)

    def on_epoch_end(self) -> None:
        self.save_parameters(self.output)

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters())
        return optimizer

    def training_step(self, train_batch, batch_idx):
        result, move, x = train_batch
        out = self.forward(x)
        loss = self.criterion(out, move.squeeze())
        tensorboard_logs = {"avg_val_loss": loss}
        self.accuracy(out, move.squeeze())
        self.log('train_acc_step', self.accuracy)
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)

    def validation_step(self, val_batch, batch_idx):
        result, move, x = val_batch
        out = self.forward(x)
        self.accuracy(out, move.squeeze())
        self.log('val_loss', self.accuracy)
        return {"val_loss": self.accuracy}

    def validation_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.accuracy)


    def init_weights(self):
        self.net.apply(init_weights)

    def save_parameters(self, output):
        buffer_weights = bytearray()
        buffer_bias = bytearray()
        num_weights = 0
        num_bias = 0
        file = open(output, "wb")
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().numpy().flatten("F")
                bias = layer.bias.detach().numpy().flatten("F")
                buffer_weights += weights.tobytes()
                buffer_bias += bias.tobytes()
                num_weights += len(weights)
                num_bias += len(bias)

        file.write(struct.pack("I", num_weights))
        file.write(buffer_weights)
        file.write(struct.pack("I", num_bias))
        file.write(buffer_bias)
        file.close()

        return

    def save(self, output):
        torch.save(self.state_dict(), output)


class LitDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, buffer_size=1500000, batch_size=1000):
        super(LitDataModule, self).__init__()
        self.train_set = NetBatchDataSet(batch_size, buffer_size, train_data, is_val_set=False)
        self.val_set = NetBatchDataSet(batch_size, 1000000, val_data, is_val_set=True)
        self.train_data = train_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=None, batch_sampler=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=None, batch_sampler=None, shuffle=False)


class PattDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, buffer_size=1500000, batch_size=1000):
        super(PattDataModule, self).__init__()
        self.train_set = PattBatchDataSet(batch_size, buffer_size, train_data, is_val_set=False)
        self.val_set = PattBatchDataSet(batch_size, 1000000, val_data, is_val_set=True)
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
        libname = pathlib.Path().absolute() / "libpyhelper.so"
        self.c_lib = ctypes.CDLL(libname)
        if not is_val_set:
            temp = self.c_lib.init_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                            ctypes.c_char_p(self.file_path.encode('utf-8')), ctypes.c_bool(False))
        else:
            temp = self.c_lib.init_val_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                                ctypes.c_char_p(self.file_path.encode('utf-8')), ctypes.c_bool(False))
        self.file_size = temp

    def __iter__(self):
        return self

    def __len__(self):
        return self.file_size // self.batch_size


class NetBatchDataSet(BatchDataSet):

    def __init__(self, batch_size, buffer_size, file_path, is_val_set=False):
        super(NetBatchDataSet, self).__init__(batch_size, buffer_size, file_path, is_val_set)

    def __next__(self):
        results = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        moves = np.zeros(shape=(self.batch_size, 1), dtype=np.int64)
        inputs = np.zeros(shape=(self.batch_size, 120), dtype=np.float32)
        res_p = results.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        inp_p = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        moves_p = moves.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        if not self.is_val_set:
            self.c_lib.get_next_batch(res_p, moves_p, inp_p)
        else:
            self.c_lib.get_next_val_batch(res_p, moves_p, inp_p)

        return results, moves, inputs


class PattBatchDataSet(BatchDataSet):

    def __init__(self, batch_size, buffer_size, file_path, is_val_set=False):
        super(PattBatchDataSet, self).__init__(batch_size, buffer_size, file_path, True, is_val_set)

    def __next__(self):
        results = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        num_wp = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        num_bp = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        num_wk = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        num_bk = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)

        patt_op_small = np.zeros(shape=(self.batch_size, 9), dtype=np.int64)
        patt_end_small = np.zeros(shape=(self.batch_size, 9), dtype=np.int64)

        patt_op_big = np.zeros(shape=(self.batch_size, 6), dtype=np.int64)
        patt_end_big = np.zeros(shape=(self.batch_size, 6), dtype=np.int64)

        res_p = results.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        num_wp_p = num_wp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        num_bp_p = num_bp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        num_wk_p = num_wk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        num_bk_p = num_bk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        patt_op_small_p = patt_op_small.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        patt_end_small_p = patt_end_small.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        patt_op_big_p = patt_op_big.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        patt_end_big_p = patt_end_big.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        if not self.is_val_set:
            self.c_lib.get_next_batch_patt(res_p, num_wp_p, num_bp_p, num_wk_p, num_bk_p, patt_op_big_p, patt_end_big_p,
                                           patt_op_small_p,
                                           patt_end_small_p)
        else:
            self.c_lib.get_next_val_batch_patt(res_p, num_wp_p, num_bp_p, num_wk_p, num_bk_p, patt_op_big_p,
                                               patt_end_big_p,
                                               patt_op_small_p,
                                               patt_end_small_p)

        return results, num_wp, num_bp, num_wk, num_bk, patt_op_big, patt_end_big, patt_op_small, patt_end_small
