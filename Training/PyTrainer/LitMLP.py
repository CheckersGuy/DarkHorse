import torch.nn as nn
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ranger import Ranger
import struct
import ctypes
import pathlib
import numpy as np
import torch.nn.functional as F


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)


class Relu1(nn.Module):
    def __init__(self):
        super(Relu1, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class ConvNet(pl.LightningModule):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU()
                                   )

        self.decoder = nn.Sequential(nn.Linear(128 * 4 * 8, 2048),nn.BatchNorm1d(2048), nn.ReLU(),nn.Linear(2048, 2048),nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, 1), nn.Sigmoid())

    def forward(self, x):
        # tensor has the form batchsizex8x4
        out = self.conv1(x)
        out_hat = out.view(x.size(0), -1)
        out_hat = self.decoder(out_hat)
        return out_hat

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        result, move, x = train_batch
        out = self.forward(x)
        loss = F.mse_loss(out, result)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        result, move, x = val_batch
        out = self.forward(x)
        loss = F.mse_loss(out, result)
        self.log('val_loss', loss)
        return loss


class Network(pl.LightningModule):

    def __init__(self, hidden, output="form_network7.weights"):
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        return [optimizer], [scheduler]

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
        # buffer_weights = bytearray()
        # buffer_bias = bytearray()
        # num_weights = 0
        # num_bias = 0
        # file = open(output, "wb")
        # for layer in self.net:
        #     if isinstance(layer, torch.nn.Linear):
        #         weights = layer.weight.detach().numpy().flatten("F")
        #         bias = layer.bias.detach().numpy().flatten("F")
        #         buffer_weights += weights.tobytes()
        #         buffer_bias += bias.tobytes()
        #         num_weights += len(weights)
        #         num_bias += len(bias)
        #
        # file.write(struct.pack("I", num_weights))
        # file.write(buffer_weights)
        # file.write(struct.pack("I", num_bias))
        # file.write(buffer_bias)
        # file.close()

        return

    def save(self, output):
        torch.save(self.state_dict(), output)


class PolicyNetwork(pl.LightningModule):

    def __init__(self, hidden, output="verypolicy.weights"):
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
        # 0.992
        optimizer = Ranger(self.parameters(), betas=(.9, 0.999), eps=1.0e-7, gc_loc=False, use_gc=False, lr=2e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.90)
        return [optimizer], [scheduler]

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

    def __init__(self, train_data, val_data, buffer_size=1500000, batch_size=1000, p_range=None):
        super(LitDataModule, self).__init__()
        if p_range is None:
            p_range = [6, 24]
        self.train_set = NetBatchDataSet2(batch_size, buffer_size, p_range, train_data, is_val_set=False)
        self.val_set = NetBatchDataSet2(batch_size, 1000000, p_range, val_data, is_val_set=True)
        self.train_data = train_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=None, batch_sampler=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=None, batch_sampler=None, shuffle=False)


class BatchDataSet(torch.utils.data.IterableDataset):

    def __init__(self, batch_size, buffer_size, p_range, file_path, is_val_set=False):
        if p_range is None:
            p_range = [6, 24]
        super(BatchDataSet, self).__init__()
        self.batch_size = batch_size
        self.is_val_set = is_val_set
        self.buffer_size = buffer_size
        self.file_path = file_path
        libname = pathlib.Path().absolute().__str__() + "/libpyhelper.so"
        self.c_lib = ctypes.CDLL(libname)
        if not is_val_set:
            temp = self.c_lib.init_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                            ctypes.c_uint64(p_range[0]), ctypes.c_uint64(p_range[1]),
                                            ctypes.c_char_p(self.file_path.encode('utf-8')), ctypes.c_bool(False))
        else:
            temp = self.c_lib.init_val_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                                ctypes.c_uint64(p_range[0]), ctypes.c_uint64(p_range[1]),
                                                ctypes.c_char_p(self.file_path.encode('utf-8')), ctypes.c_bool(False))
        self.file_size = temp

    def __iter__(self):
        return self

    def __len__(self):
        return self.file_size // self.batch_size


class NetBatchDataSet(BatchDataSet):

    def __init__(self, batch_size, buffer_size, p_range, file_path, is_val_set=False):
        super(NetBatchDataSet, self).__init__(batch_size, buffer_size, p_range, file_path, is_val_set)

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


class NetBatchDataSet2(BatchDataSet):

    def __init__(self, batch_size, buffer_size, p_range, file_path, is_val_set=False):
        super(NetBatchDataSet2, self).__init__(batch_size, buffer_size, p_range, file_path, is_val_set)

    def __next__(self):
        results = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        moves = np.zeros(shape=(self.batch_size, 1), dtype=np.int64)
        inputs = np.zeros(shape=(self.batch_size, 128), dtype=np.float32)
        res_p = results.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        inp_p = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        moves_p = moves.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        if not self.is_val_set:
            self.c_lib.get_next_batch2(res_p, moves_p, inp_p)
        else:
            self.c_lib.get_next_val_batch2(res_p, moves_p, inp_p)

        inputs = torch.Tensor(inputs)
        inputs = inputs.view(self.batch_size, 4, 8, 4)
        return results, moves, inputs
