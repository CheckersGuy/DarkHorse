import torch.nn as nn
import torch
from torch.optim import optimizer
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ranger import Ranger
import struct
import ctypes
import pathlib
import numpy as np
from enum import IntEnum







class Relu1(nn.Module):
    def __init__(self):
        super(Relu1, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)




class WDLNetwork(pl.LightningModule):

    def __init__(self, hidden, output="basemodel"):
        super(WDLNetwork, self).__init__()
        self.layers = []
        self.output = output
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
 
        for i in range(len(hidden) - 2):
            self.layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            self.layers.append(Relu1())

        self.layers.append(nn.Linear(hidden[len(hidden) - 2], hidden[len(hidden) - 1]))
        self.net =nn.Sequential(*self.layers)
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0
        self.gamma = 0.93
        print(self.net)

    def forward(self, x):
        return self.net.forward(x)

    def accuracy(self, logits, target):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), target).to(torch.float32)) / len(target)
        return 100 * acc

  
    def step(self):
        with torch.no_grad():
            for layer in self.layers[1:]:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.clamp_(self.min_weight_hidden, self.max_weight_hidden)




    def configure_optimizers(self):
        #optimizer = Ranger(self.parameters())
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer],[scheduler]

    def on_train_epoch_end(self):
        self.save_quantized("epoch.quant")

    def training_step(self, train_batch, batch_idx):
        self.step()
        wdl_values, move, x = train_batch
        output = self.forward(x)
        loss = self.criterion(output, wdl_values.squeeze(dim=1))
        acc = self.accuracy(output, wdl_values.squeeze())
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.step()
        wdl_values, move, x = val_batch
        output = self.forward(x)
        loss = self.criterion(output, wdl_values.squeeze(dim=1))
        acc = self.accuracy(output, wdl_values.squeeze())
        self.log('val_acc_step', acc)
        return loss

    def save_quantized(self, output):
        self.step()
        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max

        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)
        buffer_weights = bytearray()
        buffer_bias = bytearray()
        num_weights = 0
        num_bias = 0
        file = open(output, "wb")
        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0
        np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0
        np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        buffer_weights.extend(weights.tobytes())
        buffer_bias.extend(bias.tobytes())
        num_weights += len(weights)
        num_bias += len(bias)

        for layer in self.layers[1:]:
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().numpy().flatten()
                weights = weights * 64.0
                np.clip(weights, min16, max16)
                weights = weights.astype(np.int16)
                bias = layer.bias.detach().numpy().flatten()
                bias = bias * (127.0 * 64.0)
                bias = bias.astype(np.int32)
                buffer_weights.extend(weights.tobytes())
                buffer_bias.extend(bias.tobytes())
                num_weights += len(weights)
                num_bias += len(bias)

        file.write(struct.pack("I", num_weights))
        file.write(buffer_weights)
        file.write(struct.pack("I", num_bias))
        file.write(buffer_bias)
        file.close()
        self.to(device_gpu)
        return




class Network(pl.LightningModule):

    def __init__(self, hidden, output="basemodel"):
        super(Network, self).__init__()
        self.layers = []
        self.output = output
        for i in range(len(hidden) - 2):
            self.layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            self.layers.append(Relu1())

        self.layers.append(nn.Linear(hidden[len(hidden) - 2], hidden[len(hidden) - 1]))
        self.layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.layers)
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0
        self.gamma = 0.9
        print(self.net)

    def forward(self, x):
        return self.net.forward(x)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        with torch.no_grad():
            for layer in self.layers[1:]:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.clamp_(self.min_weight_hidden, self.max_weight_hidden)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        result, move, x = train_batch
        out = self.forward(x)
        loss =torch.pow(torch.abs(out-result),2.5).mean()
        tensorboard_logs = {"avg_val_loss": loss}
        self.log('train_loss', loss)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        result, move, x = val_batch
        out = self.forward(x)
        loss = torch.pow(torch.abs(out - result), 2.5).mean()
        self.log('val_loss', loss.detach())
        return {"val_loss": loss.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"loss": avg_loss, "log": tensorboard_logs}

    def init_weights(self):
        self.net.apply(init_weights)

    def save_quantized(self, output):

        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max

        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)
        buffer_weights = bytearray()
        buffer_bias = bytearray()
        num_weights = 0
        num_bias = 0
        file = open(output, "wb")
        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0
        np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0
        np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        buffer_weights += weights.tobytes()
        buffer_bias += bias.tobytes()
        num_weights += len(weights)
        num_bias += len(bias)

        for layer in self.layers[1:]:
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().numpy().flatten()
                weights = weights * 64.0
                np.clip(weights, min16, max16)
                weights = weights.astype(np.int16)
                bias = layer.bias.detach().numpy().flatten()
                bias = bias * (127 * 64)
                bias = bias.astype(np.int32)
                buffer_weights += weights.tobytes()
                buffer_bias += bias.tobytes()
                num_weights += len(weights)
                num_bias += len(bias)

        file.write(struct.pack("I", num_weights))
        file.write(buffer_weights)
        file.write(struct.pack("I", num_bias))
        file.write(buffer_bias)
        file.close()
        self.to(device_gpu)
        return



class PolicyNetwork(pl.LightningModule):

    def __init__(self, hidden, output="form_network20.weights"):
        super(PolicyNetwork, self).__init__()
        self.layers = []
        self.output = output
        for i in range(len(hidden) - 2):
            self.layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            self.layers.append(Relu1())

        self.layers.append(nn.Linear(hidden[len(hidden) - 2], hidden[len(hidden) - 1]))
        self.net = nn.Sequential(*self.layers)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0
        print(self.net)

    def forward(self, x):
        return self.net.forward(x)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        with torch.no_grad():
            for layer in self.layers[1:]:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.clamp_(self.min_weight_hidden, self.max_weight_hidden)

    def on_epoch_end(self) -> None:
        self.save_parameters(self.output)
        self.save("policy.pt")
        self.save_quantized("policy.quant")

    def accuracy(self, logits, target):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), target).to(torch.float32)) / len(target)
        return 100 * acc

    def training_step(self, train_batch, batch_idx):
        result, move, x = train_batch
        policy = self.forward(x)
        loss_policy = self.criterion(policy, move.squeeze(dim=1))
        acc = self.accuracy(policy, move.squeeze())
        self.log('train_acc_step', acc, prog_bar=True)
        return loss_policy

    def validation_step(self, val_batch, batch_idx):
        result, move, x = val_batch
        policy = self.forward(x)
        loss_policy = self.criterion(policy, move.squeeze(dim=1))
        self.log('val_loss', loss_policy)
        return loss_policy

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), betas=(.9, 0.999), eps=1.0e-7)
        return optimizer

    def save_quantized(self, output):

        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max

        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)
        buffer_weights = bytearray()
        buffer_bias = bytearray()
        num_weights = 0
        num_bias = 0
        file = open(output, "wb")
        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0
        np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0
        np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        buffer_weights += weights.tobytes()
        buffer_bias += bias.tobytes()
        num_weights += len(weights)
        num_bias += len(bias)

        for layer in self.layers[1:]:
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().numpy().flatten()
                weights = weights * 64.0
                np.clip(weights, min16, max16)
                weights = weights.astype(np.int16)
                bias = layer.bias.detach().numpy().flatten()
                bias = bias * (127 * 64)
                np.clip(bias, min16, max16)
                bias = bias.astype(np.int16)
                buffer_weights += weights.tobytes()
                buffer_bias += bias.tobytes()
                num_weights += len(weights)
                num_bias += len(bias)

        file.write(struct.pack("I", num_weights))
        file.write(buffer_weights)
        file.write(struct.pack("I", num_bias))
        file.write(buffer_bias)
        file.close()
        self.to(device_gpu)
        return

    def save_parameters(self, output):
        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)
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
        self.to(device_gpu)

    def save(self, output):
        torch.save(self.state_dict(), output)


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


class WDLDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, buffer_size=1500000, batch_size=1000):
        super(WDLDataModule, self).__init__()
        self.train_set = WDLDataSet(batch_size, buffer_size, train_data, False)
        self.val_set = WDLDataSet(batch_size, 1000000, val_data, True)
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
        if not is_val_set:
            temp = self.c_lib.init_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                            ctypes.c_char_p(self.file_path.encode('utf-8')))
        else:
            temp = self.c_lib.init_val_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                                ctypes.c_char_p(self.file_path.encode('utf-8')))
        self.file_size = temp

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
        moves = np.zeros(shape=(self.batch_size, 1), dtype=np.int64)
        inputs = np.zeros(shape=(self.batch_size, input_size), dtype=np.float32)
        res_p = results.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        inp_p = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        moves_p = moves.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        if not self.is_val_set:
            self.c_lib.get_next_batch(res_p, moves_p, inp_p)
        else:
            self.c_lib.get_next_val_batch(res_p, moves_p, inp_p)

        return results, moves, inputs


class WDLDataSet(BatchDataSet):

    def __init__(self, batch_size, buffer_size, file_path, is_val_set=False):
        super(WDLDataSet, self).__init__(batch_size, buffer_size, file_path, is_val_set)

    def __next__(self):
        input_size = 120
        results = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        moves = np.zeros(shape=(self.batch_size, 1), dtype=np.int64)
        inputs = np.zeros(shape=(self.batch_size, input_size), dtype=np.float32)
        res_p = results.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        inp_p = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        moves_p = moves.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        if not self.is_val_set:
            self.c_lib.get_next_batch(res_p, moves_p, inp_p)
        else:
            self.c_lib.get_next_val_batch(res_p, moves_p, inp_p)



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
        return wdl_values, torch.Tensor(moves), torch.Tensor(inputs)
