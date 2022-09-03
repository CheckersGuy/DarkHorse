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
from adan_pytorch import Adan
from enum import Enum
from enum import IntEnum


class InputFormat(IntEnum):
    V1 = 0,
    V2 = 1,
    PATTERN = 2


class ResBlock(pl.LightningModule):

    def __init__(self, in_channels, out_channels, down_sample=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if down_sample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3,
                                   padding=1)
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=3,
                                   padding=1)
            self.short_cut = nn.Sequential()
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1)

    def forward(self, x):
        shortcut = self.short_cut(x)
        x = nn.Mish()(self.bn1(self.conv1(x)))
        x = nn.Mish()(self.bn2(self.conv2(x)))
        x = x + shortcut
        return nn.Mish()(x)


class ResNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(kernel_size=5, in_channels=4, out_channels=128, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.Mish()
        )
        self.input_format = InputFormat.V2
        self.every = 1000
        self.second = nn.Sequential(ResBlock(in_channels=128, out_channels=128, down_sample=False),
                                    ResBlock(in_channels=128, out_channels=128, down_sample=False)
                                    )
        self.third = nn.Sequential(ResBlock(in_channels=128, out_channels=128, down_sample=False),
                                   ResBlock(in_channels=128, out_channels=128, down_sample=False))
        self.fourth = nn.Sequential(ResBlock(in_channels=128, out_channels=128, down_sample=False),
                                    ResBlock(in_channels=128, out_channels=128, down_sample=False)
                                    )

        self.last = nn.AdaptiveAvgPool2d(1)
        self.policy_out = nn.Sequential(nn.Linear(128, 32 * 32))
        self.value_out = nn.Sequential(nn.Linear(128, 128), nn.Mish(), nn.Linear(128, 1))

    def forward(self, input):
        y = self.first(input)
        y = self.second(y)
        y = self.third(y)
        y = self.fourth(y)
        y = self.last(y)
        y = y.view(input.size(0), -1)
        policy = self.policy_out(y)
        value = self.value_out(y)
        value = torch.sigmoid(value)
        return policy, value

    def accuracy(self, logits, target):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), target).to(torch.float32)) / len(target)
        return acc

    def training_step(self, train_batch, batch_idx):
        result, move, x = train_batch
        policy, value = self.forward(x)
        loss_value = F.mse_loss(value, result)
        loss_policy = F.cross_entropy(policy, move.squeeze(dim=1))

        acc = self.accuracy(policy, move.squeeze())
        self.log('train_acc_step', acc, prog_bar=True)
        combined_loss = (loss_policy + loss_value)
        self.log('combined_loss', combined_loss, prog_bar=True)
        self.log('value_loss', loss_value, prog_bar=True)
        return combined_loss

    def validation_step(self, val_batch, batch_idx):
        result, move, x = val_batch
        policy, value = self.forward(x)
        loss_value = F.mse_loss(value, result)
        loss_policy = F.cross_entropy(policy, move.squeeze(dim=1))
        combined_loss = 0.5 * (loss_policy + loss_value)
        self.log('val_loss', combined_loss)
        return combined_loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(lr=0.01, momentum=0.9, params=self.parameters())
        optimizer = Ranger(params=self.parameters(), weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.90)
        return optimizer

    def accuracy(self, logits, target):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), target).to(torch.float32)) / len(target)
        acc = 100 * acc
        return acc

    def validation_epoch_end(self, outputs):
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)


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
        self.input_format = InputFormat.V2
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

        self.decoder = nn.Sequential(nn.Linear(128 * 4 * 8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 256),
                                     nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        # tensor has the form batchsizex8x4
        out = self.conv1(x)
        out_hat = out.view(x.size(0), -1)
        out_hat = self.decoder(out_hat)
        return out_hat

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters())
        return optimizer

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

    def __init__(self, hidden, output="form_network20.weights"):
        super(Network, self).__init__()
        self.layers = []
        self.output = output
        for i in range(len(hidden) - 2):
            self.layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            self.layers.append(Relu1())

        self.layers.append(nn.Linear(hidden[len(hidden) - 2], hidden[len(hidden) - 1]))
        self.layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.layers)
        self.criterion = torch.nn.MSELoss()
        self.init_weights()
        self.input_format = InputFormat.V1
        self.max_weight_hidden =127.0/64.0
        self.min_weight_hidden = -127.0/64.0
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
        self.save("small.pt")
        self.save_quantized("test2.quant")

    def configure_optimizers(self):
        # optimizer = Adan(
        #      self.parameters(),
        #      lr=1e-3,  # learning rate
        #      betas=(0.1, 0.1, 0.001),  # beta 1-2-3 as described in paper
        #      weight_decay=0.  # weight decay
        #  )
        optimizer = Ranger(self.parameters(), betas=(.9, 0.999), eps=1.0e-7)
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

    def save_quantized(self, output):

        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max

        min8 = np.iinfo(np.int8).min
        max8 = np.iinfo(np.int8).max

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
                print(weights)
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
        self.input_format = InputFormat.V1
        print(self.net)

    def accuracy(self, logits, target):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), target).to(torch.float32)) / len(target)
        acc = 100 * acc
        return acc

    def forward(self, x):
        return self.net.forward(x)

    def on_epoch_end(self) -> None:
        self.save_parameters(self.output)

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), betas=(.9, 0.999), eps=1.0e-7)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        result, move, x = train_batch
        out = self.forward(x)
        loss = F.cross_entropy(out, move.squeeze(dim=1))
        tensorboard_logs = {"avg_val_loss": loss}
        acc = self.accuracy(out, move.squeeze())
        self.log('train_acc_step', acc, prog_bar=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        result, move, x = val_batch
        out = self.forward(x)
        acc = self.accuracy(out, move.squeeze())
        self.log('val_loss', acc)
        return {"val_loss": acc}

    def init_weights(self):
        self.net.apply(init_weights)

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
        return

    def save(self, output):
        torch.save(self.state_dict(), output)


class LitDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, buffer_size=1500000, batch_size=1000, p_range=None,
                 input_format=InputFormat.V1):
        super(LitDataModule, self).__init__()
        if p_range is None:
            p_range = [6, 24]

        self.train_set = NetBatchDataSet(batch_size, buffer_size, p_range, train_data, False,
                                         input_format)
        self.val_set = NetBatchDataSet(batch_size, 1000000, p_range, val_data, True, input_format)
        self.train_data = train_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=None, batch_sampler=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=None, batch_sampler=None, shuffle=False)


class BatchDataSet(torch.utils.data.IterableDataset):

    def __init__(self, batch_size, buffer_size, p_range, file_path, is_val_set=False, input_format=InputFormat.V1):
        if p_range is None:
            p_range = [6, 24]
        super(BatchDataSet, self).__init__()
        self.input_format = input_format
        self.batch_size = batch_size
        self.is_val_set = is_val_set
        self.buffer_size = buffer_size
        self.file_path = file_path
        libname = pathlib.Path().absolute().__str__() + "/libpyhelper.so"
        self.c_lib = ctypes.CDLL(libname)
        if not is_val_set:
            temp = self.c_lib.init_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                            ctypes.c_uint64(p_range[0]), ctypes.c_uint64(p_range[1]),
                                            ctypes.c_char_p(self.file_path.encode('utf-8')),
                                            ctypes.c_int32(input_format))
        else:
            temp = self.c_lib.init_val_streamer(ctypes.c_uint64(self.buffer_size), ctypes.c_uint64(self.batch_size),
                                                ctypes.c_uint64(p_range[0]), ctypes.c_uint64(p_range[1]),
                                                ctypes.c_char_p(self.file_path.encode('utf-8')),
                                                ctypes.c_int32(input_format))
        self.file_size = temp

    def __iter__(self):
        return self

    def __len__(self):
        return self.file_size // self.batch_size


# needs to be fixed somehow
class NetBatchDataSet(BatchDataSet):

    def __init__(self, batch_size, buffer_size, p_range, file_path, is_val_set=False, input_format=InputFormat.V1):
        super(NetBatchDataSet, self).__init__(batch_size, buffer_size, p_range, file_path, is_val_set,
                                              input_format)

    def __next__(self):
        input_size = 120 if self.input_format == InputFormat.V1 else 128
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

        if self.input_format == InputFormat.V2:
            inputs = torch.Tensor(inputs)
            moves = torch.LongTensor(moves)
            results = torch.tensor(results)
            inputs = inputs.view(self.batch_size, 4, 8, 4)

        return results, moves, inputs
