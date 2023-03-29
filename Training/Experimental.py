import torch.nn as nn
import torch
from torch.nn.modules import activation
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ranger import Ranger
import struct
import ctypes
import pathlib
import numpy as np
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





class Network(pl.LightningModule):

    def __init__(self):
        super(Network, self).__init__()
        self.layers = []
      
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0
        self.number_of_steps =1000
        self.batch_size =4*8192
        self.num_epochs=120
        self.gamma = 0.96

        self.layer_one =nn.Linear(120,1024)
        self.layer_sec = nn.Linear(512,8);
        self.layer_third = nn.Linear(8,32);
        self.layer_last = nn.Linear(32,1)
        self.layers = [self.layer_one,self.layer_sec,self.layer_third,self.layer_last]

    def forward(self, x):
        accu = self.layer_one.forward(x)
        accu = torch.clamp(accu,0.0,1.0)**2
        #output of the accumulator + activation
        x,y, = torch.split(accu,512,dim=1)
        out = torch.mul(x,y)
        out = self.layer_sec.forward(out)
        out = torch.clamp(out,0.0,1.0)**2
        out = self.layer_third.forward(out)
        out = torch.clamp(out,0.0,1.0)**2
        out = self.layer_last.forward(out)
        out = torch.sigmoid(out)
        return out



    def step(self):
        with torch.no_grad():
            for layer in self.layers[1:]:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.clamp_(self.min_weight_hidden, self.max_weight_hidden)


    def configure_optimizers(self):
        #optimizer = Ranger(self.parameters())
        optimizer = torch.optim.AdamW(self.parameters())
        #optimizer = ranger21.Ranger21(self.parameters(),lr=1e-3, eps=1.0e-7,
        #                              use_warmup=False,warmdown_active=False,
        #                              weight_decay=0.0,
        #                              num_batches_per_epoch=self.number_of_steps/self.batch_size,num_epochs=self.num_epochs)
        #optimizer = Lion(self.parameters(),lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer],[scheduler]

    def training_step(self, train_batch, batch_idx):
        result, move, x = train_batch
        out = self.forward(x)
        loss =torch.pow(torch.abs(out-result),2.0).mean()
        tensorboard_logs = {"avg_val_loss": loss}
        self.log('train_loss', loss)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        result, move, x = val_batch
        out = self.forward(x)
        loss = torch.pow(torch.abs(out - result), 2.0).mean()
        self.log('val_loss', loss.detach())
        return {"val_loss": loss.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"loss": avg_loss, "log": tensorboard_logs}


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





