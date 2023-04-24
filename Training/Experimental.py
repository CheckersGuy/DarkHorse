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
import ranger21
from lion_pytorch import Lion
L1 =1024
L2 = 16
L3 = 32

class Relu1(nn.Module):
    def __init__(self):
        super(Relu1, self).__init__()

    def forward(self, x):
        return (127.0/128.0)*torch.clamp(x,0.0,1.0)**2



class Network(pl.LightningModule):

    def __init__(self):
        super(Network, self).__init__()
        self.layers = []
      
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0
        self.number_of_steps =1000
        self.num_epochs=120
        self.gamma = 0.965


        self.num_buckets = 8
        self.accu = nn.Linear(120,L1)

        self.layer_one =nn.Linear(L1,L2*self.num_buckets)
        self.layer_sec = nn.Linear(L2,L3*self.num_buckets);
        self.output = nn.Linear(L3,1*self.num_buckets)
        self.layers = [self.accu,self.layer_one,self.layer_sec,self.output]
        self.init_layers()

    def forward(self, x,buckets):
        offset = torch.arange(0,x.shape[0]*self.num_buckets,self.num_buckets, device=buckets.device)
        indices = buckets.flatten()+offset

        ac = self.accu.forward(x)
        ac_out =(127.0/128.0)* torch.clamp(ac,0.0,1.0)**2

       # ac = self.accu.forward(x)
       # ac = (127.0/128.0)*torch.clamp(ac,0.0,1.0)**2
       #  ac_x,ac_y = ac.split(L1//2,dim = 1)
       # ac_out = ac_x.mul(ac_y)*(127.0/128.0)



        l1s = self.layer_one(ac_out).reshape((-1,self.num_buckets,L2))
        l1c = l1s.view(-1,L2)[indices]
        l1c = (127.0/128.0)*torch.clamp(l1c,0.0,1.0)**2
        
        l2s = self.layer_sec(l1c).reshape((-1,self.num_buckets,L3))
        l2c = l2s.view(-1,L3)[indices]
        l2c = (127.0/128.0)*torch.clamp(l2c,0.0,1.0)**2

        l3s = self.output(l2c).reshape((-1,self.num_buckets,1))
        l3c = l3s.view(-1,1)[indices]
        out = torch.sigmoid(l3c)

        return out


    def init_layers(self):
        l1_weight = self.layer_one.weight
        l1_bias = self.layer_one.bias
        l2_weight = self.layer_sec.weight
        l2_bias = self.layer_sec.bias
        output_weight = self.output.weight
        output_bias = self.output.bias

        with torch.no_grad():
            output_bias.fill_(0.0)
            for i in range(1,self.num_buckets):
                l1_weight[i*L2 : (i+1)*L2, : ] = l1_weight[0 : L2, :]
                l1_bias[i*L2 : (i+1)*L2 ] = l1_bias[0 : L2]
                l2_weight[i*L3 : (i+1)*L3, : ] = l2_weight[0 : L3, :]
                l2_bias[i*L3 : (i+1)*L3 ] = l1_bias[0 : L3]
                output_weight[i:i+1,:] = output_weight[0:1,:]



    def write_header(self,file_out):
        num_hidden = 3;
        file_out.write(struct.pack("I", num_hidden))
        file_out.write(struct.pack("I", self.num_buckets))
        file_out.write(struct.pack("I", L1))
        file_out.write(struct.pack("I", L2))
        file_out.write(struct.pack("I", L3))

    def step(self):
        with torch.no_grad():
            for layer in self.layers[1:]:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.clamp_(self.min_weight_hidden, self.max_weight_hidden)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer],[scheduler]



    def training_step(self, train_batch, batch_idx):
        self.step()
        result, move,buckets, x = train_batch
        out = self.forward(x,buckets)
        loss =torch.pow(torch.abs(out-result),2.5).mean()
        tensorboard_logs = {"avg_val_loss": loss}
        self.log('train_loss', loss)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        result, move,buckets, x = val_batch
        out = self.forward(x,buckets)
        loss = torch.pow(torch.abs(out - result), 2.0).mean()
        self.log('val_loss', loss.detach())
        return {"val_loss": loss.detach()}

    def validation_epoch_end(self, outputs):
        self.save_quantized_bucket("bucketelem.quant")
        torch.save(self.state_dict(),"buckeelem.pt")
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
                print(layer.weight.size())
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


    def save_quantized_bucket(self, output):
        self.step()
        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max

        #write the header file

        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)
        buffer_weights = bytearray()
        buffer_bias = bytearray()
        num_weights = 0
        num_bias = 0
        file = open(output, "wb")

        #saving the header file
        self.write_header(file)

        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0*4.0
        weights = np.round(weights)
        weights = np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0*4.0
        bias = np.round(bias)
        bias = np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        buffer_weights.extend(weights.tobytes())
        buffer_bias.extend(bias.tobytes())
        num_weights += len(weights)
        num_bias += len(bias)


        for i in range(self.num_buckets):
            for layer in self.layers[1:]:
                weights = layer.weight
                size = layer.weight.size()
                #print(size)
                rows = size[0]//self.num_buckets
                bucket_weights = torch.split(weights,rows)
                bucket_bias =torch.split(layer.bias,len(layer.bias)//self.num_buckets)
                #print(buckets[i].size())
                #quantization
                weights = bucket_weights[i].detach().numpy().flatten()
                weights = weights*64.0
                weights = np.round(weights)
                weights = np.clip(weights,min16,max16)
                weights = weights.astype(np.int16)
                print(bucket_bias[i].size())
                bias = bucket_bias[i].detach().numpy().flatten()
                bias = bias*(127.0 * 64.0)
                bias = np.round(bias)
                bias = bias.astype(np.int32)
                buffer_weights.extend(weights.tobytes())
                buffer_bias.extend(bias.tobytes())
        
        file.write(buffer_weights)
        file.write(buffer_bias)
        file.close()
        self.to(device_gpu)

                
        return
        for layer in self.layers[1:]:
            print(layer.weight.size())
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

        file.write(buffer_weights)
        file.write(buffer_bias)
        file.close()
        self.to(device_gpu)
        return









