import torch.nn as nn
import torch
from torch.nn.modules import activation
import pytorch_lightning as pl
from ranger import Ranger
import struct
import numpy as np
import torch.nn as nn
import torch
import pytorch_lightning as pl
import struct
import numpy as np
from focal_loss.focal_loss import FocalLoss
import adabelief_pytorch
from ranger_adabelief import RangerAdaBelief
from sophia import SophiaG
L1 =2*1024
L2 =16
L3 = 32


class Network(pl.LightningModule):

    def __init__(self):
        super(Network, self).__init__()
        self.layers = []
      
        self.max_weight_hidden = 1027.0 / 64.0
        self.min_weight_hidden = -1027.0 / 64.0
        self.gamma = 0.98


        self.num_buckets =1
        self.accu = nn.Linear(120,L1)

        self.layer_one =nn.Linear(L1//2,L2*self.num_buckets)
        self.layer_sec = nn.Linear(L2,L3*self.num_buckets);
        self.output = nn.Linear(L3,1*self.num_buckets)
        self.layers = [self.accu,self.layer_one,self.layer_sec,self.output]
        self.init_layers()

    def forward(self, x,buckets):
        offset = torch.arange(0,x.shape[0]*self.num_buckets,self.num_buckets, device=buckets.device)
        indices = buckets.flatten()+offset

        #ac = self.accu.forward(x)
       # ac_out =(127.0/128.0)* torch.clamp(ac,0.0,1.0)**2

        ac = self.accu.forward(x)
        ac = (127.0/128.0)*torch.clamp(ac,0.0,1.0)**2
        ac_x,ac_y = ac.split(L1//2,dim = 1)
        ac_out = ac_x.mul(ac_y)*(127.0/128.0)



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
        num_layers = len(self.layers)
        file_out.write(struct.pack("I", num_layers))
        file_out.write(struct.pack("I", self.num_buckets))
                    
        file_out.write(struct.pack("I", self.accu.in_features))
        file_out.write(struct.pack("I", self.accu.out_features))

        for layer in self.layers[1:]:
            file_out.write(struct.pack("I", layer.in_features))
            file_out.write(struct.pack("I", layer.out_features//self.num_buckets))


       
    def step(self):
        with torch.no_grad():
            for layer in self.layers[1:]:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.clamp_(self.min_weight_hidden, self.max_weight_hidden)


    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3,weight_decay=0)
       # optimizer = SophiaG(self.parameters(), lr=2e-4, weight_decay=0,rho=0.02,betas=(0.9,0.999))
       # optimizer = adabelief_pytorch.AdaBelief(self.parameters(),lr=1e-3,weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,steps_per_epoch=60000,epochs=120,max_lr=3e-3,cycle_momentum=False)
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
        loss = torch.pow(torch.abs(out - result), 2.5).mean()
        self.log('val_loss', loss.detach())
        return {"val_loss": loss.detach()}

    def validation_epoch_end(self, outputs):
        self.save_quantized_bucket("data3.quant")
        torch.save(self.state_dict(),"data3.pt")
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"loss": avg_loss, "log": tensorboard_logs}


    def save_quantized_bucket(self, output):
        self.step()
               #write the header file
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


class Network2(pl.LightningModule):

    def __init__(self):
        super(Network2, self).__init__()
        self.layers = []
      
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0
        self.gamma = 0.96


        self.accu = nn.Linear(120,L1)
        self.layer_one =nn.Linear(L1//2,L2)
        self.layer_sec = nn.Linear(L2,L3);
        self.output = nn.Linear(L3,1)
        self.layers = [self.accu,self.layer_one,self.layer_sec,self.output]
        self.init_layers()

    def forward(self, x,buckets):
        ac = self.accu.forward(x)

        ac = (127.0/128.0)*torch.clamp(ac,0.0,1.0)**2
        ac_x,ac_y = ac.split(L1//2,dim = 1)
        ac_out = ac_x.mul(ac_y)*(127.0/128.0)

        l1s = self.layer_one(ac_out)
        l1c = (127.0/128.0)*torch.clamp(l1s,0.0,1.0)**2
        
        l2s = self.layer_sec(l1c)
        l2c = (127.0/128.0)*torch.clamp(l2s,0.0,1.0)**2

        l3s = self.output(l2c)
        out = torch.sigmoid(l3s)

        return out


    def init_layers(self):
        #need to rework init
        pass


    def write_header(self,file_out):
        num_layers = len(self.layers)
        file_out.write(struct.pack("I", num_layers))
        file_out.write(struct.pack("I", self.accu.in_features))
        file_out.write(struct.pack("I", self.accu.out_features))

        for layer in self.layers[1:]:
            file_out.write(struct.pack("I", layer.in_features))
            file_out.write(struct.pack("I", layer.out_features))


       
    def step(self):
        with torch.no_grad():
            for layer in self.layers[1:]:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.clamp_(self.min_weight_hidden, self.max_weight_hidden)


    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3,weight_decay=0)
        optimizer = Ranger(self.parameters(),lr=2e-3,betas=(.9, 0.999),weight_decay=0,use_gc=False,gc_loc=False)
        #optimizer = RangerAdaBelief(self.parameters(),lr=1e-3)
        #optimizer = adabelief_pytorch.AdaBelief(self.parameters(),lr=1e-3,betas=(0.9,0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,steps_per_epoch=60000,epochs=120,max_lr=3e-3,cycle_momentum=False)
        return [optimizer],[scheduler]


    def training_step(self, train_batch, batch_idx):
        self.step()
        result, move,buckets, x = train_batch
        out = self.forward(x,buckets)
        loss =torch.pow(torch.abs(out-result),2.0).mean()
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
        self.save_quantized_bucket("data3.quant")
        torch.save(self.state_dict(),"data3.pt")
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"loss": avg_loss, "log": tensorboard_logs}


    def save_quantized_bucket(self, output):
        self.step()
               #write the header file
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


        for layer in self.layers[1:]:
            weights = layer.weight
            weights = layer.weight.detach().numpy().flatten()
            weights = weights*64.0
            weights = np.round(weights)
            weights = np.clip(weights,min16,max16)
            weights = weights.astype(np.int16)
            bias = layer.bias.detach().numpy().flatten()
            bias = bias*(127.0 * 64.0)
            bias = np.round(bias)
            bias = bias.astype(np.int32)
            buffer_weights.extend(weights.tobytes())
            buffer_bias.extend(bias.tobytes())
        
        file.write(buffer_weights)
        file.write(buffer_bias)
        file.close()
        self.to(device_gpu)






