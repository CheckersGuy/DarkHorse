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
import string_sum
from torch.utils.data import DataLoader
L1 =2*512
L2 =32
L3 = 32

class Network(pl.LightningModule):

    def __init__(self):
        super(Network, self).__init__()
        self.layers = []
        self.val_outputs=[] 
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0/ 64.0
        self.gamma = 0.975


        self.num_buckets =12
        self.accu = nn.Linear(120,L1)

        self.layer_one =nn.Linear(L1//2,L2*self.num_buckets)
        self.layer_sec = nn.Linear(L2,L3*self.num_buckets);
        self.output = nn.Linear(L3,1*self.num_buckets)
        self.layers = [self.accu,self.layer_one,self.layer_sec,self.output]
        self.init_layers()

    def forward(self, x,buckets):
        offset = torch.arange(0,x.shape[0]*self.num_buckets,self.num_buckets, device=buckets.device)
        indices = buckets.flatten()+offset

        ac= self.accu.forward(x)
        ac = torch.clamp(ac,0.0,1.0)

        ac_x,ac_y = ac.split(L1//2,dim = 1)
        ac_out = ac_x.mul(ac_y)*(127.0/128.0)

        l1s = self.layer_one(ac_out).reshape((-1,self.num_buckets,L2))
        l1c = l1s.view(-1,L2)[indices]
        l1c = torch.clamp(l1c,0.0,1.0)
        
        l2s = self.layer_sec(l1c).reshape((-1,self.num_buckets,L3))
        l2c = l2s.view(-1,L3)[indices]
        l2c = torch.clamp(l2c,0.0,1.0)

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
        optimizer = Ranger(self.parameters(),lr=3e-3,betas=(.9, 0.999),use_gc=False,gc_loc=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer],[scheduler]



    def training_step(self, train_batch, batch_idx):
        self.step()
        result,evalu, move,buckets,psqt_buckets, x = train_batch
        evalu = torch.sigmoid(evalu.to(dtype=torch.float32)/64.0)
        out = self.forward(x,buckets)
        loss =torch.pow(torch.abs(out-result),2.0).mean()
        self.log('train_loss', loss.detach(),prog_bar=True)
        return {"loss": loss}


    def validation_step(self, val_batch, batch_idx):
        result,evalu, move,buckets,psqt_buckets, x = val_batch
        out = self.forward(x,buckets)
        loss = torch.pow(torch.abs(out - result), 2.0).mean()
        self.log('val_loss', loss.detach())
        self.val_outputs.append(loss)
        return {"val_loss": loss.detach()}

    def on_validation_epoch_end(self):
        self.save_quantized_bucket("final6big.quant")
        avg_loss = torch.stack(self.val_outputs).mean()
        self.val_outputs.clear()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        self.log('loss', avg_loss, prog_bar=True)
        print(avg_loss)
        return {"loss": avg_loss, "log": tensorboard_logs}

    def save_quantized_bucket(self, output):
        self.step()
               #write the header file
        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max
        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)

        ft_weights_buf = bytearray()
        ft_bias_buf = bytearray()
        hidden_buf = bytearray()

        num_weights = 0
        num_bias = 0
        file = open(output, "wb")

        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0
        weights = np.round(weights)
        weights = np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0
        bias = np.round(bias)
        bias = np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        ft_weights_buf.extend(weights.tobytes())
        ft_bias_buf.extend(bias.tobytes())
 
        for layer in self.layers[1:]:
            weights = layer.weight.detach()
            weights = torch.clip(weights,self.min_weight_hidden,self.max_weight_hidden)*64.0
            weights = torch.round(weights)

            bias = layer.bias.detach()
            bias = bias*(127.0 * 64.0)
            bias = torch.round(bias)

            size = layer.weight.size()
            rows = size[0]//self.num_buckets
            cols = len(layer.bias)//self.num_buckets
            bucket_weights = torch.split(weights,rows)
            bucket_bias =torch.split(bias,cols)
            for i in range(self.num_buckets):                                   
                hidden_buf.extend(bucket_weights[i].numpy().flatten().astype(np.int8).tobytes())
                hidden_buf.extend(bucket_bias[i].numpy().flatten().astype(np.int32).tobytes())
        
        file.write(ft_weights_buf)
        file.write(ft_bias_buf)
        file.write(hidden_buf)
        file.close()
        self.to(device_gpu)

        return


class MLHNetwork(pl.LightningModule):

    def __init__(self):
        super(MLHNetwork, self).__init__()
        self.layers = []
        self.val_outputs=[] 
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0/ 64.0
        self.gamma = 0.975


        self.num_buckets =12
        self.accu = nn.Linear(120,L1)

        self.layer_one =nn.Linear(L1//2,L2*self.num_buckets)
        self.layer_sec = nn.Linear(L2,L3*self.num_buckets);
        self.output = nn.Linear(L3,1*self.num_buckets)
        self.layers = [self.accu,self.layer_one,self.layer_sec,self.output]
        self.init_layers()

    def forward(self, x,buckets):
        offset = torch.arange(0,x.shape[0]*self.num_buckets,self.num_buckets, device=buckets.device)
        indices = buckets.flatten()+offset

        ac= self.accu.forward(x)
        ac = torch.clamp(ac,0.0,1.0)

        ac_x,ac_y = ac.split(L1//2,dim = 1)
        ac_out = ac_x.mul(ac_y)*(127.0/128.0)

        l1s = self.layer_one(ac_out).reshape((-1,self.num_buckets,L2))
        l1c = l1s.view(-1,L2)[indices]
        l1c = torch.clamp(l1c,0.0,1.0)
        
        l2s = self.layer_sec(l1c).reshape((-1,self.num_buckets,L3))
        l2c = l2s.view(-1,L3)[indices]
        l2c = torch.clamp(l2c,0.0,1.0)

        l3s = self.output(l2c).reshape((-1,self.num_buckets,1))
        l3c = l3s.view(-1,1)[indices]
        out = l3c
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
        optimizer = Ranger(self.parameters(),lr=3e-2,betas=(.9, 0.999),use_gc=False,gc_loc=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer],[scheduler]



    def training_step(self, train_batch, batch_idx):
        self.step()
        result,mlh_target, move,buckets,psqt_buckets, x = train_batch
        mlh_target = torch.clamp(mlh_target.to(dtype=torch.float32),0,300)
        mlh_target = mlh_target/300.0 #value range from 0 to 60
        out = self.forward(x,buckets)
        loss =torch.pow(torch.abs(out-mlh_target),2.0).mean()
        self.log('train_loss', loss.detach(),prog_bar=True)
        return {"loss": loss}


    def validation_step(self, val_batch, batch_idx):
        torch.save(self.state_dict(),"mlh.pt")
        result,mlh_target, move,buckets,psqt_buckets, x = val_batch
        mlh_target = torch.clamp(mlh_target.to(dtype=torch.float32),0,300)
        mlh_target = mlh_target/300.0
        out = self.forward(x,buckets)
        loss = torch.pow(torch.abs(out - mlh_target), 2.0).mean()
        self.log('val_loss', loss.detach())
        self.val_outputs.append(loss)
        return {"val_loss": loss.detach()}

    def on_validation_epoch_end(self):
        self.save_quantized_bucket("mlh.quant")
        avg_loss = torch.stack(self.val_outputs).mean()
        self.val_outputs.clear()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        self.log('loss', avg_loss, prog_bar=True)
        print(avg_loss)
        return {"loss": avg_loss, "log": tensorboard_logs}

    def save_quantized_bucket(self, output):
        self.step()
               #write the header file
        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max
        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)

        ft_weights_buf = bytearray()
        ft_bias_buf = bytearray()
        hidden_buf = bytearray()

        num_weights = 0
        num_bias = 0
        file = open(output, "wb")

        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0
        weights = np.round(weights)
        weights = np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0
        bias = np.round(bias)
        bias = np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        ft_weights_buf.extend(weights.tobytes())
        ft_bias_buf.extend(bias.tobytes())
 
        for layer in self.layers[1:]:
            weights = layer.weight.detach()
            weights = torch.clip(weights,self.min_weight_hidden,self.max_weight_hidden)*64.0
            weights = torch.round(weights)

            bias = layer.bias.detach()
            bias = bias*(127.0 * 64.0)
            bias = torch.round(bias)

            size = layer.weight.size()
            rows = size[0]//self.num_buckets
            cols = len(layer.bias)//self.num_buckets
            bucket_weights = torch.split(weights,rows)
            bucket_bias =torch.split(bias,cols)
            for i in range(self.num_buckets):                                   
                hidden_buf.extend(bucket_weights[i].numpy().flatten().astype(np.int8).tobytes())
                hidden_buf.extend(bucket_bias[i].numpy().flatten().astype(np.int32).tobytes())
        
        file.write(ft_weights_buf)
        file.write(ft_bias_buf)
        file.write(hidden_buf)
        file.close()
        self.to(device_gpu)

        return



class BatchDataSet(torch.utils.data.IterableDataset):

    def __init__(self, batch_size, buffer_size, file_path, shuffle=True):
        super(BatchDataSet, self).__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.file_path = file_path
        self.loader =string_sum.BatchProvider(self.file_path,self.buffer_size,self.batch_size,shuffle)

        self.num_samples = self.loader.num_samples
        print("Called initialization")

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples//self.batch_size

    def __next__(self):
        input_size = 120
        results = np.zeros(self.batch_size, dtype=np.float32)
        evals =  np.zeros(self.batch_size, dtype=np.int16)
        moves = np.zeros(self.batch_size, dtype=np.int64)
        buckets = np.zeros(self.batch_size, dtype=np.int64)
        psqt_buckets = np.zeros(self.batch_size, dtype=np.int64)
        inputs = np.zeros(self.batch_size*input_size, dtype=np.float32)
        self.loader.testing(inputs,results,evals,buckets,psqt_buckets)

        return results.reshape(self.batch_size,1),evals.reshape(self.batch_size,1), moves.reshape(self.batch_size,1),buckets.reshape(self.batch_size,1), psqt_buckets.reshape(self.batch_size,1),inputs.reshape(self.batch_size,input_size)



class LitDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, buffer_size=1500000, batch_size=1000):
        super(LitDataModule, self).__init__()
        self.train_set = BatchDataSet(batch_size, buffer_size, train_data, True)
        self.val_set = BatchDataSet(batch_size, 100000, val_data, False)
        self.train_data = train_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=None, batch_sampler=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=None, batch_sampler=None, shuffle=False)








