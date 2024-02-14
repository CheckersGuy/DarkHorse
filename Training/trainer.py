import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import Experimental
import string_sum
import torch
#import generator_pb2



if __name__ == "__main__":
    batch_size = 2*8192 
    epochs = 200
    model = Experimental.Network()
    data_loader = Experimental.LitDataModule(train_data="/mnt/e/newtry11rescoredmlhshuffled.samples",
                                      val_data="TrainData/val.train.raw.rescored",
                                       batch_size=batch_size, buffer_size=50000000)
 #   provider = string_sum.BatchProvider("TrainData/shuffled2.train.raw.rescored",50000000,batch_size,True)
    #print(provider.num_samples)

   # model.load_state_dict(torch.load("bucket.pt"))
    #model.save_quantized_bucket("bucket.quant")
    #model.save_quantized("bla.quant")

    

    
    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{Networks/medium}")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, callbacks=[check_point_callback])

  

    trainer.fit(model, data_loader,ckpt_path="Networks/adambig.ckpt");


fen_string ="W:W8,12,23,31:BK5,K15,20,K25"
string_sum.print_fen_string(fen_string)

model = Experimental.MLHNetwork()
model.load_state_dict(torch.load("mlh3.pt"))
input = np.zeros(120,dtype=np.float32)
bucket = string_sum.input_from_fen(input,fen_string)
print(torch.from_numpy(input).unsqueeze(dim=0))
out = model.forward(torch.from_numpy(input).unsqueeze(dim=0),torch.from_numpy(np.array([bucket])).unsqueeze(dim=0))
print(out*300)
print(torch.argmax(out,dim=1))

