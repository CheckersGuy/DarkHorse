import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import Experimental
import string_sum
#import generator_pb2


if __name__ == "__main__":
    batch_size = 2*8192 
    epochs = 420
    model = Experimental.Network()
    data_loader = Experimental.LitDataModule(train_data="TrainData/shuffled.train.raw.rescored",
                                      val_data="TrainData/val.train.raw.rescored",
                                       batch_size=batch_size, buffer_size=5000000)
 #   provider = string_sum.BatchProvider("TrainData/shuffled2.train.raw.rescored",50000000,batch_size,True)
    #print(provider.num_samples)

   # model.load_state_dict(torch.load("bucket.pt"))
    #model.save_quantized_bucket("bucket.quant")
    #model.save_quantized("bla.quant")

    

    
    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{Networks/medium}")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, callbacks=[check_point_callback])

    trainer.fit(model, data_loader,ckpt_path="Networks/medium=0-v111.ckpt")


