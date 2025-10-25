import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import Experimental
import string_sum
import torch
#import generator_pb2








if __name__ == "__main__":
    batch_size = 8192 
    epochs = 605
    model = Experimental.PolicyNetwork()
    data_loader = Experimental.LitDataModule(train_data="/home/robin/Downloads/policy.data",
    val_data="/home/robin/Downloads/another1msbatch.samples",
    batch_size=batch_size, buffer_size=50000000)
    

    
    
    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{Networks/medium}")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, callbacks=[check_point_callback],limit_val_batches=0)

    trainer.fit(model, data_loader);







