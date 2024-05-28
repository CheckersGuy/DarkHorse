import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import Experimental
import string_sum
import torch
#import generator_pb2



if __name__ == "__main__":
    batch_size = 8192 
    epochs = 105
    model = Experimental.PolicyNetwork()
    data_loader = Experimental.LitDataModule(train_data="/mnt/e/nextuppolicy.samples",
    val_data="/mnt/e/validation.samples",
    batch_size=batch_size, buffer_size=40000000)

    
    
    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{Networks/medium}")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, callbacks=[check_point_callback],limit_val_batches=0)

    trainer.fit(model, data_loader,ckpt_path="Networks/depresso.ckpt");



