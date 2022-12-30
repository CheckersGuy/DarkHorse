import LitMLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import Helper as h
import torch.nn.functional as F
import numpy as np
import ctypes
import pathlib
# test_model()
#
# model = LitMLP.Network([120, 1024, 8, 32, 1],output="basemodel")
# model.init_weights()
# model.save_model_weights()

#removing input format completly since I will only have one from now on
#test the get_next_batch function seperately


libname = pathlib.Path().absolute().__str__() + "/libpyhelper.so"
c_lib = ctypes.CDLL(libname)

if __name__ == "__main__":
    model = LitMLP.Network(output="bigagain9",hidden=[120, 1024, 8, 32, 1])

    data_loader = LitMLP.LitDataModule(train_data="../TrainData/reinfformatted.train",
                                       val_data="../TrainData/val.train",
                                       batch_size=8000, buffer_size=50000000)

    # val_loader =  data_loader.val_dataloader()
    # batch = next(iter(val_loader))

    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{medium}")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=3000, callbacks=[check_point_callback])
    trainer.fit(model, data_loader)
    #model = LitMLP.PatternModel()





