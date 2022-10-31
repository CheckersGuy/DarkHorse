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
    #model = LitMLP.PatternModel()
    model = LitMLP.Network([120, 256, 32, 32, 1])
    #model.load_state_dict(torch.load("basemodel.pt"))
    #model = LitMLP.ResNet()


    data_loader = LitMLP.LitDataModule(train_data="../TrainData/reinfformatted.train",
                                       val_data="../TrainData/val.train",
                                       batch_size=32000, buffer_size=30000000)

    # val_loader =  data_loader.val_dataloader()
    # batch = next(iter(val_loader))

    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{newtiny}")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=300, callbacks=[check_point_callback])
    trainer.fit(model, data_loader)




