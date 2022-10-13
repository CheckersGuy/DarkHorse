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


# libname = pathlib.Path().absolute().__str__() + "/libpyhelper.so"
# c_lib = ctypes.CDLL(libname)
#
# temp = c_lib.init_streamer(ctypes.c_uint64(10000), ctypes.c_uint64(32000),
#                                 ctypes.c_uint64(0), ctypes.c_uint64(0),
#                                 ctypes.c_char_p("\\\wsl.localhost\\Ubuntu-22.04\\home\\leagu\\DarkHorse\\Training\\TrainData\\reinfformatted.train".encode('utf-8')),
#                                 ctypes.c_int32(0))

if __name__ == "__main__":
    #model = LitMLP.PatternModel()
    model = LitMLP.PolicyNetwork([120, 1024, 8, 32, 128])
    #model.load_state_dict(torch.load("basemodel.pt"))
    #model = LitMLP.ResNet()


    data_loader = LitMLP.LitDataModule(train_data="../TrainData/reinfformatted.train",
                                       val_data="../TrainData/weird9formatted.train",
                                       batch_size=32000, buffer_size=60000000, p_range=[0, 24],
                                       input_format=model.input_format)

    # val_loader =  data_loader.val_dataloader()
    # batch = next(iter(val_loader))

    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{policy}")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=300, callbacks=[check_point_callback])
    trainer.fit(model, data_loader)




