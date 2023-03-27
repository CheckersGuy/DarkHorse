from subprocess import run
import grpc
import LitMLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import Helper as h
import torch.nn.functional as F
import numpy as np
import ctypes
import pathlib
import os.path
import os
import sys
import logging
logging.basicConfig(filename="trainer.log",encoding="utf-8",level = logging.DEBUG)

libname = pathlib.Path().absolute().__str__() + "/libpyhelper.so"
c_lib = ctypes.CDLL(libname)
import generator_pb2

def merge_data(files,output):
    out_batch = generator_pb2.Batch()
    for file in files:
        with open(file,"rb") as f:
            data = f.read()
            batch = generator_pb2.Batch()
            batch.ParseFromString(data)
            out_batch.games.extend(batch.games)
    
    with open(output,"wb") as f:
        data = out_batch.SerializeToString()
        f.write(data)




if __name__ == "__main__":
    batch_size = 4*8192
    epochs = 30
    model = LitMLP.Network(output="nonwdl",hidden=[120, 1024, 8, 32, 1])
    model.batch_size=batch_size
    model.number_of_steps=23208612
    model.num_epochs=epochs
    data_loader = LitMLP.LitDataModule(train_data="TrainData/merged.train",
                                       val_data="TrainData/val.train",
                                       batch_size=batch_size, buffer_size=35000000)


    # val_loader =  data_loader.val_dataloader()
    # batch = next(iter(val_loader))

    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{Networks/medium}")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, callbacks=[check_point_callback])


    trainer.fit(model, data_loader)
    model.save_quantized("Networks/{}.quant".format("nonwdltest2"))
    torch.save(model.state_dict(),"Networks/{}.pt".format("nonwdlnext"))
    
    #model = LitMLP.PatternModel()


merge_data(["TrainData/testwindow.train","TrainData/merged.train"],"TrainData/merged.train")


