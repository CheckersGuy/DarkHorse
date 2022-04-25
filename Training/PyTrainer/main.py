import LitMLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
#2366956
if __name__ == "__main__":
    data_loader = LitMLP.LitDataModule(train_data="/home/robin/DarkHorse/Training/TrainData/verylargexxxx.train",
                                       val_data="/home/robin/DarkHorse/Training/TrainData/smalldataset7.train",
                                       batch_size=2000, buffer_size=10000000, p_range=[6, 24])
    device = torch.device("cpu")

    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{epoch}")

    model = LitMLP.Network([120, 256, 32, 32, 1])
    #model = LitMLP.PolicyNetwork([120, 256, 32, 32, 100])
    model.to(device)
    trainer = pl.Trainer(max_epochs=2000, callbacks=[check_point_callback])
    trainer.fit(model, data_loader,ckpt_path="epoch=3.ckpt")
