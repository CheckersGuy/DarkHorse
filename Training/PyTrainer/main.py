import LitMLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

if __name__ == "__main__":
    data_loader = LitMLP.LitDataModule(train_data="/home/robin/DarkHorse/Training/TrainData/small_dataset4.train",
                                       val_data="/home/robin/DarkHorse/Training/TrainData/medium_dataset.train",
                                       batch_size=20000, buffer_size=100000000)
    device = torch.device("cpu")

    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".",filename="{epoch}")

    model = LitMLP.Network([120, 256, 32, 32, 1])
    model.to(device)
    trainer = pl.Trainer(max_epochs=200, callbacks=[check_point_callback])
    trainer.fit(model, data_loader)
