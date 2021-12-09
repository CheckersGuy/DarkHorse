import LitMLP
import pytorch_lightning as pl
import torch




if __name__ == "__main__":
    data_loader = LitMLP.LitDataModule(train_data="/home/robin/DarkHorse/Training/TrainData/big_data_shuffl.train",
                                        val_data="/home/robin/DarkHorse/Training/TrainData/big_data.val",
                                        batch_size=20000, buffer_size=16000000)
    device = torch.device("cpu")
    model = LitMLP.Network([120, 256, 32, 32, 1])
    model.to(device)
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, data_loader)
