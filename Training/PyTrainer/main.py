import LitMLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import Helper as h


def test_model():
    model = LitMLP.ConvNet.load_from_checkpoint("conv3=0-v1.ckpt")
    fen_string ="B:W18,20,21,23,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,9,11,14,16"
    h.print_fen_string(fen_string)
    with torch.inference_mode():
        model.eval()
        input = h.create_input(fen_string)
        input = torch.unsqueeze(input, 0)
        print(input.shape)
        out = model.forward(input)
        print(out)

if __name__ == "__main__":
    data_loader = LitMLP.LitDataModule(train_data="../TrainData/verylargexxxx.train",
                                       val_data="../TrainData/smalldataset7.train",
                                       batch_size=512, buffer_size=20000000, p_range=[6, 24])

    check_point_callback = ModelCheckpoint(every_n_epochs=1, dirpath=".", filename="{conv4}")

    # model = LitMLP.Network([120, 256, 32, 32, 1])
    model = LitMLP.ConvNet()

    # model = LitMLP.PolicyNetwork([120, 256, 32, 32, 100])
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=2000, callbacks=[check_point_callback])
    trainer.fit(model, data_loader)
