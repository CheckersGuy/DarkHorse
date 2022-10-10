import LitMLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import Helper as h
import torch.nn.functional as F
import numpy as np
def test_model():
    model = LitMLP.ResNet.load_from_checkpoint("conv4=0-v2.ckpt")
    fen_string = "W:WK6,W12:B13,2"
    h.print_fen_string(fen_string)
    with torch.inference_mode():
        model.eval()
        input = h.create_input(fen_string)
        input = torch.unsqueeze(input, 0)
        print(input.shape)
        policy, value = model.forward(input)
        print(value)
        value = torch.argmax(policy).item()
        pol_value = policy.squeeze()
        pol_value = F.softmax(pol_value, dim=0)
        print(pol_value[value])
        f = value // 32
        t = value % 32
        print(f)
        print(t)
        print("Policy: ", pol_value)


# test_model()
#
# model = LitMLP.Network([120, 1024, 8, 32, 1],output="basemodel")
# model.init_weights()
# model.save_model_weights()




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




