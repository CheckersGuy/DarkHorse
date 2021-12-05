import LitMLP
import pytorch_lightning as pl
import torch
import jax
import numpy as np
from jax import grad, jit, vmap
from LitMLP import PattBatchDataSet
import jax.numpy as jnp
import timeit
from jax import random
from LitMLP import PattBatchDataSet

batched_patterns = jnp.zeros(shape=(20, 10))


@jit
def pattern_eval(weights, patterns):
    # very simple at first
    # no tapered evaluation
    result = jnp.take(weights, patterns)
    eval = jnp.sum(result)
    return jax.nn.sigmoid(eval)


# weights = jnp.arange(100, dtype=np.float32)
# pattern = np.array([20, 10, 13])
#
# gr = grad(pattern_eval)(weights, pattern)
#
# print(len(gr))

# see how this works with vmap

if __name__ == "__main__":
    data_loader = LitMLP.LitDataModule(train_data="/home/robin/DarkHorse/Training/TrainData/big_data_shuffl.train",
                                        val_data="/home/robin/DarkHorse/Training/TrainData/big_data.val",
                                        batch_size=20000, buffer_size=16000000)
    device = torch.device("cpu")
    model = LitMLP.Network([120, 512, 16, 32, 1])
    model.to(device)
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, data_loader)
