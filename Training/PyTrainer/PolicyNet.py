import torch.nn as nn
import torch
import numpy as np
import Helper as h
import struct


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)


class Relu1(nn.Module):
    def __init__(self):
        super(Relu1, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class Network(nn.Module):

    def __init__(self, hidden):
        super(Network, self).__init__()
        layers = []
        for i in range(len(hidden) - 2):
            layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            layers.append(Relu1())

        layers.append(nn.Linear(hidden[len(hidden) - 2], hidden[len(hidden) - 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net.forward(x)

    @staticmethod
    def transform(wp, bp, k, color, moved):
        black_men = bp & (~k)
        white_men = wp & (~k)
        black_king = bp & k
        white_king = wp & k

        if color == -1:
            tempp = black_men
            tempk = black_king
            black_men = h.invert_pieces(white_men)
            white_men = h.invert_pieces(tempp)
            black_king = h.invert_pieces(white_king)
            white_king = h.invert_pieces(tempk)

        white_p = np.array([white_men])
        black_p = np.array([black_men])
        kings_white = np.array([white_king])
        kings_black = np.array([black_king])
        ind_kings = 2 ** np.arange(32).reshape(-1, 1)
        ind_white = 2 ** np.arange(4, 32).reshape(-1, 1)
        ind_black = 2 ** np.arange(0, 28).reshape(-1, 1)
        comp_kings_black = (np.bitwise_and(kings_black, ind_kings) != 0).astype(np.float32).flatten()
        comp_kings_white = (np.bitwise_and(kings_white, ind_kings) != 0).astype(np.float32).flatten()
        comp_wp = (np.bitwise_and(white_p, ind_white) != 0).astype(np.float32)
        comp_wp = comp_wp.flatten()
        comp_bp = (np.bitwise_and(black_p, ind_black) != 0).astype(np.float32).flatten()

        output = np.concatenate((comp_wp, comp_bp, comp_kings_white, comp_kings_black))
        return torch.LongTensor([moved]), torch.from_numpy(output)

    def init_weights(self):
        self.net.apply(init_weights)

    def evaluate(self, wp, bp, k, color):
        result, inp = self.transform(wp, bp, k, color, - 1)
        return self.forward(inp)

    def save_parameters(self, output):
        buffer_weights = bytearray()
        buffer_bias = bytearray()
        num_weights = 0
        num_bias = 0
        file = open(output, "wb")
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().numpy().flatten("F")
                bias = layer.bias.detach().numpy().flatten("F")
                buffer_weights += weights.tobytes()
                buffer_bias += bias.tobytes()
                num_weights += len(weights)
                num_bias += len(bias)

        file.write(struct.pack("I", num_weights))
        file.write(buffer_weights)
        file.write(struct.pack("I", num_bias))
        file.write(buffer_bias)
        file.close()

        return

    def save(self, output):
        torch.save(self.state_dict(), output)
