import math
import torch.nn as nn
import torch
from torch.nn.modules import activation
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from ranger import Ranger
import os
import struct
import numpy as np
import torch.nn as nn
import torch
import pytorch_lightning as pl
import struct
import numpy as np
import string_sum
from torch.utils.data import DataLoader
from muon import MuonWithAuxAdam
from BucketLinear import BucketLinear
from BucketLinear import BucketedMixin
from rangerlite import RangerLite
# below will be moved into the network

import torch.nn.functional as F

NEG = -1e9  # finite "minus infinity": 0 * NEG = 0, whereas 0 * -inf = NaN

L1 = 2*2048
L2 = 32
L3 = 32


class Network(BucketedMixin, pl.LightningModule):

    def __init__(self, run_name="square_relu"):
        super(Network, self).__init__()
        self.delta_l1 = 0.0            # start with 0: plain factorization first
        self.delta_lr_mult = 1.0
        self.val_outputs = []
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0

        self.accu = nn.Linear(120, L1)
        self.num_buckets = 128
        self.gamma = 0.998
        self.run_name = run_name
        self.layer_one = BucketLinear(L1 // 2, L2, self.num_buckets)
        self.layer_sec = BucketLinear(L2, L3, self.num_buckets)
        # Policy: BucketLinear(L3, 128, nb)
        self.output = BucketLinear(L3, 1, self.num_buckets)
        self.layers = [self.accu, self.layer_one, self.layer_sec, self.output]

    def forward(self, x, buckets):
        offset = torch.arange(
            0, x.shape[0]*self.num_buckets, self.num_buckets, device=buckets.device)
        indices = buckets.flatten()+offset

        ac = self.accu.forward(x)
        ac = torch.clamp(ac, 0.0, 1.0)

        ac_x, ac_y = ac.split(L1//2, dim=1)
        ac_out = ac_x.mul(ac_y)*(127.0/128.0)

        l1s = self.layer_one(ac_out).reshape((-1, self.num_buckets, L2))
        l1c = l1s.view(-1, L2)[indices]
        l1c = torch.clamp(l1c, 0.0, 1.0)

        l2s = self.layer_sec(l1c).reshape((-1, self.num_buckets, L3))
        l2c = l2s.view(-1, L3)[indices]
        l2c = torch.clamp(l2c, 0.0, 1.0)

        l5s = self.output(l2c).reshape((-1, self.num_buckets, 1))
        l5c = l5s.view(-1, 1)[indices]
        out = torch.sigmoid(l5c)
        return out

    def init_layers(self):
        l1_weight = self.layer_one.weight
        l1_bias = self.layer_one.bias
        l2_weight = self.layer_sec.weight
        l2_bias = self.layer_sec.bias
        output_weight = self.output.weight
        output_bias = self.output.bias

        with torch.no_grad():
            output_bias.fill_(0.0)
            for i in range(1, self.num_buckets):
                l1_weight[i*L2: (i+1)*L2, :] = l1_weight[0: L2, :]
                l1_bias[i*L2: (i+1)*L2] = l1_bias[0: L2]
                l2_weight[i*L3: (i+1)*L3, :] = l2_weight[0: L3, :]
                l2_bias[i*L3: (i+1)*L3] = l2_bias[0: L3]
                output_weight[i:i+1, :] = output_weight[0:1, :]

    def write_header(self, file_out):
        num_layers = len(self.layers)
        file_out.write(struct.pack("I", num_layers))
        file_out.write(struct.pack("I", self.num_buckets))

        file_out.write(struct.pack("I", self.accu.in_features))
        file_out.write(struct.pack("I", self.accu.out_features))

        for layer in self.layers[1:]:
            file_out.write(struct.pack("I", layer.in_features))
            file_out.write(struct.pack(
                "I", layer.out_features//self.num_buckets))

    def configure_optimizers(self):
        # optimizer = Ranger(self.parameters(),lr=1.0, eps=1.0e-3, use_gc=False,gc_loc=False,weight_decay=0)
        optimizer = RangerLite(self.parameters(), lr=5.0e-3,
                               eps=1.0e-5, weight_decay=0.0)
        groups = self.bucket_param_groups(5.0e-3, self.delta_lr_mult)
        optimizer = RangerLite(groups, lr=5.0e-3, eps=1.0e-5, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        self.step()
        result, eval, move, buckets, x, legal_moves = train_batch
        eval = 1.0 * torch.sigmoid(eval/127.0)
        out = self.forward(x, buckets)
        loss = torch.pow(torch.abs(out-eval), 2).mean()
        self.log('train_loss', loss.detach(), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        result, eval, move, buckets, psqt_buckets, x, legal_moves = val_batch
        out = self.forward(x, buckets)
        eval = 1.0 * torch.sigmoid(eval/127.0)
        loss = torch.pow(torch.abs(out - eval), 2).mean()
        self.log('val_loss', loss.detach())
        self.val_outputs.append(loss)
        return {"val_loss": loss.detach()}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_outputs).mean()
        self.val_outputs.clear()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        self.log('loss', avg_loss, prog_bar=True)
        return {"loss": avg_loss, "log": tensorboard_logs}

    def on_train_epoch_end(self) -> None:
        display = "{name_run}_{epoch_count}.quant".format(
            name_run=self.run_name, epoch_count=self.current_epoch)
        self.save_quantized_bucket(display)
        return

    def save_quantized_bucket(self, output):
        self.step()
        # write the header file
        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max
        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)

        ft_weights_buf = bytearray()
        ft_bias_buf = bytearray()
        hidden_buf = bytearray()

        num_weights = 0
        num_bias = 0
        file = open(output, "wb")

        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0
        weights = np.round(weights)
        weights = np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0
        bias = np.round(bias)
        bias = np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        ft_weights_buf.extend(weights.tobytes())
        ft_bias_buf.extend(bias.tobytes())

        for layer in self.layers[1:]:
            weights = layer.weight.detach()
            weights = torch.clip(
                weights, self.min_weight_hidden, self.max_weight_hidden) * 64.0
            weights = torch.round(weights)

            bias = layer.bias.detach()
            bias = bias*(127.0 * 64.0)
            bias = torch.round(bias)

            size = layer.weight.size()
            rows = size[0]//self.num_buckets
            cols = len(layer.bias)//self.num_buckets
            bucket_weights = torch.split(weights, rows)
            bucket_bias = torch.split(bias, cols)
            for i in range(self.num_buckets):
                hidden_buf.extend(bucket_weights[i].numpy(
                ).flatten().astype(np.int8).tobytes())
                hidden_buf.extend(bucket_bias[i].numpy(
                ).flatten().astype(np.int32).tobytes())

        file.write(ft_weights_buf)
        file.write(ft_bias_buf)
        file.write(hidden_buf)
        file.close()
        self.to(device_gpu)

        return


class MLHNetwork(BucketedMixin, pl.LightningModule):

    def __init__(self, run_name):
        super(MLHNetwork, self).__init__()
        self.delta_l1 = 0.0            # start with 0: plain factorization first
        self.delta_lr_mult = 1.0
        self.val_outputs = []
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0
        self.gamma = 0.93
        self.run_name = run_name
        self.num_buckets = 128
        self.accu = nn.Linear(120, L1)
        self.layer_one = BucketLinear(L1 // 2, L2, self.num_buckets)
        self.layer_sec = BucketLinear(L2, L3, self.num_buckets)
        # Policy: BucketLinear(L3, 128, nb)
        self.output = BucketLinear(L3, 1, self.num_buckets)
        self.layers = [self.accu, self.layer_one, self.layer_sec, self.output]

        self.init_layers()

    def forward(self, x, buckets):
        offset = torch.arange(
            0, x.shape[0]*self.num_buckets, self.num_buckets, device=buckets.device)
        indices = buckets.flatten()+offset

        ac = self.accu.forward(x)
        ac = torch.clamp(ac, 0.0, 1.0)

        ac_x, ac_y = ac.split(L1//2, dim=1)
        ac_out = ac_x.mul(ac_y)*(127.0/128.0)

        l1s = self.layer_one(ac_out).reshape((-1, self.num_buckets, L2))
        l1c = l1s.view(-1, L2)[indices]
        l1c = torch.clamp(l1c, 0.0, 1.0)

        l2s = self.layer_sec(l1c).reshape((-1, self.num_buckets, L3))
        l2c = l2s.view(-1, L3)[indices]
        l2c = torch.clamp(l2c, 0.0, 1.0)

        l5s = self.output(l2c).reshape((-1, self.num_buckets, 1))
        l5c = l5s.view(-1, 1)[indices]
        out = l5c
        return out

    def init_layers(self):
        l1_weight = self.layer_one.weight
        l1_bias = self.layer_one.bias
        l2_weight = self.layer_sec.weight
        l2_bias = self.layer_sec.bias
        output_weight = self.output.weight
        output_bias = self.output.bias

        with torch.no_grad():
            output_bias.fill_(0.0)
            for i in range(1, self.num_buckets):
                l1_weight[i*L2: (i+1)*L2, :] = l1_weight[0: L2, :]
                l1_bias[i*L2: (i+1)*L2] = l1_bias[0: L2]
                l2_weight[i*L3: (i+1)*L3, :] = l2_weight[0: L3, :]
                l2_bias[i*L3: (i+1)*L3] = l2_bias[0: L3]
                output_weight[i:i+1, :] = output_weight[0:1, :]

    def write_header(self, file_out):
        num_layers = len(self.layers)
        file_out.write(struct.pack("I", num_layers))
        file_out.write(struct.pack("I", self.num_buckets))

        file_out.write(struct.pack("I", self.accu.in_features))
        file_out.write(struct.pack("I", self.accu.out_features))

        for layer in self.layers[1:]:
            file_out.write(struct.pack("I", layer.in_features))
            file_out.write(struct.pack(
                "I", layer.out_features//self.num_buckets))

    def configure_optimizers(self):
        # optimizer = Ranger(self.parameters(),lr=1.0, eps=1.0e-3, use_gc=False,gc_loc=False,weight_decay=0)
        groups = self.bucket_param_groups(5.0e-3, self.delta_lr_mult)
        optimizer = RangerLite(groups, lr=5.0e-3, eps=1.0e-5, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        self.step()
        result, eval, mlh_target, buckets, x, legal_moves = train_batch
        mlh_target = torch.clamp(mlh_target.to(dtype=torch.float32), 0, 300)
        mlh_target = mlh_target/300.0
        out = self.forward(x, buckets)
        loss = torch.pow(torch.abs(out-mlh_target), 2.0).mean()
        self.log('train_loss', loss.detach(), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        result, eval, mlh_target, buckets, x, legal_moves = val_batch
        mlh_target = torch.clamp(mlh_target.to(dtype=torch.float32), 0, 300)
        mlh_target = mlh_target/300.0
        out = self.forward(x, buckets)
        loss = torch.pow(torch.abs(out - mlh_target), 2.0).mean()
        self.log('val_loss', loss.detach())
        self.val_outputs.append(loss)
        return {"val_loss": loss.detach()}

    def on_train_epoch_end(self) -> None:
        display = "{name_run}_{epoch_count}.quant".format(
            name_run=self.run_name, epoch_count=self.current_epoch)
        self.save_quantized_bucket(display)
        return

    def save_quantized_bucket(self, output):
        self.step()
        # write the header file
        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max
        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)

        ft_weights_buf = bytearray()
        ft_bias_buf = bytearray()
        hidden_buf = bytearray()

        num_weights = 0
        num_bias = 0
        file = open(output, "wb")

        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0
        weights = np.round(weights)
        weights = np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0
        bias = np.round(bias)
        bias = np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        ft_weights_buf.extend(weights.tobytes())
        ft_bias_buf.extend(bias.tobytes())

        for layer in self.layers[1:]:
            weights = layer.weight.detach()
            weights = torch.clip(
                weights, self.min_weight_hidden, self.max_weight_hidden)*64.0
            weights = torch.round(weights)

            bias = layer.bias.detach()
            bias = bias*(127.0 * 64.0)
            bias = torch.round(bias)

            size = layer.weight.size()
            rows = size[0]//self.num_buckets
            cols = len(layer.bias)//self.num_buckets
            bucket_weights = torch.split(weights, rows)
            bucket_bias = torch.split(bias, cols)
            for i in range(self.num_buckets):
                hidden_buf.extend(bucket_weights[i].numpy(
                ).flatten().astype(np.int8).tobytes())
                hidden_buf.extend(bucket_bias[i].numpy(
                ).flatten().astype(np.int32).tobytes())

        file.write(ft_weights_buf)
        file.write(ft_bias_buf)
        file.write(hidden_buf)
        file.close()
        self.to(device_gpu)

        return


class FastRangerLite(torch.optim.RAdam):
    """Same update as rangerlite.RangerLite (RAdam + lookahead every k steps), but
    fast: the RAdam part runs compiled with capturable=True (step counters on the
    GPU, no host syncs) and the lookahead uses one Python step counter instead of
    an .item() per parameter (all parameters get gradients every step, so their
    counters are identical). cached_params is kept in the optimizer state like
    RangerLite, so checkpoints stay compatible."""

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.999), eps=1e-5, weight_decay=0.0,
                 lookahead_steps=6, lookahead_alpha=0.5, device="cuda", cudagraphs=False):
        params = list(params)
        for g in params:     # tensor LR: changing it (scheduler) needs no recompile
            if isinstance(g, dict):
                g["lr"] = torch.tensor(float(g.get("lr", lr)), device=device)
        super().__init__(params, lr=torch.tensor(float(lr), device=device), betas=betas,
                         eps=eps, weight_decay=weight_decay, decoupled_weight_decay=True,
                         capturable=True, foreach=True)
        self.lookahead_steps = lookahead_steps
        self.lookahead_alpha = lookahead_alpha
        self._n = None
        mode = "reduce-overhead" if cudagraphs else None
        self._radam_step = torch.compile(super().step, mode=mode)
        self._lookahead = torch.compile(self._lookahead_impl, mode=mode)

    def load_state_dict(self, state_dict):
        # a checkpoint (also from RangerLite) brings its own group settings
        # (capturable=False, float lr); restore ours so the compiled step keeps
        # running on the capturable path, and re-derive the step counter
        super().load_state_dict(state_dict)
        for g in self.param_groups:
            dev = g["params"][0].device
            g["capturable"] = True
            g["foreach"] = True
            g["lr"] = torch.tensor(float(g["lr"]), device=dev)
            for p in g["params"]:
                st = self.state.get(p, {})
                if "step" in st:
                    st["step"] = st["step"].to(device=dev, dtype=torch.float32)
        self._n = None

    def _all_params(self):
        return [p for g in self.param_groups for p in g["params"] if p.grad is not None]

    @torch.no_grad()
    def _lookahead_impl(self, ps, cached):
        torch._foreach_mul_(ps, self.lookahead_alpha)
        torch._foreach_add_(ps, cached, alpha=1.0 - self.lookahead_alpha)
        torch._foreach_copy_(cached, ps)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        ps = self._all_params()
        if self._n is None:                           # also after loading a checkpoint
            st = self.state[ps[0]]
            self._n = int(st["step"].item() + 0.1) if "step" in st else 0
        # parameters before their first update (like RangerLite); stored in the state
        # only after the RAdam step, which initialises its own state when it is empty
        pre = {p: p.detach().clone() for p in ps if "cached_params" not in self.state[p]}
        self._radam_step()
        for p, c in pre.items():
            self.state[p]["cached_params"] = c
        self._n += 1
        if self._n % self.lookahead_steps == 0:
            self._lookahead(ps, [self.state[p]["cached_params"] for p in ps])
        return loss


BLOCK_ROWS = 32   # rows per bucket block (32 measured fastest on the RTX 3060 Ti)


def bucket_blocks(buckets, num_buckets=128, rows=BLOCK_ROWS):
    """Groups a batch by bucket into blocks of `rows` samples that share a bucket.
    Returns (blk_rows [nblk, rows] sample index per slot, padding = B -> a zero row,
             blk_bucket [nblk], pos [B] flat slot of each sample).
    Runs in the DataLoader worker (numpy), so the GPU step needs no sync."""
    b = np.asarray(buckets).reshape(-1).astype(np.int64)
    B = len(b)
    order = np.argsort(b, kind="stable")
    sb = b[order]
    cnt = np.bincount(sb, minlength=num_buckets)
    nblk_per = (cnt + rows - 1) // rows
    blk_start = np.concatenate(([0], np.cumsum(nblk_per)[:-1]))
    first = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    rank = np.arange(B) - first[sb]                  # rank inside its bucket
    slot = (blk_start[sb] + rank // rows) * rows + rank % rows
    nblk = int(nblk_per.sum())
    blk_rows = np.full(nblk * rows, B, dtype=np.int64)
    blk_rows[slot] = order
    pos = np.empty(B, dtype=np.int64)
    pos[order] = slot
    blk_bucket = np.repeat(np.arange(num_buckets, dtype=np.int64), nblk_per)
    return blk_rows.reshape(nblk, rows), blk_bucket, pos


def bucket_blocks_gpu(buckets, num_buckets=128, rows=BLOCK_ROWS):
    """GPU version of bucket_blocks with a FIXED block count
    nblk_max = ceil(B/rows) + num_buckets (an upper bound of sum ceil(cnt_b/rows)),
    so no host sync is needed and all shapes are static (torch.compile friendly).
    Unused blocks only hold padding rows (index B -> zero row) and are discarded."""
    b = buckets.reshape(-1).long()
    B = b.numel()
    nblk_max = (B + rows - 1) // rows + num_buckets
    order = torch.argsort(b, stable=True)
    sb = b[order]
    # scatter_add instead of torch.bincount: bincount syncs with the host (it
    # sizes its output from max()), which would stall the CPU every step
    cnt = torch.zeros(num_buckets, dtype=torch.long, device=b.device).scatter_add_(
        0, b, torch.ones_like(b))
    nblk_per = (cnt + rows - 1) // rows
    blk_end = torch.cumsum(nblk_per, 0)
    blk_start = blk_end - nblk_per
    first = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(B, device=b.device) - first[sb]
    slot = (blk_start[sb] + rank // rows) * rows + rank % rows
    blk_rows = torch.full((nblk_max * rows,), B, dtype=torch.long, device=b.device)
    blk_rows[slot] = order
    pos = torch.empty(B, dtype=torch.long, device=b.device)
    pos[order] = slot
    blk_bucket = torch.searchsorted(blk_end, torch.arange(nblk_max, device=b.device),
                                    right=True).clamp_(max=num_buckets - 1)
    return blk_rows.view(nblk_max, rows), blk_bucket, pos


class PolicyNetwork(BucketedMixin, pl.LightningModule):

    def __init__(self, run_name, l1=256, label_mode="soft", temperature=50.0,
                 tie_tol=10.0, blunder_cp=50.0):
        """l1 = width of ONE accumulator half (the C++ template's L1)."""
        super().__init__()
        assert label_mode in ("soft", "hard")
        self.save_hyperparameters()
        self.l1 = l1
        self.label_mode = label_mode
        self.temperature = temperature
        self.tie_tol = tie_tol          # cp: moves this close to the best count as a tie-hit
        self.blunder_cp = blunder_cp    # cp: regret above this counts as a blunder
        self.run_name = run_name
        self.gamma = 0.93
        self.num_buckets = 128
        self.delta_l1 = 0.0
        self.delta_lr_mult = 1.0
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0

        self.accu = nn.Linear(120, 2 * l1)
        self.layer_one = BucketLinear(l1, L2, self.num_buckets)
        self.layer_sec = BucketLinear(L2, L3, self.num_buckets)
        self.n_out = 128
        self.output = BucketLinear(L3, self.n_out, self.num_buckets)
        self.layers = [self.accu, self.layer_one, self.layer_sec, self.output]
        # init_layers() removed, see the note at the end

    # ---------------------------------------------------------------- model
    def _select(self, t, buckets, width):
        B = t.shape[0]
        t = t.reshape(B, self.num_buckets, width)
        return t[torch.arange(B, device=t.device), buckets.flatten()]

    @staticmethod
    def _block_layer(layer, X, blk_bucket):
        # X [nblk, rows, in] -> [nblk, rows, out], each block with its bucket's weights
        W = (layer.shared_weight.unsqueeze(0) + layer.delta_weight)[blk_bucket]
        bias = (layer.shared_bias.unsqueeze(0) + layer.delta_bias)[blk_bucket]
        return torch.baddbmm(bias.unsqueeze(1), X, W.transpose(1, 2))

    # quantisation-aware training (default since 2026-11-25): the forward pass computes
    # exactly the engine's integer pipeline (fake-quantised weights, integer
    # activations, bit-exact with the engine), gradients pass the rounding unchanged
    # (straight-through). Training, validation metrics and the exported net all see
    # the int8 net the engine runs. qat = False gives the plain float forward
    # (analysis / tests only).
    qat = True

    @staticmethod
    def _ste(v, q):
        return v + (q - v).detach()

    def _qat_block_layer(self, layer, X, blk_bucket):
        # integer units: X = engine uint8 activations, result = engine int32 sum
        W = (layer.shared_weight.unsqueeze(0) + layer.delta_weight)[blk_bucket]
        W = torch.clamp(W, self.min_weight_hidden, self.max_weight_hidden) * 64.0
        W = self._ste(W, torch.round(W))                            # int8 weights
        b = (layer.shared_bias.unsqueeze(0) + layer.delta_bias)[blk_bucket] * (127.0 * 64.0)
        b = self._ste(b, torch.round(b))                            # int32 biases
        return torch.baddbmm(b.unsqueeze(1), X, W.transpose(1, 2))

    def _forward_qat(self, x, blocks):
        """Engine arithmetic (Network.h / Layer.h / LinearSparse.h / Simd.h), computed
        in integer units so every intermediate value is exact in fp32:
        accumulator int16 = round(127 w), a = clamp(acc, 0, 127), input of layer one
        (a_x * a_y) >> 7; hidden layers: int8 weights round(64 w), sum >> 6, clamp
        0..127; output: sum / 64 (truncated, value net) or >> 6 (policy net).
        Returns the output in the float units of forward() (engine int / 127)."""
        W = self.accu.weight * 127.0
        b = self.accu.bias * 127.0
        acc = F.linear(x, self._ste(W, torch.round(W)), self._ste(b, torch.round(b)))
        a = torch.clamp(acc, 0.0, 127.0)
        ax, ay = a.split(self.l1, dim=1)
        p = ax * ay / 128.0
        h = self._ste(p, torch.floor(p))
        blk_rows, blk_bucket, pos = blocks
        h = torch.cat([h, h.new_zeros(1, h.shape[1])])[blk_rows]
        for layer in (self.layer_one, self.layer_sec):
            y = self._qat_block_layer(layer, h, blk_bucket) / 64.0
            h = torch.clamp(self._ste(y, torch.floor(y)), 0.0, 127.0)
        y = self._qat_block_layer(self.output, h, blk_bucket) / 64.0
        y = self._ste(y, torch.trunc(y) if self.n_out == 1 else torch.floor(y)) / 127.0
        return y.reshape(-1, y.shape[-1])[pos]

    def forward(self, x, buckets, blocks=None):
        """blocks = (blk_rows, blk_bucket, pos) from bucket_blocks(): only the
        sample's own bucket is computed (same result as the dense path, ~3.5x
        faster). Without blocks: dense path over all buckets + select."""
        if self.qat:
            if blocks is None:
                blocks = bucket_blocks_gpu(buckets, self.num_buckets)
            return self._forward_qat(x, blocks)
        ac = torch.clamp(self.accu(x), 0.0, 1.0)
        ac_x, ac_y = ac.split(self.l1, dim=1)
        ac_out = ac_x.mul(ac_y) * (127.0 / 128.0)

        if blocks is not None:
            blk_rows, blk_bucket, pos = blocks
            h = torch.cat([ac_out, ac_out.new_zeros(1, ac_out.shape[1])])[blk_rows]
            h = torch.clamp(self._block_layer(self.layer_one, h, blk_bucket), 0.0, 1.0)
            h = torch.clamp(self._block_layer(self.layer_sec, h, blk_bucket), 0.0, 1.0)
            y = self._block_layer(self.output, h, blk_bucket)
            return y.reshape(-1, y.shape[-1])[pos]

        l1c = torch.clamp(self._select(
            self.layer_one(ac_out), buckets, L2), 0.0, 1.0)
        l2c = torch.clamp(self._select(
            self.layer_sec(l1c), buckets, L3), 0.0, 1.0)
        # raw outputs [B, n_out] (policy logits / value logit)
        return self._select(self.output(l2c), buckets, self.n_out)

    def write_header(self, file_out):
        file_out.write(struct.pack("I", len(self.layers)))
        file_out.write(struct.pack("I", self.num_buckets))
        file_out.write(struct.pack("I", self.accu.in_features))
        file_out.write(struct.pack("I", self.accu.out_features))
        for layer in self.layers[1:]:
            file_out.write(struct.pack("I", layer.in_features))
            file_out.write(struct.pack(
                "I", layer.out_features // self.num_buckets))

    # optimizer / schedule settings (train_policy.py --optimizer/--lr/--schedule/--warmup)
    opt_name = "rangerlite"     # rangerlite | radam (no lookahead) | adamw
    base_lr = 5.0e-3
    schedule = "step"           # step: StepLR per epoch (gamma); cosine: per step
    total_steps = None          # cosine: number of optimizer steps of the run
    warmup_frac = 0.0           # cosine: linear warm-up share of total_steps
    final_lr_ratio = 0.93 ** 29  # cosine ends where the 30-epoch step schedule ended

    def configure_optimizers(self):
        lr = self.base_lr
        groups = self.bucket_param_groups(lr, self.delta_lr_mult)
        if self.opt_name == "adamw":
            optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.95, 0.999), eps=1.0e-5,
                                          weight_decay=0.0, fused=True)
        elif self.use_compile:
            # same update as RangerLite, compiled (see FastRangerLite);
            # radam = the same without lookahead
            optimizer = FastRangerLite(groups, lr=lr, eps=1.0e-5, cudagraphs=self.use_cudagraphs,
                                       lookahead_steps=6 if self.opt_name == "rangerlite" else 10**12)
        else:
            assert self.opt_name == "rangerlite", "--no-compile only supports rangerlite"
            optimizer = RangerLite(groups, lr=lr, eps=1.0e-5, weight_decay=0.0)
        if self.schedule == "cosine":
            total = max(1, int(self.total_steps))
            warm = int(self.warmup_frac * total)
            lo = self.final_lr_ratio

            def factor(step):
                if step < warm:
                    return (step + 1) / warm
                t = min(1.0, (step - warm) / max(1, total - warm))
                return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * t))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]

    # ----------------------------------------------------- loss and metrics
    def _shared_step(self, batch, stage):
        _, buckets, x, q_score, q_mask = batch[:5]
        bs = x.shape[0]
        blocks = tuple(batch[5:8]) if len(batch) >= 8 else bucket_blocks_gpu(buckets, self.num_buckets)
        logits = self.forward(x, buckets, blocks)

        n_moves = q_mask.sum(dim=1)
        keep = n_moves >= 2          # one-move rows carry no policy signal
        self.log(f"{stage}/labeled_frac", keep.float().mean(),
                 on_step=(stage == "train"), on_epoch=(stage != "train"), batch_size=bs)
        if not keep.any():
            return None
        n = int(keep.sum())

        x, logits, q, m, n_moves = x[keep], logits[keep], q_score[keep], q_mask[keep], n_moves[keep]
        logits = logits.masked_fill(~m, NEG)
        # metrics only: clamp so mate scores (~±13000) don't dominate the regret means
        q_metric = q.clamp(-500, 500).masked_fill(~m, NEG)
        q = q.masked_fill(~m, NEG)

        log_p = F.log_softmax(logits, dim=1)
        best = q.argmax(dim=1)                                    # hard label
        target = torch.softmax(q / self.temperature, dim=1)       # soft label

        ce_hard = F.nll_loss(log_p, best)
        ce_soft = -(target * log_p).sum(dim=1).mean()
        loss = ce_soft if self.label_mode == "soft" else ce_hard

        with torch.no_grad():
            p = log_p.exp()
            best_score = q_metric.max(dim=1).values
            pred = logits.argmax(dim=1)
            pred_score = q_metric.gather(1, pred[:, None]).squeeze(1)
            regret = best_score - pred_score
            # illegal slots: zero regret
            q_safe = torch.where(m, q_metric, best_score[:, None])

            top3 = logits.topk(3, dim=1).indices
            hit = top3 == best[:, None]
            # tie-aware top-k: hit if any of the k highest-ranked moves is within
            # tie_tol of the best score (illegal slots get NEG and never count)
            tie_hit = (best_score[:, None] - q_metric.gather(1, top3)) <= self.tie_tol
            t_ent = -(target * torch.log(target.clamp_min(1e-12))).sum(dim=1)
            p_ent = -(p * log_p).sum(dim=1)

            stats = {
                "top1": hit[:, 0].float().mean(),
                "top2": hit[:, :2].any(dim=1).float().mean(),
                "top3": hit.any(dim=1).float().mean(),
                "top1_tie_aware": (regret <= self.tie_tol).float().mean(),
                "top2_tie_aware": tie_hit[:, :2].any(dim=1).float().mean(),
                "top3_tie_aware": tie_hit.any(dim=1).float().mean(),
                "regret_mean": regret.mean(),
                "blunder_rate": (regret > self.blunder_cp).float().mean(),
                "expected_regret": (p * (best_score[:, None] - q_safe)).sum(dim=1).mean(),
                "p_best": p.gather(1, best[:, None]).mean(),
                "confidence": p.max(dim=1).values.mean(),
                "pred_entropy": p_ent.mean(),
                "target_entropy": t_ent.mean(),
                "ce_hard": ce_hard.detach(),
                "ce_soft": ce_soft.detach(),
                "kl_soft": (ce_soft - t_ent.mean()).detach(),
                "avg_moves": n_moves.float().mean(),
            }

        is_train = stage == "train"
        self.log(f"{stage}/loss", loss.detach(), on_step=is_train, on_epoch=not is_train,
                 prog_bar=True, batch_size=n)
        for k, v in stats.items():
            self.log(f"{stage}/{k}", v, on_step=is_train, on_epoch=not is_train,
                     prog_bar=(k == "top1"), batch_size=n)

        if not is_train:
            # breakdowns by game phase and by branching factor (validation only)
            pieces = x.sum(dim=1)
            bins = {
                "phase_open": pieces >= 17,
                "phase_mid": (pieces >= 9) & (pieces <= 16),
                "phase_end": pieces <= 8,
                "moves_2": n_moves == 2,
                "moves_3_4": (n_moves >= 3) & (n_moves <= 4),
                "moves_5plus": n_moves >= 5,
            }
            for name, sel in bins.items():
                if sel.any():
                    self.log(f"val_by/{name}_top1", hit[sel, 0].float().mean(),
                             on_epoch=True, batch_size=int(sel.sum()))
                    self.log(f"val_by/{name}_top1_tie_aware", tie_hit[sel, 0].float().mean(),
                             on_epoch=True, batch_size=int(sel.sum()))
                    self.log(f"val_by/{name}_regret", regret[sel].mean(),
                             on_epoch=True, batch_size=int(sel.sum()))
        return loss

    def _train_loss(self, batch):
        """Same loss as _shared_step (mean over rows with >= 2 legal moves), but
        without GPU syncs: masked weighting instead of keep.any() / boolean indexing."""
        _, buckets, x, q_score, q_mask = batch[:5]
        blocks = tuple(batch[5:8]) if len(batch) >= 8 else bucket_blocks_gpu(buckets, self.num_buckets)
        logits = self.forward(x, buckets, blocks)
        keep = (q_mask.sum(dim=1) >= 2).float()
        logits = logits.masked_fill(~q_mask, NEG)
        q = q_score.masked_fill(~q_mask, NEG)
        log_p = F.log_softmax(logits, dim=1)
        if self.label_mode == "soft":
            ce = -(torch.softmax(q / self.temperature, dim=1) * log_p).sum(dim=1)
        else:
            ce = -log_p.gather(1, q.argmax(dim=1, keepdim=True)).squeeze(1)
        return (ce * keep).sum() / keep.sum().clamp_min(1.0)

    def _logging_step(self):
        # True on the steps Lightning writes step metrics (every log_every_n_steps);
        # the lean loop (lean_loop.py) sets _lean_log_now itself
        if getattr(self, "_lean", False):
            return self._lean_log_now
        return self.trainer._logger_connector.should_update_logs

    use_compile = True   # torch.compile the training loss (train_policy.py --no-compile)
    use_cudagraphs = False   # compile with mode="reduce-overhead" (default; train_policy.py --no-cudagraphs turns it off)

    def training_step(self, batch, batch_idx):
        if self.use_compile:
            if not hasattr(self, "_step_c"):
                self._step_c = torch.compile(          # weight clamp, ~4x faster
                    self.step, mode="reduce-overhead" if self.use_cudagraphs else None)
            self._step_c()
        else:
            self.step()
        if not self._logging_step():
            if self.use_compile:
                if not hasattr(self, "_train_loss_c"):
                    # static shapes (fixed block count) -> one compile, no recompiles
                    self._train_loss_c = torch.compile(
                        self._train_loss, dynamic=False,
                        mode="reduce-overhead" if self.use_cudagraphs else None)
                return self._train_loss_c(batch)
            return self._train_loss(batch)
        # logging step: full metrics (has GPU syncs, so only every log_every_n_steps)
        loss = self._shared_step(batch, "train")
        if loss is None:
            return (self.accu.weight * 0).sum()   # keeps the graph alive
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    # -------------------------------------------------------- housekeeping
    def on_before_optimizer_step(self, optimizer):
        if not self._logging_step():
            return
        norms = [p.grad.detach().norm(2)
                 for p in self.parameters() if p.grad is not None]
        if norms:
            self.log("train/grad_norm",
                     torch.norm(torch.stack(norms), 2), on_step=True)

    def _log_weight_stats(self):
        for name, layer in (("l1", self.layer_one), ("l2", self.layer_sec), ("out", self.output)):
            w = layer.weight.detach()
            self.log(f"weights/{name}_saturated",
                     (w.abs() >= self.max_weight_hidden - 1e-6).float().mean())
            self.log(f"weights/{name}_delta_ratio",
                     layer.delta_weight.abs().mean() /
                     layer.shared_weight.abs().mean().clamp_min(1e-9))
            exp = getattr(self.logger, "experiment", None)
            if exp is not None and hasattr(exp, "add_histogram"):
                exp.add_histogram(
                    f"weights/{name}", w.float().cpu(), self.current_epoch)

    def on_train_epoch_end(self):
        self._log_weight_stats()   # before save_quantized_bucket moves the model to the CPU
        display = "{}_{}.quant".format(self.run_name, self.current_epoch)
        self.save_quantized_bucket(display)

    def save_quantized_bucket(self, output):
        self.step()
        # write the header file
        min16 = np.iinfo(np.int16).min
        max16 = np.iinfo(np.int16).max
        device = torch.device("cpu")
        device_gpu = torch.device("cuda")
        self.to(device)

        ft_weights_buf = bytearray()
        ft_bias_buf = bytearray()
        hidden_buf = bytearray()

        num_weights = 0
        num_bias = 0
        file = open(output, "wb")

        layer = self.layers[0]
        weights = layer.weight.detach().numpy().flatten("F")
        weights = weights * 127.0
        weights = np.round(weights)
        weights = np.clip(weights, min16, max16)
        weights = weights.astype(np.int16)
        bias = layer.bias.detach().numpy().flatten("F")
        bias = bias * 127.0
        bias = np.round(bias)
        bias = np.clip(bias, min16, max16)
        bias = bias.astype(np.int16)
        ft_weights_buf.extend(weights.tobytes())
        ft_bias_buf.extend(bias.tobytes())

        for layer in self.layers[1:]:
            weights = layer.weight.detach()
            weights = torch.clip(
                weights, self.min_weight_hidden, self.max_weight_hidden)*64.0
            weights = torch.round(weights)

            bias = layer.bias.detach()
            bias = bias*(127.0 * 64.0)
            bias = torch.round(bias)

            size = layer.weight.size()
            rows = size[0]//self.num_buckets
            cols = len(layer.bias)//self.num_buckets
            bucket_weights = torch.split(weights, rows)
            bucket_bias = torch.split(bias, cols)
            for i in range(self.num_buckets):
                hidden_buf.extend(bucket_weights[i].numpy(
                ).flatten().astype(np.int8).tobytes())
                hidden_buf.extend(bucket_bias[i].numpy(
                ).flatten().astype(np.int32).tobytes())

        file.write(ft_weights_buf)
        file.write(ft_bias_buf)
        file.write(hidden_buf)
        file.close()
        self.to(device_gpu)

        return


class ValueNetwork(PolicyNetwork):
    """Value net <l1>,32,32,1 = C++ Network<l1, 32, 32, 1, 128> (engine: l1 2048),
    trained with the policy pipeline (compact loader, blocked bucket layers,
    torch.compile, FastRangerLite, lean loop). Same model, target and export as the
    old Network class: MSE between sigmoid(output) and sigmoid(eval / 127), eval =
    the stored root value of the POL1 sample clamped to +-500 (as BatchProvider.testing).
    Batches: (evals [B,1], buckets [B,1], inputs [B,120])."""

    EVAL_SCALE = 127.0

    def __init__(self, run_name, l1=2048):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.l1 = l1
        self.run_name = run_name
        self.gamma = 0.93
        self.num_buckets = 128
        self.delta_l1 = 0.0
        self.delta_lr_mult = 1.0
        self.max_weight_hidden = 127.0 / 64.0
        self.min_weight_hidden = -127.0 / 64.0
        self.n_out = 1
        self.accu = nn.Linear(120, 2 * l1)
        self.layer_one = BucketLinear(l1, L2, self.num_buckets)
        self.layer_sec = BucketLinear(L2, L3, self.num_buckets)
        self.output = BucketLinear(L3, 1, self.num_buckets)
        self.layers = [self.accu, self.layer_one, self.layer_sec, self.output]

    def _logit(self, batch):
        evals, buckets, x = batch[:3]
        blocks = tuple(batch[3:6]) if len(batch) >= 6 else bucket_blocks_gpu(buckets, self.num_buckets)
        return self.forward(x, buckets, blocks).view(-1), evals.view(-1), x

    def _train_loss(self, batch):
        logit, ev, _ = self._logit(batch)
        target = torch.sigmoid(ev / self.EVAL_SCALE)
        return (torch.sigmoid(logit) - target).pow(2).mean()

    TIE_TOL = 10.0   # cp: a choice this close to the best child counts as correct

    def training_step(self, batch, batch_idx):
        # the compiled loss must not see the variable-size child lists (recompiles)
        if not self._logging_step():
            batch = batch[:3]
        return super().training_step(batch, batch_idx)

    @torch.no_grad()
    def value_choice(self, pos, rows, encs, scores, B):
        """the value net picks among the quiet children (after the move the opponent has
        no capture; the static net only sees quiet positions in the search) by
        evaluating them: regret (cp) vs the best quiet-child score per root with >= 2
        quiet children, scores clamped to +-500 like the policy metrics"""
        cpos, cbuck = value_children(pos, rows, encs)
        quiet = ~white_has_jumps(*cpos.view(-1, 3).unbind(1))
        cpos, cbuck = cpos.view(-1, 3)[quiet].reshape(-1), cbuck[quiet]
        rows, scores = rows[quiet], scores[quiet]
        n = rows.numel()
        cpos = cpos.view(-1, 3)
        vals = torch.empty(n, device=pos.device)
        for s in range(0, n, B):          # chunks of B rows: same shapes as a batch
            e = min(n, s + B)
            p = torch.zeros(B, 3, dtype=cpos.dtype, device=pos.device)
            b = torch.zeros(B, 1, dtype=cbuck.dtype, device=pos.device)
            p[: e - s] = cpos[s:e]
            b[: e - s, 0] = cbuck[s:e]
            logit = self.forward(expand_inputs(p.reshape(-1), B), b,
                                 bucket_blocks_gpu(b, self.num_buckets)).view(-1)
            vals[s:e] = logit[: e - s]
        v = -vals                          # child value from the root mover's view
        sc = scores.clamp(-500, 500)
        R = int(pos.numel() // 3)
        neg = torch.full((R,), -1e9, device=pos.device)
        cnt = torch.zeros(R, device=pos.device).scatter_add_(0, rows, torch.ones_like(v))
        vmax = neg.clone().scatter_reduce_(0, rows, v, "amax", include_self=True)
        best = neg.clone().scatter_reduce_(0, rows, sc, "amax", include_self=True)
        # ties (equal values) count as a random choice among them: mean score
        tied = (v >= vmax[rows]).float()
        tsum = torch.zeros(R, device=pos.device).scatter_add_(0, rows, sc * tied)
        tcnt = torch.zeros(R, device=pos.device).scatter_add_(0, rows, tied)
        chosen = tsum / tcnt.clamp_min(1)
        keep = cnt >= 2
        return (best - chosen)[keep]

    def _shared_step(self, batch, stage):
        logit, ev, x = self._logit(batch)
        n = ev.numel()
        pred = torch.sigmoid(logit)
        target = torch.sigmoid(ev / self.EVAL_SCALE)
        err = pred - target
        loss = err.pow(2).mean()
        is_train = stage == "train"
        with torch.no_grad():
            cp = (logit * self.EVAL_SCALE).clamp(-500, 500)
            decided = ev.abs() >= 25
            stats = {
                "mae_cp": (cp - ev).abs().mean(),
                "sign_acc": ((cp > 0) == (ev > 0)).float().mul(decided).sum()
                / decided.sum().clamp_min(1),
                "bias": err.mean(),
                "pred_mean": pred.mean(),
                "target_mean": target.mean(),
            }
        self.log(f"{stage}/loss", loss.detach(), on_step=is_train, on_epoch=not is_train,
                 prog_bar=True, batch_size=n)
        for k, v in stats.items():
            self.log(f"{stage}/{k}", v, on_step=is_train, on_epoch=not is_train, batch_size=n)
        if len(batch) == 4:
            regret = self.value_choice(*batch[3], n)
            if regret.numel():
                c = regret.numel()
                self.log(f"{stage}/value_acc", (regret <= self.TIE_TOL).float().mean(),
                         on_step=is_train, on_epoch=not is_train, batch_size=c)
                self.log(f"{stage}/value_regret", regret.mean(),
                         on_step=is_train, on_epoch=not is_train, batch_size=c)
        if not is_train:
            # breakdowns by game phase and by eval band (calibration: bias = mean
            # pred - target in win-probability space; validation only)
            pieces = x.sum(dim=1)
            a = ev.abs()
            bins = {
                "phase_open": pieces >= 17,
                "phase_mid": (pieces >= 9) & (pieces <= 16),
                "phase_end": pieces <= 8,
                "band_0_50": a <= 50,
                "band_50_200": (a > 50) & (a <= 200),
                "band_200_500": (a > 200) & (a < 500),
                "band_500": a >= 500,
            }
            sq = err.pow(2)
            for name, sel in bins.items():
                c = int(sel.sum())
                if c:
                    self.log(f"val_by/{name}_loss", sq[sel].mean(), on_epoch=True, batch_size=c)
                    self.log(f"val_by/{name}_bias", err[sel].mean(), on_epoch=True, batch_size=c)
                    self.log(f"val_by/{name}_mae_cp", (cp[sel] - ev[sel]).abs().mean(),
                             on_epoch=True, batch_size=c)
        return loss


class PolicyBatchDataSet(torch.utils.data.IterableDataset):
    """one_pass=True: finite, restarts from the file start on every iter().
    Use it for validation so every validation epoch sees the same samples."""

    def __init__(self, batch_size, buffer_size, file_path, shuffle=True, one_pass=False,
                 rotate=False):
        super().__init__()
        # rotate: every pass over the file starts at a random record, so the shuffle
        # buffers (blocks of buffer_size consecutive samples) differ between epochs
        self.rotate = rotate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.file_path = file_path
        self.shuffle = shuffle
        self.one_pass = one_pass
        # the Rust loader is created lazily in the process that iterates (the
        # DataLoader worker), so the parent never holds a second buffer
        self.loader = None
        self._pid = None
        with open(file_path, "rb") as f:          # POL1 header: u32 magic, u64 count
            magic, self.num_samples = struct.unpack("<IQ", f.read(12))
        assert magic == 0x504F4C31, f"{file_path} is not a policy-sample file"
        self._served = 0

    def _make_loader(self):
        if self.rotate:
            try:
                return string_sum.PolicyBatchProvider(
                    self.file_path, self.buffer_size, self.batch_size, self.shuffle, True)
            except TypeError:   # extension built before the rotate option
                print("warning: string_sum without epoch rotation, reading without it", flush=True)
        return string_sum.PolicyBatchProvider(
            self.file_path, self.buffer_size, self.batch_size, self.shuffle)

    def __iter__(self):
        if self.one_pass or self.loader is None or self._pid != os.getpid():
            self.loader = self._make_loader()
            self._pid = os.getpid()
            self._served = 0
        return self

    def __len__(self):
        return self.num_samples // self.batch_size

    def fill_into(self, inputs, evals, buckets, q_score, q_mask):
        """Fills caller-owned arrays (flat numpy views, e.g. of pinned tensors) and
        returns the bucket blocks; raises StopIteration at the end of a one-pass set."""
        if self.one_pass and self._served >= len(self):
            raise StopIteration
        self._served += 1
        self.loader.fill(inputs, evals, buckets, q_score, q_mask)

    def fill_compact_into(self, slot):
        """fill_into for compact slots (see ThreadedBatchLoader); returns #children."""
        if self.one_pass and self._served >= len(self):
            raise StopIteration
        self._served += 1
        return self.loader.fill_compact(slot["pos"].numpy(), slot["evals"].numpy(),
                                        slot["buckets"].numpy(), slot["child_row"].numpy(),
                                        slot["child_enc"].numpy(), slot["child_score"].numpy())

    def fill_value_into(self, slot):
        """value-net batch (bitboards, root value, bucket; no children)"""
        if self.one_pass and self._served >= len(self):
            raise StopIteration
        self._served += 1
        self.loader.fill_value(slot["pos"].numpy(), slot["evals"].numpy(),
                               slot["buckets"].numpy())

    def __next__(self):
        if self.one_pass and self._served >= len(self):
            raise StopIteration
        self._served += 1
        B = self.batch_size
        inputs = np.zeros(B * 120, dtype=np.float32)
        evals = np.zeros(B, dtype=np.float32)
        buckets = np.zeros(B, dtype=np.int64)
        q_score = np.zeros(B * 128, dtype=np.float32)
        q_mask = np.zeros(B * 128, dtype=bool)
        self.loader.fill(inputs, evals, buckets, q_score, q_mask)
        blk_rows, blk_bucket, pos = bucket_blocks(buckets)
        return (evals.reshape(B, 1), buckets.reshape(B, 1),
                inputs.reshape(B, 120), q_score.reshape(B, 128),
                q_mask.reshape(B, 128), blk_rows, blk_bucket, pos)


# ---- value-net move choice (value_acc / value_regret) --------------------------------
# Child positions of the quiet root moves are built on the GPU from the policy move
# encodings (Move::get_move_encoding: 4*from + dir; dir 0: <<3/<<5 (MASK_L3/L5), 1: <<4,
# 2: >>4, 3: >>3/>>5 (MASK_R3/R5)), with Position::make_move semantics (white to move,
# promotion on squares 0-3), then colour-flipped to the side to move (getMirrored =
# 32-bit reversal) and bucketed like Pos::bucket_index (pext of the black pieces).
_MASK_L3, _MASK_L5, _MASK_R3, _MASK_R5 = 14737632, 117901063, 117901056, 3772834016
_PROMO_WHITE = 0xF
_BUCKET_SQUARES = (6, 9, 10, 13, 14, 17, 18)


def _rev32(v):
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1)
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2)
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4)
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8)
    return ((v >> 16) & 0xFFFF) | ((v & 0xFFFF) << 16)


def value_children(pos, rows, encs):
    """root bitboards [B*3] (white to move) + child rows/encodings -> child bitboards
    [n*3] (side to move = white after the flip) and their buckets [n]"""
    wp, bp, k = pos.view(-1, 3).unbind(1)
    WP, BP, K = wp[rows], bp[rows], k[rows]
    f = torch.ones_like(encs) << (encs // 4)
    d = encs % 4
    t0 = torch.where((f & _MASK_L3) != 0, f << 3, f << 5)
    t3 = torch.where((f & _MASK_R3) != 0, f >> 3, f >> 5)
    t = torch.where(d == 0, t0, torch.where(d == 1, f << 4, torch.where(d == 2, f >> 4, t3)))
    t = t & 0xFFFFFFFF
    WP2 = (WP & ~f) | t
    is_king = (K & f) != 0
    K2 = torch.where(is_king, (K & ~f) | t, torch.where((t & _PROMO_WHITE) != 0, K | t, K))
    cw, cb, ck = _rev32(BP), _rev32(WP2), _rev32(K2)
    bucket = torch.zeros_like(cb)
    for i, sq in enumerate(_BUCKET_SQUARES):
        bucket |= ((cb >> sq) & 1) << i
    return torch.stack([cw, cb, ck], dim=1).reshape(-1), bucket


def white_has_jumps(wp, bp, k):
    """Position::get_jumpers<WHITE>() != 0 on bitboards with white to move (int64)"""
    nocc = ~(wp | bp) & 0xFFFFFFFF
    kings = wp & k

    def fwd_black(x):
        return ((x & _MASK_L3) << 3) | ((x & _MASK_L5) << 5)

    def fwd_white(x):
        return ((x & _MASK_R3) >> 3) | ((x & _MASK_R5) >> 5)
    m = fwd_black((nocc << 4) & bp) & wp
    m |= (fwd_black(nocc) & bp) << 4 & wp
    m |= fwd_white((nocc >> 4) & bp) & kings
    m |= ((fwd_white(nocc) & bp) >> 4) & kings
    return (m & 0xFFFFFFFF) != 0


def expand_inputs(pos, B):
    """bitboards [B*3] (wp, bp, k, white to move) -> the 120 piece-square inputs
    (feature layout of AccumulatorState::apply / the Rust fill)"""
    wp, bp, k = pos.view(B, 3).unbind(1)
    sh = torch.arange(32, device=pos.device)

    def bits(v):
        return ((v[:, None] >> sh) & 1).float()

    x = torch.empty(B, 120, device=pos.device)
    x[:, 0:28] = bits(wp & ~k)[:, 4:32]     # white pawns, squares 4..31
    x[:, 28:56] = bits(bp & ~k)[:, 0:28]    # black pawns, squares 0..27
    x[:, 56:88] = bits(wp & k)              # white kings
    x[:, 88:120] = bits(bp & k)             # black kings
    return x


def expand_compact(pos, child_row, child_enc, child_score, B):
    """GPU side of PolicyBatchProvider.fill_compact: bitboards -> 120 inputs, child
    lists -> 128 scores and the legal-move mask. Produces exactly what fill() writes."""
    x = expand_inputs(pos, B)
    idx = child_row * 128 + child_enc
    q_score = torch.zeros(B * 128, device=pos.device)
    q_score[idx] = child_score
    q_mask = torch.zeros(B * 128, dtype=torch.bool, device=pos.device)
    q_mask[idx] = True
    return x, q_score.view(B, 128), q_mask.view(B, 128)


class ThreadedBatchLoader:
    """Feeds the GPU from a background thread (no worker process: passing ~9 MB
    per batch between processes cost ~20 ms on a busy CPU). The Rust loader
    writes straight into a ring of pinned host buffers; the main thread copies a
    buffer to the GPU asynchronously and records a CUDA event, and the thread
    refills that buffer only after the event (copy finished).
    Training sets keep one thread across epochs (prefetched batches carry over);
    one-pass sets (validation) restart from the file start on every iter()."""

    # max non-capture children per batch for the compact buffers (avg ~7.5 per position)
    CHILD_CAP_PER_SAMPLE = 64

    def __init__(self, dataset, depth=4, device="cuda", compact=True, kind="policy"):
        # kind="value": bitboards + root value + bucket only (fill_value), yields
        # (evals [B,1], buckets [B,1], inputs [B,120])
        assert kind in ("policy", "value")
        self.kind = kind
        self.ds = dataset
        self.depth = depth
        self.device = device
        # compact: Rust writes bitboards + child lists (~0.3 MB per batch), the GPU
        # expands them (expand_compact); otherwise the full 12 MB arrays (fill())
        self.compact = compact
        self._free = None
        self._ready = None

    def __len__(self):
        return len(self.ds)

    def _start(self):
        import queue
        import threading
        if self._free is not None:
            self._free.put(None)          # stop the previous thread
            t = getattr(self, "_thread", None)
            if t is not None:
                t.join(timeout=60)
        B = self.ds.batch_size
        self._free, self._ready = queue.Queue(), queue.Queue()
        cap = B * self.CHILD_CAP_PER_SAMPLE
        for _ in range(self.depth + 1):
            if self.compact or self.kind == "value":
                # value batches carry the child lists too (value_acc / value_regret)
                self._free.put({
                    "pos": torch.empty(B * 3, dtype=torch.int64).pin_memory(),
                    "evals": torch.empty(B, dtype=torch.float32).pin_memory(),
                    "buckets": torch.empty(B, dtype=torch.int64).pin_memory(),
                    "child_row": torch.empty(cap, dtype=torch.int64).pin_memory(),
                    "child_enc": torch.empty(cap, dtype=torch.int64).pin_memory(),
                    "child_score": torch.empty(cap, dtype=torch.float32).pin_memory(),
                    "event": None, "n": 0,
                })
                continue
            slot = {
                "evals": torch.empty(B, dtype=torch.float32).pin_memory(),
                "buckets": torch.empty(B, dtype=torch.int64).pin_memory(),
                "inputs": torch.empty(B * 120, dtype=torch.float32).pin_memory(),
                "q_score": torch.empty(B * 128, dtype=torch.float32).pin_memory(),
                "q_mask": torch.empty(B * 128, dtype=torch.bool).pin_memory(),
                "event": None,
            }
            self._free.put(slot)
        iter(self.ds)                      # (re)creates the Rust loader here
        free, ready = self._free, self._ready

        def work():
            while True:
                slot = free.get()
                if slot is None:
                    return
                if slot["event"] is not None:
                    slot["event"].synchronize()
                if self.kind == "value" or self.compact:
                    try:
                        slot["n"] = self.ds.fill_compact_into(slot)
                    except StopIteration:
                        ready.put(None)
                        return
                    except BaseException as e:
                        ready.put(e)
                        return
                    ready.put(slot)
                    continue
                # the Rust fill writes only the non-zero entries (expects zeroed
                # arrays); numpy, not torch.zero_(): torch's OpenMP threads stall
                # on a fully loaded CPU (~200 ms vs 1 ms)
                arrs = {k: slot[k].numpy()
                        for k in ("evals", "buckets", "inputs", "q_score", "q_mask")}
                for v in arrs.values():
                    v.fill(0)
                try:
                    self.ds.fill_into(arrs["inputs"], arrs["evals"], arrs["buckets"],
                                      arrs["q_score"], arrs["q_mask"])
                except StopIteration:
                    ready.put(None)
                    return
                except BaseException as e:  # surface loader errors in the main thread
                    ready.put(e)
                    return
                ready.put(slot)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def close(self):
        """Stops the fill thread and waits for it. Needed before the interpreter
        exits: a daemon thread killed inside the Rust fill (GIL released) aborts
        the process ("FATAL: exception not rethrown")."""
        if self._free is not None:
            self._free.put(None)
            t = getattr(self, "_thread", None)
            if t is not None:
                t.join(timeout=60)
            self._free = None

    def __iter__(self):
        if self.ds.one_pass or self._free is None:
            self._start()
        B, dev = self.ds.batch_size, self.device
        for _ in range(len(self)):
            item = self._ready.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            slot = item
            if self.kind == "value":
                n = slot["n"]
                g = {k: slot[k].to(dev, non_blocking=True) for k in ("pos", "evals", "buckets")}
                rows, encs, scores = (slot[k][:n].to(dev, non_blocking=True)
                                      for k in ("child_row", "child_enc", "child_score"))
                ev = torch.cuda.Event()
                ev.record()
                slot["event"] = ev
                self._free.put(slot)
                # 4th element: (root bitboards, child rows, encodings, scores) for the
                # move-choice metrics; the compiled training loss only sees batch[:3]
                yield (g["evals"].view(B, 1), g["buckets"].view(B, 1), expand_inputs(g["pos"], B),
                       (g["pos"], rows, encs, scores))
                continue
            if self.compact:
                n = slot["n"]
                g = {k: slot[k].to(dev, non_blocking=True) for k in ("pos", "evals", "buckets")}
                rows, encs, scores = (slot[k][:n].to(dev, non_blocking=True)
                                      for k in ("child_row", "child_enc", "child_score"))
                ev = torch.cuda.Event()
                ev.record()
                slot["event"] = ev
                self._free.put(slot)
                x, q, m = expand_compact(g["pos"], rows, encs, scores, B)
                yield (g["evals"].view(B, 1), g["buckets"].view(B, 1), x, q, m)
                continue
            g = {k: slot[k].to(dev, non_blocking=True)
                 for k in ("evals", "buckets", "inputs", "q_score", "q_mask")}
            ev = torch.cuda.Event()
            ev.record()
            slot["event"] = ev
            self._free.put(slot)
            # bucket blocks are built on the GPU (bucket_blocks_gpu) in the step
            yield (g["evals"].view(B, 1), g["buckets"].view(B, 1), g["inputs"].view(B, 120),
                   g["q_score"].view(B, 128), g["q_mask"].view(B, 128))


class PolicyLitDataModule(pl.LightningDataModule):
    """kind="value": the same POL1 files for the value net (root value as target)"""

    def __init__(self, train_data, val_data, buffer_size=5_000_000, batch_size=8192,
                 kind="policy", rotate=True):
        super().__init__()
        self.kind = kind
        self.train_set = PolicyBatchDataSet(
            batch_size, buffer_size, train_data, True, rotate=rotate)
        self.val_set = PolicyBatchDataSet(
            batch_size, 10**9, val_data, False, one_pass=True)

    # background-thread loaders with pinned buffers (see ThreadedBatchLoader);
    # created once so the training stream continues across epochs
    def train_dataloader(self):
        if not hasattr(self, "_train_loader"):
            self._train_loader = ThreadedBatchLoader(self.train_set, kind=self.kind)
        return self._train_loader

    def val_dataloader(self):
        if not hasattr(self, "_val_loader"):
            self._val_loader = ThreadedBatchLoader(self.val_set, kind=self.kind)
        return self._val_loader


class BatchDataSet(torch.utils.data.IterableDataset):

    def __init__(self, batch_size, buffer_size, file_path, shuffle=True):
        super(BatchDataSet, self).__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.file_path = file_path
        self.loader = string_sum.BatchProvider(
            self.file_path, self.buffer_size, self.batch_size, shuffle)

        self.num_samples = self.loader.num_samples
        print("Called initialization")

    def __iter__(self):
        return self

    def __len__(self):
        return 100000000//self.batch_size

    def __next__(self):
        input_size = 120
        results = np.zeros(self.batch_size, dtype=np.float32)
        evals = np.zeros(self.batch_size, dtype=np.float32)
        moves = np.zeros(self.batch_size, dtype=np.int64)
        buckets = np.zeros(self.batch_size, dtype=np.int64)
        inputs = np.zeros(self.batch_size*input_size, dtype=np.float32)
        legal_moves = np.zeros(self.batch_size * 128, dtype=bool)
        self.loader.testing(inputs, legal_moves, results,
                            evals, moves, buckets)

        return results.reshape(self.batch_size, 1), evals.reshape(self.batch_size, 1), moves.reshape(self.batch_size, 1), buckets.reshape(self.batch_size, 1), inputs.reshape(self.batch_size, input_size), legal_moves.reshape(self.batch_size, 128)


class LitDataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, buffer_size=1500000, batch_size=1000):
        super(LitDataModule, self).__init__()
        self.train_set = BatchDataSet(
            batch_size, buffer_size, train_data, True)
        self.val_set = BatchDataSet(batch_size, 100000, val_data, False)
        self.train_data = train_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=None, batch_sampler=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=None, batch_sampler=None, shuffle=False)
