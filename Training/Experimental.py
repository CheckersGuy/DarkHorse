import torch.nn as nn
import torch
from torch.nn.modules import activation
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from ranger import Ranger
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
        self.output = BucketLinear(L3, 128, self.num_buckets)
        self.layers = [self.accu, self.layer_one, self.layer_sec, self.output]
        # init_layers() removed, see the note at the end

    # ---------------------------------------------------------------- model
    def _select(self, t, buckets, width):
        B = t.shape[0]
        t = t.reshape(B, self.num_buckets, width)
        return t[torch.arange(B, device=t.device), buckets.flatten()]

    def forward(self, x, buckets):
        ac = torch.clamp(self.accu(x), 0.0, 1.0)
        ac_x, ac_y = ac.split(self.l1, dim=1)
        ac_out = ac_x.mul(ac_y) * (127.0 / 128.0)

        l1c = torch.clamp(self._select(
            self.layer_one(ac_out), buckets, L2), 0.0, 1.0)
        l2c = torch.clamp(self._select(
            self.layer_sec(l1c), buckets, L3), 0.0, 1.0)
        # raw logits [B,128]
        return self._select(self.output(l2c), buckets, 128)

    def write_header(self, file_out):
        file_out.write(struct.pack("I", len(self.layers)))
        file_out.write(struct.pack("I", self.num_buckets))
        file_out.write(struct.pack("I", self.accu.in_features))
        file_out.write(struct.pack("I", self.accu.out_features))
        for layer in self.layers[1:]:
            file_out.write(struct.pack("I", layer.in_features))
            file_out.write(struct.pack(
                "I", layer.out_features // self.num_buckets))

    def configure_optimizers(self):
        groups = self.bucket_param_groups(5.0e-3, self.delta_lr_mult)
        optimizer = RangerLite(groups, lr=5.0e-3, eps=1.0e-5, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]

    # ----------------------------------------------------- loss and metrics
    def _shared_step(self, batch, stage):
        _, buckets, x, q_score, q_mask = batch
        bs = x.shape[0]
        logits = self.forward(x, buckets)

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

    def training_step(self, batch, batch_idx):
        self.step()
        loss = self._shared_step(batch, "train")
        if loss is None:
            return (self.accu.weight * 0).sum()   # keeps the graph alive
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    # -------------------------------------------------------- housekeeping
    def on_before_optimizer_step(self, optimizer):
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


class PolicyBatchDataSet(torch.utils.data.IterableDataset):
    """one_pass=True: finite, restarts from the file start on every iter().
    Use it for validation so every validation epoch sees the same samples."""

    def __init__(self, batch_size, buffer_size, file_path, shuffle=True, one_pass=False):
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.file_path = file_path
        self.shuffle = shuffle
        self.one_pass = one_pass
        self.loader = self._make_loader()
        self.num_samples = self.loader.num_samples
        self._served = 0

    def _make_loader(self):
        return string_sum.PolicyBatchProvider(
            self.file_path, self.buffer_size, self.batch_size, self.shuffle)

    def __iter__(self):
        if self.one_pass:
            self.loader = self._make_loader()
            self._served = 0
        return self

    def __len__(self):
        return self.num_samples // self.batch_size

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
        return (evals.reshape(B, 1), buckets.reshape(B, 1),
                inputs.reshape(B, 120), q_score.reshape(B, 128),
                q_mask.reshape(B, 128))


class PolicyLitDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, buffer_size=5_000_000, batch_size=8192):
        super().__init__()
        self.train_set = PolicyBatchDataSet(
            batch_size, buffer_size, train_data, True)
        self.val_set = PolicyBatchDataSet(
            batch_size, 10**9, val_data, False, one_pass=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=None, shuffle=False)


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
