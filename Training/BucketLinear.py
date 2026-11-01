import torch
import torch.nn.functional as F
import torch.nn as nn


class BucketLinear(nn.Module):
    """num_buckets weight sets, W_b = W_shared + dW_b. Drop-in for the old
    nn.Linear(in, out_per_bucket * num_buckets)."""

    def __init__(self, in_features, out_per_bucket, num_buckets):
        super().__init__()
        self.in_features = in_features
        self.out_per_bucket = out_per_bucket
        self.num_buckets = num_buckets
        self.out_features = out_per_bucket * num_buckets   # write_header relies on this
        # same default init as before
        ref = nn.Linear(in_features, out_per_bucket)
        self.shared_weight = nn.Parameter(ref.weight.detach().clone())
        self.shared_bias = nn.Parameter(ref.bias.detach().clone())
        self.delta_weight = nn.Parameter(torch.zeros(
            num_buckets, out_per_bucket, in_features))
        self.delta_bias = nn.Parameter(
            torch.zeros(num_buckets, out_per_bucket))

    @property
    def weight(self):   # [nb*out, in], bucket-major: same layout the export splits on
        return (self.shared_weight.unsqueeze(0) + self.delta_weight).reshape(
            self.out_features, self.in_features)

    @property
    def bias(self):
        return (self.shared_bias.unsqueeze(0) + self.delta_bias).reshape(self.out_features)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def clamp_fused(self, lo, hi):
        # the int8 range applies to the FUSED weight, so clamp the sum and
        # push the correction into delta (shared stays in range too)
        self.shared_weight.clamp_(lo, hi)
        fused = (self.shared_weight.unsqueeze(0) +
                 self.delta_weight).clamp_(lo, hi)
        self.delta_weight.copy_(fused - self.shared_weight.unsqueeze(0))

    def delta_penalty(self):
        return self.delta_weight.abs().mean()


class BucketedMixin:
    def step(self):                       # replaces the per-class step()
        with torch.no_grad():
            for layer in self.layers[1:]:
                layer.clamp_fused(self.min_weight_hidden,
                                  self.max_weight_hidden)

    def delta_loss(self):
        return sum(l.delta_penalty() for l in self.layers[1:])

    def bucket_param_groups(self, lr, delta_lr_mult=1.0):
        shared, delta = [], []
        for n, p in self.named_parameters():
            (delta if n.endswith(("delta_weight", "delta_bias")) else shared).append(p)
        return [{"params": shared, "lr": lr}, {"params": delta, "lr": lr * delta_lr_mult}]
