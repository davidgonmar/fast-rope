import torch
from torch import nn, Tensor
import rope_cuda

device = torch.device("cuda")


class RotaryEmbeddingNaive(nn.Module):
    """
    Positional encoding to add to embeddings introduced in "Rotary Positional Embedding".
    Uses the complex number interpretation of the rotation.

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        thetas = torch.arange(0, d_model // 2, dtype=torch.float)  # (d_model // 2)
        thetas = 10000 ** (-2 * (thetas - 1) / d_model)  # (d_model // 2)
        # pos must be like (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2) (t = theta)
        # where i = 0, 1, ..., d/2
        # so repeat interleave thetas to go from
        # (mt1, mt2, ..., mtd/2) to (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2)
        thetas = thetas.repeat_interleave(2)  # (d_model // 2)

        freqs = torch.outer(
            torch.arange(0, seq_len, dtype=torch.float), thetas
        )  # (seq_len, d_model)

        # angular wrapping to avoid large values, wont affect the rotation
        freqs = freqs % (2 * torch.pi)

        freqs = torch.view_as_complex(
            freqs.reshape(seq_len, d_model // 2, 2)  # (seq_len, d_model // 2, 2)
        )  # (seq_len, d_model // 2)

        freqs = torch.exp(freqs)  # now freqs is of the for
        # [[exp(i * m0 * theta0), exp(i * m0 * theta1), ..., exp(i * m0 * theta2), exp(i * m0 * theta3), ...]
        # [exp(i * m1 * theta0), exp(i * m1 * theta1), ..., exp(i * m1 * theta2), exp(i * m1 * theta3), ...]]
        # where m0, m1, ... are the positions and theta0, theta1, ... are the thetas
        # remember that exp(i * theta) = cos(theta) + i * sin(theta), and multiplying two complex numbers
        # is done by multiplying the magnitudes and adding the angles
        # so, for example x = a + bi and take entry 0, 0 of freqs, we have
        # x * freqs[0, 0] = (a + bi) * (cos(m0 * theta0) + i * sin(m0 * theta0))
        # = a * cos + a * i * sin + b * i * cos - b * sin
        # this represents a rotation of x by m0 * theta0
        # now, as the paper says, we subdivide the d-dimentional space into d/2 2-dimensional subspaces (d = d_model)
        # and individually rotate each subspace by the corresponding theta (remember we have d/2 thetas for d/2 subspaces)
        # and also take into account the position m

        self.register_buffer("freqs", freqs)

    @staticmethod
    def _apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
        # x is of shape (batch_size, seq_len, d_model)
        # freqs is of shape at least (seq_len)
        # so we want only to add the positional encoding to the seq_len dimension
        batch_size, seq_len, d_model = x.size()

        xcomp = torch.view_as_complex(x.view(batch_size, seq_len, d_model // 2, 2))

        x_rot = (
            xcomp * freqs.unsqueeze(0)[:, :seq_len, :]
        )  # handle case where seq_len < self.seq_len

        return torch.view_as_real(x_rot).reshape(batch_size, seq_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.freqs.device:
            self.freqs = self.freqs.to(x.device)
        return self._apply_rotary_emb(x, self.freqs)


class RotaryEmbeddingFast(nn.Module):
    """
    Positional encoding to add to embeddings introduced in "Rotary Positional Embedding".

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        thetas = torch.arange(0, d_model // 2, dtype=torch.float)  # (d_model // 2)
        thetas = 10000 ** (-2 * (thetas - 1) / d_model)  # (d_model // 2)
        # pos must be like (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2) (t = theta)
        # where i = 0, 1, ..., d/2
        # so repeat interleave thetas to go from
        # (mt1, mt2, ..., mtd/2) to (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2)
        thetas = thetas.repeat_interleave(2)

        freqs = torch.outer(
            torch.arange(0, seq_len, dtype=torch.float), thetas
        )  # (seq_len, d_model)
        # angular wrapping to avoid large values, wont affect the rotation
        freqs = freqs % (2 * torch.pi)

        self.register_buffer("freqs", freqs)

    @staticmethod
    def _apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
        return rope_cuda.rope_forward(x, freqs)

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        if device != self.freqs.device:
            self.freqs = self.freqs.to(device)
        return self._apply_rotary_emb(x, self.freqs)


class RotaryEmbeddingNaive2(nn.Module):
    """
    Positional encoding to add to embeddings introduced in "Rotary Positional Embedding".

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        thetas = torch.arange(0, d_model // 2, dtype=torch.float)  # (d_model // 2)
        thetas = 10000 ** (-2 * (thetas - 1) / d_model)  # (d_model // 2)
        # pos must be like (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2) (t = theta)
        # where i = 0, 1, ..., d/2
        # so repeat interleave thetas to go from
        # (mt1, mt2, ..., mtd/2) to (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2)
        thetas = thetas.repeat_interleave(2)

        freqs = torch.outer(
            torch.arange(0, seq_len, dtype=torch.float), thetas
        )  # (seq_len, d_model)

        self.register_buffer("freqs", freqs)

    @staticmethod
    def _apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
        # x is of shape (batch_size, seq_len, d_model)
        # freqs is of shape at least (seq_len)
        # so we want only to add the positional encoding to the seq_len dimension
        batch_size, seq_len, d_model = x.size()

        # we want (-x1, x0, -x3, x2, ..., -xd-1, xd-2) as in the paper (here indexed from 0 instead of 1)
        # from (x0, x1, ..., xd-1)
        x_even, x_odd = x.view(batch_size, seq_len, d_model // 2, 2).unbind(
            dim=-1
        )  # (batch_size, seq_len, d_model // 2)
        # x_even is (x0, x2, ..., xd-2)
        # x_odd is (x1, x3, ..., xd-1)
        x_rot = torch.stack((-x_odd, x_even), dim=-1).reshape(
            batch_size, seq_len, d_model
        )  # (batch_size, seq_len, d_model)

        freq_cos = torch.cos(freqs).reshape(1, -1, d_model)[
            :, :seq_len, :
        ]  # handle case where seq_len < self.seq_len
        freq_sin = torch.sin(freqs).reshape(1, -1, d_model)[
            :, :seq_len, :
        ]  # shape (1, seq_len, d_model)

        return x * freq_cos + x_rot * freq_sin

    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.freqs.device:
            self.freqs = self.freqs.to(x.device)
        return self._apply_rotary_emb(x, self.freqs)


if __name__ == "__main__":
    # first warmup and correctness test
    d_model = 128
    seq_len = 512
    batch_size = 4
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    rotary_emb_naive = RotaryEmbeddingNaive(d_model, seq_len, 0.1)
    rotary_emb_fast = RotaryEmbeddingFast(d_model, seq_len, 0.1)
    rotary_emb_naive2 = RotaryEmbeddingNaive2(d_model, seq_len, 0.1)
    y_naive = rotary_emb_naive(x)
    y_fast = rotary_emb_fast(x)
    y_naive2 = rotary_emb_naive2(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        y_naive, y_fast, equal_nan=True, rtol=1e-2, atol=1e-2
    )  # maybe figure out better tolerances
    torch.testing.assert_close(
        y_naive, y_naive2, equal_nan=True, rtol=1e-2, atol=1e-2
    )  # maybe figure out better tolerances
    # then benchmark
    import time
    import numpy as np

    x = torch.randn(batch_size, seq_len, d_model).to(device)
    rotary_emb_naive = RotaryEmbeddingNaive(d_model, seq_len, 0.1)
    rotary_emb_fast = RotaryEmbeddingFast(d_model, seq_len, 0.1)

    num_iters = 1000
    times_naive = []
    times_fast = []
    times_naive2 = []

    for _ in range(num_iters):
        start = time.time()
        y_naive = rotary_emb_naive(x)
        torch.cuda.synchronize()
        times_naive.append(time.time() - start)

        start = time.time()
        y_fast = rotary_emb_fast(x)
        torch.cuda.synchronize()
        times_fast.append(time.time() - start)

        start = time.time()
        y_naive2 = rotary_emb_naive2(x)
        torch.cuda.synchronize()
        times_naive2.append(time.time() - start)

    print("Naive mean time:", np.mean(times_naive))
    print("Fast mean time:", np.mean(times_fast))
    print("Speedup:", np.mean(times_naive) / np.mean(times_fast))
