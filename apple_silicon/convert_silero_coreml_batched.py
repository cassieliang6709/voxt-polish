"""Export 32-chunk (1.024s) batched Silero VAD v6 to CoreML MLProgram (.mlpackage)
to amortize ANE / CoreML dispatch overhead by 32x, and benchmark vs single-chunk & MLX.
"""
import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class SileroVADv6Batched32(nn.Module):
    def __init__(self, weights: dict[str, np.ndarray], batch_chunks: int = 32):
        super().__init__()
        self.batch_chunks = batch_chunks
        self.register_buffer(
            "stft_w",
            torch.from_numpy(np.transpose(weights["stft_conv.weight"], (0, 2, 1)).copy()),
        )
        self.conv1 = self._make_conv(weights, "conv1", stride=1, padding=1)
        self.conv2 = self._make_conv(weights, "conv2", stride=2, padding=1)
        self.conv3 = self._make_conv(weights, "conv3", stride=2, padding=1)
        self.conv4 = self._make_conv(weights, "conv4", stride=1, padding=1)
        self.register_buffer("lstm_wx", torch.from_numpy(weights["lstm.Wx"].copy()))
        self.register_buffer("lstm_wh", torch.from_numpy(weights["lstm.Wh"].copy()))
        self.register_buffer("lstm_b", torch.from_numpy(weights["lstm.bias"].copy()))
        self.final_conv = self._make_conv(weights, "final_conv", stride=1, padding=0)
        self.register_buffer("reflect_idx", torch.arange(574, 510, -1, dtype=torch.long))

    @staticmethod
    def _make_conv(weights, name, stride, padding):
        w = np.transpose(weights[f"{name}.weight"], (0, 2, 1)).copy()
        b = weights[f"{name}.bias"].copy()
        conv = nn.Conv1d(w.shape[1], w.shape[0], w.shape[2], stride=stride, padding=padding)
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(w))
            conv.bias.copy_(torch.from_numpy(b))
        return conv

    def forward(self, windows: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor):
        # windows: [B, 576] (all B chunks run Conv1d front-end in ONE batched pass!)
        reflected = torch.index_select(windows, 1, self.reflect_idx)
        x = torch.cat([windows, reflected], dim=1).unsqueeze(1)  # [B, 1, 640]
        stft = F.conv1d(x, self.stft_w, stride=128, padding=0)   # [B, 258, 4]
        re = stft[:, :129, :]
        im = stft[:, 129:, :]
        mag = torch.sqrt(re * re + im * im)

        h = F.relu(self.conv1(mag))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))  # [B, 128, 1]
        feats = h.squeeze(-1)      # [B, 128]

        # Precompute input gates for all B steps in one GEMM!
        x_gates = F.linear(feats, self.lstm_wx, self.lstm_b)  # [B, 512]
        ht = h_in  # [1, 128]
        ct_state = c_in
        hs = []
        for t in range(self.batch_chunks):
            gates = x_gates[t : t + 1, :] + F.linear(ht, self.lstm_wh)
            i_g, f_g, g_g, o_g = torch.chunk(gates, 4, dim=-1)
            ct_state = torch.sigmoid(f_g) * ct_state + torch.sigmoid(i_g) * torch.tanh(g_g)
            ht = torch.sigmoid(o_g) * torch.tanh(ct_state)
            hs.append(ht)
        h_seq = F.relu(torch.cat(hs, dim=0)).unsqueeze(-1)  # [B, 128, 1]
        probs = torch.sigmoid(self.final_conv(h_seq)).squeeze(-1).squeeze(-1)  # [B]
        return probs, ht, ct_state


def main():
    repo_root = Path(__file__).resolve().parent.parent
    omni_models = Path.home() / "Library/Application Support/SGLang Omni/models"
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--safetensors",
        default=str(
            omni_models
            / "models--mlx-community--silero-vad-v6/snapshots/2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06/model.safetensors"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(repo_root / "apple_silicon/SileroVADv6_B32.mlpackage"),
    )
    ap.add_argument("--batch-chunks", type=int, default=32)
    args = ap.parse_args()

    weights = {}
    with safe_open(args.safetensors, framework="numpy") as f:
        for k in f.keys():
            if k.startswith("vad_16k."):
                weights[k[len("vad_16k."):]] = f.get_tensor(k)

    B = args.batch_chunks
    model = SileroVADv6Batched32(weights, B).eval()
    ex_w = torch.zeros(B, 576, dtype=torch.float32)
    ex_h = torch.zeros(1, 128, dtype=torch.float32)
    ex_c = torch.zeros(1, 128, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, (ex_w, ex_h, ex_c))

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        inputs=[
            ct.TensorType(name="windows", shape=(B, 576), dtype=np.float32),
            ct.TensorType(name="h_in", shape=(1, 128), dtype=np.float32),
            ct.TensorType(name="c_in", shape=(1, 128), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="probs"),
            ct.TensorType(name="h_out"),
            ct.TensorType(name="c_out"),
        ],
    )
    mlmodel.save(args.out)
    print(json.dumps({"saved": args.out, "batch_chunks": B}))


if __name__ == "__main__":
    main()
