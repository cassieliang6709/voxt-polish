"""Convert Silero VAD v6 (from Voxt's MLX safetensors) to CoreML MLProgram (.mlpackage)
targeting Apple Neural Engine (ANE) for zero-GPU-contention heterogeneous audio pipelines.

Exact math matched line-for-line against `sglang_omni_mlx/native/src/silero_vad/silero_vad.cpp`:
  window[576] = [context(64) || chunk(512)]
  -> right-only reflect pad(64) -> [640]
  -> STFT conv1d(256, hop=128) -> mag = sqrt(re^2 + im^2) [1, 129, 4]
  -> conv1(129->128, k=3, s=1, p=1) + ReLU
  -> conv2(128->64,  k=3, s=2, p=1) + ReLU
  -> conv3(64->64,   k=3, s=2, p=1) + ReLU
  -> conv4(64->128,  k=3, s=1, p=1) + ReLU  -> [1, 128, 1]
  -> LSTMCell(128, 128) + ReLU
  -> final_conv(128->1, k=1) + Sigmoid -> speech probability scalar
"""
import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class SileroVADv6Step(nn.Module):
    def __init__(self, weights: dict[str, np.ndarray]):
        super().__init__()
        # MLX Conv1d weight layout: (out_ch, kernel, in_ch) -> PyTorch: (out_ch, in_ch, kernel)
        self.register_buffer(
            "stft_w",
            torch.from_numpy(np.transpose(weights["stft_conv.weight"], (0, 2, 1)).copy()),
        )
        self.conv1 = self._make_conv(weights, "conv1", stride=1, padding=1)
        self.conv2 = self._make_conv(weights, "conv2", stride=2, padding=1)
        self.conv3 = self._make_conv(weights, "conv3", stride=2, padding=1)
        self.conv4 = self._make_conv(weights, "conv4", stride=1, padding=1)

        # LSTM gates: MLX LstmStep uses standard PyTorch i, f, g, o split
        self.register_buffer("lstm_wx", torch.from_numpy(weights["lstm.Wx"].copy()))
        self.register_buffer("lstm_wh", torch.from_numpy(weights["lstm.Wh"].copy()))
        self.register_buffer("lstm_b", torch.from_numpy(weights["lstm.bias"].copy()))

        self.final_conv = self._make_conv(weights, "final_conv", stride=1, padding=0)
        # Right-only reflect indices: 574, 573, ..., 511 (len 64)
        self.register_buffer(
            "reflect_idx", torch.arange(574, 510, -1, dtype=torch.long)
        )

    @staticmethod
    def _make_conv(weights, name, stride, padding):
        w = np.transpose(weights[f"{name}.weight"], (0, 2, 1)).copy()
        b = weights[f"{name}.bias"].copy()
        conv = nn.Conv1d(w.shape[1], w.shape[0], w.shape[2], stride=stride, padding=padding)
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(w))
            conv.bias.copy_(torch.from_numpy(b))
        return conv

    def forward(self, audio_window: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor):
        # audio_window: [1, 576] (64 context + 512 chunk)
        # h_in, c_in:   [1, 128]
        reflected = torch.index_select(audio_window, 1, self.reflect_idx)
        x = torch.cat([audio_window, reflected], dim=1).unsqueeze(1)  # [1, 1, 640]
        stft = F.conv1d(x, self.stft_w, stride=128, padding=0)  # [1, 258, 4]
        re = stft[:, :129, :]
        im = stft[:, 129:, :]
        mag = torch.sqrt(re * re + im * im)  # [1, 129, 4]

        h = F.relu(self.conv1(mag))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))  # [1, 128, 1]
        feat = h.squeeze(-1)  # [1, 128]

        gates = F.linear(feat, self.lstm_wx, self.lstm_b) + F.linear(h_in, self.lstm_wh)
        i_g, f_g, g_g, o_g = torch.chunk(gates, 4, dim=-1)
        c_out = torch.sigmoid(f_g) * c_in + torch.sigmoid(i_g) * torch.tanh(g_g)
        h_out = torch.sigmoid(o_g) * torch.tanh(c_out)

        out = F.relu(h_out).unsqueeze(-1)  # [1, 128, 1]
        prob = torch.sigmoid(self.final_conv(out)).squeeze(-1)  # [1, 1]
        return prob, h_out, c_out


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
        default=str(repo_root / "apple_silicon/SileroVADv6.mlpackage"),
    )
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    weights = {}
    with safe_open(args.safetensors, framework="numpy") as f:
        for k in f.keys():
            if k.startswith("vad_16k."):
                weights[k[len("vad_16k."):]] = f.get_tensor(k)

    model = SileroVADv6Step(weights).eval()
    ex_w = torch.zeros(1, 576, dtype=torch.float32)
    ex_h = torch.zeros(1, 128, dtype=torch.float32)
    ex_c = torch.zeros(1, 128, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(model, (ex_w, ex_h, ex_c))
        pt_prob, _, _ = traced(ex_w, ex_h, ex_c)

    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        inputs=[
            ct.TensorType(name="audio_window", shape=(1, 576), dtype=np.float32),
            ct.TensorType(name="h_in", shape=(1, 128), dtype=np.float32),
            ct.TensorType(name="c_in", shape=(1, 128), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="prob"),
            ct.TensorType(name="h_out"),
            ct.TensorType(name="c_out"),
        ],
    )
    out_path = Path(args.out)
    mlmodel.save(str(out_path))

    # Quick parity check between PyTorch FP32 and CoreML ANE on random + silence
    rng = np.random.default_rng(42)
    h_pt = torch.zeros(1, 128)
    c_pt = torch.zeros(1, 128)
    h_cm = np.zeros((1, 128), dtype=np.float32)
    c_cm = np.zeros((1, 128), dtype=np.float32)
    diffs = []
    for _ in range(20):
        w = (rng.standard_normal((1, 576)).astype(np.float32) * 0.1)
        with torch.no_grad():
            p_pt, h_pt, c_pt = model(torch.from_numpy(w), h_pt, c_pt)
        res = mlmodel.predict({"audio_window": w, "h_in": h_cm, "c_in": c_cm})
        h_cm = res["h_out"].astype(np.float32)
        c_cm = res["c_out"].astype(np.float32)
        diffs.append(abs(float(p_pt.item()) - float(res["prob"].item())))

    print(
        json.dumps(
            {
                "saved": str(out_path),
                "precision": args.precision,
                "pt_zero_prob": round(float(pt_prob.item()), 6),
                "max_abs_diff_vs_pt_20steps": round(max(diffs), 6),
                "mean_abs_diff_vs_pt_20steps": round(sum(diffs) / len(diffs), 6),
            }
        )
    )


if __name__ == "__main__":
    main()
