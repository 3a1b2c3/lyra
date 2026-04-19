"""Export Lyra-2 VAE encoder chunk to ONNX and benchmark against PyTorch.

The VAE encoder is stateful (feat_cache flows between chunks). We export a single
chunk call with explicit cache tensors as inputs/outputs so ORT can run each step.

Usage:
  python scripts/export_vae_onnx.py --vae_pth checkpoints/vae/vae.pth
  python scripts/export_vae_onnx.py --vae_pth checkpoints/vae/vae.pth --benchmark
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add TRT and cuDNN DLL dirs before any onnxruntime import
_TRT_BIN = "C:/bin/TensorRT-10.15.1.29.Windows.amd64.cuda-12.9/TensorRT-10.15.1.29/bin"
_TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
for _d in [_TRT_BIN, _TORCH_LIB]:
    if os.path.isdir(_d):
        os.add_dll_directory(_d)


def load_vae(vae_pth, device, dtype):
    from lyra_2._src.tokenizers.wan2pt1 import WanVAE
    print(f"Loading VAE from {vae_pth} ...")
    vae = WanVAE(vae_pth=vae_pth, device=device, dtype=dtype, is_amp=False)
    print("VAE loaded.")
    return vae


class EncoderChunkWrapper(torch.nn.Module):
    """Wraps a single Encoder3d chunk call with explicit cache I/O for ONNX export."""
    def __init__(self, encoder, conv_num):
        super().__init__()
        self.encoder = encoder
        self.conv_num = conv_num

    def forward(self, x, *cache_in):
        cache = list(cache_in)
        feat_idx = [0]
        out = self.encoder(x, feat_cache=cache, feat_idx=feat_idx)
        # Return output + updated cache tensors
        cache_out = [c if c is not None else torch.zeros(1) for c in cache]
        return (out,) + tuple(cache_out)


def get_cache_shapes(vae_core, device, dtype, H, W):
    """Run one encoder pass to capture feat_cache tensor shapes."""
    x_dummy = torch.zeros(1, 3, 1, H, W, device=device, dtype=dtype)
    vae_core.clear_cache()
    vae_core._enc_conv_idx = [0]
    _ = vae_core.encoder(x_dummy, feat_cache=vae_core._enc_feat_map, feat_idx=vae_core._enc_conv_idx)
    shapes = []
    for t in vae_core._enc_feat_map:
        shapes.append(t.shape if t is not None else None)
    vae_core.clear_cache()
    return shapes


def export(vae_core, device, dtype, H, W, out_path):
    print(f"\nExporting encoder chunk ({H}x{W}) to {out_path} ...")

    # Warm up cache to get shapes
    cache_shapes = get_cache_shapes(vae_core, device, dtype, H, W)
    conv_num = vae_core._conv_num

    # Build dummy inputs — need enough temporal frames for CausalConv3d kernels
    t_dummy = getattr(vae_core, 'temporal_window', 4) + 1
    x = torch.zeros(1, 3, t_dummy, H, W, device=device, dtype=dtype)
    cache_in = []
    for s in cache_shapes:
        if s is None:
            cache_in.append(torch.zeros(1, device=device, dtype=dtype))
        else:
            cache_in.append(torch.zeros(*s, device=device, dtype=dtype))

    wrapper = EncoderChunkWrapper(vae_core.encoder, conv_num).to(device).to(dtype)
    wrapper.eval()

    input_names = ["x"] + [f"cache_{i}" for i in range(len(cache_in))]
    output_names = ["out"] + [f"cache_out_{i}" for i in range(len(cache_in))]

    dynamic_axes = {"x": {0: "batch"}}

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (x, *cache_in),
                out_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        print(f"Exported to {out_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def benchmark_pytorch(vae_core, device, dtype, H, W, n_frames=81, repeats=3):
    """Benchmark full encode of n_frames at HxW."""
    from lyra_2._src.tokenizers.wan2pt1 import WanVAE
    x = torch.zeros(1, 3, n_frames, H, W, device=device, dtype=dtype)
    scale = (torch.zeros(16, device=device), torch.ones(16, device=device))

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            vae_core.encode(x, scale)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    print(f"PyTorch encode ({n_frames}f, {H}x{W}): {min(times):.2f}s (best of {repeats})")


def benchmark_ort(onnx_path, device, dtype, H, W, cache_shapes, t_dummy):
    import onnxruntime as ort
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"ORT providers: {sess.get_providers()}")

    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    x_np = np.zeros((1, 3, t_dummy, H, W), dtype=np_dtype)
    cache_np = []
    for s in cache_shapes:
        if s is None:
            cache_np.append(np.zeros(1, dtype=np_dtype))
        else:
            cache_np.append(np.zeros(s, dtype=np_dtype))

    inputs = {"x": x_np}
    inputs.update({f"cache_{i}": c for i, c in enumerate(cache_np)})

    # Warmup
    for _ in range(3):
        sess.run(None, inputs)

    times = []
    for _ in range(5):
        t0 = time.time()
        sess.run(None, inputs)
        times.append(time.time() - t0)
    print(f"ORT single chunk {t_dummy}f ({H}x{W}): {min(times)*1000:.1f}ms (best of 5)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_pth", default="checkpoints/vae/vae.pth")
    parser.add_argument("--out", default="checkpoints/vae/encoder_chunk.onnx")
    parser.add_argument("--resolution", default="320,576")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    H, W = [int(x) for x in args.resolution.split(",")]
    dtype = torch.float16  # TRT prefers fp16 over bfloat16

    vae = load_vae(args.vae_pth, args.device, dtype)
    vae_core = vae.model

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t_dummy = getattr(vae_core, 'temporal_window', 4) + 1
    success = export(vae_core, args.device, dtype, H, W, args.out)

    if success and args.benchmark:
        cache_shapes = get_cache_shapes(vae_core, args.device, dtype, H, W)
        # PyTorch single chunk baseline
        x = torch.zeros(1, 3, t_dummy, H, W, device=args.device, dtype=dtype)
        vae_core.clear_cache()
        times = []
        for _ in range(5):
            vae_core.clear_cache()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                vae_core.encoder(x, feat_cache=vae_core._enc_feat_map, feat_idx=[0])
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        print(f"PyTorch single chunk {t_dummy}f ({H}x{W}): {min(times)*1000:.1f}ms (best of 5)")
        benchmark_ort(args.out, args.device, dtype, H, W, cache_shapes, t_dummy)


if __name__ == "__main__":
    main()
