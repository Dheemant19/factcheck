"""Check that this machine can run the project (day-one test on the laptop and the cluster).

    python scripts/check_env.py

Prints library versions, the GPU and its compute capability, and runs a bf16 matmul and a
4-bit linear layer on the GPU. Nothing is downloaded.
"""

from __future__ import annotations

import importlib
import platform


def main() -> None:
    import torch

    print(f"python        {platform.python_version()} ({platform.system()})")
    print(f"torch         {torch.__version__}  (built for CUDA {torch.version.cuda})")
    for name in ("transformers", "datasets", "peft", "accelerate", "sentence_transformers",
                 "optuna", "bitsandbytes", "shap", "umap", "streamlit"):  # fmt: skip
        try:
            module = importlib.import_module(name)
            print(f"{name:13s} {getattr(module, '__version__', 'installed')}")
        except ImportError:
            print(f"{name:13s} MISSING")

    if not torch.cuda.is_available():
        print("\nCUDA not available: training will run on CPU (fine for tests only).")
        return

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    print(
        f"\nGPU           {torch.cuda.get_device_name(0)}  ({arch}, "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
    )
    if arch not in torch.cuda.get_arch_list():
        print(
            f"WARNING: this torch build has no kernels for {arch}. RTX 50-series needs a "
            "CUDA 12.8+ build: uv pip install torch --index-url https://download.pytorch.org/whl/cu132"
        )
        return

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"bf16 matmul   ok (bf16 supported: {torch.cuda.is_bf16_supported()})", (a @ a).shape)

    try:
        import bitsandbytes as bnb

        layer = bnb.nn.Linear4bit(64, 32, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
        layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
        print("4-bit linear  ok (QLoRA and the 4-bit app LLM can run here)")
    except Exception as err:  # noqa: BLE001
        print(
            f"4-bit linear  FAILED: {err}\n  -> the app falls back to encoders + Qwen3-4B (plan, Risks)"
        )


if __name__ == "__main__":
    main()
