import sys
import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


CONFIGS = {
    # A3: SDPA + torch.compile(inductor), full FT
    "a3": {
        "model_dir": "bert-base-squadv2-a3-sdpa-compile-t4",
        "lr": 3e-5,
        "use_compile": True,
        "compile_backend": "inductor",
        "gradient_checkpointing": False,
        "use_8bit_optim": False,
    },
    # C1: 8-bit AdamW + SDPA + compile
    "c1": {
        "model_dir": "bert-base-squadv2-c1-8bit",
        "lr": 5e-5,
        "use_compile": True,
        "compile_backend": "inductor",
        "gradient_checkpointing": False,
        "use_8bit_optim": True,
    },
    # C2: gradient checkpointing + compile
    "c2": {
        "model_dir": "bert-base-squadv2-c2-gc",
        "lr": 5e-5,
        "use_compile": True,
        "compile_backend": "inductor",
        "gradient_checkpointing": True,
        "use_8bit_optim": False,
    },
}


def make_batch(tokenizer, batch_size=8, seq_len=384, device="cuda"):
    # Synthetic but realistic-looking batch
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, device=device)
    # Just put the answer at position 1 for all examples
    start_positions = torch.ones(batch_size, device=device, dtype=torch.long)
    end_positions = torch.ones(batch_size, device=device, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_positions,
        "end_positions": end_positions,
    }


def build_optimizer(model, cfg):
    if cfg["use_8bit_optim"]:
        if bnb is None:
            raise RuntimeError(
                "bitsandbytes not available but use_8bit_optim=True. "
                "Install bitsandbytes or turn this flag off."
            )
        optim = bnb.optim.AdamW8bit(model.parameters(), lr=cfg["lr"])
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    return optim


def measure_one(config_name: str):
    if config_name not in CONFIGS:
        raise SystemExit(f"Unknown config '{config_name}'. Use one of {list(CONFIGS.keys())}")

    cfg = CONFIGS[config_name]
    model_dir = cfg["model_dir"]

    if not os.path.isdir(model_dir):
        raise SystemExit(f"Model directory '{model_dir}' not found in CWD={os.getcwd()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Measuring VRAM for {config_name} ({model_dir}) on device {device} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)

    if cfg["gradient_checkpointing"]:
        print("  -> Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        # recommended when using GC
        model.config.use_cache = False

    if cfg["use_compile"]:
        print(f"  -> Wrapping model with torch.compile(backend='{cfg['compile_backend']}')")
        model = torch.compile(model, backend=cfg["compile_backend"])

    model.to(device)
    model.train()

    optimizer = build_optimizer(model, cfg)

    batch = make_batch(tokenizer, device=device)

    # Warmup step to trigger graph capture / allocations
    print("  -> Warmup step (not measured)...")
    optimizer.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    # Now measure a single "steady-state" train step
    torch.cuda.reset_peak_memory_stats(device)

    print("  -> Measured step...")
    optimizer.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory

    to_mib = lambda b: b / (1024 ** 2)

    print("\n=== RESULT ===")
    print(f"  Config              : {config_name} ({model_dir})")
    print(f"  Peak allocated      : {peak_alloc} bytes  (~{to_mib(peak_alloc):.1f} MiB)")
    print(f"  Peak reserved       : {peak_reserved} bytes  (~{to_mib(peak_reserved):.1f} MiB)")
    print(f"  Device total memory : {total} bytes  (~{to_mib(total):.1f} MiB)")
    print("======================")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python measure_vram_qa.py [a3|c1|c2]")
        sys.exit(1)
    measure_one(sys.argv[1])