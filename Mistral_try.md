Yes — with your BF16 checkpoint verified, I would not start with AWQ/GPTQ again. I would first build a stable W4A16 compressed-tensors checkpoint that mostly quantizes the huge MoE expert weights, keeps sensitive routing/MLA pieces BF16, and serves with MLA KV cache unquantized first.

For this model, KV cache is probably not the main blocker if vLLM actually uses MLA latent cache. Mistral-Small-4-119B-2603 has num_hidden_layers=36, kv_lora_rank=256, and qk_rope_head_dim=64, so the MLA cache is roughly (256 + 64) * 36 * bytes per token, about 23 KB/token in BF16 or 11.5 KB/token in FP8. That means 16K context is only around 360 MB BF16 KV, and even 256K is around 5.5 GiB BF16 KV, before vLLM block/allocator overhead. The real risk is weights + vLLM overhead, not 10K–16K KV. 

Recommended first recipe

Target:

W4A16 INT4 weights
group_size = 32
symmetric = true
observer = mse if supported
format = compressed-tensors / pack-quantized
quantize: routed MoE experts + shared experts
keep BF16: router mlp.gate, q_a_proj, kv_a_proj_with_mqa, norms, lm_head, embeddings, vision
KV cache: BF16/auto first
backend: TRITON_MLA or auto on A100

Why this shape:

W4A16 keeps activations in 16-bit and compresses weights by roughly 3.7×, which is the correct family for your memory goal. 

LLM Compressor’s oneshot supports moe_calibrate_all_experts=True, and its docs say this is required for accurate MoE quantization; the sequential pipeline is the intended path for models too large to fit normally. 

Do not quantize the MoE router mlp.gate to INT4. The working MLX examples either keep it higher precision or effectively unquantized.

Do not start with TurboQuant/RotorQuant. vLLM’s own recent TurboQuant study says FP8 KV is the safest default, and TurboQuant trades accuracy/latency/throughput for extra KV capacity. 



---

Phase 1: data-free W4A16 MSE/RTN-style quantization

This is the first checkpoint I would try. It should fit or be very close to your 72 GB usable limit, and it has fewer ways to corrupt the model than AWQ/GPTQ.

# quantize_mistral4_w4a16_safe.py

import os
import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

try:
    from compressed_tensors.offload import load_offloaded_model
except Exception:
    load_offloaded_model = None


MODEL_ID = os.environ.get("MODEL_ID", "/models/Mistral-Small-4-119B-2603-BF16")
OUT_DIR = os.environ.get("OUT_DIR", "./Mistral-Small-4-119B-2603-W4A16-g32-safe")
OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", "./offload_mistral4_w4")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

tokenizer = getattr(processor, "tokenizer", processor)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_model():
    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto_offload",
        offload_folder=OFFLOAD_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    return Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        **kwargs,
    )


if load_offloaded_model is not None:
    with load_offloaded_model():
        model = load_model()
else:
    model = load_model()


# W4A16: int4 weights, bf16/fp16 activations.
# group_size=32 is conservative and matches many compressed-tensors W4 deployments.
w4_args = QuantizationArgs(
    num_bits=4,
    type="int",
    strategy="group",
    group_size=32,
    symmetric=True,
    dynamic=False,
    observer="mse",  # remove this line if your compressed-tensors version rejects it
)

# First pass: quantize only the large MoE feed-forward paths.
# This gives most of the memory reduction while avoiding fragile MLA/router pieces.
w4_expert_scheme = QuantizationScheme(
    targets=[
        # Transformers/vLLM-style names
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.experts\\..*gate_proj$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.experts\\..*up_proj$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.experts\\..*down_proj$",

        # Some conversions expose switch_mlp instead of experts
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.switch_mlp\\..*gate_proj$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.switch_mlp\\..*up_proj$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.switch_mlp\\..*down_proj$",

        # Shared expert path: small compared with routed experts, but usually safe to quantize.
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.shared_experts\\.gate_proj$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.shared_experts\\.up_proj$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.shared_experts\\.down_proj$",
    ],
    weights=w4_args,
)

recipe = [
    QuantizationModifier(
        config_groups={
            "w4a16_experts_g32": w4_expert_scheme,
        },
        ignore=[
            # Keep output/input embeddings BF16 for first working run.
            "lm_head",
            "re:.*embed_tokens.*",

            # Keep multimodal path BF16.
            "re:.*vision.*",
            "re:.*vision_model.*",
            "re:.*image.*",
            "re:.*multi_modal_projector.*",
            "re:.*mm_projector.*",
            "re:.*patch_merger.*",

            # Keep MLA attention BF16 in the first run.
            "re:.*self_attn.*",
            "re:.*attention.*",

            # Critical: MoE router, not expert gate_proj.
            "re:.*language_model\\.model\\.layers\\.[0-9]+\\.mlp\\.gate$",
            "re:.*\\.mlp\\.gate$",
            "re:.*router.*",

            # Norms stay BF16.
            "re:.*norm.*",
            "re:.*layernorm.*",
            "re:.*layer_norm.*",
        ],
    )
]

oneshot(
    model=model,
    processor=processor,
    tokenizer=tokenizer,
    recipe=recipe,
    dataset=None,
    pipeline="datafree",
    output_dir=OUT_DIR,
)

# Depending on llmcompressor version, output_dir may already save.
# This makes it explicit.
model.save_pretrained(OUT_DIR, save_compressed=True)
processor.save_pretrained(OUT_DIR)

print(f"Saved to {OUT_DIR}")

Run:

MODEL_ID=/path/to/your/Mistral-Small-4-119B-2603-BF16 \
OUT_DIR=/path/to/Mistral-Small-4-119B-2603-W4A16-g32-safe \
OFFLOAD_DIR=/local_nvme/offload_mistral4_w4 \
python quantize_mistral4_w4a16_safe.py

Before serving, check the real model size:

du -sh /path/to/Mistral-Small-4-119B-2603-W4A16-g32-safe

For your 72 GB usable VRAM, I would prefer the checkpoint to be ≤62–64 GiB loaded, not just 65 GB on disk. vLLM needs memory for CUDA graphs/compilation, non-quantized modules, temporary buffers, and KV blocks.


---

Phase 1 serving command on A100-80

Start small. No FP8 KV. No TurboQuant. No long context.

vllm serve /path/to/Mistral-Small-4-119B-2603-W4A16-g32-safe \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype auto \
  --attention-backend TRITON_MLA \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --reasoning-parser mistral

The official model card recommends vLLM nightly, Transformers from main, and the Mistral parser flags for this model family. Their example is multi-GPU with FLASH_ATTN_MLA, but on A100 you should use auto or TRITON_MLA, not FLASH_ATTN_MLA. 

Check logs for:

Using Triton MLA backend
use_mla=True
compressed-tensors

If output is garbage at this point, the cause is likely compressed-tensors metadata / wrong modules quantized / tokenizer-template issue, not KV cache.


---

Phase 2: if model is still too large

If Phase 1 is too close to 72 GB, quantize selected attention projections, but still keep q_a_proj and kv_a_proj_with_mqa BF16.

Add this second scheme:

w4_attention_scheme = QuantizationScheme(
    targets=[
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.self_attn\\.q_b_proj$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.self_attn\\.kv_b_proj$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.self_attn\\.o_proj$",

        # Some MLX/Mistral4 conversions split kv_b_proj into these:
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.self_attn\\.embed_q$",
        "re:.*language_model\\.model\\.layers\\.[0-9]+\\.self_attn\\.unembed_out$",
    ],
    weights=w4_args,
)

Then modify:

config_groups={
    "w4a16_experts_g32": w4_expert_scheme,
    "w4a16_attention_safe_g32": w4_attention_scheme,
}

And remove this broad ignore:

"re:.*self_attn.*",
"re:.*attention.*",

Replace it with:

# Keep sensitive first-stage MLA projections BF16.
"re:.*self_attn\\.q_a_proj$",
"re:.*self_attn\\.kv_a_proj_with_mqa$",
"re:.*self_attn\\.q_a_layernorm$",
"re:.*self_attn\\.kv_a_layernorm$",

I would not quantize q_a_proj or kv_a_proj_with_mqa in the first working A100 build. They are small enough that saving a little memory there is not worth the risk.


---

Phase 3: only after clean output, try calibrated AWQ

After the RTN/MSE-style checkpoint generates sane text, try AWQ for the same expert targets. Keep the same ignore list. Use 256–512 real chat samples, not random text. LLM Compressor defaults to 512 calibration samples, batch size 1, and has moe_calibrate_all_experts=True; for MoE this matters. 

I would only AWQ the MLP expert paths first, not attention:

try:
    from llmcompressor.modifiers.transform.awq import AWQModifier, AWQMapping
except ImportError:
    from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

awq_mappings = [
    AWQMapping(
        "re:.*post_attention_layernorm$",
        [
            "re:.*mlp\\.experts\\..*gate_proj$",
            "re:.*mlp\\.experts\\..*up_proj$",
            "re:.*mlp\\.switch_mlp\\..*gate_proj$",
            "re:.*mlp\\.switch_mlp\\..*up_proj$",
            "re:.*mlp\\.shared_experts\\.gate_proj$",
            "re:.*mlp\\.shared_experts\\.up_proj$",
        ],
    ),
    AWQMapping(
        "re:.*mlp\\.experts\\..*up_proj$",
        ["re:.*mlp\\.experts\\..*down_proj$"],
    ),
    AWQMapping(
        "re:.*mlp\\.switch_mlp\\..*up_proj$",
        ["re:.*mlp\\.switch_mlp\\..*down_proj$"],
    ),
    AWQMapping(
        "re:.*mlp\\.shared_experts\\.up_proj$",
        ["re:.*mlp\\.shared_experts\\.down_proj$"],
    ),
]

recipe = [
    AWQModifier(mappings=awq_mappings),
    QuantizationModifier(
        config_groups={
            "w4a16_experts_g32": w4_expert_scheme,
        },
        ignore=ignore,
    ),
]

Run with:

oneshot(
    model=model,
    processor=processor,
    tokenizer=tokenizer,
    dataset=calib_ds,
    recipe=recipe,
    num_calibration_samples=512,
    max_seq_length=2048,
    batch_size=1,
    pipeline="sequential",
    moe_calibrate_all_experts=True,
    output_dir=OUT_DIR,
)


---

KV cache plan

For your stated goal, I would use this order:

Step 1 — baseline

--kv-cache-dtype auto

This should use BF16/FP16 KV. With MLA, 10K–16K context should be cheap.

Step 2 — if memory is tight

--kv-cache-dtype fp8

vLLM supports FP8 KV cache and documents fp8_e4m3 and fp8_e5m2 options on CUDA. 

I would not start with:

--calculate-kv-scales

It is deprecated in current vLLM CLI docs, and I would avoid another variable while debugging. 

Step 3 — TurboQuant only if you really need it

Try only after the model is stable with BF16 or FP8 KV:

--kv-cache-dtype turboquant_4bit_nc

vLLM’s TurboQuant docs list presets such as turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, and turboquant_3bit_nc; their own blog recommends FP8 as the safest default, says turboquant_4bit_nc may be practical under memory pressure, and warns against the aggressive 3-bit variants without careful validation. 

For this model on A100, I would not plan around RotorQuant. Treat RotorQuant/MLX as useful evidence that low-bit KV can work, but use vLLM-supported KV modes first.


---

Practical run order

I would do exactly this:

1. Quantize routed experts + shared experts to W4A16 group32 symmetric.
2. Keep router mlp.gate BF16.
3. Keep q_a_proj and kv_a_proj_with_mqa BF16.
4. Serve with TRITON_MLA, kv-cache-dtype auto, max-model-len 16K.
5. If sane, raise max-model-len to 32K / 64K.
6. If OOM, add FP8 KV.
7. If still OOM, quantize q_b/kv_b/o_proj.
8. Only then test AWQ calibration.
9. Only then test TurboQuant.

My expected outcome: with expert W4A16 and MLA BF16 KV, 16K should fit on your 72 GB usable A100. If it does not, the checkpoint is probably loading too many modules in BF16, not a KV-cache issue.
