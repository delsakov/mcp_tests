llmcompressor doesn't actually need a hardcoded "Mistral 4" definition. As long as you have a recent version of transformers, Mistral Small 4 loads under the standard MistralForCausalLM or MixtralForCausalLM MoE architecture. llmcompressor dynamically traces the Hugging Face AST to find the linear layers.
Since you are dealing with a 119B model and targeting INT4 (AWQ or GPTQ) for your A100-80GB, you need to leverage llmcompressor's **model offloading** (introduced in version 0.10) to avoid OOM during calibration, and you must **explicitly protect the router weights** you validated so they stay in BF16.
Here is the recipe to execute this using the oneshot pipeline:
## 1. Environment Setup
Make sure you are on llmcompressor >= 0.10 and the latest transformers, as this includes the custom compressed-tensors offloading required to calibrate a 119B model without needing a massive multi-GPU cluster.
```bash
pip install --upgrade llmcompressor transformers compressed-tensors

```
## 2. The Quantization Recipe
You will use the GPTQModifier or AWQModifier. For MoE models, llmcompressor aggressively targets all Linear modules by default. **You must use the ignore list** to tell it to skip the embedding layers, the language modeling head, and crucially, the MoE gates/routers.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from compressed_tensors.offload import load_offloaded_model
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier # or AWQModifier

model_id = "mistralai/Mistral-Small-4-119B-2603"
save_path = "./Mistral-Small-4-119B-INT4"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 1. Load the model using the offloading context manager
# This prevents RAM/VRAM OOM by dynamically paging layers
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        torch_dtype="auto"
    )

# 2. Configure INT4 Quantization
# We target W4A16 (4-bit weights, 16-bit activations)
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    group_size=128,
    ignore=[
        "lm_head",          # Keep output head in BF16
        "tok_embeddings",   # Keep embeddings in BF16
        "re:.*gate$"        # CRITICAL: Keep MoE routers in BF16
    ]
)

# 3. Setup Calibration Data
# Use a lightweight dataset to calculate the quantization scales
def format_data(example):
    return tokenizer(example["text"], padding=False, truncation=True, max_length=2048)

from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(512))
ds = ds.map(format_data)

# 4. Run the compression
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

# 5. Save the quantized model
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

```
### Why this works for your setup:
 * **re:.*gate$:** This regex is the secret sauce for MoE models in llmcompressor. It ensures that Mistral's routing algorithm (which determines which 4 of the 128 experts fire) remains untouched in BF16, preserving the exact routing behavior you already verified.
 * **auto_offload:** The 119B FP16 model is ~238 GB. This flag automatically stages layers between your disk, system CPU RAM, and the A100 VRAM, pulling them onto the GPU only when that specific layer is being calibrated.
 * **Format:** The resulting output will be saved in safetensors format with compressed-tensors metadata. vLLM can load this natively.
Once quantized, the model footprint will drop to ~65 GB. You can then load it into vLLM alongside RotorQuant for your 3-bit KV cache.















How to apply this to Gemma-4-31B-it
You can build this exact pipeline using the llmcompressor library. Instead of just loading a single modifier, you stack the SpinQuantModifier on top of the QuantizationModifier in your recipe.

Here is the exact Python recipe to apply SpinQuant + GPTQ to your Gemma-4 model:

Python
from llmcompressor.modifiers.experimental import SpinQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

# The Recipe Stack: Rotate first, then Quantize
recipe = [
    # Step 1: Smooth the outliers
    SpinQuantModifier(
        targets=["Linear"],
        rotations=["R1", "R2"], # Apply to both weights and activations
        transform_block_size=128, 
        transform_type="hadamard"
    ),
    # Step 2: Compress using GPTQ
    QuantizationModifier(
        targets=["Linear"],
        scheme="W4A16",      # 4-bit weights, 16-bit activations
        algorithm="GPTQ",    # Explicitly enforce GPTQ calibration
        ignore=["lm_head"],  # Protect the final output head
        group_size=64        # Maintain the outlier cage for safety
    )
]

# Execute the pipeline (requires a loaded model and calibration dataset)
oneshot(
    model=model,
    dataset=calibration_dataset,
    recipe=recipe,
    max_seq_length=4096,
    num_calibration_samples=512,
)
A Quick Deployment Note for Gemma-4
Because Gemma-4 utilizes a unique Hybrid Attention architecture (mixing sliding window and global layers), the SpinQuantModifier needs to correctly read the head_dim from the model's config.json to calculate the rotation blocks. Make sure you are on the absolute latest nightly build of llmcompressor, as the community only recently patched the config parsing logic to fully support the newer Gemma variants.
