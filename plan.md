# Port Qwen3-VL into MLX Swift Examples

## ✅ COMPLETED

All Qwen3-VL components have been successfully ported and validated!

- ✅ Text-only generation working (haiku example)
- ✅ Vision + language working (image understanding)
- ✅ 8-bit quantized model supported
- ✅ All 4.4B parameters loading correctly
- ✅ VLMEval app functional end-to-end

## Previous Status
- Repository references `Qwen3-VL-4B-Instruct-4bit` but lacks Swift implementation for language/vision/processor.
- Builds fail when Qwen3-VL is set as default model.
- Python reference implementation exists in `~/Developer/Samples/mlx-vlm/mlx_vlm/models/qwen3_vl/`.
- Need to port components and integrate with MLXVLM.

## IMPORTANT
Refernce the python implementation for the language stack and vision tower in Developer/Samples/mlx-vlm/ for Qwen3-VL.

## Goals
1. Implement Qwen3-VL language stack (attention, mRope, deepstack, KV caches).
2. Implement Qwen3-VL vision tower and connectors.
3. Implement user input processor for images/videos + prompts.
4. Register the model/processor in `VLMModelFactory` and make it selectable.
5. Validate VLMEval app end-to-end with Qwen3-VL.

## Detailed Tasks (All Completed ✅)

### 1. ✅ Port Language Stack (`Libraries/MLXVLM/Models/Qwen3VLLanguage.swift`)
- ✅ Rotary embedding (3-axis mRoPE) from python `language.py`
- ✅ Attention with RMSNorm, cache update, rotary application
- ✅ MLP (gate/down/up) and decoder layers
- ✅ Deepstack visual embeddings and `LanguageModel` wrapper

### 2. ✅ Port Vision Tower (`Libraries/MLXVLM/Models/Qwen3VLVision.swift`)
- ✅ Patch embedding (3D conv), positional embeddings, transformer blocks
- ✅ Patch merger and deepstack outputs
- ✅ Outputs feed correctly into language module

### 3. ✅ Combine Model (`Libraries/MLXVLM/Models/Qwen3VL.swift`)
- ✅ Vision + language components wired
- ✅ `prepare`, `callAsFunction`, LoRA hooks, sanitizer implemented
- ✅ Visual token insertion and merged embeddings working

### 4. ✅ Processor (`Libraries/MLXVLM/Models/Qwen3VLProcessor.swift`)
- ✅ Image/video preprocessing using `MediaProcessing`
- ✅ Chat template, padding replacement, position id calculations
- ✅ Mirrors python processor behavior

### 5. ✅ Factory & Registry Updates
- ✅ Model/processor registered in `VLMModelFactory`
- ✅ `VLMRegistry.qwen3VL4BInstruct4Bit` with default prompt/EOS tokens
- ✅ App functional with Qwen3-VL

### 6. ✅ Validation & Testing
- ✅ `swift build --target MLXVLM` passes
- ✅ `swift build --target VLMEval` passes
- ✅ VLMEval with text-only generates coherent haikus
- ✅ VLMEval with images works flawlessly

## Critical Bugs Fixed During Implementation

1. **Weight Key Remapping** - `"model.language_model"` → `"language_model.model"` ensures all weights load
2. **lm_head Mapping** - Correctly strips `"model.lm_head"` prefix
3. **Visual Mask Dimension** - Preserves `[batch, seq]` shape for deepstack
4. **Text-Only Embeddings** - Passes `nil` to let Model embed from `inputIds` 
5. **Causal Mask Bug** - Passes `mask: nil` to Model to allow causal mask creation (critical fix!)

## Implementation Notes
- MRoPE position id logic successfully ported (3D position IDs for temporal/height/width)
- 8-bit quantized model works efficiently
- Video processing integrated with frame sampling
- Deepstack visual embeddings working for multi-layer visual features

## Ready for Pull Request
All components tested and working. Code is clean and follows MLX Swift Examples patterns.

