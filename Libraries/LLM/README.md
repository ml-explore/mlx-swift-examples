#  LLM

This is a port of several models from:

- https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/

using the Hugging Face swift transformers package to provide tokenization:

- https://github.com/huggingface/swift-transformers

The [Models.swift](Models.swift) provides minor overrides and customization --
if you require overrides for the tokenizer or prompt customizations they can be
added there.

This is set up to load models from Hugging Face, e.g. https://huggingface.co/mlx-community

The following models have been tried:

- mlx-community/Mistral-7B-v0.1-hf-4bit-mlx
- mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX
- mlx-community/phi-2-hf-4bit-mlx
- mlx-community/quantized-gemma-2b-it

Currently supported model types are:

- Llama / Mistral
- Gemma
- Phi

See [Configuration.swift](Configuration.swift) for more info.

See [llm-tool](../../Tools/llm-tool)

# LoRA

[Lora.swift](Lora.swift) contains an implementation of LoRA based on this example:

- https://github.com/ml-explore/mlx-examples/tree/main/lora

See [llm-tool/LoraCommands.swift](../../Tools/llm-tool/LoraCommands.swift) for an example of a driver and
[llm-tool](../../Tools/llm-tool) for examples of how to run it.
