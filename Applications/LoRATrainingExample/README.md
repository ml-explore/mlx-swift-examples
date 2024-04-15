#  LoRATrainingExample

Example application that:

- downloads the `mlx-community/Mistral-7B-v0.1-hf-4bit-mlx` model from huggingface
- loads the train/valid/test data from `$SRCROOT/Data/lora` (this is copied into the build but you can imagine how it might be downloaded)
- adds LoRA adapters and trains the model
- let's you evaluate a prompt against the model

This roughly equates to the command line example in [Tools/llm-tool](../../Tools/llm-tool) and
you can read more about LoRA there.

This evaluates the LoRA adapted model rather than a fused model.  This doesn't persist
the LoRA weights or the fused model -- it will retrain it each time the program is launched.

### Troubleshooting

The `mlx-community/Mistral-7B-v0.1-hf-4bit-mlx` model requires a little over 4G of
memory to load an train -- this may require ~6G of physical RAM.


