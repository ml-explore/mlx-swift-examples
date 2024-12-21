#  LLMEval

An example that:

- downloads a huggingface model (phi-2) and tokenizer
- evaluates a prompt
- displays the output as it generates text

You will need to set the Team on the LLMEval target in order to build and run on iOS.

Some notes about the setup:

- this downloads models from hugging face so LLMEval -> Signing & Capabilities has the "Outgoing Connections (Client)" set in the App Sandbox
- LLM models are large so this uses the Increased Memory Limit entitlement on iOS to allow ... increased memory limits for devices that have more memory
- `MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)` is used to limit the buffer cache size
- The Phi2 4 bit model is small enough to run on some iPhone models
    - this can be changed by editing `let modelConfiguration = ModelConfiguration.phi4bit`

### Trying Different Models

The example application uses Phi2 model by default, see [ContentView.swift](ContentView.swift#L58):

```
    /// this controls which model loads -- phi4bit is one of the smaller ones so this will fit on
    /// more devices
    let modelConfiguration = ModelConfiguration.phi4bit
```

There are some pre-configured models in [MLXLLM/LLMModelFactory.swift](../../Libraries/MLXLLM/LLMModelFactory.swift#L78)
and you can load any weights from Hugging Face where there
is a model architecture defined and you have enough
memory.

### Troubleshooting

If the program crashes with a very deep stack trace you may need to build
in Release configuration.  This seems to depend on the size of the model.

There are a couple options:

- build Release
- force the model evaluation to run on the main thread, e.g. using @MainActor
- build `Cmlx` with optimizations by modifying `mlx/Package.swift` and adding `.unsafeOptions(["-O3"]),` around line 87

See discussion here: https://github.com/ml-explore/mlx-swift-examples/issues/3

### Performance

Different models have difference performance characteristics. For example Gemma 2B may outperform Phi-2 in terms of tokens / second.

You may also find that running outside the debugger boosts performance.  You can do this in Xcode by pressing cmd-opt-r and unchecking "Debug Executable".
