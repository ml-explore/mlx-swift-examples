#  LLMEval

An example that:

- downloads a huggingface model and tokenizer
- evaluates a prompt
- displays the output as it generates text

You will need to set the Team on the LLMEval target in order to build and run on iOS.

Some notes about the setup:

- this downloads models from hugging face so LLMEval -> Signing & Capabilities has the "Outgoing Connections (Client)" set in the App Sandbox
- LLM models are large so this uses the Increased Memory Limit entitlement on iOS to allow ... increased memory limits for devices that have more memory
- `Memory.cacheLimit = 20 * 1024 * 1024` is used to limit the buffer cache size

`MLXChatExample` is a more full featured multi-turn chat example that supports VLMs.
`LLMBasic` is a **minimal** LLM chat example.

### Trying Different Models

The example app uses an 8 billion parameter quantized Qwen3 model by default, see [LLMEvaluator.swift](ViewModels/LLMEvaluator.swift#L52):

```
    var modelConfiguration = LLMRegistry.qwen3_8b_4bit
```

There are some pre-configured models in [MLXLLM/LLMModelFactory.swift](../../Libraries/MLXLLM/LLMModelFactory.swift#L78)
and you can load any weights from Hugging Face where there
is a model architecture defined and you have enough
memory.

For example:
```
    /// phi4bit is one of the smaller models so will fit on more devices
    var modelConfiguration = LLMRegistry.phi4bit
```

### Performance

Different models have difference performance characteristics. For example Gemma 2B may outperform Phi-2 in terms of tokens / second.

You may also find that running outside the debugger boosts performance. You can do this in Xcode by pressing cmd-opt-r and unchecking "Debug Executable".
