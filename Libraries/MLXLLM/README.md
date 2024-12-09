# MLXLLM

This is a port of several models from:

- https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/

using the Hugging Face swift transformers package to provide tokenization:

- https://github.com/huggingface/swift-transformers

The [LLMModelFactory.swift](LLMModelFactory.swift) provides minor overrides and customization --
if you require overrides for the tokenizer or prompt customizations they can be
added there.

This is set up to load models from Hugging Face, e.g. https://huggingface.co/mlx-community

The following models have been tried:

- mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX
- mlx-community/Llama-3.2-1B-Instruct-4bit
- mlx-community/Llama-3.2-3B-Instruct-4bit
- mlx-community/Meta-Llama-3-8B-Instruct-4bit
- mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
- mlx-community/Mistral-7B-Instruct-v0.3-4bit
- mlx-community/Mistral-Nemo-Instruct-2407-4bit
- mlx-community/OpenELM-270M-Instruct
- mlx-community/Phi-3.5-MoE-instruct-4bit
- mlx-community/Phi-3.5-mini-instruct-4bit
- mlx-community/Qwen1.5-0.5B-Chat-4bit
- mlx-community/SmolLM-135M-Instruct-4bit
- mlx-community/gemma-2-2b-it-4bit
- mlx-community/gemma-2-9b-it-4bit
- mlx-community/phi-2-hf-4bit-mlx
- mlx-community/quantized-gemma-2b-it

Currently supported model types are:

- Cohere
- Gemma
- Gemma2
- InternLM2
- Llama / Mistral
- OpenELM
- Phi
- Phi3
- PhiMoE
- Qwen2
- Starcoder2

See [llm-tool](../../Tools/llm-tool)

# Adding a Model

If the model follows the typical LLM pattern:

- `config.json`, `tokenizer.json`, and `tokenizer_config.json`
- `*.safetensors`

You can follow the pattern of the models in the [Models](Models) directory
and create a `.swift` file for your new model:

## Create a Configuration

Create a configuration struct to match the `config.json` (any parameters needed).

```swift
public struct YourModelConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    
    // use this pattern for values that need defaults
    public let _layerNormEps: Float?
    public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case _layerNormEps = "layer_norm_eps"
    }
}
```

## Create the Model Class

Create the model class.  The top-level public class should have a
structure something like this:

```swift
public class YourModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {

    public let kvHeads: [Int]

    @ModuleInfo var model: YourModelInner

    public func loraLinearLayers() -> LoRALinearLayers {
        // TODO: modify as needed
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ args: YourModelConfiguration) {
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = YourModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // TODO: modify as needed
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }
}
```

## Register the Model

In [LLMModelFactory.swift](LLMModelFactory.swift) register the model type itself
(this is independent of the model id):

```swift
public class ModelTypeRegistry: @unchecked Sendable {
...
    private var creators: [String: @Sendable (URL) throws -> any LanguageModel] = [
        "yourModel": create(YourModelConfiguration.self, YourModel.init),
```

Add a constant for the model in the `ModelRegistry` (not strictly required but useful
for callers to refer to it in code):

```swift
public class ModelRegistry: @unchecked Sendable {
...
    static public let yourModel_4bit = ModelConfiguration(
        id: "mlx-community/YourModel-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?"
    )
```

and finally add it to the all list -- this will let users find the model
configuration by id:

```swift
    private static func all() -> [ModelConfiguration] {
        [
            codeLlama13b4bit,
...
            yourModel_4bit,
```

# Using a Model

See [MLXLMCommon/README.md](../MLXLMCommon/README.md#using-a-model).

# LoRA

[Lora.swift](Lora.swift) contains an implementation of LoRA based on this example:

- https://github.com/ml-explore/mlx-examples/tree/main/lora

See [llm-tool/LoraCommands.swift](../../Tools/llm-tool/LoraCommands.swift) for an example of a driver and
[llm-tool](../../Tools/llm-tool) for examples of how to run it.
