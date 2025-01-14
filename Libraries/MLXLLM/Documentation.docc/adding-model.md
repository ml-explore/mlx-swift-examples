#  Adding a Model

If the model follows the typical LLM pattern you can add a new
model in a few steps.

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
