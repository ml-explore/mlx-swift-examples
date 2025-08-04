# Porting Models

There are a number of ways to implement new models in MLX in Swift:

- Build from scratch
- Port from other ML frameworks
    - [MLX Documentation](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html)
- Port from [existing MLX models in Python](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models)

This document talks primarily about the latter.

## Porting Models from MLX in Python

Let's consider a concrete example, [gemma.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma.py). For reference, here is the current port [Gemma.swift](https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/MLXLLM/Models/Gemma.swift).

### Imports

When creating the new model, you need to import the right modules. Typically these are sufficient:

```swift
import Foundation
import MLX
import MLXLMCommon
import MLXNN
```

- Foundation
    - Used for standard Swift features like `Codable`. This lets us easily read the JSON configuration.
- MLX
    - The base [`MLXArray`](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx) framework
- MLXLMCommon
    - Language model support code (this library)
    - Provides weight loading, token generation, etc.
- MLXNN
    - The base [Module and NN](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlxnn) package for MLX

### Configuration

Next, port the configuration type from Python. You can see it at the top of the `gemma.py` file:

```python
@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 10000
    rope_traditional: bool = False
```

This will be loaded from a JSON file and used to configure the model, including layer parameters like `rms_norm_eps` and structure like `num_hidden_layers`.

This translates naturally into a `Codable` struct in Swift with a few details:

- The keys in the JSON file will be `snake_case`. The simplest way to accommodate that is to specify `CodingKeys` to name them explicitly.
- Some of the parameters have default values.

```swift
public struct GemmaConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    private let _ropeTheta: Float?
    public var ropeTheta: Float { _ropeTheta ?? 10_000 }
    private let _ropeTraditional: Bool?
    public var ropeTraditional: Bool { _ropeTraditional ?? false }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case _ropeTheta = "rope_theta"
        case _ropeTraditional = "rope_traditional"
    }
}
```

* Note: The type is called `ModelArgs` in Python, and this is scoped to the file because of the way Python importing works. In Swift we need this type to be public, so we give it a unique name, `GemmaConfiguration`.

Default values are implemented with this pattern:

```swift
private let _ropeTheta: Float?
public var ropeTheta: Float { _ropeTheta ?? 10_000 }
```

And `CodingKeys` maps `rope_theta` to `_ropeTheta`:

```swift
enum CodingKeys: String, CodingKey {
...
    case _ropeTheta = "rope_theta"
```

This will read `rope_theta` from the JSON file if present and fall back to a default value of `10_000`.

### Understanding Property Wrappers

MLX Swift uses property wrappers to handle automatic registration of modules and parameters. This is different from Python MLX, which uses runtime discovery.

#### Python MLX: Automatic Discovery

In Python MLX, you simply assign modules and parameters as attributes, and MLX automatically discovers them:

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Just assign - MLX auto-discovers these
        self.linear = nn.Linear(256, 256)  # sub-module
        self.weight = mx.ones([256])       # parameter
```

#### Swift MLX: Property Wrappers

In Swift MLX, you must explicitly declare modules and parameters using property wrappers:

```swift
class MyModule: Module {
    @ModuleInfo var linear: Linear     // sub-module
    @ParameterInfo var weight: MLXArray // parameter
    
    init() {
        self._linear.wrappedValue = Linear(256, 256)
        self._weight.wrappedValue = MLXArray.ones([256])
        super.init()
    }
}
```

#### When to Use Each

- **`@ModuleInfo`**: For neural network layers (anything with `callAsFunction`)
  - Examples: `Linear`, `Attention`, `MLP`, `RMSNorm`, arrays of layers
- **`@ParameterInfo`**: For raw `MLXArray` tensors you use manually
  - Examples: weight matrices, bias vectors, scaling factors

#### Weight Key Mapping

Both property wrappers support custom keys for weight loading:

```swift
@ModuleInfo(key: "self_attn") var attention: Attention
@ParameterInfo(key: "custom_weight") var weight: MLXArray
```

The `key` parameter maps to the corresponding key in the saved model weights. If no key is specified, it uses the property name.

#### Initialization Syntax

When using property wrappers, initialization uses the `wrappedValue` syntax:

```swift
// Without property wrapper:
self.weight = MLXArray.ones([dimensions])

// With property wrapper:
self._weight.wrappedValue = MLXArray.ones([dimensions])
```

This explicit registration system provides type safety and ensures all modules and parameters are properly tracked for operations like weight loading, quantization, and gradient computation.

#### Advanced Property Wrapper Patterns

Beyond the basic usage, there are several advanced patterns you'll encounter:

**Optional Modules**
```swift
@ModuleInfo(key: "lm_head") var lmHead: Linear?
@ModuleInfo(key: "text_projection") var textProjection: Linear?
```
Used when modules are conditionally created based on configuration.

**Arrays of Modules**
```swift
@ModuleInfo(key: "layers") var layers: [TransformerBlock]
@ModuleInfo(key: "down_blocks") var downBlocks: [EncoderDecoderBlock2D]
```
For dynamic numbers of repeated layers.

**Complex Module Types**
```swift
@ModuleInfo(key: "mid_blocks") var midBlocks: (ResnetBlock2D, Attention, ResnetBlock2D)
```
Tuples and other composite types are supported.

**Optional Parameters**
```swift
@ParameterInfo var bias: MLXArray?
@ModuleInfo(key: "bias") var bias: MLXArray?
```
For optional parameters that may not exist in all model variants.

**Special Case: @ModuleInfo with MLXArray**
```swift
@ModuleInfo(key: "weight") var weight: MLXArray
@ModuleInfo(key: "scales") var scales: MLXArray
```
In rare cases (like quantized layers), `@ModuleInfo` is used with `MLXArray` instead of `@ParameterInfo`. This typically occurs with specialized quantization or expert layers where the arrays are treated as sub-modules for weight loading purposes.

**Computed vs Loaded Parameters**
```swift
// Parameter loaded from weights - uses @ParameterInfo
@ParameterInfo(key: "correct_output_scale") var correctOutputScale: MLXArray

// Computed parameter - uses private underscore prefix (ignored during loading)
private let _routerInputScale: MLXArray
```

This distinction is crucial:
- `@ParameterInfo` properties are expected to exist in the weight files and will be loaded automatically
- Private properties with underscore prefix (`_`) are ignored during weight loading and must be computed during initialization
- Initialization syntax differs:
  ```swift
  // For @ParameterInfo (loaded from weights):
  self._correctOutputScale.wrappedValue = MLXArray(...)
  
  // For private computed parameters:
  self._routerInputScale = MLXArray(...)
  ```

Use private underscore properties when you need to compute values based on configuration or other parameters, but don't want the weight loading to fail because these "parameters" don't exist in the saved weights.

### Porting Layers without Children

Now we can begin porting the layers (Modules). Here is an example layer with no child layers (e.g. `Linear`) but with parameters (e.g. `MLXArray`):

```python
class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
```

And the equivalent Swift code:

```swift
private class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
    }
}
```

* Note: The Modules that make up the layers in the model are typically declared as `private`. Many models use similarly named layers, and this keeps the names isolated to each model's namespace.

Here is a detailed breakdown of the conversion. Consider the Python initializer for the class:

```python
def __init__(self, dims: int, eps: float = 1e-5):
    super().__init__()
    self.weight = mx.ones((dims,))
    self.eps = eps
```

In Python, storing a value in a property is how instance variables (properties) are created. This is implemented as a dictionary behind the scenes. Swift requires that properties be declared and given types:

```swift
let weight: MLXArray
let eps: Float

public init(dimensions: Int, eps: Float = 1e-5) {
    self.weight = MLXArray.ones([dimensions])
    self.eps = eps
}
```

Note that the weight is given an initial value and shape based on the parameters to the initializer. In typical inference use, these values will be replaced when the weights are loaded (``loadWeights(modelDirectory:model:quantization:)``).

* Note: If the property names in Python don't make good Swift names, you can use the `@ParameterInfo` property wrapper to specify the key: `@ParameterInfo(key: "some_weight") var weight: MLXArray`

If using the `@ParameterInfo` to override the parameter key, be aware that the syntax for initializing the value changes:

```swift
// Was: self.weight = MLXArray.ones([dimensions])
self._weight.wrappedValue = MLXArray.ones([dimensions])
```

Next, the `__call__` method in Python is a direct conversion to Swift:

```python
def __call__(self, x):
    return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
```

becomes:

```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
}
```

* Note: [This reference](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/converting-python) shows the mapping between Python method and function names and Swift.

### Porting Layers with Children

Consider this module from Python that uses the previously defined `RMSNorm` module:

```python
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
```

and the full conversion to Swift:

```swift
private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: GemmaConfiguration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}
```

As seen previously, the `__init__` method converts to an initializer for the class:

```python
def __init__(self, args: ModelArgs):
    super().__init__()
    self.num_attention_heads = args.num_attention_heads
    self.hidden_size = args.hidden_size
    self.self_attn = Attention(args)
    self.mlp = MLP(args.hidden_size, args.intermediate_size)
    ...
```

This initializer takes `ModelArgs` in Python. This is the name of the configuration type, which we call `GemmaConfiguration`.

```swift
@ModuleInfo(key: "self_attn") var attention: Attention
let mlp: MLP

...

public init(_ args: GemmaConfiguration) {
    self._attention.wrappedValue = Attention(args)
    self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
    ...
```

Note that some of the lines from Python are omitted:

```python
self.num_attention_heads = args.num_attention_heads
self.hidden_size = args.hidden_size
```

These properties are not used in the code and can be discarded. If there are many properties that _are_ needed, it may be more convenient in Swift to simply store the `GemmaConfiguration`, which provides access to the typed properties inside.

* Note: Much like parameters with non-Swift names, you can (and typically do) use `@ModuleInfo` to give the same naming hint: `@ModuleInfo(key: "self_attn") var attention: Attention`

Finally we convert the `__call__` method from Python:

```python
def __call__(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    r = self.self_attn(self.input_layernorm(x), mask, cache)
    h = x + r
    r = self.mlp(self.post_attention_layernorm(h))
    out = h + r
    return out
```

Note that the `r` variable is assigned twice, so we make this a `var` in Swift:

```swift
public func callAsFunction(
    _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
) -> MLXArray {
    var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    r = mlp(postAttentionLayerNorm(h))
    return h + r
}
```

Sometimes the input parameter is assigned in this function, to keep the code structure as close as possible to the Python:

```python
def __call__(self, x) -> mx.array:
    x = x + 1
    x = mx.sqrt(x)
    return x
```

I suggest porting this code as:

```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = x
    x = x + 1
    x = sqrt(x)
    return x
}
```

### Porting Layers: Configuration and Structure

Sometimes the configuration determines the structure of the model:

```python
class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        ...
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        ...
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        ...
```

In this case, the number of `TransformerBlock` layers depends on the configuration value `num_hidden_layers` to create an array of children. In `__call__` it iterates through these children, chaining the calls together sequentially.

In Swift you do the same thing:

```swift
private class GemmaModelInner: Module {
    ...
    fileprivate let layers: [TransformerBlock]

    public init(_ args: GemmaConfiguration) {
        ...
        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                TransformerBlock(args)
            }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        ...
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        ...
    }
```

You have an array `layers` that matches the same property in Python. When calling the Module, you simply iterate through the `layers` property, chaining the calls together.

### Model Class

Finally we reach the top level of of the model. In Python there is typically a class named `Model`. This is the public entry point into the model. There is typically very little code in this module, though it may prepare the inputs and outputs.

There is also usually a class named e.g. `GemmaModel`, which is the implementation of the model itself:

```python
class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
    ...

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = GemmaModel(args)
        self.args = args
    ...
```

In Swift we need to expose the `Model` class as a public type and leave the `GemmaModel` as a private implementation detail. Typically these are named like this in Swift:

```swift
private class GemmaModelInner: Module {
    let args: GemmaConfiguration
    let vocabularySize: Int
    let numHiddenLayers: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [TransformerBlock]
    fileprivate let norm: RMSNorm

    public init(_ args: GemmaConfiguration) {
    ...
}

public class GemmaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let modelType: String
    private let model: GemmaModelInner

    public init(_ args: GemmaConfiguration) {
        self.modelType = args.modelType
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = GemmaModelInner(args)
    }
    ...
}
```

### Registration

The last step before we can use the model is to register the types so that everything can be found from the configuration file.

Since this is an LLM (as opposed to a VLM), we register the type that will show in the configuration file in the `LLMTypeRegistry`:

```swift
public class LLMTypeRegistry: ModelTypeRegistry, @unchecked Sendable {

    private static func all() -> [String: @Sendable (URL) throws -> any LanguageModel] {
    [
        ...
        "gemma": create(GemmaConfiguration.self, GemmaModel.init),
```

Now we can load the model using `llm-tool` or the `LLMEval` example application, or implement this ourselves in code:

```swift
let modelConfiguration = ModelConfiguration(id: "mlx-community/quantized-gemma-2b-it")

// This will download the weights from Hugging Face Hub and load the model
let container = try await MLXModelFactory.shared.loadContainer(configuration: modelConfiguration)

// Prepare the prompt and parameters used to generate the response
let generateParameters = GenerateParameters()
let input = UserInput(prompt: "Are cherries sweet?")

// Run inference
let result = try await modelContainer.perform { [input] context in
    // Convert the UserInput into LMInput
    let input = try context.processor.prepare(input: input)

    return generate(input: input, parameters: generateParameters, context: context) { tokens in
        // This could potentially use NaiveStreamingDetokenizer and print
        // text as it was generated
        if tokens.count >= 20 {
            return .stop
        } else {
            return .more
        }
    }
}

print(result.output)
```

## Notes

### Porting a Model Similar to an Existing Model

If you are porting a model and it is similar to an existing (already ported) model, you can often take a shortcut and look at the diffs.

For example `gemma.py` and `gemma2.py` are related. If you look at the diff between them, it comprises roughly a dozen changes. If you already have `Gemma.swift`, you can copy that to `Gemma2.swift` (and make the appropriate naming changes) and then examine the diffs on the Python side and make the same changes.

Many models are related to each other, so this can be a very effective way to create new models.

### Debugging a Ported Model

What do you do when your ported model spews random text or crashes with a broadcast error?

Let's start with the latter: A broadcast error means that the shapes of the `MLXArray`s are incorrect. For example, if the broadcast error shows up in `Attention` (which is often the case), then you can make a helper function in Python:

```python
def trace(name, x):
    print(f"{name}: {x.shape}")
```

and one in Swift:

```swift
func trace(_ name: String, _ x: MLXArray) {
    print("\(name): \(x.shape)")
}
```

In the Python code you can print the known working shapes:

```python
trace("queries", queries)
trace("keys", keys)
# etc. as needed
output = scaled_dot_product_attention(
    queries, keys, values, cache=cache, scale=self.scale, mask=mask
)
```

and the same in Swift:

```swift
trace("queries", queries)
trace("keys", keys)
// etc. as needed
let output = MLXFast.scaledDotProductAttention(
    queries: queries, keys: keys, values: values, scale: scale, mask: mask
)
```

Often it will be a shape like `[1, 128, 256]` vs. `[128, 256]` (in this case, there is a missing `[.newAxis]` somewhere in the code). It may be something more complicated, but either way you know which value is incorrect and you can track it down.

Incorrect output can be investigated in a similar fashion, but you can start by making sure the inputs are correct: compare the integer tokens from the Python side to what the Swift side generates. The implementations of `transformers` and `swift-transformers` are similar but not identical. If needed, the token array from the Python program can be used directly.

After making sure the inputs are identical, make sure the generation parameters (temperature, seed, etc.) are the same.

If the output still differs, then you must look at the contents of the arrays during inference, not just the shapes. You can modify the `trace` functions like this:

```python
def trace(name, x):
    print(f"{name}: {x.shape} {x.sum().item()}")
```

and:

```swift
func trace(_ name: String, _ x: MLXArray) {
    print("\(name): \(x.shape) \(x.sum().item(Float.self))")
}
```

This uses `sum()` to give an aggregate value. If these produce the same (or nearly the same) values, then the contents of the array are _probably_ the same. If they are wildly different, the contents of the array are _certainly_ different. If the arrays contain larger numbers, you might try different aggregation functions or look at slices of the array, e.g. the first row.

You can use these calls on the values passed into the `__call__`/`callAsFunction` methods, and you can also use them on the parameters of the layers themselves.

The nice thing about this technique is that it doesn't require understanding how the model works. You have a reference implementation on the Python side, and you only need to identify where it is different. Once you determine the point where it is different, you can track backward and figure out why (again, it is just calling functions and doing math).

### Optional Modules and Parameters

Models sometimes have optional modules or parameters based on their configuration.
For example, [qwen2.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py#L161C1-L163C1) only creates the `lm_head` module if `tie_word_embeddings` is `False`:

```python
if not args.tie_word_embeddings:
    self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
```

In Swift it is important to do the same thing using `Optional`:

```swift
public class Qwen2Model: Module, LLMModel, KVCacheDimensionProvider {
    ...

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen2Configuration) {
        ...
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = ...
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        ...
    }
```

If the `lmHead` module is created but not used, the parameter load will fail validation because the `lm_head` keys will be missing.

### Pre-computed MLXArrays

In some cases it is convenient to pre-compute some `MLXArray` but not treat it as a loadable parameter. In particular, we do not want the loading of parameters to fail because this MLXArray is "missing".

For example, in PaliGemma there is a constant `positionIds` based on the `imageSize` and `patchSize` configuration. If we name the property with a leading underscore, it will not be considered a valid parameter and will be ignored when loading parameters:

```swift
fileprivate class VisionEmbeddings: Module, UnaryLayer {

    ...
    let positions: Int
    let _positionIds: MLXArray

    public init(_ config: PaliGemmaConfiguration.VisionConfiguration) {
        ...
        let d = config.imageSize / config.patchSize
        self.positions = d * d
        self._positionIds = MLXArray(0 ..< positions)[.newAxis, 0...]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        ...
        let embeddings = patchEmbeddings + self.positionEmbedding(self._positionIds)
        ...
    }
}
```
