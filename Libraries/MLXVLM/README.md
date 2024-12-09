# MLXVLM

This is a port of several models from:

- https://github.com/Blaizzy/mlx-vlm

using the Hugging Face swift transformers package to provide tokenization:

- https://github.com/huggingface/swift-transformers

The [VLMModelFactory.swift](VLMModelFactory.swift) provides minor overrides and customization --
if you require overrides for the tokenizer or prompt customizations they can be
added there.

This is set up to load models from Hugging Face, e.g. https://huggingface.co/mlx-community

The following models have been tried:

- mlx-community/paligemma-3b-mix-448-8bit
- mlx-community/Qwen2-VL-2B-Instruct-4bit

Currently supported model types are:

- paligemma
- qwen2_vl

See [llm-tool](../../Tools/llm-tool)

# Adding a Model

If the model follows the typical VLM pattern:

- `config.json`, `tokenizer.json`, and `tokenizer_config.json`
- `*.safetensors`

You can follow the pattern of the models in the [Models](Models) directory
and create a `.swift` file for your new model:

## Create a Model Configuration

Create a configuration struct for both the Text and Vision models
that matches the structure in `config.json`.  A struct like this
is recommended:

```swift
public struct YourModelConfiguration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
        public let hiddenSize: Int

        // use this pattern for values that need defaults
        public let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        
        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case _layerNormEps = "layer_norm_eps"
        }
    }
    
    public struct VisionConfiguration: Codable, Sendable {
        ...
    }
    
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let vocabularySize: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case vocabularySize = "vocab_size"
    }
}
```

## Create a Processor Configuration

VLMs also require a image/video preprocessor.  Create a configuration to match 
the `preprocessor_config.json` file:

```swift
public struct YourModelProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let width: Int
        public let height: Int

        var cgSize: CGSize { .init(width: width, height: height) }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
    }
}
```

this will be consumed by:

```swift
public class YourModelProcessor: UserInputProcessor {
...
```

discussed later.

## Create the Vision, Text and VLM Classes

VLMs have language and vision models that are aggregated into a single
top level model.

For purposes of name spacing you might put the Language and Vision
models into an `enum` to create something structured like this:

```swift
// MARK: - Language

private enum Language {

    fileprivate class Attention: Module {
        ...
    }
    
    ...
    
    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: YourModel

        var kvHeads: [Int]
        var headDim: MLX.IntOrPair

        public init(_ args: YourModelConfiguration.TextConfiguration) {
            self.model = YourModel(args)

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
            }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            mask: MLXArray? = nil
        ) -> LMOutput {
            ...
            return LMOutput(logits: ...)
        }
    }
}
```

Similarly the Vision model can go into an `enum` namespace:

```swift
// MARK: - Vision

private enum Vision {

    fileprivate class Attention: Module {
        ...
    }
    
    fileprivate class VisionModel: Module {

        @ModuleInfo(key: "vision_model") var visionModel: InternalVisionModel

        public init(_ config: YourModelConfiguration.VisionConfiguration) {
            self._visionModel.wrappedValue = InternalVisionModel(config)
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, MLXArray?
        ) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
        }
    }
}
```

The exact signatures on the `init()` and `callAsFunction()` can vary as needed --
these models are not exposed to callers.

The top level model is the only piece of the model with public API and it
should implement `VLMModel` (aka `LanguageModel`).  Here is an outline of how
the top level model might work:

```swift
public class YourModel: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: YourModelConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var headDim: MLX.IntOrPair { languageModel.headDim }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ config: YourModelConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        // TODO prepare the cache and resulting logits based on the
        // text prompt and any media assets
        guard let image = input.image else { throw VLMError.imageRequired }
        guard let mask = input.text.mask else { throw VLMError.maskRequired }
        let inputIds = input.text.tokens

        let inputEmbedding = inputEmbeddings(
            inputIds: inputIds, pixelValues: image.pixels, mask: mask)

        let result = languageModel(
            inputIds, cache: cache, inputEmbedding: inputEmbedding, mask: mask)

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        // TODO evaluate a step in the language model
        languageModel(inputs, cache: cache).logits
    }
}
```

## Create the UserInputProcessor

VLMs require custom `UserInputProcessor` instances to manipulate the prompts and
media as needed.  For example it might:

- apply resampling and normalization to the images
- convert the images into an `MLXArray` and build a `THW` struct describing the layout
- modify the prompt by injecting `<image>` tokens that the model expects

In the python implementations, much of this code typically lives in the `transformers`
package from huggingface -- inspection will be required to determine which code
is called and what it does.  You can examine the processors in the `Models` directory:
they reference the files and functions that they are based on.

The `UserInputProcessor` is initialized with the `ProcessorConfiguration` (defined above)
and has a prepare method:

```swift
public func prepare(input: UserInput) throws -> LMInput
```

This is a slight paraphrase of the `PaligemmaUserInputProcessor` as an example:

```swift
public class YourModelProcessor: UserInputProcessor {

    private let config: YourModelProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: YourModelProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func prepare(image: CIImage, processing: UserInput.Processing?) -> MLXArray {
        // based on image_processing_siglip from transformers
        var image = image

        // we want to do all of the image processing in an sRGB tone curve
        // rather than a linear space as that is what transformers / torch_vision
        // do (implicitly by using sRGB rasters directly)
        image = MediaProcessing.inSRGBToneCurveSpace(image)

        // apply user instructions
        image = MediaProcessing.apply(image, processing: processing)

        image = MediaProcessing.resampleBicubic(image, to: config.size.cgSize)
        image = MediaProcessing.normalize(
            image, mean: config.imageMeanTuple, std: config.imageStdTuple)

        return MediaProcessing.asMLXArray(image)
    }

    public func prepare(input: UserInput) throws -> LMInput {
        switch input.images.count {
        case 0: throw VLMError.imageRequired
        case 1: break
        default: throw VLMError.singleImageAllowed
        }

        // this doesn't have a chat template so just use the last message.
        var prompt = input.prompt.asMessages().last?["content"] ?? ""

        // based on transformers/processing_paligemma
        let count = input.images.count * config.imageSequenceLength
        prompt =
            Array(repeating: "<image>", count: count).joined() + (tokenizer.bosToken ?? "") + prompt
            + "\n"

        let promptTokens = try tokenizer.encode(text: prompt)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray)

        let pixels = try prepare(image: input.images[0].asCIImage(), processing: input.processing)

        return LMInput(text: .init(tokens: promptArray, mask: mask), image: .init(pixels: pixels))
    }

}
```

Note that the python code may rely on the chat template to inject the image tokens
(paligemma does not).  This may have to be expressed in swift code as the current
interface does not support the structured parameters used for this (see Qwen2VL 
processor for an example).

## Register the Model

In [VLMModelFactory.swift](VLMModelFactory.swift) register the model type itself
(this is independent of the model id):

```swift
public class ModelTypeRegistry: @unchecked Sendable {
...
    private var creators: [String: @Sendable (URL) throws -> any LanguageModel] = [
        "yourModel": create(YourModelConfiguration.self, YourModel.init),
```

Similarly, register the UserInputProcessor type (`preprocessor_config.json`):

```swift
public class ProcessorTypeRegistry: @unchecked Sendable {
...
    private var creators:
        [String: @Sendable (URL, any Tokenizer) throws -> any UserInputProcessor] = [
            "YourModelProcessor": create(
                YourModelProcessorConfiguration.self, YourModelProcessor.init),
```

Add a constant for the model in the ModelRegistry (not strictly required but useful
for callers to refer to it in code):

```swift
public class ModelRegistry: @unchecked Sendable {
...
    static public let yourModel_4bit = ModelConfiguration(
        id: "mlx-community/YourModel-4bit",
        defaultPrompt: "Describe the image in English"
    )
```

and finally add it to the all list -- this will let users find the model
configuration by id:

```swift
    private static func all() -> [ModelConfiguration] {
        [
            paligemma3bMix4488bit,
...
            yourModel_4bit,
```

# Using a Model

See [MLXLMCommon/README.md](../MLXLMCommon/README.md#using-a-model).
