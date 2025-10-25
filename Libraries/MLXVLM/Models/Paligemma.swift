// Copyright © 2024 Apple Inc.

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/paligemma

import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Language

private enum Language {
    fileprivate class Attention: Module {

        let args: PaliGemmaConfiguration.TextConfiguration
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPE

        public init(_ args: PaliGemmaConfiguration.TextConfiguration) {
            self.args = args

            let dim = args.hiddenSize
            let heads = args.attentionHeads
            let kvHeads = args.kvHeads

            let headDim = args.hiddenSize / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self.rope = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(gelu(gate(x)) * up(x))
        }
    }

    fileprivate class TransformerBlock: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm

        public init(_ args: PaliGemmaConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class GemmaModel: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [TransformerBlock]
        fileprivate let norm: Gemma.RMSNorm

        let hiddenScale: Float

        public init(_ args: PaliGemmaConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.hiddenScale = pow(Float(args.hiddenSize), 0.5)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    TransformerBlock(args)
                }
            self.norm = Gemma.RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            mask: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbedding ?? embedTokens(inputs)
            h = h * hiddenScale

            let mask =
                if mask == nil || (cache?[0].offset ?? 0) > 0 {
                    createAttentionMask(h: h, cache: cache)
                } else {
                    MLXFast.ScaledDotProductAttentionMaskMode.none
                }

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: GemmaModel

        var kvHeads: [Int]

        public init(_ args: PaliGemmaConfiguration.TextConfiguration) {
            self.model = GemmaModel(args)

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            mask: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding, mask: mask)
            out = model.embedTokens.asLinear(out)
            return LMOutput(logits: out)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            weights.filter {
                !$0.key.contains("self_attn.rotary_emb.inv_freq")
            }
        }
    }
}

// MARK: - Vision

private enum Vision {
    fileprivate class Attention: Module {

        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "out_proj") var wo: Linear

        public init(dims: Int, numHeads: Int, bias: Bool = true) {
            precondition(dims % numHeads == 0, "Dimensions must be divisible by numHeads")

            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dims, dims, bias: bias)
            self._wk.wrappedValue = Linear(dims, dims, bias: bias)
            self._wv.wrappedValue = Linear(dims, dims, bias: bias)
            self._wo.wrappedValue = Linear(dims, dims, bias: bias)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        ) -> MLXArray {
            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            let (B, L) = (queries.dim(0), queries.dim(1))
            let S = keys.dim(1)

            queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class PhiMLP: Module, UnaryLayer {

        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        public init(_ config: PaliGemmaConfiguration.VisionConfiguration) {
            self.fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            self.fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(geluApproximate(fc1(x)))
        }
    }

    fileprivate class EncoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: PhiMLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        public init(_ config: PaliGemmaConfiguration.VisionConfiguration) {
            self._attention.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.attentionHeads, bias: true)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            self.mlp = PhiMLP(config)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        ) -> MLXArray {
            var r = attention(layerNorm1(x), mask: mask)
            let h = x + r
            r = mlp(layerNorm2(h))
            return h + r
        }
    }

    fileprivate class Encoder: Module {
        var layers: [EncoderLayer]

        public init(_ config: PaliGemmaConfiguration.VisionConfiguration) {
            self.layers = (0 ..< config.hiddenLayers).map { _ in
                EncoderLayer(config)
            }
        }

        public func callAsFunction(
            _ x: MLXArray, outputHiddenStates: Bool = false,
            mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        ) -> (MLXArray, [MLXArray]?) {
            var encoderStates: [MLXArray]? = outputHiddenStates ? [] : nil
            var h = x
            var x = x
            for l in layers {
                x = l(x, mask: mask)
                if outputHiddenStates {
                    encoderStates?.append(x)
                }
                h = x[0]
            }
            return (h, encoderStates)
        }
    }

    fileprivate class VisionEmbeddings: Module, UnaryLayer {

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        let positions: Int
        let _positionIds: MLXArray

        public init(_ config: PaliGemmaConfiguration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels, outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize), stride: .init(config.patchSize)
            )
            let d = config.imageSize / config.patchSize
            self.positions = d * d
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: positions, dimensions: config.hiddenSize
            )
            self._positionIds = MLXArray(0 ..< positions)[.newAxis, 0...]
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var patchEmbeddings = self.patchEmbedding(x)
            patchEmbeddings = patchEmbeddings.flattened(start: 1, end: 2)
            let embeddings = patchEmbeddings + self.positionEmbedding(self._positionIds)
            return embeddings
        }
    }

    fileprivate class SigLipVisionModel: Module {

        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        public init(_ config: PaliGemmaConfiguration.VisionConfiguration) {
            self.embeddings = VisionEmbeddings(config)
            self.encoder = Encoder(config)
            self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, MLXArray?
        ) {
            let x = embeddings(x)

            let (encoderOutput, hiddenStates) = encoder(x, outputHiddenStates: outputHiddenStates)
            let poolerOutput = postLayerNorm(encoderOutput)

            return (poolerOutput, x, hiddenStates?.last)
        }
    }

    fileprivate class VisionModel: Module {

        @ModuleInfo(key: "vision_model") var visionModel: SigLipVisionModel

        public init(_ config: PaliGemmaConfiguration.VisionConfiguration) {
            precondition(
                config.modelType == "siglip_vision_model",
                "Unsupported modelType: \(config.modelType)")
            self._visionModel.wrappedValue = SigLipVisionModel(config)
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, MLXArray?
        ) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim != 4 {
                return false
            }

            let (outChannels, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
            return outChannels >= kH && outChannels >= kW && kH == kW
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    // Remove unused position_ids
                    continue
                } else if k.contains("patch_embedding.weight") {
                    // PyTorch conv2d weight tensors have shape:
                    //   [out_channels, in_channels, kH, KW]
                    // MLX conv2d expects the weight be of shape:
                    //   [out_channels, kH, KW, in_channels]
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Processor

/// PaliGemma VLM `UserInputProcessor`.
///
/// This is meant to be used with ``PaliGemma`` and is typically created by ``VLMModelFactory``.
public class PaliGemmaProcessor: UserInputProcessor {

    private let config: PaliGemmaProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: PaliGemmaProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func prepare(image: CIImage, processing: UserInput.Processing?) throws -> MLXArray {
        // based on image_processing_siglip from transformers
        var image = image

        // we want to do all of the image processing in an sRGB tone curve
        // rather than a linear space as that is what transformers / torch_vision
        // do (implicitly by using sRGB rasters directly)
        image = MediaProcessing.inSRGBToneCurveSpace(image)

        // apply user instructions
        image = MediaProcessing.apply(image, processing: processing)

        image = try MediaProcessing.resampleBicubic(image, to: config.size.cgSize)
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
        var prompt = prompt(from: input)

        // based on transformers/processing_paligemma
        let count = input.images.count * config.imageSequenceLength
        prompt =
            Array(repeating: "<image>", count: count).joined() + (tokenizer.bosToken ?? "") + prompt
            + "\n"

        let promptTokens = try tokenizer.encode(text: prompt)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        let pixels = try prepare(image: input.images[0].asCIImage(), processing: input.processing)

        return LMInput(text: .init(tokens: promptArray, mask: mask), image: .init(pixels: pixels))
    }

    private func prompt(from userInput: UserInput) -> String {
        switch userInput.prompt {
        case .text(let text):
            text
        case .messages(let messages):
            messages.last?["content"] as? String ?? ""
        case .chat(let messages):
            messages.last?.content ?? ""
        }
    }

}

// MARK: - Model

private class PaliGemmaMultiModalProjector: Module, UnaryLayer {

    @ModuleInfo var linear: Linear

    public init(_ config: PaliGemmaConfiguration.VisionConfiguration) {
        self.linear = Linear(config.hiddenSize, config.projectionDimensions, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(x)
    }
}

/// PaliGemma VLM
///
/// This is typically created by ``VLMModelFactory``.
public class PaliGemma: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        PaliGemmaMultiModalProjector

    public let config: PaliGemmaConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: PaliGemmaConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
        self._multiModalProjector.wrappedValue = PaliGemmaMultiModalProjector(
            config.visionConfiguration)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, mask: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        guard let pixelValues else {
            return (inputIds, mask)
        }

        let inputEmbedding = languageModel.model.embedTokens(inputIds)
        let (hiddenState, _, _) = self.visionModel(
            pixelValues.transposed(0, 2, 3, 1).asType(inputEmbedding.dtype),
            outputHiddenStates: true
        )

        var imageFeatures = hiddenState[.newAxis, .ellipsis].asType(inputEmbedding.dtype)
        imageFeatures = multiModalProjector(imageFeatures)

        return prepareInputsForMultimodal(
            imageFeatures: imageFeatures, inputEmbedding: inputEmbedding,
            inputIds: inputIds, attentionMask: mask)
    }

    private func prepareInputsForMultimodal(
        imageFeatures: MLXArray, inputEmbedding: MLXArray, inputIds: MLXArray,
        attentionMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        let embedDimension = imageFeatures.dim(2)
        let (batchSize, sequenceLength) = inputIds.shape2
        var scaledImageFeatures = imageFeatures / pow(Float(config.hiddenSize), 0.5)

        let textMask = (inputIds .!= config.imageTokenIndex) & (inputIds .!= config.padTokenId)
        let imageMask = inputIds .== config.imageTokenIndex
        let padMask = inputIds .== config.padTokenId

        // expand masks to match embedding dimension
        var textMaskExpanded = expandedDimensions(textMask, axis: -1)
        var padMaskExpanded = expandedDimensions(padMask, axis: -1)

        // insert padding and text token embeddings
        var finalEmbedding = which(textMaskExpanded, inputEmbedding, 0)
        finalEmbedding = which(padMaskExpanded, 0, finalEmbedding)

        let padSize = finalEmbedding.dim(1) - scaledImageFeatures.dim(1)
        scaledImageFeatures = padded(scaledImageFeatures, widths: [0, .init((0, padSize)), 0])

        // insert image embeddings - the image mask is always less or equal to the sentence in length
        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        finalEmbedding = which(imageMaskExpanded, scaledImageFeatures, finalEmbedding)

        finalEmbedding = which(padMaskExpanded, 0, finalEmbedding)

        let attentionMaskExpanded1 = expandedDimensions(attentionMask, axis: 1)
        let attentionMaskExpanded2 = expandedDimensions(attentionMask, axis: 2)
        var finalAttentionMask4d = attentionMaskExpanded1 * attentionMaskExpanded2
        finalAttentionMask4d = expandedDimensions(finalAttentionMask4d, axis: 1)

        return (finalEmbedding, finalAttentionMask4d)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard let image = input.image else { throw VLMError.imageRequired }
        guard let mask = input.text.mask else { throw VLMError.maskRequired }
        let inputIds = input.text.tokens

        let (inputEmbedding, finalAttentionMask4d) = inputEmbeddings(
            inputIds: inputIds, pixelValues: image.pixels, mask: mask)

        let result = languageModel(
            inputIds, cache: cache, inputEmbedding: inputEmbedding, mask: finalAttentionMask4d)

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        languageModel.sanitize(weights: visionModel.sanitize(weights: weights))
    }
}

// MARK: - Configuration

/// Confguration for ``PaliGemma``
public struct PaliGemmaConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let vocabularySize: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
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
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case _rmsNormEps = "rms_norm_eps"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let patchSize: Int
        public let projectionDimensions: Int
        public let imageSize: Int
        private let _channels: Int?
        public var channels: Int { _channels ?? 3 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case projectionDimensions = "projection_dim"
            case imageSize = "image_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let vocabularySize: Int
    public let ignoreIndex: Int
    public let imageTokenIndex: Int
    public let hiddenSize: Int
    public let padTokenId: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case ignoreIndex = "ignore_index"
        case imageTokenIndex = "image_token_index"
        case hiddenSize = "hidden_size"
        case padTokenId = "pad_token_id"
    }
}

/// Configuration for ``PaliGemmaProcessor``
public struct PaliGemmaProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let width: Int
        public let height: Int

        var cgSize: CGSize { .init(width: width, height: height) }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let imageSequenceLength: Int

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
        case imageSequenceLength = "image_seq_length"
    }
}
