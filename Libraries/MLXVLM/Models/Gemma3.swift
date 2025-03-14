import CoreImage
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma3

// MARK: - Text Configuration

public struct Gemma3TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let slidingWindow: Int
    public let ropeScaling: [String: StringOrNumber]?

    // Default values
    public var attentionHeads: Int = 8
    public var headDim: Int = 256
    public var rmsNormEps: Float = 1.0e-6
    public var vocabularySize: Int = 262208
    public var kvHeads: Int = 4
    public var ropeGlobalBaseFreq: Float = 1_000_000.0
    public var ropeLocalBaseFreq: Float = 10_000.0
    public var ropeTraditional: Bool = false
    public var queryPreAttnScalar: Float = 256
    public var mmTokensPerImage: Int = 256
    public var slidingWindowPattern: Int = 6

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case slidingWindow = "sliding_window"
        case ropeScaling = "rope_scaling"
    }
}

// MARK: - Vision Configuration

public struct Gemma3VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let patchSize: Int
    public let imageSize: Int
    public let visionUseHead: Bool
    public let skipVision: Bool

    // Default values
    public var numChannels: Int = 3
    public var layerNormEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case imageSize = "image_size"
        case visionUseHead = "vision_use_head"
        case skipVision = "skip_vision"
    }
}

// MARK: - Quantization Configuration

public struct QuantizationConfig: Codable, Sendable {
    public let groupSize: Int
    public let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

// MARK: - Model Configuration

public struct Gemma3Configuration: Codable, Sendable {
    public let textConfiguration: Gemma3TextConfiguration
    public let visionConfiguration: Gemma3VisionConfiguration
    public let modelType: String
    public let architectures: [String]
    public let imageTokenIndex: Int
    public let mmTokensPerImage: Int
    public let boiTokenIndex: Int
    public let eoiTokenIndex: Int
    public let eosTokenId: [Int]
    public let torchDtype: String
    public let transformersVersion: String
    public let quantization: QuantizationConfig?
    public let initializerRange: Float

    // Default values
    public var vocabularySize: Int = 257152
    public var ignoreIndex: Int = -100
    public var hiddenSize: Int = 2048
    public var padTokenId: Int = 0

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case architectures
        case imageTokenIndex = "image_token_index"
        case mmTokensPerImage = "mm_tokens_per_image"
        case boiTokenIndex = "boi_token_index"
        case eoiTokenIndex = "eoi_token_index"
        case eosTokenId = "eos_token_id"
        case torchDtype = "torch_dtype"
        case transformersVersion = "transformers_version"
        case quantization
        case initializerRange = "initializer_range"
    }
}

// MARK: - Attention

private class Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: GemmaUtils.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: GemmaUtils.RMSNorm

    @ModuleInfo var rope: RoPE

    init(config: Gemma3TextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.repeats = numHeads / numKVHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx

        self.scale = pow(config.queryPreAttnScalar, -0.5)

        self._queryProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = GemmaUtils.RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = GemmaUtils.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

        let baseFreq = isSliding ? config.ropeLocalBaseFreq : config.ropeGlobalBaseFreq
        self._rope.wrappedValue = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            base: baseFreq
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        var mask = mask

        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Sliding window
        if isSliding && mask != nil {
            let keyLen = keys.dim(-2)
            if mask!.dim(-1) != keyLen {
                mask = mask![0..., 0..., 0..., 0 ..< keyLen]
            }
        }

        // Handle key-value head repetition
        if repeats > 1 {
            // Repeat keys and values to match the number of query heads
            keys = repeated(keys, count: repeats, axis: 1)
            values = repeated(values, count: repeats, axis: 1)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

// MARK: - MLP

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - TransformerBlock

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Attention
    @ModuleInfo var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: GemmaUtils.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: GemmaUtils.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: GemmaUtils.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: GemmaUtils.RMSNorm

    let numAttentionHeads: Int
    let hiddenSize: Int

    init(config: Gemma3TextConfiguration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize

        self._selfAttention.wrappedValue = Attention(config: config, layerIdx: layerIdx)
        self.mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = GemmaUtils.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = GemmaUtils.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = GemmaUtils.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = GemmaUtils.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let r = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        let r2 = mlp(preFeedforwardLayerNorm(h))
        let out = h + postFeedforwardLayerNorm(r2)
        return out
    }
}

// MARK: - GemmaModel

private class GemmaModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [TransformerBlock]
    @ModuleInfo var norm: GemmaUtils.RMSNorm

    let config: Gemma3TextConfiguration

    init(_ config: Gemma3TextConfiguration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            TransformerBlock(config: config, layerIdx: layerIdx)
        }

        self.norm = GemmaUtils.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputEmbedding: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding = inputEmbedding {
            h = inputEmbedding
        } else if let inputs = inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("Either inputs or inputEmbedding must be provided")
        }

        // Scale embeddings
        h = h * pow(Float(config.hiddenSize), 0.5)

        var mask = mask
        if mask == nil {
            // Create attention mask
            if let cache = cache, cache.count >= config.slidingWindowPattern {
                let j = config.slidingWindowPattern
                mask = createAttentionMask(h: h, cache: Array(cache[(j - 1) ..< j]))
            } else {
                mask = createAttentionMask(h: h, cache: nil)
            }
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - LanguageModel

private class LanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo var model: GemmaModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: Gemma3TextConfiguration
    var kvHeads: [Int]

    init(_ config: Gemma3TextConfiguration) {
        self.config = config
        self.model = GemmaModel(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)

        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> LMOutput {
        let out = model(inputs, inputEmbedding: inputEmbedding, mask: mask, cache: cache)
        let logits = lmHead(out)
        return LMOutput(logits: logits)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        if sanitizedWeights["lm_head.weight"] == nil {
            sanitizedWeights["language_model.lm_head.weight"] =
                sanitizedWeights["language_model.model.embed_tokens.weight"]
        }

        return sanitizedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

// MARK: - Vision Model Components

private class VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outputProj: Linear

    let numHeads: Int
    let scale: Float

    init(
        dimensions: Int,
        numHeads: Int,
        queryInputDimensions: Int? = nil,
        keyInputDimensions: Int? = nil,
        valueInputDimensions: Int? = nil,
        valueDimensions: Int? = nil,
        valueOutputDimensions: Int? = nil,
        bias: Bool = true
    ) {
        if dimensions % numHeads != 0 {
            fatalError("The input feature dimensions should be divisible by the number of heads")
        }

        self.numHeads = numHeads
        let headDim = dimensions / numHeads
        self.scale = pow(Float(headDim), -0.5)

        let queryInputDims = queryInputDimensions ?? dimensions
        let keyInputDims = keyInputDimensions ?? dimensions
        let valueInputDims = valueInputDimensions ?? keyInputDims
        let valueDims = valueDimensions ?? dimensions
        let valueOutputDims = valueOutputDimensions ?? dimensions

        self._queryProj.wrappedValue = Linear(queryInputDims, dimensions, bias: bias)
        self._keyProj.wrappedValue = Linear(keyInputDims, dimensions, bias: bias)
        self._valueProj.wrappedValue = Linear(valueInputDims, valueDims, bias: bias)
        self._outputProj.wrappedValue = Linear(valueDims, valueOutputDims, bias: bias)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        let (B, L, _) = (queries.dim(0), queries.dim(1), queries.dim(2))
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

        return outputProj(output)
    }
}

private class VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo var activationFn: GELU

    init(config: Gemma3VisionConfiguration) {
        self.activationFn = GELU(approximation: .precise)
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = fc1(x)
        x = activationFn(x)
        return fc2(x)
    }
}

private class EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: VisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: VisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    let embedDim: Int

    init(config: Gemma3VisionConfiguration) {
        self.embedDim = config.hiddenSize

        self._selfAttention.wrappedValue = VisionAttention(
            dimensions: config.hiddenSize,
            numHeads: config.attentionHeads,
            bias: true
        )

        self._layerNorm1.wrappedValue = LayerNorm(dimensions: embedDim, eps: config.layerNormEps)
        self.mlp = VisionMLP(config: config)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: embedDim, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let r = selfAttention(layerNorm1(x), mask: mask)
        let h = x + r
        let r2 = mlp(layerNorm2(h))
        return h + r2
    }
}

private class Encoder: Module {
    @ModuleInfo var layers: [EncoderLayer]

    init(config: Gemma3VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            EncoderLayer(config: config)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false,
        mask: MLXArray? = nil
    ) -> (MLXArray, [MLXArray]?) {
        var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
        var h = x

        for layer in layers {
            h = layer(h)
            if outputHiddenStates {
                encoderStates?.append(h)
            }
        }

        return (h[0], encoderStates)
    }
}

private class VisionEmbeddings: Module, UnaryLayer {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let config: Gemma3VisionConfiguration
    let embedDim: Int
    let imageSize: Int
    let patchSize: Int
    let numPatches: Int
    let numPositions: Int

    init(config: Gemma3VisionConfiguration) {
        self.config = config
        self.embedDim = config.hiddenSize
        self.imageSize = config.imageSize
        self.patchSize = config.patchSize

        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: embedDim,
            kernelSize: IntOrPair(patchSize),
            stride: IntOrPair(patchSize)
        )

        self.numPatches = (imageSize / patchSize) * (imageSize / patchSize)
        self.numPositions = numPatches

        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: numPositions,
            dimensions: embedDim
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var patchEmbeddings = patchEmbedding(x)
        patchEmbeddings = flattened(patchEmbeddings, start: 1, end: 2)

        let positionIds = MLXArray(Array(0 ..< numPositions))[.newAxis, 0...]
        var embeddings = patchEmbeddings
        embeddings = embeddings + positionEmbedding(positionIds)

        return embeddings
    }
}

private class SigLipVisionModel: Module {
    @ModuleInfo var embeddings: VisionEmbeddings
    @ModuleInfo var encoder: Encoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    init(config: Gemma3VisionConfiguration) {
        self.embeddings = VisionEmbeddings(config: config)
        self.encoder = Encoder(config: config)
        self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> (MLXArray, MLXArray, [MLXArray]?) {
        let x = embeddings(x)

        let (encoderOutput, encoderStates) = encoder(
            x,
            outputHiddenStates: outputHiddenStates,
            mask: nil
        )

        let poolerOutput = postLayerNorm(encoderOutput)

        return (poolerOutput, x, encoderStates)
    }
}

private class VisionModel: Module {
    @ModuleInfo(key: "vision_model") var visionModel: SigLipVisionModel

    let modelType: String

    init(config: Gemma3VisionConfiguration) {
        self.modelType = config.modelType
        self._visionModel.wrappedValue = SigLipVisionModel(config: config)
    }

    func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> (MLXArray, MLXArray, [MLXArray]?) {
        visionModel(x, outputHiddenStates: outputHiddenStates)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]

        for (k, v) in weights {
            if k.contains("patch_embedding.weight") {
                // Check if already in MLX format
                if isMLXWeight(v) {
                    sanitizedWeights[k] = v
                } else {
                    // Convert from PyTorch format to MLX format
                    sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                }
            } else {
                sanitizedWeights[k] = v
            }
        }

        return sanitizedWeights
    }

    private func isMLXWeight(_ array: MLXArray) -> Bool {
        if array.ndim != 4 {
            return false
        }

        let (outChannels, kH, kW, _) = (array.dim(0), array.dim(1), array.dim(2), array.dim(3))

        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }
}

// MARK: - Multimodal Projector

class Gemma3MultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "mm_input_projection_weight") var mmInputProjectionWeight: MLXArray
    @ModuleInfo(key: "mm_soft_emb_norm") var mmSoftEmbNorm: GemmaUtils.RMSNorm
    @ModuleInfo var avgPool: AvgPool2d

    let patchesPerImage: Int
    let tokensPerSide: Int
    let kernelSize: Int

    init(config: Gemma3Configuration) {
        self._mmInputProjectionWeight.wrappedValue = ones([
            config.visionConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
        ])

        self._mmSoftEmbNorm.wrappedValue = GemmaUtils.RMSNorm(
            dimensions: config.visionConfiguration.hiddenSize,
            eps: config.visionConfiguration.layerNormEps
        )

        self.patchesPerImage =
            config.visionConfiguration.imageSize / config.visionConfiguration.patchSize

        self.tokensPerSide = Int(sqrt(Double(config.mmTokensPerImage)))
        self.kernelSize = patchesPerImage / tokensPerSide

        self.avgPool = AvgPool2d(
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(kernelSize)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, _, l) = (x.dim(0), x.dim(1), x.dim(2))

        var reshapedVisionOutputs = x.transposed(0, 2, 1)
        reshapedVisionOutputs = reshapedVisionOutputs.reshaped(
            b, l, patchesPerImage, patchesPerImage
        )

        // Transpose to place h, w in indices 1, 2
        reshapedVisionOutputs = reshapedVisionOutputs.transposed(0, 2, 3, 1)
        var pooledVisionOutputs = avgPool(reshapedVisionOutputs)
        pooledVisionOutputs = pooledVisionOutputs.transposed(0, 3, 1, 2).flattened(start: 2)
        pooledVisionOutputs = pooledVisionOutputs.transposed(0, 2, 1)

        let normedVisionOutputs = mmSoftEmbNorm(pooledVisionOutputs)

        let projectedVisionOutputs = einsum(
            "btm,md->btd",
            normedVisionOutputs,
            mmInputProjectionWeight
        )

        return projectedVisionOutputs.asType(x.dtype)
    }
}

// MARK: - Gemma 3 Model

public class Gemma3: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: LanguageModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: Gemma3MultiModalProjector

    public let config: Gemma3Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Gemma3Configuration) {
        self.config = config

        self._visionTower.wrappedValue = VisionModel(config: config.visionConfiguration)
        self._languageModel.wrappedValue = LanguageModel(config.textConfiguration)
        self._multiModalProjector.wrappedValue = Gemma3MultiModalProjector(config: config)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray? = nil,
        pixelValues: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray?) {
        guard let pixelValues else {
            return (languageModel.model.embedTokens(inputIds!), nil)
        }

        let inputsEmbeds = languageModel.model.embedTokens(inputIds!)

        let (hiddenState, _, _) = visionTower(
            pixelValues.transposed(0, 2, 3, 1).asType(inputsEmbeds.dtype),
            outputHiddenStates: true
        )

        var imageFeatures = hiddenState[.newAxis, 0..., 0...]
        imageFeatures = multiModalProjector(imageFeatures)

        return prepareInputsForMultimodal(
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds!,
            attentionMask: mask
        )
    }

    private func prepareInputsForMultimodal(
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let (_, _, embedDim) = (imageFeatures.dim(0), imageFeatures.dim(1), imageFeatures.dim(2))

        let batchSize = inputIds.dim(0)
        let sequenceLength = inputIds.dim(1)
        let scaledImageFeatures = imageFeatures / sqrt(Float(config.hiddenSize))
        var finalEmbedding = MLXArray.zeros([batchSize, sequenceLength, embedDim])

        let padTokenId = config.padTokenId

        // Create masks for text, image, and padding tokens
        let textMask = MLX.logicalAnd(
            MLX.notEqual(inputIds, MLXArray(config.imageTokenIndex)),
            MLX.notEqual(inputIds, MLXArray(padTokenId))
        )
        let imageMask = MLX.equal(inputIds, MLXArray(config.imageTokenIndex))
        let padMask = MLX.equal(inputIds, MLXArray(padTokenId))

        // Expand masks to match embedding dimension
        var textMaskExpanded = expandedDimensions(textMask, axis: -1)
        textMaskExpanded = repeated(textMaskExpanded, count: embedDim, axis: -1)
        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = repeated(imageMaskExpanded, count: embedDim, axis: -1)
        var padMaskExpanded = expandedDimensions(padMask, axis: -1)
        padMaskExpanded = repeated(padMaskExpanded, count: embedDim, axis: -1)

        // Insert text token embeddings
        finalEmbedding = MLX.where(textMaskExpanded, inputsEmbeds, finalEmbedding)

        // Pad scaled_image_features to match the sequence length
        let padSize = finalEmbedding.dim(1) - scaledImageFeatures.dim(1)
        let paddedImageFeatures = MLX.padded(
            scaledImageFeatures,
            widths: [IntOrPair((0, 0)), IntOrPair((0, padSize)), IntOrPair((0, 0))]
        )

        // Insert image embeddings
        finalEmbedding = MLX.where(imageMaskExpanded, paddedImageFeatures, finalEmbedding)

        // Apply padding mask to ensure zeros in pad positions
        finalEmbedding = MLX.where(
            padMaskExpanded, MLXArray.zeros(like: finalEmbedding), finalEmbedding)

        var finalAttentionMask4d: MLXArray? = nil
        if let attentionMask = attentionMask {
            let attentionMaskExpanded1 = expandedDimensions(attentionMask, axis: 1)
            let attentionMaskExpanded2 = expandedDimensions(attentionMask, axis: 2)
            finalAttentionMask4d = attentionMaskExpanded1 * attentionMaskExpanded2
            finalAttentionMask4d = expandedDimensions(finalAttentionMask4d!, axis: 1)
        }

        return (finalEmbedding, finalAttentionMask4d)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard let imagePixels = input.image?.pixels else {
            // Text-only input
            return .logits(languageModel(input.text.tokens, cache: cache))
        }

        let (inputEmbeddings, finalAttentionMask4d) = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: imagePixels,
            mask: input.text.mask
        )

        let result = languageModel(
            nil,
            cache: cache,
            inputEmbedding: inputEmbeddings,
            mask: finalAttentionMask4d
        )

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        visionTower.sanitize(weights: weights)
    }
}

public class Gemma3Processor: UserInputProcessor {
    private let config: Gemma3ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // Implementation unchanged
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        let processedImages =
            try images
            .map { MediaProcessing.inSRGBToneCurveSpace($0) }
            .map {
                try MediaProcessing.resampleBicubic(
                    $0, to: CGSize(width: config.imageSize, height: config.imageSize))
            }
            .map {
                MediaProcessing.normalize(
                    $0, mean: config.imageMeanTuple, std: config.imageStdTuple)
            }
            .map { MediaProcessing.asMLXArray($0) }

        let pixelValues = concatenated(processedImages)

        return (pixelValues, THW(images.count, config.imageSize, config.imageSize))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = input.prompt.asMessages()

        print("Messages before tokenization:", messages)

        // Tokenize the messages using the chat template
        let promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        print("Prompt token IDs:", promptTokens)
        let decoded = try tokenizer.decode(tokens: promptTokens)
        print("Decoded prompt tokens: \(decoded)")

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        var finalPromptTokens = promptTokens

        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated,
                frames: imagePixelsAndFrames.map { $0.1 }
            )

            // Find all occurrences of the beginning of image token
            let boiTokenId = 255999  // From config.json

            var boiTokenIndices = [Int]()

            for (i, token) in promptTokens.enumerated() {
                if token == boiTokenId {
                    boiTokenIndices.append(i)
                }
            }

            // Make sure we have the right number of image tokens
            guard boiTokenIndices.count == input.images.count else {
                throw VLMError.processing(
                    "Number of image tokens (\(boiTokenIndices.count)) does not match number of images (\(input.images.count))"
                )
            }

            // Replace each BOI token with the full sequence
            var result = [Int]()
            var currentIndex = 0

            for boiIndex in boiTokenIndices {
                // Add tokens before the BOI token
                result.append(contentsOf: promptTokens[currentIndex ..< boiIndex])

                // Add newlines before the BOI token
                result.append(108)  // Token for two newlines

                // Add the BOI token
                result.append(boiTokenId)

                // Add the image tokens
                let imageTokenId = 262144  // From config.json
                result.append(
                    contentsOf: Array(repeating: imageTokenId, count: config.imageSeqLength))

                // Add the EOI token
                let eoiTokenId = 256000  // From config.json
                result.append(eoiTokenId)

                // Add newlines after the EOI token
                result.append(108)  // Token for two newlines

                // Update current index to after the BOI token
                currentIndex = boiIndex + 1
            }

            // Add any remaining tokens
            if currentIndex < promptTokens.count {
                result.append(contentsOf: promptTokens[currentIndex...])
            }

            finalPromptTokens = result

            print("Final prompt token IDs:", finalPromptTokens)
            let decodedFinal = try tokenizer.decode(tokens: finalPromptTokens)
            print("Decoded final prompt tokens: \(decodedFinal)")
        }

        let promptArray = MLXArray(finalPromptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage
        )
    }
}

public struct Gemma3ProcessorConfiguration: Codable, Sendable {
    // Fields from the preprocessor_config.json
    public let processorClass: String
    public let imageProcessorType: String
    public let doNormalize: Bool
    public let doRescale: Bool
    public let doResize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let resample: Int
    public let rescaleFactor: Float
    public let size: ImageSize

    // Optional fields
    public let doConvertRgb: Bool?
    public let doPanAndScan: Bool?
    public let panAndScanMaxNumCrops: Int?
    public let panAndScanMinCropSize: Int?
    public let panAndScanMinRatioToActivate: Float?

    // Hard-coded value from Gemma3 config.json
    // TODO: Check if there's a better solution than hard-coding this
    public let imageTokenId: Int = 255999  // 262144

    public struct ImageSize: Codable, Sendable {
        public let height: Int
        public let width: Int
    }

    // Computed properties for convenience
    public var imageSize: Int { size.height }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessorType = "image_processor_type"
        case doNormalize = "do_normalize"
        case doRescale = "do_rescale"
        case doResize = "do_resize"
        case doConvertRgb = "do_convert_rgb"
        case doPanAndScan = "do_pan_and_scan"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case resample
        case rescaleFactor = "rescale_factor"
        case size
        case panAndScanMaxNumCrops = "pan_and_scan_max_num_crops"
        case panAndScanMinCropSize = "pan_and_scan_min_crop_size"
        case panAndScanMinRatioToActivate = "pan_and_scan_min_ratio_to_activate"
        // imageTokenId is not decoded from JSON
    }
}

extension Gemma3: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        return languageModel.model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
