import CoreImage
import MLX
import MLXFast
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
    public let finalLogitSoftcapping: Float?

    public let vocabularySize: Int = 262208
    public let rmsNormEps: Float = 1.0e-6

    // Decoded from JSON when present, with fallback if not

    private let _attentionHeads: Int?
    private let _kvHeads: Int?
    private let _headDim: Int?
    private let _queryPreAttnScalar: Float?

    // Not included in 4B model config.json, included for 12B and 27B models
    public var attentionHeads: Int {
        _attentionHeads ?? 8
    }

    // Not included in 4B model config.json, included for 12B and 27B models
    public var kvHeads: Int {
        _kvHeads ?? 4
    }

    // Not included in 4B and 12B model config.json, included for 27B model
    public var headDim: Int {
        _headDim ?? 256
    }

    // Not included in 4B and 12B model config.json, included for 27B model
    public var queryPreAttnScalar: Float {
        _queryPreAttnScalar ?? 256
    }

    public let ropeGlobalBaseFreq: Float = 1_000_000.0
    public let ropeLocalBaseFreq: Float = 10_000.0
    public let ropeTraditional: Bool = false
    public let mmTokensPerImage: Int = 256
    public let slidingWindowPattern: Int = 6
    public let maxPositionEmbeddings: Int = 4096

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case slidingWindow = "sliding_window"
        case ropeScaling = "rope_scaling"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case _attentionHeads = "num_attention_heads"
        case _kvHeads = "num_key_value_heads"
        case _headDim = "head_dim"
        case _queryPreAttnScalar = "query_pre_attn_scalar"
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

    public let numChannels: Int = 3
    public let layerNormEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case imageSize = "image_size"
    }
}

// MARK: - Model Configuration

public struct Gemma3Configuration: Codable, Sendable {
    public let textConfiguration: Gemma3TextConfiguration
    public let visionConfiguration: Gemma3VisionConfiguration
    public let modelType: String
    public let mmTokensPerImage: Int
    public let quantization: BaseConfiguration.Quantization?

    private let _vocabularySize: Int?
    private let _padTokenId: Int?

    // Computed properties that use the text configuration or provide defaults

    public var vocabularySize: Int {
        _vocabularySize ?? textConfiguration.vocabularySize
    }

    public var hiddenSize: Int {
        textConfiguration.hiddenSize
    }

    public var padTokenId: Int {
        _padTokenId ?? 0
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case mmTokensPerImage = "mm_tokens_per_image"
        case quantization

        case _vocabularySize = "vocab_size"
        case _padTokenId = "pad_token_id"
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

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

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

        self._queryNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // Gemma3 uses sliding window attention pattern
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
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        // Reshape for multi-head attention
        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        // Apply normalization
        queries = queryNorm(queries)
        keys = keyNorm(keys)

        // Apply rotary position embedding
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Handle sliding window masking
        var finalMask = mask
        if case .array(let maskArray) = mask, maskArray.shape.last! != keys.shape[2] {
            let keyLen = keys.shape[2]
            let slicedMask = maskArray[.ellipsis, (-keyLen)...]
            finalMask = .array(slicedMask)
        }

        // Scaled dot-product attention with native GQA support
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: finalMask
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
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    let numAttentionHeads: Int
    let hiddenSize: Int

    init(config: Gemma3TextConfiguration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize

        self._selfAttention.wrappedValue = Attention(config: config, layerIdx: layerIdx)
        self.mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let r = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = Gemma.clipResidual(x, postAttentionLayerNorm(r))
        let r2 = mlp(preFeedforwardLayerNorm(h))
        let out = Gemma.clipResidual(h, postFeedforwardLayerNorm(r2))
        return out
    }
}

// MARK: - GemmaModel

private class GemmaModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [TransformerBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

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

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputEmbedding: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding = inputEmbedding {
            h = inputEmbedding
        } else if let inputs = inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("Either inputs or inputEmbedding must be provided")
        }

        // Apply embedding scaling
        let scale = MLXArray(sqrtf(Float(config.hiddenSize)), dtype: .bfloat16).asType(
            inputs?.dtype ?? h.dtype)
        h = h * scale

        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        // Create attention masks for global and sliding window layers
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = .none

        if mask == nil {
            let j = config.slidingWindowPattern
            if j > 0 && j <= layerCache!.count {
                let globalCacheSlice = layerCache![(j - 1) ..< j].compactMap { $0 }
                fullMask = createAttentionMask(h: h, cache: globalCacheSlice)
            }
            slidingWindowMask = createAttentionMask(h: h, cache: layerCache?.compactMap { $0 })
        }

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)

            let localMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                localMask = mask
            } else if isGlobal {
                localMask = fullMask
            } else {
                localMask = slidingWindowMask
            }

            h = layer(h, mask: localMask, cache: layerCache?[i])
        }

        return norm(h)
    }
}

// MARK: - LanguageModel

private class LanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo var model: GemmaModel
    @ModuleInfo(key: "lm_head") var lmHead: Module  // Can be Linear or QuantizedLinear

    let config: Gemma3TextConfiguration
    var kvHeads: [Int]

    init(_ config: Gemma3TextConfiguration) {
        self.config = config
        self.model = GemmaModel(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
    }

    /// Creates appropriate cache types for each layer
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        var caches: [any KVCache] = []
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        let slidingWindowPattern = config.slidingWindowPattern
        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)
            if isGlobalLayer {
                caches.append(StandardKVCache())
            } else {
                caches.append(RotatingKVCache(maxSize: slidingWindow, keep: 0))
            }
        }
        return caches
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> LMOutput {
        let optionalCache = cache?.map { $0 as KVCache? }
        let out = model(inputs, inputEmbedding: inputEmbedding, mask: mask, cache: optionalCache)

        // Call the lmHead (works whether it's Linear or QuantizedLinear)
        var finalLogits: MLXArray
        if let linear = lmHead as? Linear {
            finalLogits = linear(out)
        } else if let quantized = lmHead as? QuantizedLinear {
            finalLogits = quantized(out)
        } else {
            fatalError("lmHead must be Linear or QuantizedLinear")
        }

        // Apply final logit softcapping if configured
        if let softcap = config.finalLogitSoftcapping, softcap > 0 {
            let scale = MLXArray(softcap)
            finalLogits = tanh(finalLogits / scale) * scale
        }

        return LMOutput(logits: finalLogits)
    }

    func sanitize(
        weights: [String: MLXArray], quantizationConfig: BaseConfiguration.Quantization? = nil
    )
        -> [String: MLXArray]
    {
        var processedWeights = weights

        // Check if we have quantized weights
        let hasQuantizedLmHead = hasQuantizedWeights(
            layerPath: "language_model.lm_head", in: weights)

        if hasQuantizedLmHead {
            // Use quantization config from model configuration if available
            let q = quantizationConfig?.asTuple ?? (64, 4, .affine)

            // Only quantize layers that actually have quantized weights
            quantize(model: self) { path, module in
                // Check each specific layer path for quantized weights
                let fullPath = "language_model.\(path)"
                if weights["\(fullPath).scales"] != nil
                    && weights["\(fullPath).weight"]?.dtype == .uint32
                {
                    return q
                }
                return nil
            }
        } else {
            // Handle weight tying for regular (non-quantized) lm_head
            if processedWeights["language_model.lm_head.weight"] == nil {
                if let embedWeight = processedWeights["language_model.model.embed_tokens.weight"] {
                    processedWeights["language_model.lm_head.weight"] = embedWeight
                }
            }
        }

        // Remove unused precomputed rotary freqs
        return processedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    /// Check if a layer has quantized weights
    private func hasQuantizedWeights(layerPath: String, in weights: [String: MLXArray]) -> Bool {
        let scalesKey = "\(layerPath).scales"
        let biasesKey = "\(layerPath).biases"
        let weightKey = "\(layerPath).weight"

        let hasScales = weights[scalesKey] != nil
        let hasBiases = weights[biasesKey] != nil
        let hasWeight = weights[weightKey]?.dtype == .uint32

        return hasScales && hasBiases && hasWeight
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

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
        -> MLXArray
    {
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

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
        -> MLXArray
    {
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
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> (MLXArray, [MLXArray]?) {
        var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
        var h = x

        for layer in layers {
            h = layer(h, mask: mask)
            if outputHiddenStates {
                encoderStates?.append(h)
            }
        }

        return (h, encoderStates)
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

        // Check if we have the expected number of patches (safety net for config mismatches)
        let actualNumPatches = patchEmbeddings.dim(1)
        let useNumPositions = min(actualNumPatches, numPositions)

        // Use position IDs from 0 to numPositions
        let positionIds = MLXArray(Array(0 ..< useNumPositions))[.newAxis, 0...]
        var embeddings = patchEmbeddings

        // Add position embeddings only to the patches we have positions for
        if useNumPositions == actualNumPatches {
            // Normal case: add position embeddings to all patches
            embeddings = embeddings + positionEmbedding(positionIds)
        } else {
            // Safety case: only add to first N patches to avoid broadcast error
            let positionedPatches =
                embeddings[0..., ..<useNumPositions, 0...] + positionEmbedding(positionIds)
            let remainingPatches = embeddings[0..., useNumPositions..., 0...]
            embeddings = concatenated([positionedPatches, remainingPatches], axis: 1)
        }

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
            mask: .none
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

    /// Check if array is already in MLX format for conv2d weights
    private func checkArrayShape(_ arr: MLXArray) -> Bool {
        let shape = arr.shape

        // Check if the shape has 4 dimensions
        guard shape.count == 4 else { return false }

        let (outChannels, kH, kW, _) = (shape[0], shape[1], shape[2], shape[3])

        // Check if out_channels is the largest, and kH and kW are the same
        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = [String: MLXArray]()

        for (k, v) in weights {
            // Handle vision model quantized weights if they exist
            if k.contains("vision_tower") && hasQuantizedWeights(layerPath: k, in: weights) {
                // Keep quantized weights as-is - they will be handled by QuantizedLinear at runtime
                sanitizedWeights[k] = v
            } else if k.contains("patch_embedding.weight") {
                // PyTorch conv2d weight tensors have shape:
                //   [out_channels, in_channels, kH, KW]
                // MLX conv2d expects the weight be of shape:
                //   [out_channels, kH, KW, in_channels]
                if checkArrayShape(v) {
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

    /// Check if a layer has quantized weights (copied from LanguageModel)
    private func hasQuantizedWeights(layerPath: String, in weights: [String: MLXArray]) -> Bool {
        let scalesKey = "\(layerPath).scales"
        let biasesKey = "\(layerPath).biases"
        let weightKey = "\(layerPath).weight"

        return weights[scalesKey] != nil && weights[biasesKey] != nil
            && weights[weightKey]?.dtype == .uint32
    }
}

// MARK: - Multimodal Projector

class Gemma3MultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "mm_input_projection_weight") var mmInputProjectionWeight: MLXArray
    @ModuleInfo(key: "mm_soft_emb_norm") var mmSoftEmbNorm: Gemma.RMSNorm
    @ModuleInfo var avgPool: AvgPool2d

    let config: Gemma3Configuration
    let patchesPerImage: Int
    let tokensPerSide: Int
    let kernelSize: Int

    init(config: Gemma3Configuration) {
        self.config = config

        self._mmInputProjectionWeight.wrappedValue = ones([
            config.visionConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
        ])

        self._mmSoftEmbNorm.wrappedValue = Gemma.RMSNorm(
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

        // Use fixed config values
        var reshapedVisionOutputs = x.transposed(0, 2, 1)
        reshapedVisionOutputs = reshapedVisionOutputs.reshaped(
            b, l, patchesPerImage, patchesPerImage
        )

        // Transpose to place spatial dimensions in indices 1, 2
        reshapedVisionOutputs = reshapedVisionOutputs.transposed(0, 2, 3, 1)
        // Use fixed average pooling
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

/// Inserts image features into text embeddings at specified token positions
/// Implements the multimodal fusion approach used in Gemma3 VLM
private func maskedScatter(
    finalEmbedding: MLXArray,
    imageMaskExpanded: MLXArray,
    scaledImageFeatures: MLXArray
) -> MLXArray {
    // Reshape the tensors to 1D
    let finalEmbeddingShape = finalEmbedding.shape
    let scaledImageFeaturesFlattened = scaledImageFeatures.flattened()
    let finalEmbeddingFlattened = finalEmbedding.flattened()
    let imageMaskExpandedFlattened = imageMaskExpanded.flattened()

    let maskValues = imageMaskExpandedFlattened.asArray(Bool.self)
    let imagePositionIndices = maskValues.enumerated().compactMap { index, value in
        value ? UInt32(index) : nil
    }

    guard !imagePositionIndices.isEmpty else {
        return finalEmbedding
    }

    // Scatter the scaled image features into the special image token positions
    let imagePositions = MLXArray(imagePositionIndices)
    guard scaledImageFeaturesFlattened.shape[0] == imagePositions.shape[0] else {
        fatalError(
            """
            Critical error in maskedScatter: Size mismatch between image features and positions.
            Image features: \(scaledImageFeaturesFlattened.shape[0])
            Image positions: \(imagePositions.shape[0])
            """)
    }
    finalEmbeddingFlattened[imagePositions] = scaledImageFeaturesFlattened
    return finalEmbeddingFlattened.reshaped(finalEmbeddingShape)
}

// MARK: - Gemma 3 Model

public class Gemma3: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: LanguageModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: Gemma3MultiModalProjector

    public let config: Gemma3Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    /// Create cache with proper types for each layer
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        return languageModel.newCache(parameters: parameters)
    }

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

        // Process image through vision tower
        let processedPixels = pixelValues.transposed(0, 2, 3, 1).asType(inputsEmbeds.dtype)

        let (hiddenState, _, _) = visionTower(
            processedPixels,
            outputHiddenStates: true
        )

        let imageFeatures = multiModalProjector(hiddenState)

        let (finalEmbedding, finalAttentionMask4d) = prepareInputsForMultimodal(
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds!,
            attentionMask: mask
        )

        return (finalEmbedding, finalAttentionMask4d)
    }

    private func prepareInputsForMultimodal(
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let embedDim = inputsEmbeds.dim(2)
        let batchSize = inputIds.dim(0)
        let sequenceLength = inputIds.dim(1)

        // Scale image features to match text embedding magnitude
        let scaledImageFeatures = imageFeatures / sqrt(Float(config.textConfiguration.hiddenSize))

        // Use input embeddings as starting point
        var finalEmbedding = inputsEmbeds

        let padTokenId = config.padTokenId
        let imageTokenId = 262144  // Image token used after expansion

        // Create masks for different token types
        let textMask = MLX.logicalAnd(
            MLX.notEqual(inputIds, MLXArray(imageTokenId)),
            MLX.notEqual(inputIds, MLXArray(padTokenId))
        )
        let imageMask = MLX.equal(inputIds, MLXArray(imageTokenId))
        let padMask = MLX.equal(inputIds, MLXArray(padTokenId))

        // Expand masks to match embedding dimension
        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = repeated(imageMaskExpanded, count: embedDim, axis: -1)

        // Apply pad mask to final embedding
        var padMaskExpanded = expandedDimensions(padMask, axis: -1)
        padMaskExpanded = repeated(padMaskExpanded, count: embedDim, axis: -1)
        finalEmbedding = MLX.where(
            padMaskExpanded, MLXArray.zeros(like: finalEmbedding), finalEmbedding)

        // Insert image token embeddings using masked_scatter
        finalEmbedding = maskedScatter(
            finalEmbedding: finalEmbedding,
            imageMaskExpanded: imageMaskExpanded,
            scaledImageFeatures: scaledImageFeatures
        )

        var finalAttentionMask4d: MLXArray? = nil
        if let attentionMask = attentionMask {
            let attentionMaskExpanded1 = expandedDimensions(attentionMask, axis: 1)
            let attentionMaskExpanded2 = expandedDimensions(attentionMask, axis: 2)
            finalAttentionMask4d = attentionMaskExpanded1 * attentionMaskExpanded2
            finalAttentionMask4d = expandedDimensions(finalAttentionMask4d!, axis: 1)
        }

        return (finalEmbedding.asType(inputsEmbeds.dtype), finalAttentionMask4d)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard let imagePixels = input.image?.pixels else {
            // Text-only input
            let convertedCache = cache.compactMap { $0 as? KVCache }
            let result = languageModel(
                input.text.tokens, cache: convertedCache, inputEmbedding: nil, mask: nil)
            return .logits(result)
        }

        let (inputEmbeddings, _) = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: imagePixels,
            mask: input.text.mask
        )

        let convertedCache = cache.compactMap { $0 as? KVCache }
        // Use causal masking for text generation
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = .causal

        let result = languageModel(
            nil,  // Pass nil for tokens when using embeddings
            cache: convertedCache,
            inputEmbedding: inputEmbeddings,
            mask: maskMode
        )

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        return languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let lmHeadKeys = weights.keys.filter { $0.contains("lm_head") }

        // Also check attention layer structures
        let attnKeys = weights.keys.filter {
            $0.contains("self_attn")
                && ($0.contains("q_proj") || $0.contains("k_proj") || $0.contains("v_proj")
                    || $0.contains("o_proj"))
        }

        // Handle language model sanitization first (quantization, weight tying, etc.)
        var processedWeights = languageModel.sanitize(
            weights: weights, quantizationConfig: config.quantization)

        // Handle vision model sanitization (conv2d weight reshaping, etc.)
        processedWeights = visionTower.sanitize(weights: processedWeights)

        return processedWeights
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
        var userProcessing = processing ?? UserInput.Processing()
        // Always use the vision configuration's imageSize. Ignore UserInput resize setting.
        let targetSize = CGSize(width: config.imageSize, height: config.imageSize)

        // Force the correct size for vision model alignment
        userProcessing.resize = targetSize

        let processedImages = try images.map { image in
            let processedImage = MediaProcessing.apply(image, processing: userProcessing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage = try MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
            let normalizedImage = MediaProcessing.normalize(
                resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
            return MediaProcessing.asMLXArray(normalizedImage)
        }

        let pixelValues = concatenated(processedImages)

        return (pixelValues, THW(images.count, config.imageSize, config.imageSize))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Use structured content message generator for Gemma3's chat template
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        // Process images if any
        var processedImage: LMInput.ProcessedImage?

        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated,
                frames: imagePixelsAndFrames.map { $0.1 }
            )

            // Expand single <start_of_image> token to multiple image tokens
            let startOfImageTokenId = 255999
            let imageTokenId = 262144
            let numImageTokens = config.imageSeqLength  // 256

            var expandedTokens: [Int] = []

            for token in promptTokens {
                if token == startOfImageTokenId {
                    // Replace with 256 image tokens
                    expandedTokens.append(
                        contentsOf: Array(repeating: imageTokenId, count: numImageTokens))
                } else {
                    expandedTokens.append(token)
                }
            }

            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
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

    // Image token identifier from model configuration
    public let imageTokenId: Int = 262144

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
    }
}

extension Gemma3: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
