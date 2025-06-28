//
//  Gemma3n.swift
//  mlx-swift-examples
//
//  Created by Anthony DePasquale on 27.06.2025.
//

import CoreImage
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configuration Classes

// Protocol for multimodal configs that can be used with Gemma3nMultimodalEmbedder
public protocol MultimodalConfig {
    var hiddenSize: Int { get }
    var rmsNormEps: Float { get }
    var vocabOffset: Int { get }
    var vocabSize: Int { get }
}

public struct AudioConfig: Codable, Sendable, MultimodalConfig {
    // Constants with default values (always present)
    public let inputFeatSize: Int = 80
    public let hiddenSize: Int = 1536
    public let confAttentionChunkSize: Int = 12
    public let confAttentionContextLeft: Int = 13
    public let confAttentionContextRight: Int = 0
    public let confAttentionInvalidLogitsValue: Float = -1e9
    public let confAttentionLogitCap: Float = 50.0
    public let confNumAttentionHeads: Int = 8
    public let confNumHiddenLayers: Int = 12
    public let confConvKernelSize: Int = 5
    public let confPositionalBiasSize: Int = 256
    public let confReductionFactor: Int = 4
    public let confResidualWeight: Float = 0.5
    public let sscpConvChannelSize: [Int] = [128, 32]
    public let sscpConvGroupNormEps: Float = 1e-3
    public let sscpConvKernelSize: [[Int]] = [[3, 3], [3, 3]]
    public let sscpConvStrideSize: [[Int]] = [[2, 2], [2, 2]]
    public let vocabSize: Int = 128
    public let sscpConvEps: Float = 1e-3
    public let rmsNormEps: Float = 1e-6
    public let gradientClipping: Float = 10000000000.0
    public let vocabOffset: Int = 262272  // 262_144 + 128 (text vocab size + vision vocab size)

    enum CodingKeys: String, CodingKey {
        case inputFeatSize = "input_feat_size"
        case hiddenSize = "hidden_size"
        case confAttentionChunkSize = "conf_attention_chunk_size"
        case confAttentionContextLeft = "conf_attention_context_left"
        case confAttentionContextRight = "conf_attention_context_right"
        case confAttentionInvalidLogitsValue = "conf_attention_invalid_logits_value"
        case confAttentionLogitCap = "conf_attention_logit_cap"
        case confNumAttentionHeads = "conf_num_attention_heads"
        case confNumHiddenLayers = "conf_num_hidden_layers"
        case confConvKernelSize = "conf_conv_kernel_size"
        case confPositionalBiasSize = "conf_positional_bias_size"
        case confReductionFactor = "conf_reduction_factor"
        case confResidualWeight = "conf_residual_weight"
        case sscpConvChannelSize = "sscp_conv_channel_size"
        case sscpConvGroupNormEps = "sscp_conv_group_norm_eps"
        case sscpConvKernelSize = "sscp_conv_kernel_size"
        case sscpConvStrideSize = "sscp_conv_stride_size"
        case vocabSize = "vocab_size"
        case sscpConvEps = "sscp_conv_eps"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case vocabOffset = "vocab_offset"
    }
}

public struct VisionConfig: Codable, Sendable, MultimodalConfig {
    // Constants with default values (always present)
    public let modelType: String = "gemma3n_vision"
    public let numHiddenLayers: Int = 12
    public let hiddenSize: Int = 2048
    public let intermediateSize: Int = 8192
    public let numAttentionHeads: Int = 16
    public let patchSize: Int = 16
    public let imageSize: Int = 224
    public let numChannels: Int = 3
    public let rmsNormEps: Float = 1e-6
    public let vocabSize: Int = 128
    public let vocabOffset: Int = 262144

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numHiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case imageSize = "image_size"
        case numChannels = "num_channels"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabOffset = "vocab_offset"
    }
}

public struct TextConfig: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: [Int]
    private let _numAttentionHeads: Int?
    private let _headDim: Int?
    private let _rmsNormEps: Float?
    private let _vocabSize: Int?
    private let _vocabSizePerLayerInput: Int?
    private let _numKeyValueHeads: Int?
    private let _laurelRank: Int?
    private let _fracSharedLayers: Float?
    private let _altupActiveIdx: Int?
    private let _padTokenId: Int?
    private let _altupNumInputs: Int?
    public let altupCoefClip: Float?
    private let _altupCorrectScale: Bool?
    private let _hiddenSizePerLayerInput: Int?
    private let _ropeLocalBaseFreq: Float?
    private let _ropeTraditional: Bool?
    private let _ropeTheta: Float?
    private let _queryPreAttnScalar: Float?
    private let _slidingWindow: Int?
    public let ropeScaling: [String: StringOrNumber]?
    public let activationSparsityPattern: [Float]?
    public let layerTypes: [String]?
    private let _mmTokensPerImage: Int?
    private let _slidingWindowPattern: Int?
    private let _finalLogitSoftcapping: Float?
    private let _queryRescaleScalar: Float?
    private let _numKvSharedLayers: Int?
    private let _maxPositionEmbeddings: Int?
    private let _attnLogitSoftcapping: Float?

    // Computed properties with defaults
    public var numAttentionHeads: Int {
        _numAttentionHeads ?? 2
    }

    public var headDim: Int {
        _headDim ?? 256
    }

    public var rmsNormEps: Float {
        _rmsNormEps ?? 1.0e-6
    }

    public var vocabSize: Int {
        _vocabSize ?? 262400
    }

    public var vocabSizePerLayerInput: Int {
        _vocabSizePerLayerInput ?? 262144
    }

    public var numKeyValueHeads: Int {
        _numKeyValueHeads ?? 4
    }

    public var laurelRank: Int {
        _laurelRank ?? 64
    }

    public var fracSharedLayers: Float {
        _fracSharedLayers ?? 0.5
    }

    public var altupActiveIdx: Int {
        _altupActiveIdx ?? 0
    }

    public var padTokenId: Int {
        _padTokenId ?? 0
    }

    public var altupNumInputs: Int {
        _altupNumInputs ?? 4
    }

    public var altupCorrectScale: Bool {
        _altupCorrectScale ?? true
    }

    public var hiddenSizePerLayerInput: Int {
        _hiddenSizePerLayerInput ?? 1024
    }

    public var ropeLocalBaseFreq: Float {
        _ropeLocalBaseFreq ?? 10000.0
    }

    public var ropeTraditional: Bool {
        _ropeTraditional ?? false
    }

    public var ropeTheta: Float {
        _ropeTheta ?? 1000000.0
    }

    public var queryPreAttnScalar: Float {
        _queryPreAttnScalar ?? 0.0625
    }

    public var slidingWindow: Int {
        _slidingWindow ?? 1024
    }

    public var mmTokensPerImage: Int {
        _mmTokensPerImage ?? 256
    }

    public var slidingWindowPattern: Int {
        _slidingWindowPattern ?? 5
    }

    public var finalLogitSoftcapping: Float {
        _finalLogitSoftcapping ?? 30.0
    }

    public var queryRescaleScalar: Float {
        _queryRescaleScalar ?? 1.0
    }

    public var numKvSharedLayers: Int {
        _numKvSharedLayers ?? 0
    }

    public var maxPositionEmbeddings: Int {
        _maxPositionEmbeddings ?? 32768
    }

    public var attnLogitSoftcapping: Float {
        _attnLogitSoftcapping ?? 0.0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case _numAttentionHeads = "num_attention_heads"
        case _headDim = "head_dim"
        case _rmsNormEps = "rms_norm_eps"
        case _vocabSize = "vocab_size"
        case _vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case _numKeyValueHeads = "num_key_value_heads"
        case _laurelRank = "laurel_rank"
        case _fracSharedLayers = "frac_shared_layers"
        case _altupActiveIdx = "altup_active_idx"
        case _padTokenId = "pad_token_id"
        case _altupNumInputs = "altup_num_inputs"
        case altupCoefClip = "altup_coef_clip"
        case _altupCorrectScale = "altup_correct_scale"
        case _hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case _ropeLocalBaseFreq = "rope_local_base_freq"
        case _ropeTraditional = "rope_traditional"
        case _ropeTheta = "rope_theta"
        case _queryPreAttnScalar = "query_pre_attn_scalar"
        case _slidingWindow = "sliding_window"
        case ropeScaling = "rope_scaling"
        case _mmTokensPerImage = "mm_tokens_per_image"
        case _slidingWindowPattern = "sliding_window_pattern"
        case activationSparsityPattern = "activation_sparsity_pattern"
        case _finalLogitSoftcapping = "final_logit_softcapping"
        case _queryRescaleScalar = "query_rescale_scalar"
        case _numKvSharedLayers = "num_kv_shared_layers"
        case _maxPositionEmbeddings = "max_position_embeddings"
        case _attnLogitSoftcapping = "attn_logit_softcapping"
        case layerTypes = "layer_types"
    }
}

public struct ModelConfig: Codable, Sendable {
    // Required configs (no defaults in Python)
    public let textConfig: TextConfig
    public let visionConfig: VisionConfig
    public let audioConfig: AudioConfig
    public let modelType: String

    // Fields with default values (can be overridden from JSON)
    private let _vocabSize: Int?
    private let _ignoreIndex: Int?
    private let _imageTokenIndex: Int?
    private let _audioTokenId: Int?
    private let _imageTokenId: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?
    private let _visionSoftTokensPerImage: Int?
    private let _audioSoftTokensPerImage: Int?

    // Optional field
    public let eosTokenId: [Int]?

    // Computed properties with defaults
    public var vocabSize: Int {
        _vocabSize ?? 257152
    }

    public var ignoreIndex: Int {
        _ignoreIndex ?? -100
    }

    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 262145
    }

    public var audioTokenId: Int {
        _audioTokenId ?? 262273
    }

    public var imageTokenId: Int {
        _imageTokenId ?? 262145
    }

    public var hiddenSize: Int {
        _hiddenSize ?? 2048
    }

    public var padTokenId: Int {
        _padTokenId ?? 0
    }

    public var visionSoftTokensPerImage: Int {
        _visionSoftTokensPerImage ?? 256
    }

    public var audioSoftTokensPerImage: Int {
        _audioSoftTokensPerImage ?? 188
    }

    // Custom initializer to allow manual construction
    public init(
        textConfig: TextConfig,
        visionConfig: VisionConfig,
        audioConfig: AudioConfig,
        modelType: String,
        vocabSize: Int? = nil,
        ignoreIndex: Int? = nil,
        imageTokenIndex: Int? = nil,
        audioTokenId: Int? = nil,
        imageTokenId: Int? = nil,
        hiddenSize: Int? = nil,
        padTokenId: Int? = nil,
        visionSoftTokensPerImage: Int? = nil,
        audioSoftTokensPerImage: Int? = nil,
        eosTokenId: [Int]? = nil
    ) {
        self.textConfig = textConfig
        self.visionConfig = visionConfig
        self.audioConfig = audioConfig
        self.modelType = modelType
        self._vocabSize = vocabSize
        self._ignoreIndex = ignoreIndex
        self._imageTokenIndex = imageTokenIndex
        self._audioTokenId = audioTokenId
        self._imageTokenId = imageTokenId
        self._hiddenSize = hiddenSize
        self._padTokenId = padTokenId
        self._visionSoftTokensPerImage = visionSoftTokensPerImage
        self._audioSoftTokensPerImage = audioSoftTokensPerImage
        self.eosTokenId = eosTokenId
    }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
        case modelType = "model_type"
        case _vocabSize = "vocab_size"
        case _ignoreIndex = "ignore_index"
        case _imageTokenIndex = "image_token_index"
        case _audioTokenId = "audio_token_id"
        case _imageTokenId = "image_token_id"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
        case _visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case _audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case eosTokenId = "eos_token_id"
    }
}

// MARK: - Language Model Components

private class Gemma3nRMSNorm: Module {
    let eps: Float
    let scaleShift: Float
    @ModuleInfo var weight: MLXArray?

    init(dim: Int, eps: Float = 1e-6, scaleShift: Float = 0, withScale: Bool = true) {
        self.eps = eps
        self.scaleShift = scaleShift

        if withScale {
            self.weight = MLXArray.ones([dim])
        } else {
            self.weight = nil
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = norm(x.asType(.float32))

        if let weight {
            return (output * (weight + scaleShift)).asType(x.dtype)
        } else {
            return output.asType(x.dtype)
        }
    }

    private func norm(_ x: MLXArray) -> MLXArray {
        return x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
    }
}

private class Gemma3nLaurelBlock: Module {
    @ModuleInfo(key: "linear_left") var linearLeft: Linear
    @ModuleInfo(key: "linear_right") var linearRight: Linear
    @ModuleInfo(key: "post_laurel_norm") var postLaurelNorm: Gemma3nRMSNorm

    init(config: TextConfig) {
        self._linearLeft.wrappedValue = Linear(config.hiddenSize, config.laurelRank, bias: false)
        self._linearRight.wrappedValue = Linear(config.laurelRank, config.hiddenSize, bias: false)
        self._postLaurelNorm.wrappedValue = Gemma3nRMSNorm(
            dim: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let laurelX = linearLeft(x)
        let laurelX2 = linearRight(laurelX)
        let normedLaurelX = postLaurelNorm(laurelX2)
        return x + normedLaurelX
    }
}

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape.last! / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

private func applyRotaryPosEmb(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    unsqueezeDim: Int = 1
) -> MLXArray {
    let cosExpanded = expandedDimensions(cos, axis: unsqueezeDim)
    let sinExpanded = expandedDimensions(sin, axis: unsqueezeDim)
    return (x * cosExpanded) + (rotateHalf(x) * sinExpanded)
}

private class Gemma3nRotaryEmbedding: Module {
    let ropeType: String
    let maxSeqLenCached: Int
    let originalMaxSeqLen: Int
    let config: TextConfig
    let attentionScaling: Float
    private let _invFreq: MLXArray
    private let _originalInvFreq: MLXArray

    init(config: TextConfig) {
        if let ropeScaling = config.ropeScaling {
            let ropeTypeValue = ropeScaling["rope_type"] ?? ropeScaling["type"]
            if case .string(let typeString) = ropeTypeValue {
                self.ropeType = typeString
            } else {
                self.ropeType = "default"
            }
        } else {
            self.ropeType = "default"
        }

        self.maxSeqLenCached = config.maxPositionEmbeddings
        self.originalMaxSeqLen = config.maxPositionEmbeddings
        self.config = config
        self.attentionScaling = 1.0

        let (invFreq, _) = Self.computeDefaultRopeParameters(config: config)
        self._invFreq = MLXArray(invFreq).asType(.float32)
        self._originalInvFreq = MLXArray(invFreq).asType(.float32)

        super.init()
    }

    static func computeDefaultRopeParameters(config: TextConfig) -> ([Float], Float) {
        let base = config.ropeTheta
        let partialRotaryFactor: Float = 1.0
        let headDim = config.headDim
        let dim = Int(Float(headDim) * partialRotaryFactor)

        let attentionFactor: Float = 1.0

        let invFreqArray: [Float] = stride(from: 0, to: dim, by: 2).map { i in
            1.0 / pow(base, Float(i) / Float(dim))
        }

        return (invFreqArray, attentionFactor)
    }

    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let invFreqExpanded = expandedDimensions(_invFreq, axes: [0, 2])
        let positionIdsExpanded = expandedDimensions(positionIds.asType(.float32), axes: [1])

        let freqs = matmul(
            invFreqExpanded.asType(.float32),
            positionIdsExpanded.asType(.float32)
        ).transposed(0, 2, 1)

        let emb = concatenated([freqs, freqs], axis: -1)
        let cosEmb = cos(emb) * attentionScaling
        let sinEmb = sin(emb) * attentionScaling

        return (cosEmb.asType(x.dtype), sinEmb.asType(x.dtype))
    }
}

private class Gemma3nAttention: Module {
    let isSliding: Bool
    let attnLogitSoftcapping: Float
    let numHeads: Int
    let numKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isKvSharedLayer: Bool
    let kvSharedLayerIndex: Int?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: Gemma3nRMSNorm

    init(config: TextConfig, layerIdx: Int) {
        self.isSliding =
            (config.layerTypes
            ?? Array(repeating: "global_attention", count: config.numHiddenLayers))[layerIdx]
            == "sliding_attention"
        self.attnLogitSoftcapping = config.attnLogitSoftcapping

        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.repeats = numHeads / numKVHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx
        self.scale = 1.0

        self._qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        self._qNorm.wrappedValue = Gemma3nRMSNorm(
            dim: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = Gemma3nRMSNorm(
            dim: config.headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = Gemma3nRMSNorm(
            dim: config.headDim,
            eps: config.rmsNormEps,
            withScale: false
        )

        let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        self.isKvSharedLayer = layerIdx >= firstKvSharedLayerIdx

        if !isKvSharedLayer {
            self.kvSharedLayerIndex = nil
        } else if isSliding {
            self.kvSharedLayerIndex = firstKvSharedLayerIdx - 2
        } else {
            self.kvSharedLayerIndex = firstKvSharedLayerIdx - 1
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        caches: [KVCache?]? = nil,
        positionEmbeddings: (MLXArray, MLXArray)? = nil,
        cachePosition: MLXArray? = nil
    ) -> MLXArray {
        let inputShape = Array(x.shape.dropLast())
        let hiddenShape = inputShape + [-1, headDim]

        guard let (cos, sin) = positionEmbeddings else {
            fatalError("Position embeddings are required")
        }

        var queries = qProj(x)
        queries = queries.reshaped(hiddenShape)
        queries = qNorm(queries)
        queries = applyRotaryPosEmb(queries, cos: cos, sin: sin, unsqueezeDim: 2)
        queries = queries.transposed(0, 2, 1, 3)

        var keys: MLXArray
        var values: MLXArray

        if isKvSharedLayer,
            let kvSharedLayerIndex = kvSharedLayerIndex,
            let cache = cache,
            let caches = caches,
            kvSharedLayerIndex < caches.count,
            let sharedCache = caches[kvSharedLayerIndex]
        {
            // Use shared KV from designated cache layer
            let sharedState = sharedCache.state
            if sharedState.count >= 2 {
                keys = sharedState[0]
                values = sharedState[1]
            } else {
                // Fallback: compute KV normally if shared cache is empty
                keys = kProj(x).reshaped(hiddenShape)
                keys = kNorm(keys)
                keys = applyRotaryPosEmb(keys, cos: cos, sin: sin, unsqueezeDim: 2)
                keys = keys.transposed(0, 2, 1, 3)

                values = vProj(x).reshaped(hiddenShape)
                values = vNorm(values)
                values = values.transposed(0, 2, 1, 3)
            }
        } else {
            keys = kProj(x).reshaped(hiddenShape)
            keys = kNorm(keys)
            keys = applyRotaryPosEmb(keys, cos: cos, sin: sin, unsqueezeDim: 2)
            keys = keys.transposed(0, 2, 1, 3)

            values = vProj(x).reshaped(hiddenShape)
            values = vNorm(values)
            values = values.transposed(0, 2, 1, 3)
        }

        // Repeat keys and values for multi-head attention
        keys = repeated(keys, count: repeats, axis: 1)
        values = repeated(values, count: repeats, axis: 1)

        // Use custom attention function that supports both quantized cache and logit softcapping
        let output = gemma3nAttentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            attnLogitSoftcapping: attnLogitSoftcapping,
            mask: mask ?? .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(inputShape + [-1])

        return oProj(output)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let config: TextConfig
    let activationSparsity: Float

    init(config: TextConfig, layerIdx: Int = 0) {
        self.config = config
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize[0]

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)

        if let activationSparsityPattern = config.activationSparsityPattern {
            self.activationSparsity = activationSparsityPattern[layerIdx]
        } else {
            self.activationSparsity = 0.0
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gateProj = self.gateProj(x)
        if activationSparsity > 0.0 {
            gateProj = gaussianTopK(gateProj)
        }
        let activations = geluApproximate(gateProj)
        let upProj = self.upProj(x)
        let downProj = self.downProj(activations * upProj)
        return downProj
    }

    private func gaussianTopK(_ inputs: MLXArray) -> MLXArray {
        let p = MLXArray(activationSparsity, dtype: .float32)
        let stdMultiplier = sqrt(2.0) * erfInverse(2 * p - 1)
        let stdMultiplierCasted = stdMultiplier.asType(inputs.dtype)
        let inputsMean = mean(inputs, axis: -1, keepDims: true)
        let inputsStd = std(inputs, axis: -1, keepDims: true)
        let cutoffX = inputsMean + inputsStd * stdMultiplierCasted
        return maximum(0, inputs - cutoffX)
    }
}

private class Gemma3nAltUp: Module {
    @ModuleInfo(key: "correct_output_scale") var correctOutputScale: MLXArray
    @ModuleInfo(key: "correction_coefs") var correctionCoefs: Linear
    @ModuleInfo(key: "prediction_coefs") var predictionCoefs: Linear
    @ModuleInfo(key: "modality_router") var modalityRouter: Linear
    @ModuleInfo(key: "router_norm") var routerNorm: Gemma3nRMSNorm
    private let _routerInputScale: MLXArray

    let config: TextConfig

    init(config: TextConfig) {
        self.config = config

        self._correctOutputScale.wrappedValue = MLXArray.zeros([config.hiddenSize])
        self._correctionCoefs.wrappedValue = Linear(
            config.altupNumInputs,
            config.altupNumInputs,
            bias: false
        )
        self._predictionCoefs.wrappedValue = Linear(
            config.altupNumInputs,
            config.altupNumInputs * config.altupNumInputs,
            bias: false
        )
        self._modalityRouter.wrappedValue = Linear(
            config.hiddenSize,
            config.altupNumInputs,
            bias: false
        )
        self._routerNorm.wrappedValue = Gemma3nRMSNorm(
            dim: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        self._routerInputScale = MLXArray(pow(Float(config.hiddenSize), -1.0))

        super.init()
    }

    func computeRouterModalities(_ x: MLXArray) -> MLXArray {
        guard let routerNormWeight = routerNorm.weight else {
            fatalError("routerNorm.weight is nil in Gemma3nAltUp")
        }
        let routerInputs = routerNorm(x) * _routerInputScale.asType(routerNormWeight.dtype)

        let routed = modalityRouter(routerInputs).asType(.float32)
        return tanh(routed)
    }

    func predict(_ x: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(x[config.altupActiveIdx])

        var predictionCoefsWeight = predictionCoefs.weight.asType(.float32)

        if let altupCoefClip = config.altupCoefClip {
            predictionCoefsWeight = clip(
                predictionCoefsWeight,
                min: MLXArray(-altupCoefClip),
                max: MLXArray(altupCoefClip)
            )
        }

        let allCoefs = predictionCoefs(modalities)
            .reshaped(
                Array(modalities.shape.dropLast()) + [config.altupNumInputs, config.altupNumInputs]
            )
            .transposed(0, 1, 3, 2)

        let xPermuted = x.asType(.float32).transposed(1, 2, 3, 0)
        let predictions = matmul(xPermuted, allCoefs)
        let predictionsPermuted = predictions.transposed(3, 0, 1, 2)
        let finalPredictions = predictionsPermuted + x
        return finalPredictions.asType(x.dtype)
    }

    func correct(predictions: MLXArray, activated: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(activated)

        var correctionCoefsWeight = correctionCoefs.weight.asType(.float32)

        if let altupCoefClip = config.altupCoefClip {
            correctionCoefsWeight = clip(
                correctionCoefsWeight,
                min: MLXArray(-altupCoefClip),
                max: MLXArray(altupCoefClip)
            )
        }

        let allCoefs = correctionCoefs(modalities) + 1.0

        let activeX = predictions[config.altupActiveIdx]
        let innovation = activated - activeX

        let innovationExpanded = expandedDimensions(innovation, axis: 0)
        let innovationBroadcast = broadcast(
            innovationExpanded,
            to: [config.altupNumInputs] + Array(innovation.shape)
        )

        let allCoefsReshaped = allCoefs.transposed(2, 1, 0)
        let allCoefsExpanded = expandedDimensions(allCoefsReshaped, axis: 1)

        let corrected = innovationBroadcast * allCoefsExpanded
        let finalCorrected = corrected + predictions

        return finalCorrected.asType(activated.dtype)
    }

    func scaleCorrectOutput(_ corrected: MLXArray) -> MLXArray {
        let scale = config.altupCorrectScale ? correctOutputScale : MLXArray(1.0)
        return corrected * scale
    }

    func callAsFunction(_ x: MLXArray, activated: MLXArray) -> (MLXArray, MLXArray) {
        let predictions = predict(x)
        let corrected = correct(predictions: predictions, activated: activated)
        var output = corrected[config.altupActiveIdx]
        if config.altupCorrectScale {
            output = scaleCorrectOutput(output)
        }
        return (corrected, output)
    }
}

private class Gemma3nDecoderLayer: Module {
    let config: TextConfig
    let hiddenSize: Int
    let layerIdx: Int
    let isSliding: Bool
    let slidingWindow: Int
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma3nAttention
    @ModuleInfo var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Gemma3nRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Gemma3nRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: Gemma3nRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: Gemma3nRMSNorm
    @ModuleInfo var altup: Gemma3nAltUp
    @ModuleInfo var laurel: Gemma3nLaurelBlock
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma3nRMSNorm

    init(config: TextConfig, layerIdx: Int) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._selfAttn.wrappedValue = Gemma3nAttention(config: config, layerIdx: layerIdx)
        self.isSliding =
            (config.layerTypes
            ?? Array(repeating: "global_attention", count: config.numHiddenLayers))[layerIdx]
            == "sliding_attention"

        self._mlp.wrappedValue = MLP(config: config, layerIdx: layerIdx)
        self._inputLayernorm.wrappedValue = Gemma3nRMSNorm(
            dim: hiddenSize,
            eps: config.rmsNormEps,
        )

        self._postAttentionLayernorm.wrappedValue = Gemma3nRMSNorm(
            dim: hiddenSize,
            eps: config.rmsNormEps,
        )
        self._preFeedforwardLayernorm.wrappedValue = Gemma3nRMSNorm(
            dim: hiddenSize,
            eps: config.rmsNormEps,
        )
        self._postFeedforwardLayernorm.wrappedValue = Gemma3nRMSNorm(
            dim: hiddenSize,
            eps: config.rmsNormEps,
        )

        self._altup.wrappedValue = Gemma3nAltUp(config: config)
        self._laurel.wrappedValue = Gemma3nLaurelBlock(config: config)

        self._perLayerInputGate.wrappedValue = Linear(
            hiddenSize,
            hiddenSizePerLayerInput,
            bias: false
        )
        self._perLayerProjection.wrappedValue = Linear(
            hiddenSizePerLayerInput,
            hiddenSize,
            bias: false
        )
        self._postPerLayerInputNorm.wrappedValue = Gemma3nRMSNorm(
            dim: hiddenSize,
            eps: config.rmsNormEps,
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        caches: [KVCache?]? = nil,
        cachePosition: MLXArray? = nil,
        positionEmbeddingsGlobal: (MLXArray, MLXArray)? = nil,
        positionEmbeddingsLocal: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        var x = x
        if x.ndim == 1 {
            x = expandedDimensions(x, axis: 0)
        }

        var finalMask = mask
        if isSliding, case .array(let maskArray) = mask {
            let effectiveSeqLen = max(cachePosition?.shape[0] ?? 0, slidingWindow)
            let minDtype = MLXArray(Float.leastNormalMagnitude)

            let slidingWindowMask = tril(
                MLXArray.ones([maskArray.shape[0], effectiveSeqLen], dtype: .bool),
                k: -slidingWindow
            )
            let updatedMask = MLX.where(slidingWindowMask, minDtype, maskArray)

            let offset = max(0, (cachePosition?.max().item() ?? 0) - effectiveSeqLen + 1)
            let maskIndexes = MLXArray(0 ..< min(effectiveSeqLen, updatedMask.shape.last!)) + offset
            let slicedMask = take(updatedMask, maskIndexes.asType(.int32), axis: -1)
            finalMask = .array(slicedMask)
        }

        let predictions = altup.predict(x)
        let activePrediction = predictions[config.altupActiveIdx]

        let activePredictionNormed = inputLayernorm(activePrediction)
        let laurelOutput = laurel(activePredictionNormed)

        let positionEmbeddings = isSliding ? positionEmbeddingsLocal : positionEmbeddingsGlobal

        let attn = selfAttn(
            activePredictionNormed,
            mask: finalMask,
            cache: cache,
            caches: caches,
            positionEmbeddings: positionEmbeddings,
            cachePosition: cachePosition
        )

        let attnNormed = postAttentionLayernorm(attn)
        let attnGated = activePrediction + attnNormed
        let attnLaurel =
            (attnGated + laurelOutput) / sqrt(MLXArray(2.0, dtype: activePrediction.dtype))

        let attnNormFf = preFeedforwardLayernorm(attnLaurel)
        let attnFfw = mlp(attnNormFf)
        let attnFfwNorm = postFeedforwardLayernorm(attnFfw)
        let attnFfwLaurelGated = attnLaurel + attnFfwNorm

        var correctedPredictions = altup.correct(
            predictions: predictions, activated: attnFfwLaurelGated)

        var firstPrediction = correctedPredictions[config.altupActiveIdx]
        if config.altupCorrectScale {
            firstPrediction = altup.scaleCorrectOutput(firstPrediction)
        }

        firstPrediction = perLayerInputGate(firstPrediction)
        firstPrediction = geluApproximate(firstPrediction)

        // Per-layer input multiplication is always performed in the Python version
        guard let perLayerInput = perLayerInput else {
            fatalError(
                "per_layer_input is required but was nil. This should never happen in normal operation."
            )
        }
        firstPrediction = firstPrediction * perLayerInput

        firstPrediction = perLayerProjection(firstPrediction)
        firstPrediction = postPerLayerInputNorm(firstPrediction)

        for i in 1 ..< correctedPredictions.shape[0] {
            correctedPredictions[i] = correctedPredictions[i] + firstPrediction
        }

        return correctedPredictions
    }
}

private class Gemma3Model: Module {
    let config: TextConfig
    let hiddenSize: Int
    let vocabSize: Int
    let vocabSizePerLayerInput: Int
    let numHiddenLayers: Int
    private let _perLayerProjectionScale: MLXArray
    private let _perLayerInputScale: MLXArray
    private let _embedTokensScale: Float
    private let _embedTokensPerLayerScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma3nDecoderLayer]
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma3nRMSNorm

    @ModuleInfo(key: "altup_projections") var altupProjections: [Linear]
    @ModuleInfo(key: "altup_unembed_projections") var altupUnembedProjections: [Linear]

    @ModuleInfo var norm: Gemma3nRMSNorm
    @ModuleInfo(key: "rope_embedding") var ropeEmbedding: Gemma3nRotaryEmbedding
    @ModuleInfo(key: "rope_embedding_local") var ropeEmbeddingLocal: Gemma3nRotaryEmbedding

    init(config: TextConfig) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.vocabSize = config.vocabSize
        self.vocabSizePerLayerInput = config.vocabSizePerLayerInput
        self.numHiddenLayers = config.numHiddenLayers

        assert(vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize,
        )
        self._embedTokensScale = pow(Float(config.hiddenSize), 0.5)

        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { layerIdx in
            Gemma3nDecoderLayer(config: config, layerIdx: layerIdx)
        }

        self._embedTokensPerLayer.wrappedValue = Embedding(
            embeddingCount: config.vocabSizePerLayerInput,
            dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput,
        )
        self._embedTokensPerLayerScale = pow(Float(config.hiddenSizePerLayerInput), 0.5)

        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numHiddenLayers * config.hiddenSizePerLayerInput,
            bias: false
        )

        self._perLayerProjectionNorm.wrappedValue = Gemma3nRMSNorm(
            dim: config.hiddenSizePerLayerInput,
            eps: config.rmsNormEps,
        )

        self._altupProjections.wrappedValue = (0 ..< (config.altupNumInputs - 1)).map { _ in
            Linear(config.hiddenSize, config.hiddenSize, bias: false)
        }
        self._altupUnembedProjections.wrappedValue = (0 ..< (config.altupNumInputs - 1)).map { _ in
            Linear(config.hiddenSize, config.hiddenSize, bias: false)
        }

        self._norm.wrappedValue = Gemma3nRMSNorm(
            dim: config.hiddenSize,
            eps: config.rmsNormEps,
        )

        self._perLayerProjectionScale = MLXArray(pow(Float(hiddenSize), -0.5))
        self._perLayerInputScale = rsqrt(MLXArray(2.0))

        self._ropeEmbedding.wrappedValue = Gemma3nRotaryEmbedding(config: config)

        var localConfig = config
        // Note: Creating a modified copy for local rope - this is a simplification
        // In actual implementation, we'd need to handle the rope_local_base_freq properly
        self._ropeEmbeddingLocal.wrappedValue = Gemma3nRotaryEmbedding(config: localConfig)

        super.init()
    }

    func callAsFunction(
        inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputsEmbeds {
            h = inputsEmbeds
        } else if let inputs {
            h = embedTokens(inputs)
            h = (h * MLXArray(_embedTokensScale, dtype: .float32)).asType(h.dtype)
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }

        let perLayerInputsProcessed: MLXArray
        if let perLayerInputs {
            perLayerInputsProcessed = perLayerInputs
        } else if let inputs {
            perLayerInputsProcessed = getPerLayerInputs(inputs)
        } else {
            fatalError("Cannot generate per layer inputs without input ids")
        }

        let finalPerLayerInputs = projectPerLayerInputs(h, perLayerInputs: perLayerInputsProcessed)

        let cacheArray = cache ?? Array(repeating: nil as KVCache?, count: layers.count)

        let pastSeenTokens = cacheArray.first??.offset ?? 0
        let cachePosition = MLXArray(pastSeenTokens ..< (pastSeenTokens + h.shape[1]))

        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = .none

        if mask == nil {
            let j = config.slidingWindowPattern
            if j > 0, j <= cacheArray.count {
                let globalCacheSlice = Array(cacheArray[(j - 1) ..< j]).compactMap { $0 }
                fullMask = createAttentionMask(h: h, cache: globalCacheSlice, returnArray: true)
            }
            slidingWindowMask = createAttentionMask(
                h: h, cache: cacheArray.compactMap { $0 }, returnArray: true)
        }

        let h0 = h

        let positionIds = expandedDimensions(cachePosition, axis: 0)
        let positionEmbeddingsGlobal = ropeEmbedding(h0, positionIds: positionIds)
        let positionEmbeddingsLocal = ropeEmbeddingLocal(h0, positionIds: positionIds)

        let targetMagnitude = pow(mean(h0.square(), axis: -1, keepDims: true), 0.5)
        let epsilonTensor = MLXArray(Float.leastNormalMagnitude, dtype: h0.dtype)

        var hList = Array(repeating: h0, count: config.altupNumInputs)

        for i in 1 ..< config.altupNumInputs {
            // `i - 1` is used because altupProjections has `altupNumInputs - 1` elements.
            let altupProj = altupProjections[i - 1](hList[i])
            hList[i] = altupProj.asType(h0.dtype)
            let newMagnitude = pow(mean(hList[i].square(), axis: -1, keepDims: true), 0.5)
            hList[i] = hList[i] * (targetMagnitude / maximum(newMagnitude, epsilonTensor))
        }

        h = stacked(hList, axis: 0)

        for (i, (layer, c)) in zip(layers, cacheArray).enumerated() {
            let perLayerInput = finalPerLayerInputs[0..., 0..., i, 0...]

            let isGlobal =
                (config.layerTypes
                ?? Array(repeating: "global_attention", count: config.numHiddenLayers))[i]
                == "global_attention"

            let localMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                localMask = mask
            } else if isGlobal {
                localMask = fullMask
            } else {
                localMask = slidingWindowMask
            }

            h = layer(
                h,
                mask: localMask,
                cache: c,
                perLayerInput: perLayerInput,
                caches: cacheArray,
                cachePosition: cachePosition,
                positionEmbeddingsGlobal: positionEmbeddingsGlobal,
                positionEmbeddingsLocal: positionEmbeddingsLocal
            )
        }

        let targetMagnitudeFinal = pow(mean(h[0].square(), axis: -1, keepDims: true), 0.5)

        for i in 1 ..< config.altupNumInputs {
            // `i - 1` is used because altupUnembedProjections has `altupNumInputs - 1` elements.
            let altupUnembProj = altupUnembedProjections[i - 1](h[i])
            h[i] = altupUnembProj.asType(h0.dtype)
            let newMagnitude = pow(mean(h[i].square(), axis: -1, keepDims: true), 0.5)
            h[i] = h[i] * (targetMagnitudeFinal / maximum(newMagnitude, epsilonTensor))
        }

        h = mean(h, axis: 0)
        return norm(h)
    }

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        let perLayerInputsMask = logicalAnd(
            inputIds .>= 0,
            inputIds .< vocabSizePerLayerInput
        )
        let tokens = MLX.where(perLayerInputsMask, inputIds, MLXArray.zeros(like: inputIds))
        var result = embedTokensPerLayer(tokens)
        result = (result * MLXArray(_embedTokensPerLayerScale, dtype: .float32)).asType(
            result.dtype)
        result = result.reshaped(
            Array(inputIds.shape) + [config.numHiddenLayers, config.hiddenSizePerLayerInput]
        )
        return result
    }

    func projectPerLayerInputs(_ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?) -> MLXArray {
        var perLayerProjection = perLayerModelProjection(inputsEmbeds)
        perLayerProjection =
            perLayerProjection * _perLayerProjectionScale.asType(inputsEmbeds.dtype)

        perLayerProjection = perLayerProjection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.numHiddenLayers, config.hiddenSizePerLayerInput,
            ]
        )
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)

        guard let perLayerInputs = perLayerInputs else {
            return perLayerProjection
        }

        var adjustedPerLayerInputs = perLayerInputs
        if perLayerProjection.shape != perLayerInputs.shape {
            let targetLayers = min(
                config.numHiddenLayers, perLayerInputs.shape[perLayerInputs.shape.count - 2])
            adjustedPerLayerInputs = perLayerInputs[.ellipsis, ..<targetLayers, 0...]
        }

        return (perLayerProjection + adjustedPerLayerInputs)
            * _perLayerInputScale.asType(inputsEmbeds.dtype)
    }
}

private class LanguageModel: Module, KVCacheDimensionProvider {
    let config: TextConfig
    let modelType: String
    let finalLogitSoftcapping: Float?
    let textVocabSize: Int

    @ModuleInfo(key: "model") var model: Gemma3Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    var kvHeads: [Int]

    init(config: TextConfig) {
        self.config = config
        self.modelType = config.modelType
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self.textVocabSize = config.vocabSizePerLayerInput

        self._model.wrappedValue = Gemma3Model(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)

        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)

        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        var caches: [any KVCache] = []
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.numHiddenLayers {
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
        inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> LMOutput {
        let out = model(
            inputs: inputs,
            inputsEmbeds: inputsEmbeds,
            mask: mask,
            cache: cache,
            perLayerInputs: perLayerInputs
        )
        var finalLogits = lmHead(out)

        if let softcap = finalLogitSoftcapping, softcap > 0 {
            finalLogits = tanh(finalLogits / softcap) * softcap
        }

        return LMOutput(logits: finalLogits)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        for (k, v) in weights {
            if !k.contains("language_model.model") && !k.contains("language_model.lm_head") {
                // Transform keys that don't contain the specific patterns
                let newKey = k.replacingOccurrences(
                    of: "language_model", with: "language_model.model")
                sanitizedWeights[newKey] = v
            } else if k.contains("self_attn.rotary_emb.inv_freq") {
                // Skip rotary embedding inverse frequency weights
                continue
            } else {
                sanitizedWeights[k] = v
            }
        }
        // Handle tied lm_head weights
        if sanitizedWeights["language_model.lm_head.weight"] == nil {
            let embedTokensKey = "language_model.model.embed_tokens.weight"
            if let embedWeight = sanitizedWeights[embedTokensKey] {
                sanitizedWeights["language_model.lm_head.weight"] = embedWeight
            }
        }
        return sanitizedWeights
    }
}

// MARK: - Multimodal Embedder

private class Gemma3nMultimodalEmbedder: Module, UnaryLayer {
    let multimodalHiddenSize: Int
    let eps: Float
    let vocabOffset: Int
    let vocabSize: Int
    let textHiddenSize: Int

    @ModuleInfo var embedding: Embedding
    @ModuleInfo(key: "hard_embedding_norm") var hardEmbeddingNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "soft_embedding_norm") var softEmbeddingNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_post_projection_norm") var embeddingPostProjectionNorm:
        Gemma3nRMSNorm

    init(multimodalConfig: any MultimodalConfig, textConfig: TextConfig) {
        self.multimodalHiddenSize = multimodalConfig.hiddenSize
        self.eps = multimodalConfig.rmsNormEps
        self.vocabOffset = multimodalConfig.vocabOffset
        self.vocabSize = multimodalConfig.vocabSize
        self.textHiddenSize = textConfig.hiddenSize

        self._embedding.wrappedValue = Embedding(
            embeddingCount: vocabSize,
            dimensions: multimodalHiddenSize
        )
        self._hardEmbeddingNorm.wrappedValue = Gemma3nRMSNorm(
            dim: multimodalHiddenSize,
            eps: eps
        )
        self._softEmbeddingNorm.wrappedValue = Gemma3nRMSNorm(
            dim: multimodalHiddenSize,
            eps: eps
        )
        self._embeddingProjection.wrappedValue = Linear(
            multimodalHiddenSize,
            textHiddenSize,
            bias: false
        )
        self._embeddingPostProjectionNorm.wrappedValue = Gemma3nRMSNorm(
            dim: textHiddenSize,
            eps: eps,
            withScale: false
        )

        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray?, inputsEmbeds: MLXArray?) -> MLXArray {
        guard (inputIds == nil) != (inputsEmbeds == nil) else {
            fatalError("You must specify exactly one of inputIds or inputsEmbeds")
        }

        let embNorm: MLXArray
        if let inputsEmbeds {
            embNorm = softEmbeddingNorm(inputsEmbeds)
        } else if let inputIds {
            let hardEmb = embedding(inputIds - vocabOffset)
            embNorm = hardEmbeddingNorm(hardEmb)
        } else {
            fatalError("Either inputIds or inputsEmbeds must be provided")
        }

        let embNormProj = embeddingProjection(embNorm)
        let projected = embeddingPostProjectionNorm(embNormProj)
        return projected
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        return callAsFunction(inputIds, inputsEmbeds: nil)
    }
}

// MARK: - Helper Functions

// MARK: - Custom Attention for Gemma3n with Logit Softcapping

/// Custom attention function for Gemma3n that supports:
/// - Logit softcapping (applied before softmax)
/// - Standard KV cache support
/// - Exact alignment with Python implementation
///
/// TODO: Quantized KV Cache Integration
/// Action items for adding quantized cache support:
/// 1. Add QuantizedKVCache detection: `if let quantizedKVCache = cache as? QuantizedKVCache`
/// 2. Use quantizedKVCache.updateQuantized(keys: keys, values: values) for cache update
/// 3. Implement manual quantized attention computation with logit softcapping:
///    - Cannot use quantizedScaledDotProductAttention directly (no softcapping support)
///    - Need to manually compute: matmul(queries, dequantized_keys) with softcapping
///    - May require dequantization of keys for logit softcapping application
/// 4. Consider performance trade-offs:
///    - Manual dequantization vs quantized attention benefits
///    - Might need hybrid approach or dedicated quantized+softcapping function
/// 5. Test with QuantizedKVCache to ensure numerical accuracy matches Python
/// 6. Update documentation and examples
private func gemma3nAttentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    attnLogitSoftcapping: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    // Update cache and get cached keys/values (matches Python's cache.update_and_fetch)
    let (cachedKeys, cachedValues): (MLXArray, MLXArray)

    if let cache {
        (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    } else {
        (cachedKeys, cachedValues) = (keys, values)
    }

    // Manual attention computation to support logit softcapping
    // This matches the Python implementation exactly:
    // attn_weights = mx.matmul(queries, keys.swapaxes(2, 3)) * self.scale
    var attnWeights = matmul(queries, cachedKeys.swappedAxes(2, 3)) * scale

    // Apply logit softcapping if enabled (matches Python)
    // if self.attn_logit_softcapping is not None and self.attn_logit_softcapping > 0:
    if attnLogitSoftcapping > 0 {
        attnWeights = attnWeights / attnLogitSoftcapping
        attnWeights = tanh(attnWeights)
        attnWeights = attnWeights * attnLogitSoftcapping
    }

    // Apply mask if provided (matches Python)
    // if mask is not None: causal_mask = mask[:, : keys.shape[-2]]
    if case .array(let maskArray) = mask {
        let causalMask = maskArray[0..., ..<cachedKeys.shape[2]]
        attnWeights = attnWeights + causalMask
    }

    // Apply softmax and compute output (matches Python)
    // attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(queries.dtype)
    attnWeights = softmax(attnWeights.asType(.float32), axis: -1).asType(queries.dtype)

    // output = mx.matmul(attn_weights, values)
    return matmul(attnWeights, cachedValues)
}

private func bicubicInterpolate(
    _ x: MLXArray, to targetSize: (Int, Int), alignCorners: Bool = false
) -> MLXArray {
    // TODO: This implementation uses nested loops and sequential MLX operations, which is much slower
    // than the Python version that uses mx.fast.metal_kernel() for parallel GPU computation.
    // MLX Swift currently doesn't have custom Metal kernel creation capabilities like Python's
    // mx.fast.metal_kernel(). Consider optimizing with vectorized MLX operations or requesting
    // custom kernel support from the MLX Swift team for better performance.

    // Input: NHWC format [batch, height, width, channels]
    // Output: NHWC format [batch, target_height, target_width, channels]

    let inputShape = x.shape
    let (batchSize, inputHeight, inputWidth, channels) = (
        inputShape[0], inputShape[1], inputShape[2], inputShape[3]
    )
    let (targetHeight, targetWidth) = targetSize

    // If no resizing needed, return input
    if inputHeight == targetHeight && inputWidth == targetWidth {
        return x
    }

    // Convert to float32 for computation if needed
    let inputDtype = x.dtype
    let xFloat = x.asType(.float32)

    // Calculate scale factors
    let scaleH: Float
    let scaleW: Float

    if alignCorners && targetHeight > 1 && targetWidth > 1 {
        scaleH = Float(inputHeight - 1) / Float(targetHeight - 1)
        scaleW = Float(inputWidth - 1) / Float(targetWidth - 1)
    } else {
        scaleH = Float(inputHeight) / Float(targetHeight)
        scaleW = Float(inputWidth) / Float(targetWidth)
    }

    // Bicubic kernel function (matches Python implementation with a=-0.5)
    func cubicKernel(_ x: Float) -> Float {
        let absx = abs(x)
        let absx2 = absx * absx
        let absx3 = absx2 * absx
        let a: Float = -0.5

        if absx <= 1.0 {
            return (a + 2.0) * absx3 - (a + 3.0) * absx2 + 1.0
        } else if absx < 2.0 {
            return a * absx3 - 5.0 * a * absx2 + 8.0 * a * absx - 4.0 * a
        }
        return 0.0
    }

    // Create output array
    var result = MLXArray.zeros(
        [batchSize, targetHeight, targetWidth, channels], type: Float32.self)

    // Process each output pixel
    for outY in 0 ..< targetHeight {
        for outX in 0 ..< targetWidth {
            // Calculate input coordinates
            let inY: Float
            let inX: Float

            if alignCorners && targetHeight > 1 && targetWidth > 1 {
                inY = Float(outY) * scaleH
                inX = Float(outX) * scaleW
            } else {
                inY = (Float(outY) + 0.5) * scaleH - 0.5
                inX = (Float(outX) + 0.5) * scaleW - 0.5
            }

            // Get integer and fractional parts
            let y0 = Int(floor(inY))
            let x0 = Int(floor(inX))
            let yFrac = inY - Float(y0)
            let xFrac = inX - Float(x0)

            // Bicubic interpolation with 4x4 neighborhood
            var interpolatedPixel = MLXArray.zeros([batchSize, channels], type: Float32.self)
            var weightSum: Float = 0.0

            for i in -1 ... 2 {
                let yPos = max(0, min(y0 + i, inputHeight - 1))
                let wy = cubicKernel(yFrac - Float(i))

                for j in -1 ... 2 {
                    let xPos = max(0, min(x0 + j, inputWidth - 1))
                    let wx = cubicKernel(xFrac - Float(j))
                    let weight = wy * wx

                    if weight != 0.0 {
                        let pixelValue = xFloat[0..., yPos, xPos, 0...]
                        interpolatedPixel = interpolatedPixel + pixelValue * weight
                        weightSum += weight
                    }
                }
            }

            // Normalize by weight sum
            if weightSum > 0.0 {
                interpolatedPixel = interpolatedPixel / weightSum
            }

            // Set the result
            result[0..., outY, outX, 0...] = interpolatedPixel
        }
    }

    // Convert back to original dtype
    return result.asType(inputDtype)
}

private func maskedScatter(
    inputTensor: MLXArray,
    mask: MLXArray,
    source: MLXArray
) -> MLXArray {
    let maskBool = mask.asType(.bool)

    if !maskBool.any().item() {
        return broadcast(inputTensor, to: mask.shape)
    }

    let inputShape = mask.shape
    var resultFlat = broadcast(inputTensor, to: inputShape).flattened()
    let maskFlat = maskBool.flattened()
    let sourceFlat = source.flattened()

    let selectionMask = cumsum(maskFlat.asType(.int32)) - 1
    let sourceLen = sourceFlat.shape[0]
    let boundedIndices = selectionMask % sourceLen

    let selectedValues = take(sourceFlat, boundedIndices, axis: 0)
    resultFlat = MLX.where(maskFlat, selectedValues, resultFlat)

    return resultFlat.reshaped(inputShape)
}

// MARK: - Main Model

public class Gemma3n: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: LanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma3nVisionModel
    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma3nAudioModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma3nMultimodalEmbedder
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma3nMultimodalEmbedder

    public let config: ModelConfig

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        return languageModel.newCache(parameters: parameters)
    }

    public init(_ config: ModelConfig) {
        self.config = config

        self._languageModel.wrappedValue = LanguageModel(config: config.textConfig)
        self._visionTower.wrappedValue = Gemma3nVisionModel(config: config.visionConfig)
        self._audioTower.wrappedValue = Gemma3nAudioModel(config: config.audioConfig)
        self._embedVision.wrappedValue = Gemma3nMultimodalEmbedder(
            multimodalConfig: config.visionConfig,
            textConfig: config.textConfig
        )
        self._embedAudio.wrappedValue = Gemma3nMultimodalEmbedder(
            multimodalConfig: config.audioConfig,
            textConfig: config.textConfig
        )

        super.init()
    }

    func getInputEmbeddings(
        inputIds: MLXArray? = nil,
        pixelValues: MLXArray? = nil,
        inputFeatures: MLXArray? = nil,
        inputFeaturesMask: MLXArray? = nil
    ) -> MLXArray {
        if pixelValues == nil && inputFeatures == nil {
            return languageModel.model.embedTokens(inputIds!)
        }

        guard let inputIds = inputIds else {
            fatalError("Input IDs required for multimodal input")
        }

        var inputsEmbeds = languageModel.model.embedTokens(inputIds)

        // Ensure no gaps between text, vision, and audio embeddings, in that order
        // This matches the Python assertion
        assert(
            embedAudio.vocabOffset == config.vocabSize - config.audioConfig.vocabSize,
            "Audio vocab offset mismatch"
        )
        assert(
            embedVision.vocabOffset == config.vocabSize - config.audioConfig.vocabSize
                - config.visionConfig.vocabSize,
            "Vision vocab offset mismatch"
        )

        // Handle vision tokens
        if pixelValues != nil {
            let visionMask = logicalAnd(
                inputIds .>= config.visionConfig.vocabOffset,
                inputIds .< config.audioConfig.vocabOffset
            )

            if visionMask.any().item() {
                let visionTokens = MLX.where(visionMask, inputIds, MLXArray.zeros(like: inputIds))
                let visionEmbedsFlat = embedVision.callAsFunction(visionTokens, inputsEmbeds: nil)
                inputsEmbeds = MLX.where(
                    expandedDimensions(visionMask, axis: -1),
                    visionEmbedsFlat,
                    inputsEmbeds
                )
            }
        }

        // Handle audio tokens
        if inputFeatures != nil {
            let audioMask = inputIds .>= config.audioConfig.vocabOffset

            if audioMask.any().item() {
                let audioTokens = MLX.where(audioMask, inputIds, MLXArray.zeros(like: inputIds))
                let audioEmbedsFlat = embedAudio.callAsFunction(audioTokens, inputsEmbeds: nil)
                inputsEmbeds = MLX.where(
                    expandedDimensions(audioMask, axis: -1),
                    audioEmbedsFlat,
                    inputsEmbeds
                )
            }
        }

        // Process vision features
        if let pixelValues {
            let imageFeatures = getImageFeatures(pixelValues)
            return mergeMultimodalAndText(
                inputIds: inputIds,
                inputsEmbeds: inputsEmbeds,
                features: imageFeatures,
                tokenId: config.imageTokenId,
                modality: "image"
            )
        }

        // Process audio features
        if let inputFeatures, let inputFeaturesMask = inputFeaturesMask {
            let (audioFeatures, audioMask) = getAudioFeatures(inputFeatures, .!inputFeaturesMask)
            let audioPaddingIds = MLXArray([config.vocabSize - 1]).expandedDimensions(axis: 0)
            let audioPaddingEmbs = embedAudio.callAsFunction(audioPaddingIds, inputsEmbeds: nil)

            let maskedAudioFeatures = MLX.where(
                expandedDimensions(audioMask, axis: -1),
                audioPaddingEmbs,
                audioFeatures
            )

            let audioBatchSize = maskedAudioFeatures.shape[0]
            let audioSeqLen = maskedAudioFeatures.shape[1]
            let audioEmbedDim = maskedAudioFeatures.shape[2]
            let extraPaddingTokens = config.audioSoftTokensPerImage - audioSeqLen

            let extraPaddingFeatures = broadcast(
                audioPaddingEmbs,
                to: [audioBatchSize, extraPaddingTokens, audioEmbedDim]
            )

            let finalAudioFeatures = concatenated(
                [maskedAudioFeatures, extraPaddingFeatures], axis: 1)

            return mergeMultimodalAndText(
                inputIds: inputIds,
                inputsEmbeds: inputsEmbeds,
                features: finalAudioFeatures,
                tokenId: config.audioTokenId,
                modality: "audio"
            )
        }

        return inputsEmbeds
    }

    func getAudioFeatures(_ inputFeatures: MLXArray, _ inputFeaturesMask: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let (audioOutputs, audioMask) = audioTower(inputFeatures, inputFeaturesMask)
        return (embedAudio.callAsFunction(nil, inputsEmbeds: audioOutputs), audioMask)
    }

    func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let visionOutputs = visionTower(pixelValues, outputHiddenStates: true)

        // Python: vision_outputs.transpose(0, 3, 1, 2) - NHWC -> NCHW
        let visionOutputsNCHW = visionOutputs.transposed(0, 3, 1, 2)

        // Python: reshape and transpose to get [batch, tokens, features]
        let reshaped = visionOutputsNCHW.reshaped([
            visionOutputsNCHW.shape[0],
            config.visionConfig.hiddenSize,
            config.visionSoftTokensPerImage,
        ]).transposed(0, 2, 1)

        // Normalize and embed the soft tokens into language model space
        let scaledOutputs = reshaped * pow(Float(config.visionConfig.hiddenSize), 0.5)
        return embedVision.callAsFunction(nil, inputsEmbeds: scaledOutputs)
    }

    func mergeMultimodalAndText(
        inputIds: MLXArray?,
        inputsEmbeds: MLXArray,
        features: MLXArray,
        tokenId: Int,
        modality: String
    ) -> MLXArray {
        let specialModalityMask: MLXArray

        if let inputIds {
            specialModalityMask = expandedDimensions(inputIds .== tokenId, axis: -1)
        } else {
            // When inputIds is nil, create mask by comparing embeddings
            let embedFn: (MLXArray) -> MLXArray =
                modality == "audio"
                ? { self.embedAudio.callAsFunction($0, inputsEmbeds: nil) }
                : { self.languageModel.model.embedTokens($0) }
            let tokenEmbedding = embedFn(MLXArray([tokenId]))
            specialModalityMask = inputsEmbeds .== tokenEmbedding
        }

        let specialModalityMaskBroadcast = broadcast(specialModalityMask, to: inputsEmbeds.shape)

        let modalityTokensInText = specialModalityMaskBroadcast.sum().item(Int.self)
        let featureTokens = features.size

        guard modalityTokensInText == featureTokens else {
            fatalError(
                """
                Number of \(modality)s does not match number of special \(modality) tokens in the input text.
                Got \(modalityTokensInText) \(modality) tokens in the text and \(featureTokens) tokens from \(modality) embeddings.
                """)
        }

        let featuresTyped = features.asType(inputsEmbeds.dtype)
        return maskedScatter(
            inputTensor: inputsEmbeds, mask: specialModalityMaskBroadcast, source: featuresTyped)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels

        let inputsEmbeds = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues
        )

        let perLayerInputs = languageModel.model.getPerLayerInputs(inputIds)
        let convertedCache = cache.compactMap { $0 as? KVCache }

        let result = languageModel(
            inputs: nil,
            inputsEmbeds: inputsEmbeds,
            mask: .causal,
            cache: convertedCache,
            perLayerInputs: perLayerInputs
        )

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let convertedCache = cache?.compactMap { $0 as? KVCache }
        return languageModel(inputs: inputs, cache: convertedCache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = [String: MLXArray]()
        for (k, v) in weights {
            if k.starts(with: "model.") {
                let newKey = k.split(separator: ".").dropFirst().joined(separator: ".")
                sanitizedWeights[newKey] = v
            } else {
                sanitizedWeights[k] = v
            }
        }
        sanitizedWeights = visionTower.sanitize(weights: sanitizedWeights)
        // TODO: The audio and language sanitization is not done in the Python implementation. Is this needed?
        sanitizedWeights = audioTower.sanitize(weights: sanitizedWeights)
        sanitizedWeights = languageModel.sanitize(weights: sanitizedWeights)
        return sanitizedWeights
    }
}

// MARK: - Audio Model Components

// MARK: - Helper Functions for Padding
private func convertTorchToMLXPadWidth(_ padding: [Int], _ inputShape: [Int]) -> [IntOrPair] {
    let ndim = inputShape.count
    var padWidth = Array(repeating: IntOrPair((0, 0)), count: ndim)

    if ndim >= 1 && padding.count >= 2 {
        padWidth[ndim - 1] = IntOrPair((padding[0], padding[1]))
    }
    if ndim >= 2 && padding.count >= 4 {
        padWidth[ndim - 2] = IntOrPair((padding[2], padding[3]))
    }
    if ndim >= 3 && padding.count >= 6 {
        padWidth[ndim - 3] = IntOrPair((padding[4], padding[5]))
    }
    if ndim >= 4 && padding.count >= 8 {
        padWidth[ndim - 4] = IntOrPair((padding[6], padding[7]))
    }

    return padWidth
}

// MARK: - Audio Relative Position Embedding
private class Gemma3nAudioRelativePositionEmbedding: Module {
    let config: AudioConfig
    let numHeads: Int
    let channels: Int
    let headDim: Int
    let maxBackward: Int
    let maxForward: Int

    @ModuleInfo(key: "pos_proj") var posProj: Linear
    private let _invTimescales: MLXArray

    init(config: AudioConfig) {
        self.config = config
        self.numHeads = config.confNumAttentionHeads
        self.channels = config.hiddenSize
        self.headDim = channels / numHeads
        self.maxBackward =
            config.confAttentionContextLeft > 0 ? config.confAttentionContextLeft - 1 : 0
        self.maxForward = config.confAttentionContextRight

        self._posProj.wrappedValue = Linear(channels, numHeads * headDim, bias: false)

        let minTimescale: Float = 1.0
        let maxTimescale: Float = 1.0e4
        let numTimescales = channels / 2
        let logTimescaleIncrement =
            log(maxTimescale / minTimescale) / max(Float(numTimescales - 1), 1)
        let invTimescales =
            minTimescale
            * exp(
                MLXArray(0 ..< numTimescales).asType(.float32) * (-logTimescaleIncrement)
            )

        self._invTimescales = expandedDimensions(
            expandedDimensions(invTimescales, axis: 0),
            axis: 0
        )

        super.init()
    }

    private func getTimingSignal1dPos(_ position: MLXArray, dtype: DType) -> MLXArray {
        assert(position.ndim == 2)
        let positionFloat = expandedDimensions(position.asType(.float32), axis: -1)

        let scaledTime = positionFloat * _invTimescales
        let timingSignal = concatenated([sin(scaledTime), cos(scaledTime)], axis: -1)
        return timingSignal.asType(dtype)
    }

    private func relativeShift(
        _ termBdBeforeShift: MLXArray,
        batchSize: Int,
        numHeads: Int,
        numQueryBlocks: Int,
        queryBlockSize: Int,
        keyContextSize: Int,
        maxSpanPlus1: Int
    ) -> MLXArray {
        let padAmountLastDim = (keyContextSize + 1) - maxSpanPlus1
        let paddingTuple = [0, padAmountLastDim]

        let termBdPadded = padded(
            termBdBeforeShift,
            widths: convertTorchToMLXPadWidth(paddingTuple, Array(termBdBeforeShift.shape))
        )

        let termBdReshaped = termBdPadded.reshaped([
            batchSize,
            numHeads,
            numQueryBlocks,
            queryBlockSize * (keyContextSize + 1),
        ])

        let termBdSliced = termBdReshaped[0..., 0..., 0..., ..<(queryBlockSize * keyContextSize)]

        let termBdShifted = termBdSliced.reshaped([
            batchSize,
            numHeads,
            numQueryBlocks,
            queryBlockSize,
            keyContextSize,
        ])

        return termBdShifted
    }

    func callAsFunction(_ queries: MLXArray, _ keys: MLXArray) -> MLXArray {
        let (batchSize, numQueryBlocks, queryBlockSize, numHeads, headDim) = (
            queries.shape[0], queries.shape[1], queries.shape[2], queries.shape[3], queries.shape[4]
        )
        let keyContextSize = keys.shape[2]

        // Relative positions for sinusoidal embeddings
        let posIndices = expandedDimensions(
            MLXArray(stride(from: maxBackward, through: -maxForward - 1, by: -1)),
            axis: 0
        )
        let maxSpanPlus1 = posIndices.shape[1]

        let sinEmbTimingSignal = getTimingSignal1dPos(posIndices, dtype: queries.dtype)

        // Project sinusoidal embeddings
        let projectedSinEmb = posProj(sinEmbTimingSignal)
        let sinEmb = projectedSinEmb.reshaped([1, maxSpanPlus1, numHeads, headDim]).squeezed(
            axis: 0)

        // Term AC: Query-Key content interaction
        let queriesP = queries.transposed(0, 3, 1, 2, 4)
        let keysPT = keys.transposed(0, 3, 1, 4, 2)
        let termAc = matmul(queriesP, keysPT)

        // Term BD: Query-Position interaction
        let qTransposed = queries.transposed(0, 3, 1, 2, 4)
        let sTransposed = sinEmb.transposed(1, 2, 0)

        let qReshaped = qTransposed.reshaped([
            batchSize, numHeads, numQueryBlocks * queryBlockSize, headDim,
        ])

        let termBdUnshifedMatmul = matmul(qReshaped, sTransposed)

        let termBdUnshifed = termBdUnshifedMatmul.reshaped([
            batchSize,
            numHeads,
            numQueryBlocks,
            queryBlockSize,
            maxSpanPlus1,
        ])

        let termBdShifted = relativeShift(
            termBdUnshifed,
            batchSize: batchSize,
            numHeads: numHeads,
            numQueryBlocks: numQueryBlocks,
            queryBlockSize: queryBlockSize,
            keyContextSize: keyContextSize,
            maxSpanPlus1: maxSpanPlus1
        )

        return termAc + termBdShifted
    }
}

// MARK: - Cumulative Group Norm
private class Gemma3nCumulativeGroupNorm: Module {
    let numChannels: Int
    let featureDims: [Int]
    let eps: Float
    let useScale: Bool
    let useBias: Bool
    let reductionAxes: [Int]

    @ModuleInfo var weight: MLXArray?
    @ModuleInfo var bias: MLXArray?

    init(
        numChannels: Int,
        featureDims: [Int],
        eps: Float = 1e-3,
        useScale: Bool = true,
        useBias: Bool = false
    ) {
        self.numChannels = numChannels
        self.featureDims = featureDims
        self.eps = eps
        self.useScale = useScale
        self.useBias = useBias

        // Axes for normalization: all dimensions except Batch (0) and Time (1)
        self.reductionAxes = Array(2 ..< (2 + featureDims.count + 1))

        if useScale {
            self._weight.wrappedValue = MLXArray.ones([numChannels])
        } else {
            self._weight.wrappedValue = nil
        }

        if useBias {
            self._bias.wrappedValue = MLXArray.zeros([numChannels])
        } else {
            self._bias.wrappedValue = nil
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let expectedInputSuffix = featureDims + [numChannels]
        assert(Array(x.shape.suffix(expectedInputSuffix.count)) == expectedInputSuffix)

        if let mask {
            assert(mask.shape == Array(x.shape.prefix(2)))
            assert(mask.dtype == .bool)
        }

        let inputDtype = x.dtype
        let calcDtype = DType.float32
        let xCalc = x.asType(calcDtype)

        let maskCalc: MLXArray
        if let mask {
            let maskSuffixShape = Array(repeating: 1, count: expectedInputSuffix.count)
            maskCalc = mask.reshaped(Array(mask.shape) + maskSuffixShape).asType(calcDtype)
        } else {
            maskCalc = MLXArray.ones(like: xCalc).asType(calcDtype)
        }

        let xMaskedForSum = xCalc * maskCalc

        // Cumulative Statistics Calculation
        let sumValuesAtT = sum(xMaskedForSum, axes: reductionAxes, keepDims: true)
        let cumSumValues = cumsum(sumValuesAtT, axis: 1)

        let elementsInGroupAtT = sum(maskCalc, axes: reductionAxes, keepDims: true)
        let cumCountElements = cumsum(elementsInGroupAtT, axis: 1)
        let safeCumCountElements = clip(cumCountElements, min: MLXArray(1))

        let cumMean = cumSumValues / safeCumCountElements

        let squaredDiffFromMean = pow(xCalc - cumMean, 2)
        let sumSqDiffAtT = sum(
            squaredDiffFromMean * maskCalc,
            axes: reductionAxes,
            keepDims: true
        )
        let cumSumSqDiff = cumsum(sumSqDiffAtT, axis: 1)

        let cumVariance = cumSumSqDiff / safeCumCountElements

        var normalizedX = (xCalc - cumMean) * rsqrt(cumVariance + eps)

        if useScale, let weight = weight {
            let scale = weight.asType(calcDtype)
            let scaleViewShape = Array(repeating: 1, count: x.ndim - 1) + [numChannels]
            normalizedX = normalizedX * scale.reshaped(scaleViewShape)
        }

        if useBias, let bias = bias {
            let biasValue = bias.asType(calcDtype)
            let biasViewShape = Array(repeating: 1, count: x.ndim - 1) + [numChannels]
            normalizedX = normalizedX + biasValue.reshaped(biasViewShape)
        }

        let finalOutput = normalizedX * maskCalc
        return finalOutput.asType(inputDtype)
    }
}

// MARK: - Audio SSCP Conv Block
private class Gemma3nAudioSSCPConvBlock: Module {
    let config: AudioConfig
    let manualPadding: [Int]

    @ModuleInfo var conv: Conv2d
    @ModuleInfo var norm: Gemma3nCumulativeGroupNorm

    init(
        idx: Int,
        inputFreqDim: Int,
        config: AudioConfig,
        manualPadding: [Int] = [0, 0, 0, 0]
    ) {
        self.config = config
        self.manualPadding = manualPadding

        let inChannels = idx == 0 ? 1 : config.sscpConvChannelSize[idx - 1]
        let outChannels = config.sscpConvChannelSize[idx]
        let (kernelH, kernelW) = (
            config.sscpConvKernelSize[idx][0], config.sscpConvKernelSize[idx][1]
        )
        let (strideH, strideW) = (
            config.sscpConvStrideSize[idx][0], config.sscpConvStrideSize[idx][1]
        )

        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair((kernelH, kernelW)),
            stride: IntOrPair((strideH, strideW)),
            padding: IntOrPair((0, 0)),
            bias: false
        )

        let fInPadded = inputFreqDim + manualPadding[0] + manualPadding[1]
        let fOutConv = (fInPadded - kernelW) / strideW + 1

        self._norm.wrappedValue = Gemma3nCumulativeGroupNorm(
            numChannels: outChannels,
            featureDims: [fOutConv],
            eps: config.sscpConvEps,
            useScale: true,
            useBias: false
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let audioencodingsPadded = padded(
            x,
            widths: convertTorchToMLXPadWidth(manualPadding, Array(x.shape))
        )

        let audioencodingsConv = conv(audioencodingsPadded.transposed(0, 2, 3, 1))
        let xNormed = norm(audioencodingsConv)
        let audioencodingsNormed = xNormed.transposed(0, 3, 1, 2)
        return relu(audioencodingsNormed)
    }
}

// MARK: - Audio Subsample Conv Projection
private class Gemma3nAudioSubSampleConvProjection: Module {
    let config: AudioConfig
    let inputProjInFeatures: Int

    @ModuleInfo(key: "conv_0") var conv0: Gemma3nAudioSSCPConvBlock
    @ModuleInfo(key: "conv_1") var conv1: Gemma3nAudioSSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    init(config: AudioConfig) {
        self.config = config

        var currentFForBlockInput = config.inputFeatSize
        var calculatedBlockPadding: [[Int]] = []
        var calculatedFOutDims: [Int] = []

        for i in 0 ..< 2 {
            let (kernelH, kernelW) = (
                config.sscpConvKernelSize[i][0], config.sscpConvKernelSize[i][1]
            )
            let (strideH, strideW) = (
                config.sscpConvStrideSize[i][0], config.sscpConvStrideSize[i][1]
            )

            let padTTop = 0
            let padTBottom = kernelH - 1
            let padFLeft = 1
            let padFRight = 1

            let manualPaddingTuple = [padFLeft, padFRight, padTTop, padTBottom]
            calculatedBlockPadding.append(manualPaddingTuple)

            let fInPadded = currentFForBlockInput + padFLeft + padFRight
            let fOutAfterConv = (fInPadded - kernelW) / strideW + 1

            calculatedFOutDims.append(fOutAfterConv)
            currentFForBlockInput = fOutAfterConv
        }

        self._conv0.wrappedValue = Gemma3nAudioSSCPConvBlock(
            idx: 0,
            inputFreqDim: config.inputFeatSize,
            config: config,
            manualPadding: calculatedBlockPadding[0]
        )

        self._conv1.wrappedValue = Gemma3nAudioSSCPConvBlock(
            idx: 1,
            inputFreqDim: calculatedFOutDims[0],
            config: config,
            manualPadding: calculatedBlockPadding[1]
        )

        let finalCOut = config.sscpConvChannelSize.last!
        let finalFOut = calculatedFOutDims.last!
        self.inputProjInFeatures = finalCOut * finalFOut

        self._inputProjLinear.wrappedValue = Linear(
            inputProjInFeatures,
            config.hiddenSize,
            bias: false
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // audio_encodings is [B, T, F_in]
        // Reshape to [B, 1, T, F_in]
        let audioencodingsReshaped = expandedDimensions(x, axis: 1)
        var result = conv0(audioencodingsReshaped)
        result = conv1(result)

        let (b, cOut, tOut, fOut) = (
            result.shape[0], result.shape[1], result.shape[2], result.shape[3]
        )
        let xTransposed = result.transposed(0, 2, 3, 1)
        let outputFlattened = xTransposed.reshaped([b, tOut, fOut * cOut])
        let output = inputProjLinear(outputFlattened)
        return output
    }
}

// MARK: - Audio Attention
private class Gemma3nAudioAttention: Module {
    let config: AudioConfig
    let numHeads: Int
    let hiddenSize: Int
    let headDim: Int
    let chunkSize: Int
    let maxFutureHorizon: Int
    let maxPastHorizon: Int
    let attentionInvalidLogitsValue: Float
    let attentionLogitsSoftCap: Float
    let contextSize: Int
    let qScale: Float
    private let _localCausalValidMask: MLXArray
    private let _softcap: MLXArray

    @ModuleInfo(key: "relative_position_embedding") var relativePositionEmbedding:
        Gemma3nAudioRelativePositionEmbedding
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear

    init(config: AudioConfig) {
        self.config = config
        self.numHeads = config.confNumAttentionHeads
        self.hiddenSize = config.hiddenSize
        self.headDim = hiddenSize / numHeads
        self.chunkSize = config.confAttentionChunkSize
        self.maxFutureHorizon = config.confAttentionContextRight
        self.maxPastHorizon = max(0, config.confAttentionContextLeft - 1)
        self.attentionInvalidLogitsValue = config.confAttentionInvalidLogitsValue
        self.attentionLogitsSoftCap = config.confAttentionLogitCap
        self.contextSize = chunkSize + maxPastHorizon + maxFutureHorizon

        self._relativePositionEmbedding.wrappedValue = Gemma3nAudioRelativePositionEmbedding(
            config: config)
        self._perDimScale.wrappedValue = MLXArray.zeros([headDim])

        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)

        let qScale = pow(Float(headDim), -0.5)
        let rSoftplus0 = 1.0 / log(2.0)
        self.qScale = qScale * Float(rSoftplus0)

        let lowerCausalMask = tril(
            MLXArray.ones([contextSize, chunkSize], dtype: .bool),
            k: 0
        ).transposed()

        let upperCausalMask = tril(
            MLXArray.ones([chunkSize, contextSize], dtype: .bool),
            k: maxPastHorizon + maxFutureHorizon
        )

        let localCausalValidMaskTemp = MLXArray.ones([chunkSize, contextSize], dtype: .bool)
        self._localCausalValidMask = localCausalValidMaskTemp .&& lowerCausalMask
            .&& upperCausalMask

        self._softcap = MLXArray(attentionLogitsSoftCap, dtype: .float32)

        super.init()
    }

    private func padDim1(_ x: MLXArray, dim10Val: Int, dim11Val: Int) -> MLXArray {
        var paddingTuple = Array(repeating: 0, count: x.ndim * 2)
        let dimIdxFromEnd = x.ndim - 2
        let startIdxForDim = 2 * dimIdxFromEnd
        paddingTuple[startIdxForDim] = dim10Val
        paddingTuple[startIdxForDim + 1] = dim11Val

        return padded(
            x,
            widths: convertTorchToMLXPadWidth(paddingTuple, Array(x.shape))
        )
    }

    private func convertToBlock(_ x: MLXArray, paddingVal: Float = 0.0) -> MLXArray {
        let shape = x.shape
        let (b, t) = (shape[0], shape[1])
        let numBlocks = (t + chunkSize - 1) / chunkSize

        let paddingLen = numBlocks * chunkSize - t
        let paddedX = paddingLen > 0 ? padDim1(x, dim10Val: 0, dim11Val: paddingLen) : x

        let permutedims = [b, numBlocks, chunkSize] + Array(shape.dropFirst(2))
        return paddedX.reshaped(permutedims)
    }

    private func unfoldMLX(_ x: MLXArray, dimension: Int, size: Int, step: Int) -> MLXArray {
        let shape = x.shape
        let dimSize = shape[dimension]
        let numWindows = (dimSize - size) / step + 1

        var windows: [MLXArray] = []
        for i in 0 ..< numWindows {
            let startIdx = i * step
            let endIdx = startIdx + size

            var slices: [any MLXArrayIndex] = Array(repeating: .ellipsis, count: shape.count)
            slices[dimension] = startIdx ..< endIdx

            windows.append(x[slices])
        }

        return stacked(windows, axis: dimension + 1)
    }

    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPastHorizon
        let padRight = maxFutureHorizon + chunkSize - 1
        let paddedX = padDim1(x, dim10Val: padLeft, dim11Val: padRight)

        let frameLen = contextSize
        let frameStep = chunkSize

        let xUnfolded = unfoldMLX(paddedX, dimension: 1, size: frameLen, step: frameStep)

        if x.ndim > 2 && xUnfolded.ndim > 3 {
            return xUnfolded.transposed(0, 2, 1, 3, 4)
        }

        return xUnfolded
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let queryStates = qProj(x).reshaped(
            Array(x.shape.dropLast()) + [numHeads, headDim]
        )
        let keyStates = kProj(x).reshaped(
            Array(x.shape.dropLast()) + [numHeads, headDim]
        )
        let valueStates = vProj(x).reshaped(
            Array(x.shape.dropLast()) + [numHeads, headDim]
        )

        let perDimScaleSp = logAddExp(perDimScale, MLXArray(0.0))
        let broadcastShape = [1, 1, 1, headDim]
        let perDimScaleSpBroadcast = perDimScaleSp.reshaped(broadcastShape)
        let scaledQueryStates = queryStates * qScale * perDimScaleSpBroadcast

        let (batchSize, qTime) = (scaledQueryStates.shape[0], scaledQueryStates.shape[1])

        let queryBlocks = convertToBlock(scaledQueryStates)
        let keyBlocks = extractBlockContext(keyStates)
        let valueBlocks = extractBlockContext(valueStates)
        let numQueryBlocks = queryBlocks.shape[1]

        // Create validity mask
        let originalValidMask = .!mask
        let extractedValidMaskBlocks = extractBlockContext(originalValidMask).transposed(0, 2, 1)

        let conditionFromInputValidity = expandedDimensions(
            expandedDimensions(extractedValidMaskBlocks, axis: 1),
            axis: -2
        )

        let conditionFromCausality = expandedDimensions(
            expandedDimensions(
                expandedDimensions(_localCausalValidMask, axis: 0),
                axis: 0
            ),
            axis: 0
        )

        let finalConditionForWhere = conditionFromInputValidity .&& conditionFromCausality

        var logits = relativePositionEmbedding(queryBlocks, keyBlocks)

        // Apply attention logit softcap
        logits = logits / _softcap
        logits = tanh(logits)
        logits = logits * _softcap

        // Apply the combined mask
        logits = MLX.where(
            finalConditionForWhere,
            logits,
            MLXArray(attentionInvalidLogitsValue)
        )

        let probabilities = softmax(logits.asType(.float32), axis: -1).asType(valueBlocks.dtype)

        // Compute context vectors
        let (bDim, nDim, uDim, wDim, cDim) = (
            probabilities.shape[0], probabilities.shape[1], probabilities.shape[2],
            probabilities.shape[3], probabilities.shape[4]
        )
        let hDim = valueBlocks.shape.last!

        let probBun = probabilities.transposed(0, 2, 1, 3, 4).reshaped([-1, wDim, cDim])
        let vBun = valueBlocks.transposed(0, 1, 3, 2, 4).reshaped([-1, cDim, hDim])
        let resultBmm = matmul(probBun, vBun)

        var contextVectors = resultBmm.reshaped([bDim, uDim, nDim, wDim, hDim]).transposed(
            0, 1, 3, 2, 4)
        contextVectors = contextVectors.reshaped([
            batchSize,
            numQueryBlocks * chunkSize,
            numHeads,
            headDim,
        ])

        contextVectors = contextVectors[0..., ..<qTime, 0..., 0...]
        return contextVectors
    }
}

// MARK: - Conformer Attention
private class Gemma3nAudioConformerAttention: Module {
    let config: AudioConfig
    let postInFeatures: Int
    private let _gradientClipping: MLXArray

    @ModuleInfo(key: "pre_attn_norm") var preAttnNorm: Gemma3nRMSNorm
    @ModuleInfo var attn: Gemma3nAudioAttention
    @ModuleInfo var post: Linear
    @ModuleInfo(key: "post_norm") var postNorm: Gemma3nRMSNorm

    init(config: AudioConfig) {
        self.config = config
        let headDim = config.hiddenSize / config.confNumAttentionHeads
        self.postInFeatures = config.hiddenSize
        self._gradientClipping = MLXArray(config.gradientClipping)

        self._preAttnNorm.wrappedValue = Gemma3nRMSNorm(dim: config.hiddenSize)
        self._attn.wrappedValue = Gemma3nAudioAttention(config: config)
        self._post.wrappedValue = Linear(postInFeatures, config.hiddenSize, bias: false)
        self._postNorm.wrappedValue = Gemma3nRMSNorm(dim: config.hiddenSize)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let audioencodingsInputToAttn = x
        let clippedX = clip(x, min: -_gradientClipping, max: _gradientClipping)
        let audioencodingsNorm = preAttnNorm(clippedX)
        let audioencodingsAttnOut = attn(audioencodingsNorm, mask: mask)

        let (b, t, numHeads, headDim) = (
            audioencodingsAttnOut.shape[0], audioencodingsAttnOut.shape[1],
            audioencodingsAttnOut.shape[2], audioencodingsAttnOut.shape[3]
        )
        let audioencodingsReshaped = audioencodingsAttnOut.reshaped([b, t, numHeads * headDim])

        let postResult = post(audioencodingsReshaped)
        let clippedPost = clip(postResult, min: -_gradientClipping, max: _gradientClipping)
        return audioencodingsInputToAttn + postNorm(clippedPost)
    }
}

// MARK: - Conformer Feed Forward
private class Gemma3nAudioConformerFeedForward: Module {
    let config: AudioConfig
    private let _gradientClipping: MLXArray
    private let _postLayerScale: MLXArray

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: Linear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: Linear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: Gemma3nRMSNorm

    init(config: AudioConfig) {
        self.config = config
        self._gradientClipping = MLXArray(config.gradientClipping)
        self._postLayerScale = MLXArray(config.confResidualWeight)

        self._preLayerNorm.wrappedValue = Gemma3nRMSNorm(dim: config.hiddenSize)
        self._ffwLayer1.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 4, bias: false)
        self._ffwLayer2.wrappedValue = Linear(config.hiddenSize * 4, config.hiddenSize, bias: false)
        self._postLayerNorm.wrappedValue = Gemma3nRMSNorm(dim: config.hiddenSize)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        let clippedX = clip(x, min: -_gradientClipping, max: _gradientClipping)
        var result = preLayerNorm(clippedX)
        result = ffwLayer1(result)
        result = silu(result)
        result = ffwLayer2(result)
        let clippedResult = clip(result, min: -_gradientClipping, max: _gradientClipping)
        let normedResult = postLayerNorm(clippedResult)
        return residual + (normedResult * _postLayerScale)
    }
}

// MARK: - Conformer Light Conv1D
private class Gemma3nAudioConformerLightConv1d: Module {
    let config: AudioConfig
    private let _gradientClipping: MLXArray
    let causalPadding: Int

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: Linear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: Linear

    init(config: AudioConfig) {
        self.config = config
        self._gradientClipping = MLXArray(config.gradientClipping)
        self.causalPadding = config.confConvKernelSize - 1

        self._preLayerNorm.wrappedValue = Gemma3nRMSNorm(
            dim: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._linearStart.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize * 2,
            bias: false
        )
        self._depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.confConvKernelSize,
            stride: 1,
            padding: 0,
            groups: config.hiddenSize,
            bias: false
        )
        self._convNorm.wrappedValue = Gemma3nRMSNorm(
            dim: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._linearEnd.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(_ audioencodings: MLXArray) -> MLXArray {
        let audioencodingsResidual = audioencodings

        var result = preLayerNorm(audioencodings)
        result = linearStart(result)
        result = glu(result, axis: -1)

        // Apply manual causal padding and conv1d
        let audioencodingsTransposed = result.transposed(0, 2, 1)
        let paddedAudio = padded(
            audioencodingsTransposed,
            widths: convertTorchToMLXPadWidth(
                [causalPadding, 0], Array(audioencodingsTransposed.shape))
        )

        result = depthwiseConv1d(paddedAudio.transposed(0, 2, 1))
        result = clip(result, min: -_gradientClipping, max: _gradientClipping)
        result = convNorm(result)
        result = silu(result)
        result = linearEnd(result)

        return result + audioencodingsResidual
    }
}

// MARK: - Conformer Block
private class Gemma3nAudioConformerBlock: Module {
    let config: AudioConfig
    private let _gradientClipping: MLXArray

    @ModuleInfo(key: "ffw_layer_start") var ffwLayerStart: Gemma3nAudioConformerFeedForward
    @ModuleInfo var attention: Gemma3nAudioConformerAttention
    @ModuleInfo var lconv1d: Gemma3nAudioConformerLightConv1d
    @ModuleInfo(key: "ffw_layer_end") var ffwLayerEnd: Gemma3nAudioConformerFeedForward
    @ModuleInfo var norm: Gemma3nRMSNorm

    init(config: AudioConfig) {
        self.config = config
        self._gradientClipping = MLXArray(config.gradientClipping)

        self._ffwLayerStart.wrappedValue = Gemma3nAudioConformerFeedForward(config: config)
        self._attention.wrappedValue = Gemma3nAudioConformerAttention(config: config)
        self._lconv1d.wrappedValue = Gemma3nAudioConformerLightConv1d(config: config)
        self._ffwLayerEnd.wrappedValue = Gemma3nAudioConformerFeedForward(config: config)
        self._norm.wrappedValue = Gemma3nRMSNorm(dim: config.hiddenSize)

        super.init()
    }

    func callAsFunction(_ audioencodings: MLXArray, _ audioMelMask: MLXArray) -> MLXArray {
        var result = ffwLayerStart(audioencodings)
        result = attention(result, mask: audioMelMask)

        let validityMaskForLconv = .!audioMelMask
        let audioencodingsForLconvInput =
            result
            * expandedDimensions(
                validityMaskForLconv, axis: -1
            ).asType(result.dtype)

        result = lconv1d(audioencodingsForLconvInput)
        result = ffwLayerEnd(result)
        result = clip(result, min: -_gradientClipping, max: _gradientClipping)
        return norm(result)
    }
}

// MARK: - MobileNetV5 Architecture Components

// MARK: - Layer Scale 2D
private class LayerScale2d: Module, UnaryLayer {
    let inplace: Bool
    @ModuleInfo var gamma: MLXArray

    init(dim: Int, initValues: Float = 1e-5, inplace: Bool = false) {
        self.inplace = inplace
        self._gamma.wrappedValue = MLXArray(initValues) * MLXArray.ones([dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if inplace {
            return x * gamma
        } else {
            return x * gamma
        }
    }
}

// MARK: - RMS Norm 2D for Vision
private func rmsNorm2d(
    _ x: MLXArray,
    normalizedShape: [Int],
    weight: MLXArray? = nil,
    eps: Float = 1e-5
) -> MLXArray {
    assert(normalizedShape.count == 1)
    let dtype = x.dtype
    let v = pow(x, 2)
    let vMean = mean(v, axis: 1, keepDims: true)
    var result = x * rsqrt(vMean + eps)

    if let weight {
        let weightReshaped = weight.reshaped([1, -1, 1, 1])
        result = result.asType(dtype) * weightReshaped
    }
    return result
}

private class RMSNormAct2d: Module, UnaryLayer {
    let normalizedShape: [Int]
    let eps: Float
    let applyAct: Bool
    @ModuleInfo var weight: MLXArray
    @ModuleInfo var drop: Identity
    @ModuleInfo var act: UnaryLayer

    init(numChannels: Int, eps: Float = 1e-6, applyAct: Bool = true) {
        self.normalizedShape = [numChannels]
        self.eps = eps
        self.applyAct = applyAct

        self._weight.wrappedValue = MLXArray.ones([numChannels])
        self._drop.wrappedValue = Identity()
        self._act.wrappedValue = applyAct ? GELU() : Identity()

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Convert from NHWC to NCHW for RMS norm
        let xNCHW = x.transposed(0, 3, 1, 2)
        var result = rmsNorm2d(xNCHW, normalizedShape: normalizedShape, weight: weight, eps: eps)
        result = drop(result)
        result = act(result)
        // Convert back to NHWC
        return result.transposed(0, 2, 3, 1)
    }
}

// MARK: - Helper Functions
private func numGroups(groupSize: Int?, channels: Int) -> Int {
    guard let groupSize = groupSize, groupSize > 0 else {
        return 1  // normal conv with 1 group
    }
    // NOTE: groupSize == 1 -> depthwise conv
    assert(channels % groupSize == 0)
    let groups = channels / groupSize
    return groups
}

private func makeDivisible(
    _ v: Int, divisor: Int = 8, minValue: Int? = nil, roundLimit: Float = 0.9
) -> Int {
    let minVal = minValue ?? divisor
    let newV = max(minVal, (v + divisor / 2) / divisor * divisor)
    // Make sure that round down does not go down by more than 10%
    if Float(newV) < roundLimit * Float(v) {
        return newV + divisor
    }
    return newV
}

private func to2Tuple(_ x: Any) -> (Int, Int) {
    if let tuple = x as? (Int, Int) {
        return tuple
    } else if let int = x as? Int {
        return (int, int)
    } else {
        fatalError("Cannot convert to 2-tuple")
    }
}

// MARK: - Conv Norm Act
private class ConvNormAct: Module, UnaryLayer {
    let outChannels: Int
    @ModuleInfo var conv: Conv2d
    @ModuleInfo var bn: RMSNormAct2d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = false,
        applyAct: Bool = true,
        eps: Float = 1e-6
    ) {
        self.outChannels = outChannels

        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding),
            dilation: IntOrPair(dilation),
            groups: groups,
            bias: bias
        )

        self._bn.wrappedValue = RMSNormAct2d(
            numChannels: outChannels,
            eps: eps,
            applyAct: applyAct
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = conv(x)
        let r = bn(c)
        return r
    }
}

// MARK: - Universal Inverted Residual
private class UniversalInvertedResidual: Module, UnaryLayer {
    let hasSkip: Bool
    @ModuleInfo(key: "dw_start") var dwStart: UnaryLayer
    @ModuleInfo(key: "pw_exp") var pwExp: ConvNormAct
    @ModuleInfo(key: "dw_mid") var dwMid: UnaryLayer
    @ModuleInfo(key: "pw_proj") var pwProj: ConvNormAct
    @ModuleInfo(key: "layer_scale") var layerScale: UnaryLayer

    init(
        inChannels: Int,
        outChannels: Int,
        dwKernelSizeStart: Int = 0,
        dwKernelSizeMid: Int = 3,
        dwKernelSizeEnd: Int = 0,
        stride: Int = 1,
        dilation: Int = 1,
        groupSize: Int = 1,
        padType: String = "",
        noskip: Bool = false,
        expRatio: Float = 1.0,
        convKwargs: [String: Any]? = nil,
        dropPathRate: Float = 0.0,
        layerScaleInitValue: Float? = 1e-5
    ) {
        self.hasSkip = (inChannels == outChannels && stride == 1) && !noskip

        if stride > 1 {
            assert(dwKernelSizeStart > 0 || dwKernelSizeMid > 0 || dwKernelSizeEnd > 0)
        }

        // DW Start
        if dwKernelSizeStart > 0 {
            let dwStartStride = dwKernelSizeMid > 0 ? 1 : stride
            let dwStartGroups = numGroups(groupSize: groupSize, channels: inChannels)
            self._dwStart.wrappedValue = ConvNormAct(
                inChannels: inChannels,
                outChannels: inChannels,
                kernelSize: dwKernelSizeStart,
                stride: dwStartStride,
                padding: (dwKernelSizeStart - 1) / 2,
                dilation: dilation,
                groups: dwStartGroups,
                bias: false,
                applyAct: false,
                eps: 1e-05
            )
        } else {
            self._dwStart.wrappedValue = Identity()
        }

        // PW Expansion
        let midChannels = makeDivisible(Int(Float(inChannels) * expRatio))
        self._pwExp.wrappedValue = ConvNormAct(
            inChannels: inChannels,
            outChannels: midChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: false,
            eps: 1e-05
        )

        // DW Mid
        if dwKernelSizeMid > 0 {
            let dwMidGroups = numGroups(groupSize: groupSize, channels: midChannels)
            self._dwMid.wrappedValue = ConvNormAct(
                inChannels: midChannels,
                outChannels: midChannels,
                kernelSize: dwKernelSizeMid,
                stride: stride,
                padding: (dwKernelSizeMid - 1) / 2,
                dilation: dilation,
                groups: dwMidGroups,
                bias: false,
                eps: 1e-05
            )
        } else {
            self._dwMid.wrappedValue = Identity()
        }

        // PW Projection
        self._pwProj.wrappedValue = ConvNormAct(
            inChannels: midChannels,
            outChannels: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: false,
            applyAct: false,
            eps: 1e-05
        )

        // Layer Scale
        if let layerScaleInitValue {
            self._layerScale.wrappedValue = LayerScale2d(
                dim: outChannels, initValues: layerScaleInitValue)
        } else {
            self._layerScale.wrappedValue = Identity()
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var result = dwStart(x)
        result = pwExp(result)
        result = dwMid(result)
        result = pwProj(result)
        result = layerScale(result)

        if hasSkip {
            result = result + shortcut
        }
        return result
    }
}

// MARK: - Edge Residual
private class EdgeResidual: Module, UnaryLayer {
    let hasSkip: Bool
    @ModuleInfo(key: "conv_exp") var convExp: Conv2d
    @ModuleInfo var bn1: RMSNormAct2d
    @ModuleInfo(key: "conv_pwl") var convPwl: Conv2d
    @ModuleInfo var bn2: RMSNormAct2d

    init(
        inChannels: Int,
        outChannels: Int,
        expKernelSize: Int = 3,
        stride: Int = 1,
        dilation: Int = 1,
        groupSize: Int = 0,
        padType: String = "",
        forceInChannels: Int = 0,
        noskip: Bool = false,
        expandRatio: Float = 1.0,
        pwKernelSize: Int = 1,
        normLayer: RMSNormAct2d.Type = RMSNormAct2d.self
    ) {
        let midChannels: Int
        if forceInChannels > 0 {
            midChannels = makeDivisible(Int(Float(forceInChannels) * expandRatio))
        } else {
            midChannels = makeDivisible(Int(Float(inChannels) * expandRatio))
        }

        let groups = numGroups(groupSize: groupSize, channels: midChannels)
        self.hasSkip = (inChannels == outChannels && stride == 1) && !noskip

        let padding = (expKernelSize - 1) / 2

        self._convExp.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: midChannels,
            kernelSize: IntOrPair(expKernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding),
            dilation: IntOrPair(dilation),
            groups: groups,
            bias: false
        )

        self._bn1.wrappedValue = RMSNormAct2d(numChannels: midChannels, eps: 1e-05)

        // Point-wise linear projection
        let paddingPwl = (pwKernelSize - 1) / 2
        self._convPwl.wrappedValue = Conv2d(
            inputChannels: midChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(pwKernelSize),
            stride: IntOrPair(1),
            padding: IntOrPair(paddingPwl),
            bias: false
        )

        self._bn2.wrappedValue = RMSNormAct2d(
            numChannels: outChannels,
            eps: 1e-05,
            applyAct: false
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var result = convExp(x)
        result = bn1(result)
        result = convPwl(result)
        result = bn2(result)

        if hasSkip {
            result = result + shortcut
        }
        return result
    }
}

// MARK: - Multi-Query Attention 2D
private class MultiQueryAttention2d: Module {
    let numHeads: Int
    let queryStrides: (Int, Int)
    let kvStride: Int
    let fusedAttn: Bool
    let keyDim: Int
    let valueDim: Int
    let scale: Float

    @ModuleInfo(key: "query_proj") var queryProj: Conv2d

    @ModuleInfo(key: "key_down_conv") var keyDownConv: UnaryLayer
    @ModuleInfo(key: "key_norm") var keyNorm: UnaryLayer
    @ModuleInfo(key: "value_down_conv") var valueDownConv: UnaryLayer
    @ModuleInfo(key: "value_norm") var valueNorm: UnaryLayer

    @ModuleInfo(key: "key_proj") var keyProj: Conv2d
    @ModuleInfo(key: "value_proj") var valueProj: Conv2d
    @ModuleInfo(key: "attn_drop") var attnDrop: UnaryLayer
    @ModuleInfo(key: "output_proj") var outputProj: Conv2d
    @ModuleInfo(key: "proj_drop") var projDrop: UnaryLayer

    init(
        dim: Int,
        dimOut: Int? = nil,
        numHeads: Int = 8,
        keyDim: Int = 64,
        valueDim: Int = 64,
        queryStrides: (Int, Int) = (1, 1),
        kvStride: Int = 1,
        dilation: Int = 1,
        padding: String = "",
        dwKernelSize: Int = 3,
        attnDrop: Float = 0.0,
        projDrop: Float = 0.0
    ) {
        let dimOut = dimOut ?? dim
        self.numHeads = numHeads
        self.queryStrides = queryStrides
        self.kvStride = kvStride
        self.fusedAttn = true
        self.keyDim = keyDim
        self.valueDim = valueDim
        let headDim = keyDim
        self.scale = pow(Float(headDim), -0.5)

        // Query
        self._queryProj.wrappedValue = Conv2d(
            inputChannels: dim,
            outputChannels: numHeads * keyDim,
            kernelSize: IntOrPair(1)
        )

        // Key
        if kvStride > 1 {
            self._keyDownConv.wrappedValue = Conv2d(
                inputChannels: dim,
                outputChannels: dim,
                kernelSize: IntOrPair(dwKernelSize),
                stride: IntOrPair(kvStride),
                padding: IntOrPair((dwKernelSize - 1) / 2 * dilation),
                dilation: IntOrPair(dilation),
                groups: dim,  // Depthwise
                bias: false
            )

            self._keyNorm.wrappedValue = RMSNormAct2d(numChannels: dim, eps: 1e-6, applyAct: false)
        } else {
            self._keyDownConv.wrappedValue = Identity()
            self._keyNorm.wrappedValue = Identity()
        }
        self._keyProj.wrappedValue = Conv2d(
            inputChannels: dim,
            outputChannels: keyDim,
            kernelSize: IntOrPair(1),
            bias: false
        )

        // Value
        if kvStride > 1 {
            self._valueDownConv.wrappedValue = Conv2d(
                inputChannels: dim,
                outputChannels: dim,
                kernelSize: IntOrPair(dwKernelSize),
                stride: IntOrPair(kvStride),
                padding: IntOrPair((dwKernelSize - 1) / 2 * dilation),
                dilation: IntOrPair(dilation),
                groups: dim,  // Depthwise
                bias: false
            )
            self._valueNorm.wrappedValue = RMSNormAct2d(
                numChannels: dim, eps: 1e-6, applyAct: false)
        } else {
            self._valueDownConv.wrappedValue = Identity()
            self._valueNorm.wrappedValue = Identity()
        }
        self._valueProj.wrappedValue = Conv2d(
            inputChannels: dim,
            outputChannels: valueDim,
            kernelSize: IntOrPair(1),
            bias: false
        )

        // Attention dropout
        self._attnDrop.wrappedValue = attnDrop > 0 ? Dropout(p: attnDrop) : Identity()

        // Output projection
        self._outputProj.wrappedValue = Conv2d(
            inputChannels: valueDim * numHeads,
            outputChannels: dimOut,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            bias: false
        )

        self._projDrop.wrappedValue = projDrop > 0 ? Dropout(p: projDrop) : Identity()

        super.init()
    }

    private func reshapeInput(_ t: MLXArray) -> MLXArray {
        // Input shape MLX: [B, H, W, C]
        // MLX Reshape: [B, H, W, C] -> [B, -1, C] -> [B, 1, -1, C] -> SDPA
        let s = t.shape
        let reshaped = t.reshaped([s[0], -1, s[3]])
        return expandedDimensions(reshaped, axis: 1)
    }

    private func reshapeProjectedQuery(_ t: MLXArray, numHeads: Int, keyDim: Int) -> MLXArray {
        // Input shape MLX: [B, H, W, C] where C = numHeads * keyDim
        let (B, H, W, C) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3])
        let reshaped = t.reshaped([B, H * W, numHeads, keyDim])
        return reshaped.transposed(0, 2, 1, 3)
    }

    private func reshapeOutput(_ t: MLXArray, numHeads: Int, hPx: Int, wPx: Int) -> MLXArray {
        // Input shape: [B, NH, L, D] where L = hPx * wPx
        // Output shape MLX: [B, H, W, C] where C = NH * D
        let (B, NH, L, D) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3])
        // First transpose to [B, L, NH, D]
        let transposed = t.transposed(0, 2, 1, 3)
        // Then reshape to [B, H, W, NH*D]
        return transposed.reshaped([B, hPx, wPx, NH * D])
    }

    func callAsFunction(_ x: MLXArray, attnMask: MLXArray? = nil) -> MLXArray {
        let (B, H, W, C) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        let q = queryProj(x)
        let qReshaped = reshapeProjectedQuery(q, numHeads: numHeads, keyDim: keyDim)

        var k = keyDownConv(x)
        k = keyNorm(k)
        k = keyProj(k)
        let kReshaped = reshapeInput(k)

        var v = valueDownConv(x)
        v = valueNorm(v)
        v = valueProj(v)
        let vReshaped = reshapeInput(v)

        let o: MLXArray
        if fusedAttn {
            o = MLXFast.scaledDotProductAttention(
                queries: qReshaped,
                keys: kReshaped,
                values: vReshaped,
                scale: scale,
                mask: .none
            )
        } else {
            fatalError("Unfused attention not implemented")
        }

        let oReshaped = reshapeOutput(
            o,
            numHeads: numHeads,
            hPx: H / queryStrides.0,
            wPx: W / queryStrides.1
        )

        return outputProj(oReshaped)
    }
}

// MARK: - Mobile Attention
private class MobileAttention: Module, UnaryLayer {
    let hasSkip: Bool
    let queryStrides: (Int, Int)
    let kvStride: Int
    let hasQueryStride: Bool

    @ModuleInfo var norm: RMSNormAct2d
    @ModuleInfo var attn: MultiQueryAttention2d
    @ModuleInfo(key: "layer_scale") var layerScale: UnaryLayer
    @ModuleInfo(key: "drop_path") var dropPath: Identity

    init(
        inChannels: Int,
        outChannels: Int,
        stride: Int = 1,
        dwKernelSize: Int = 3,
        dilation: Int = 1,
        groupSize: Int = 1,
        padType: String = "",
        numHeads: Int = 8,
        keyDim: Int = 64,
        valueDim: Int = 64,
        useMultiQuery: Bool = true,
        queryStrides: (Int, Int) = (1, 1),
        kvStride: Int = 1,
        cpeDwKernelSize: Int = 3,
        noskip: Bool = false,
        actLayer: Module? = nil,
        aaLayer: Module? = nil,
        dropPathRate: Float = 0.0,
        attnDrop: Float = 0.0,
        projDrop: Float = 0.0,
        layerScaleInitValue: Float? = 1e-5,
        useBias: Bool = false
    ) {
        self.hasSkip = (stride == 1 && inChannels == outChannels) && !noskip
        self.queryStrides = queryStrides
        self.kvStride = kvStride
        self.hasQueryStride = queryStrides.0 > 1 || queryStrides.1 > 1

        // Normalization layer
        self._norm.wrappedValue = RMSNormAct2d(
            numChannels: inChannels,
            eps: 1e-05,
            applyAct: false
        )

        // Attention layer
        if useMultiQuery {
            self._attn.wrappedValue = MultiQueryAttention2d(
                dim: inChannels,
                dimOut: outChannels,
                numHeads: numHeads,
                keyDim: keyDim,
                valueDim: valueDim,
                queryStrides: queryStrides,
                kvStride: kvStride,
                dilation: dilation,
                padding: padType,
                dwKernelSize: dwKernelSize,
                attnDrop: attnDrop,
                projDrop: projDrop
            )
        } else {
            fatalError("Attention not implemented")
        }

        // Layer scaling
        if let layerScaleInitValue {
            self._layerScale.wrappedValue = LayerScale2d(
                dim: outChannels, initValues: layerScaleInitValue)
        } else {
            self._layerScale.wrappedValue = Identity()
        }

        // Drop path for residual connection
        self._dropPath.wrappedValue = Identity()

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var result = norm(x)
        result = attn(result)
        result = layerScale(result)

        // Apply skip connection if available
        if hasSkip {
            result = dropPath(result) + shortcut
        }

        return result
    }
}

// MARK: - Configuration Classes
private struct EdgeResidualConfig {
    let kernelSize: Int
    let filters: Int
    let strides: Int
    let expandRatio: Float
    let isMultiscale: Bool

    init(
        kernelSize: Int = 3, filters: Int = 32, strides: Int = 1, expandRatio: Float = 4.0,
        isMultiscale: Bool = false
    ) {
        self.kernelSize = kernelSize
        self.filters = filters
        self.strides = strides
        self.expandRatio = expandRatio
        self.isMultiscale = isMultiscale
    }
}

private func _er(
    _ kernelSize: Int, _ filters: Int, _ strides: Int = 1, _ expandRatio: Float = 4.0,
    _ isMultiscale: Bool = false
) -> EdgeResidualConfig {
    return EdgeResidualConfig(
        kernelSize: kernelSize, filters: filters, strides: strides, expandRatio: expandRatio,
        isMultiscale: isMultiscale)
}

private struct UniversalInvertedResidualConfig {
    let startDwKernelSize: Int
    let midDwKernelSize: Int
    let filters: Int
    let strides: Int
    let expandRatio: Float
    let isMultiscale: Bool

    init(
        startDwKernelSize: Int, midDwKernelSize: Int, filters: Int, strides: Int = 1,
        expandRatio: Float = 4.0, isMultiscale: Bool = false
    ) {
        self.startDwKernelSize = startDwKernelSize
        self.midDwKernelSize = midDwKernelSize
        self.filters = filters
        self.strides = strides
        self.expandRatio = expandRatio
        self.isMultiscale = isMultiscale
    }
}

private func _uir(
    _ startDwKernelSize: Int, _ midDwKernelSize: Int, _ filters: Int, _ strides: Int = 1,
    _ expandRatio: Float = 4.0, _ isMultiscale: Bool = false
) -> UniversalInvertedResidualConfig {
    return UniversalInvertedResidualConfig(
        startDwKernelSize: startDwKernelSize,
        midDwKernelSize: midDwKernelSize,
        filters: filters,
        strides: strides,
        expandRatio: expandRatio,
        isMultiscale: isMultiscale
    )
}

private struct MultiQueryAttentionBlockConfig {
    let numHeads: Int
    let kvDim: Int
    let kvStrides: Int
    let mmqaAvgPoolKv: Bool
    let isMultiscale: Bool

    init(
        numHeads: Int = 8, kvDim: Int = 16, kvStrides: Int = 1, mmqaAvgPoolKv: Bool = false,
        isMultiscale: Bool = false
    ) {
        self.numHeads = numHeads
        self.kvDim = kvDim
        self.kvStrides = kvStrides
        self.mmqaAvgPoolKv = mmqaAvgPoolKv
        self.isMultiscale = isMultiscale
    }
}

private func _mmqa(
    _ numHeads: Int, _ kvDim: Int, _ kvStrides: Int, _ mmqaAvgPoolKv: Bool = false,
    _ isMultiscale: Bool = false
) -> MultiQueryAttentionBlockConfig {
    return MultiQueryAttentionBlockConfig(
        numHeads: numHeads,
        kvDim: kvDim,
        kvStrides: kvStrides,
        mmqaAvgPoolKv: mmqaAvgPoolKv,
        isMultiscale: isMultiscale
    )
}

// MARK: - MobileNet Definition
private func gemma3nMobilenetDef() -> [[Any]] {
    return [
        // Stage 1: Edge Residuals
        [_er(3, 128, 2)] + Array(repeating: _er(3, 128, 1), count: 2),
        // Stage 2: Universal Inverted Residuals
        [_uir(3, 5, 256, 2, 6.0)] + [5, 3, 5, 3].map { _uir($0, 0, 256) },
        // Stage 3: Universal Inverted Residuals with Multi-Query Attention
        [_uir(5, 5, 640, 2, 6.0)]
            + Array(repeating: _uir(5, 0, 640), count: 7)
            + [_uir(0, 0, 640, 1, 1.0)]
            + Array(repeating: [_mmqa(12, 64, 2), _uir(0, 0, 640, 1, 2.0)], count: 13).flatMap {
                $0
            }
            + [_mmqa(12, 64, 2), _uir(0, 0, 640, 1, 2.0, true)],
        // Stage 4: Universal Inverted Residuals with Multi-Query Attention
        [_uir(5, 5, 1280, 2, 6.0)]
            + Array(repeating: [_mmqa(16, 96, 1), _uir(0, 0, 1280, 1, 2.0)], count: 18).flatMap {
                $0
            }
            + [_mmqa(16, 96, 1), _uir(0, 0, 1280, 1, 2.0, true)],
    ]
}

// MARK: - Multi-Scale Fusion Adapter
private class MobileNetV5MultiScaleFusionAdapter: Module {
    let inChannels: Int
    let outChannels: Int
    let outputResolution: (Int, Int)
    let expansionRatio: Float
    let interpolationMode: String
    let useLayerScale: Bool
    let layerScaleInitValue: Float
    let noskip: Bool

    @ModuleInfo var ffn: UniversalInvertedResidual
    @ModuleInfo var norm: RMSNormAct2d
    @ModuleInfo(key: "avg_pool") var avgPool: AvgPool2d

    init(
        inChannels: [Int],
        outChannels: Int,
        outputResolution: Int,
        expansionRatio: Float = 2.0,
        interpolationMode: String = "nearest",
        useLayerScale: Bool = false,
        layerScaleInitValue: Float = 1e-5,
        noskip: Bool = true
    ) {
        let inChannelsSum = inChannels.reduce(0, +)
        self.inChannels = inChannelsSum
        self.outChannels = outChannels
        self.outputResolution = to2Tuple(outputResolution)
        self.expansionRatio = expansionRatio
        self.interpolationMode = interpolationMode
        self.useLayerScale = useLayerScale
        self.layerScaleInitValue = layerScaleInitValue
        self.noskip = noskip

        self._ffn.wrappedValue = UniversalInvertedResidual(
            inChannels: inChannelsSum,
            outChannels: outChannels,
            dwKernelSizeStart: 0,
            dwKernelSizeMid: 0,
            noskip: noskip,
            expRatio: expansionRatio,
            layerScaleInitValue: useLayerScale ? layerScaleInitValue : nil
        )

        self._norm.wrappedValue = RMSNormAct2d(numChannels: outChannels, eps: 1e-6, applyAct: false)

        // For simplicity, using AvgPool2d for downsampling
        self._avgPool.wrappedValue = AvgPool2d(kernelSize: IntOrPair(2), stride: IntOrPair(2))

        super.init()
    }

    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        // Convert from NHWC to NCHW for processing
        let inputsNCHW = inputs.map { $0.transposed(0, 3, 1, 2) }

        // Find the highest resolution (first input)
        let highResolution = inputsNCHW[0].shape.suffix(2)
        var resizedInputs: [MLXArray] = []

        for img in inputsNCHW {
            let imgShape = img.shape.suffix(2)
            var resizedImg = img

            // Resize if needed using nearest neighbor interpolation
            if imgShape[0] < highResolution[0] || imgShape[1] < highResolution[1] {
                // Simple nearest neighbor interpolation
                let scaleH = Float(highResolution[0]) / Float(imgShape[0])
                let scaleW = Float(highResolution[1]) / Float(imgShape[1])
                // For simplicity, just repeat the image - in practice you'd implement proper interpolation
                resizedImg = img
            }

            resizedInputs.append(resizedImg)
        }

        // Concatenate on channel dimension
        let channelCatImgs = concatenated(resizedInputs, axis: 1)

        // Convert back to NHWC for MLX processing
        let channelCatImgsNHWC = channelCatImgs.transposed(0, 2, 3, 1)
        var img = ffn(channelCatImgsNHWC)

        // Handle output resolution adjustment
        let currentResolution = (img.shape[1], img.shape[2])
        if currentResolution.0 != outputResolution.0 || currentResolution.1 != outputResolution.1 {
            if currentResolution.0 % outputResolution.0 != 0
                || currentResolution.1 % outputResolution.1 != 0
            {
                // Use bicubic interpolation to match Python implementation
                img = bicubicInterpolate(img, to: outputResolution)
            } else {
                let hStrides = currentResolution.0 / outputResolution.0
                let wStrides = currentResolution.1 / outputResolution.1

                // Convert to NCHW for AvgPool2d
                let imgNCHW = img.transposed(0, 3, 1, 2)
                let pooled = AvgPool2d(
                    kernelSize: IntOrPair(hStrides),
                    stride: IntOrPair(hStrides)
                )(imgNCHW)
                img = pooled.transposed(0, 2, 3, 1)
            }

            img = noskip ? norm(img) : img
        }

        return img
    }
}

// MARK: - Vision Tower
private class VisionTower: Module {
    @ModuleInfo(key: "conv_stem") var convStem: ConvNormAct
    @ModuleInfo var blocks: [[UnaryLayer]]
    @ModuleInfo var msfa: MobileNetV5MultiScaleFusionAdapter

    let numFeatures: Int
    let headHiddenSize: Int
    let msfaIndices: (Int, Int)
    let msfaOutputResolution: (Int, Int)

    init(config: VisionConfig) {
        self._convStem.wrappedValue = ConvNormAct(
            inChannels: 3,
            outChannels: 64,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            eps: 1e-05
        )

        self.msfaIndices = (3, 4)
        self.msfaOutputResolution = (16, 16)

        let (numFeatures, blocks) = Self.buildBlocks(convStemOutChannels: 64)
        self.numFeatures = numFeatures
        self.headHiddenSize = numFeatures
        self._blocks.wrappedValue = blocks

        self._msfa.wrappedValue = MobileNetV5MultiScaleFusionAdapter(
            inChannels: [1920],
            outChannels: 2048,
            outputResolution: msfaOutputResolution.0
        )

        super.init()
    }

    static func buildBlocks(convStemOutChannels: Int) -> (Int, [[UnaryLayer]]) {
        var blocks: [[UnaryLayer]] = []
        var inChannels = convStemOutChannels

        for (stage, blockConfigs) in gemma3nMobilenetDef().enumerated() {
            var blockGroup: [UnaryLayer] = []

            for config in blockConfigs {
                if let edgeConfig = config as? EdgeResidualConfig {
                    let block = EdgeResidual(
                        inChannels: inChannels,
                        outChannels: edgeConfig.filters,
                        expKernelSize: edgeConfig.kernelSize,
                        stride: edgeConfig.strides,
                        expandRatio: edgeConfig.expandRatio
                    )
                    inChannels = edgeConfig.filters
                    blockGroup.append(block)
                } else if let uirConfig = config as? UniversalInvertedResidualConfig {
                    let block = UniversalInvertedResidual(
                        inChannels: inChannels,
                        outChannels: uirConfig.filters,
                        dwKernelSizeStart: uirConfig.startDwKernelSize,
                        dwKernelSizeMid: uirConfig.midDwKernelSize,
                        stride: uirConfig.strides,
                        expRatio: uirConfig.expandRatio
                    )
                    inChannels = uirConfig.filters
                    blockGroup.append(block)
                } else if let attentionConfig = config as? MultiQueryAttentionBlockConfig {
                    let block = MobileAttention(
                        inChannels: inChannels,
                        outChannels: inChannels,
                        stride: 1,
                        numHeads: attentionConfig.numHeads,
                        keyDim: attentionConfig.kvDim,
                        valueDim: attentionConfig.kvDim,
                        kvStride: attentionConfig.kvStrides,
                        actLayer: nil
                    )
                    blockGroup.append(block)
                }
            }
            blocks.append(blockGroup)
        }

        return (inChannels, blocks)
    }

    func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> MLXArray {
        var featIdx = 0
        // Convert from NCHW to NHWC
        var result = x.transposed(0, 2, 3, 1)
        result = convStem(result)
        var intermediates: [MLXArray] = []

        if msfaIndices.0 == featIdx || msfaIndices.1 == featIdx {
            intermediates.append(result)
        }

        // MBV5 is constructed of 4 stages, each stage is a group of blocks
        for blockGroup in blocks {
            featIdx += 1
            for block in blockGroup {
                result = block(result)
            }

            if msfaIndices.0 == featIdx || msfaIndices.1 == featIdx {
                intermediates.append(result)
            }
        }

        result = msfa(intermediates)
        return result
    }
}

// MARK: - Complete Vision Model
private class Gemma3nVisionModel: Module {
    let modelType: String
    @ModuleInfo(key: "timm_model") var timmModel: VisionTower

    init(config: VisionConfig) {
        self.modelType = config.modelType
        self._timmModel.wrappedValue = VisionTower(config: config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> MLXArray {
        return timmModel(x, outputHiddenStates: outputHiddenStates)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        var skipTranspose = false
        let testKey = "vision_tower.timm_model.blocks.0.0.conv_exp.weight"
        if let convWeight = weights[testKey], convWeight.ndim == 4,
            convWeight.shape[3] > convWeight.shape[1]
        {
            skipTranspose = true
        }
        for (k, v) in weights {
            if (k.contains("conv") && k.contains("weight"))
                || (k.contains("attn") && k.contains("proj.weight"))
            {
                if v.ndim == 4 && !skipTranspose {
                    sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                }
            }
        }
        return sanitizedWeights
    }
}

// MARK: - Complete Audio Model
private class Gemma3nAudioModel: Module {
    let config: AudioConfig

    @ModuleInfo(key: "subsample_conv_projection") var subsampleConvProjection:
        Gemma3nAudioSubSampleConvProjection
    @ModuleInfo var conformer: [Gemma3nAudioConformerBlock]

    init(config: AudioConfig) {
        self.config = config

        self._subsampleConvProjection.wrappedValue = Gemma3nAudioSubSampleConvProjection(
            config: config)

        self._conformer.wrappedValue = (0 ..< config.confNumHiddenLayers).map { i in
            return Gemma3nAudioConformerBlock(config: config)
        }

        super.init()
    }

    func callAsFunction(
        _ audioMel: MLXArray,
        _ audioMelMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        var audioencodings = subsampleConvProjection(audioMel)

        // Subsample the input audio_mel_mask to match the time dimension
        let tSub = audioencodings.shape[1]

        var timeStrideProduct = 1
        for stridePairIdx in 0 ..< config.sscpConvStrideSize.count {
            timeStrideProduct *= config.sscpConvStrideSize[stridePairIdx][0]
        }

        let indices = MLXArray(0 ..< tSub) * timeStrideProduct
        let clippedIndices = clip(indices, max: MLXArray(audioMelMask.shape[1] - 1))

        var currentMask: MLXArray
        if audioMelMask.ndim > 1 && clippedIndices.ndim == 1 {
            let expandedIndices = expandedDimensions(clippedIndices, axis: 0)
            let broadcastIndices = broadcast(
                expandedIndices,
                to: [audioMelMask.shape[0], clippedIndices.shape[0]]
            )
            currentMask = take(audioMelMask, broadcastIndices.asType(.int32), axis: 1)
        } else {
            currentMask = take(audioMelMask, clippedIndices.asType(.int32), axis: 1)
        }

        // Adjust mask length if needed
        if currentMask.shape[1] != tSub {
            if currentMask.shape[1] > tSub {
                currentMask = currentMask[0..., ..<tSub]
            } else {
                let paddingNeeded = tSub - currentMask.shape[1]
                currentMask = padded(
                    currentMask,
                    widths: convertTorchToMLXPadWidth([0, paddingNeeded], Array(currentMask.shape))
                )
            }
        }

        for block in conformer {
            audioencodings = block(audioencodings, currentMask)
        }

        if config.confReductionFactor > 1 {
            let stride = config.confReductionFactor
            audioencodings = audioencodings[0..., 0 ..< audioencodings.shape[1], stride, 0...]
            currentMask = currentMask[0..., 0 ..< currentMask.shape[1], stride]
        }

        // Final masking
        if currentMask.shape[1] != audioencodings.shape[1] {
            let targetLen = audioencodings.shape[1]
            let maskCurrentLen = currentMask.shape[1]
            if targetLen > maskCurrentLen {
                let paddingNeeded = targetLen - maskCurrentLen
                currentMask = padded(
                    currentMask,
                    widths: convertTorchToMLXPadWidth([0, paddingNeeded], Array(currentMask.shape))
                )
            } else if maskCurrentLen > targetLen {
                currentMask = currentMask[0..., ..<targetLen]
            }
        }

        audioencodings = MLX.where(
            expandedDimensions(currentMask, axis: -1),
            MLXArray(0.0),
            audioencodings
        )

        return (audioencodings, currentMask)
    }

    /// Sanitizes weights by transposing convolution layers if they are not
    /// already in the expected MLX format.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        // Iterate over the original keys to decide which ones to modify in the copy.
        for (k, v) in weights {
            if k.contains("conv.weight") {
                if checkArrayShape(v) {
                    sanitizedWeights[k] = v
                } else {
                    sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                }
            } else if k.contains("conv1d.weight") {
                if true {
                    sanitizedWeights[k] = v
                } else {
                    sanitizedWeights[k] = v.transposed(0, 2, 1)
                }
            } else {
                sanitizedWeights[k] = v
            }
        }
        return sanitizedWeights
    }
}

// MARK: - LoRA Support

extension Gemma3n: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        return languageModel.model.layers.map { layer in
            (layer.selfAttn, ["q_proj", "v_proj"])
        }
    }
}

// MARK: - VLM Factory Configuration and Processor

public struct Gemma3nConfiguration: Codable, Sendable {
    public let textConfig: TextConfig
    public let visionConfig: VisionConfig
    public let audioConfig: AudioConfig
    public let modelType: String
    private let _vocabSize: Int?
    private let _ignoreIndex: Int?
    private let _imageTokenIndex: Int?
    private let _audioTokenId: Int?
    private let _imageTokenId: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?
    private let _visionSoftTokensPerImage: Int?
    private let _audioSoftTokensPerImage: Int?
    public let eosTokenId: [Int]?
    public let quantization: QuantizationConfig?

    // Computed properties with defaults
    public var vocabSize: Int {
        _vocabSize ?? 257152
    }

    public var ignoreIndex: Int {
        _ignoreIndex ?? -100
    }

    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 262145
    }

    public var audioTokenId: Int {
        _audioTokenId ?? 262273
    }

    public var imageTokenId: Int {
        _imageTokenId ?? 262145
    }

    public var hiddenSize: Int {
        _hiddenSize ?? 2048
    }

    public var padTokenId: Int {
        _padTokenId ?? 0
    }

    public var visionSoftTokensPerImage: Int {
        _visionSoftTokensPerImage ?? 256
    }

    public var audioSoftTokensPerImage: Int {
        _audioSoftTokensPerImage ?? 188
    }

    public var vocabularySize: Int { vocabSize }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
        case modelType = "model_type"
        case _vocabSize = "vocab_size"
        case _ignoreIndex = "ignore_index"
        case _imageTokenIndex = "image_token_index"
        case _audioTokenId = "audio_token_id"
        case _imageTokenId = "image_token_id"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
        case _visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case _audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case eosTokenId = "eos_token_id"
        case quantization
    }

    public init(from modelConfig: ModelConfig, quantization: QuantizationConfig? = nil) {
        self.textConfig = modelConfig.textConfig
        self.visionConfig = modelConfig.visionConfig
        self.audioConfig = modelConfig.audioConfig
        self.modelType = modelConfig.modelType
        self._vocabSize = modelConfig.vocabSize
        self._ignoreIndex = modelConfig.ignoreIndex
        self._imageTokenIndex = modelConfig.imageTokenIndex
        self._audioTokenId = modelConfig.audioTokenId
        self._imageTokenId = modelConfig.imageTokenId
        self._hiddenSize = modelConfig.hiddenSize
        self._padTokenId = modelConfig.padTokenId
        self._visionSoftTokensPerImage = modelConfig.visionSoftTokensPerImage
        self._audioSoftTokensPerImage = modelConfig.audioSoftTokensPerImage
        self.eosTokenId = modelConfig.eosTokenId
        self.quantization = quantization
    }
}

public class Gemma3nProcessor: UserInputProcessor {
    private let config: Gemma3nProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma3nProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        var userProcessing = processing ?? UserInput.Processing()
        let targetSize = CGSize(width: config.imageSize, height: config.imageSize)
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
        // Create structured messages for Gemma3n using LIST_WITH_IMAGE_TYPE_TEXT format
        var messages: [[String: Any]] = []

        if !input.images.isEmpty {
            // Add image and text content in the format expected by Gemma3n
            let content: [[String: Any]] = [
                ["type": "image"],
                ["type": "text", "text": input.prompt.description],
            ]
            messages.append(["role": "user", "content": content])
        } else {
            // Text-only message
            messages.append(["role": "user", "content": input.prompt.description])
        }

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

            // Note: Unlike Gemma3, Gemma3n doesn't expand image tokens in the processor
            // The model handles token mapping directly in get_input_embeddings
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage
        )
    }
}

public struct Gemma3nProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let imageProcessorType: String?
    public let doNormalize: Bool
    public let doRescale: Bool
    public let doResize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let visionSoftTokensPerImage: Int
    public let resample: Int
    public let rescaleFactor: Float
    public let size: ImageSize

    // Optional fields that may be present in some configs
    public let doConvertRgb: Bool?
    public let doPanAndScan: Bool?

    public var imageTokenId: Int { 262145 }
    public var audioTokenId: Int { 262273 }

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
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case resample
        case rescaleFactor = "rescale_factor"
        case size
    }
}

extension Gemma3n {
    public convenience init(_ config: Gemma3nConfiguration) {
        let modelConfig = ModelConfig(
            textConfig: config.textConfig,
            visionConfig: config.visionConfig,
            audioConfig: config.audioConfig,
            modelType: config.modelType,
            eosTokenId: config.eosTokenId
        )
        self.init(modelConfig)
    }
}
