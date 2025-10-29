//
//  Gemma3nText.swift
//  mlx-swift-examples
//
//  Created by Max Kupriianov on 28.06.2025.
//

// Based on https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gemma3n.py

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma3nTextConfiguration: Codable {
    let modelType: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabSize: Int
    let numKeyValueHeads: Int
    let numKvSharedLayers: Int
    let queryPreAttnScalar: Float
    let vocabSizePerLayerInput: Int
    let slidingWindow: Int
    let maxPositionEmbeddings: Int
    let ropeLocalBaseFreq: Float
    let ropeTheta: Float
    let finalLogitSoftcapping: Float
    let layerTypes: [String]?
    let activationSparsityPattern: [Float]?
    let hiddenSizePerLayerInput: Int
    let altupNumInputs: Int
    let altupCoefClip: Float?
    let altupCorrectScale: Bool
    let altupActiveIdx: Int
    let laurelRank: Int
    let ropeScaling: [String: String]?
    let slidingWindowPattern: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTheta = "rope_theta"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case layerTypes = "layer_types"
        case activationSparsityPattern = "activation_sparsity_pattern"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case altupNumInputs = "altup_num_inputs"
        case altupCoefClip = "altup_coef_clip"
        case altupCorrectScale = "altup_correct_scale"
        case altupActiveIdx = "altup_active_idx"
        case laurelRank = "laurel_rank"
        case ropeScaling = "rope_scaling"
        case slidingWindowPattern = "sliding_window_pattern"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        // in the case of Gemma 3n model, the configuration matches VLMs and text config is under a text_config key
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        numKvSharedLayers = try container.decode(Int.self, forKey: .numKvSharedLayers)
        queryPreAttnScalar = try container.decode(Float.self, forKey: .queryPreAttnScalar)
        vocabSizePerLayerInput = try container.decode(Int.self, forKey: .vocabSizePerLayerInput)
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        ropeLocalBaseFreq = try container.decode(Float.self, forKey: .ropeLocalBaseFreq)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        finalLogitSoftcapping = try container.decode(Float.self, forKey: .finalLogitSoftcapping)
        layerTypes = try container.decode([String]?.self, forKey: .layerTypes)
        activationSparsityPattern = try container.decodeIfPresent(
            [Float].self, forKey: .activationSparsityPattern)
        hiddenSizePerLayerInput = try container.decode(Int.self, forKey: .hiddenSizePerLayerInput)
        altupNumInputs = try container.decode(Int.self, forKey: .altupNumInputs)
        altupCoefClip = try container.decodeIfPresent(Float.self, forKey: .altupCoefClip)
        altupCorrectScale = try container.decode(Bool.self, forKey: .altupCorrectScale)
        altupActiveIdx = try container.decode(Int.self, forKey: .altupActiveIdx)
        laurelRank = try container.decode(Int.self, forKey: .laurelRank)
        ropeScaling = try container.decodeIfPresent([String: String].self, forKey: .ropeScaling)
        slidingWindowPattern = try container.decodeIfPresent(
            Int.self, forKey: .slidingWindowPattern)
    }
}

private class RMSNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private class Gemma3nTextLaurelBlock: Module {
    @ModuleInfo(key: "linear_left") var linearLeft: Linear
    @ModuleInfo(key: "linear_right") var linearRight: Linear
    @ModuleInfo(key: "post_laurel_norm") var postLaurelNorm: RMSNorm

    init(_ config: Gemma3nTextConfiguration) {
        _linearLeft.wrappedValue = Linear(config.hiddenSize, config.laurelRank, bias: false)
        _linearRight.wrappedValue = Linear(config.laurelRank, config.hiddenSize, bias: false)
        _postLaurelNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var laurelX = linearLeft(x)
        laurelX = linearRight(laurelX)
        let normedLaurelX = postLaurelNorm(laurelX)
        return x + normedLaurelX
    }
}

private class Gemma3nAttention: Module {
    let isSliding: Bool
    let numHeads: Int
    let numKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isKvSharedLayer: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: RMSNoScale
    @ModuleInfo var rope: RoPE

    init(_ config: Gemma3nTextConfiguration, layerIdx: Int) {
        let layerTypes =
            config.layerTypes ?? Array(repeating: "global_attention", count: config.numHiddenLayers)
        self.isSliding = layerTypes[layerIdx] == "sliding_attention"

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

        self._qNorm.wrappedValue = RMSNorm(
            dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(
            dimensions: config.headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = RMSNoScale(eps: config.rmsNormEps)

        let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        self.isKvSharedLayer = layerIdx >= firstKvSharedLayerIdx

        // Use appropriate RoPE base frequency for sliding vs global attention
        let baseFreq = isSliding ? config.ropeLocalBaseFreq : config.ropeTheta
        self._rope.wrappedValue = RoPE(
            dimensions: headDim,
            traditional: false,
            base: baseFreq
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x)
        queries = queries.reshaped(B, L, -1, headDim)
        queries = qNorm(queries)

        let offset =
            if isKvSharedLayer && cache != nil {
                cache!.offset
            } else {
                cache?.offset ?? 0
            }

        var keys: MLXArray
        var values: MLXArray

        if isKvSharedLayer && cache != nil {
            let state = cache!.state
            if state.count >= 2 {
                keys = state[0]
                values = state[1]
            } else {
                keys = kProj(x).reshaped(B, L, -1, headDim)
                keys = kNorm(keys)
                keys = keys.transposed(0, 2, 1, 3)
                keys = rope(keys, offset: offset)

                values = vProj(x).reshaped(B, L, -1, headDim)
                values = vNorm(values)
                values = values.transposed(0, 2, 1, 3)

                if let cache = cache {
                    (keys, values) = cache.update(keys: keys, values: values)
                }
            }
        } else {
            keys = kProj(x).reshaped(B, L, -1, headDim)
            keys = kNorm(keys)
            keys = keys.transposed(0, 2, 1, 3)
            keys = rope(keys, offset: offset)

            values = vProj(x).reshaped(B, L, -1, headDim)
            values = vNorm(values)
            values = values.transposed(0, 2, 1, 3)

            if let cache = cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.shape[keys.shape.count - 2]
            if maskArray.shape.last! != keysSeqLen {
                let slicedMask = maskArray[.ellipsis, 0 ..< keysSeqLen].asType(queries.dtype)
                adjustedMask = .array(slicedMask)
            } else {
                adjustedMask = .array(maskArray.asType(queries.dtype))
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: adjustedMask ?? .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let config: Gemma3nTextConfiguration
    let hiddenSize: Int
    let intermediateSize: Int
    let activationSparsity: Float
    @ModuleInfo private var _stdMultiplier: MLXArray?

    init(_ config: Gemma3nTextConfiguration, layerIdx: Int) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.intermediateSize = config.intermediateSize

        if let activationSparsityPattern = config.activationSparsityPattern {
            self.activationSparsity = activationSparsityPattern[layerIdx]
        } else {
            self.activationSparsity = 0.0
        }

        if self.activationSparsity > 0 {
            self._stdMultiplier =
                sqrt(MLXArray(2.0)) * erfInverse(2 * MLXArray(self.activationSparsity) - 1)
        } else {
            self._stdMultiplier = nil
        }

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateProj = self.gateProj(x)
        let activations: MLXArray
        if activationSparsity > 0.0, let stdMultiplier = self._stdMultiplier {
            activations = geluTopK(gateProj, stdMultiplier: stdMultiplier)
        } else {
            activations = geluApproximate(gateProj)
        }
        let upProj = self.upProj(x)
        let downProj = self.downProj(activations * upProj)
        return downProj
    }

    private func geluTopK(_ inputs: MLXArray, stdMultiplier: MLXArray) -> MLXArray {
        let inputsMean = mean(inputs, axis: -1, keepDims: true)
        let inputsStd = std(inputs, axis: -1, keepDims: true)
        let cutoffX = inputsMean + inputsStd * stdMultiplier.asType(inputsStd.dtype)
        return geluApproximate(maximum(MLXArray(0), inputs - cutoffX))
    }
}

private class Gemma3nAltUp: Module {
    @ModuleInfo(key: "correct_output_scale") var correctOutputScale: MLXArray
    @ModuleInfo(key: "correction_coefs") var correctionCoefs: Linear
    @ModuleInfo(key: "prediction_coefs") var predictionCoefs: Linear
    @ModuleInfo(key: "modality_router") var modalityRouter: Linear
    @ModuleInfo(key: "router_norm") var routerNorm: RMSNorm
    private let _routerInputScale: MLXArray

    let config: Gemma3nTextConfiguration

    init(config: Gemma3nTextConfiguration) {
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
        self._routerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._routerInputScale = MLXArray(pow(Float(config.hiddenSize), -1.0))

        super.init()
    }

    func computeRouterModalities(_ x: MLXArray) -> MLXArray {
        let routerInputs = routerNorm(x) * _routerInputScale.asType(routerNorm.weight.dtype)
        let routed = modalityRouter(routerInputs).asType(.float32)
        return tanh(routed)
    }

    func predict(_ x: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(x[config.altupActiveIdx])

        var predictionWeight = predictionCoefs.weight.asType(.float32)

        if let altupCoefClip = config.altupCoefClip {
            predictionWeight = clip(
                predictionWeight,
                min: MLXArray(-altupCoefClip),
                max: MLXArray(altupCoefClip)
            )
        }

        let rawOutput = matmul(modalities, predictionWeight.transposed())
        let allCoefs =
            rawOutput
            .reshaped(
                Array(modalities.shape.dropLast()) + [config.altupNumInputs, config.altupNumInputs]
            )
            .transposed(0, 1, 3, 2)

        let xUp = x.asType(.float32)
        let xPermuted = xUp.transposed(1, 2, 3, 0)
        let predictions = matmul(xPermuted, allCoefs)
        let predictionsPermuted = predictions.transposed(3, 0, 1, 2)
        let finalPredictions = predictionsPermuted + xUp
        return finalPredictions.asType(x.dtype)
    }

    func correct(predictions: MLXArray, activated: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(activated)

        var correctionWeight = correctionCoefs.weight.asType(.float32)

        if let altupCoefClip = config.altupCoefClip {
            correctionWeight = clip(
                correctionWeight,
                min: MLXArray(-altupCoefClip),
                max: MLXArray(altupCoefClip)
            )
        }

        let allCoefs = matmul(modalities, correctionWeight.transposed()) + 1.0

        let activeX = predictions[config.altupActiveIdx]
        let innovation = activated - activeX

        let allCoefsTransposed = allCoefs.transposed(2, 1, 0)
        let corrected =
            expandedDimensions(innovation, axis: 0)
            * expandedDimensions(allCoefsTransposed, axis: 1)
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
    let config: Gemma3nTextConfiguration
    let hiddenSize: Int
    let layerIdx: Int
    let isSliding: Bool
    let slidingWindow: Int
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma3nAttention
    @ModuleInfo var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm
    @ModuleInfo var altup: Gemma3nAltUp
    @ModuleInfo var laurel: Gemma3nTextLaurelBlock
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm

    init(_ config: Gemma3nTextConfiguration, layerIdx: Int) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._selfAttn.wrappedValue = Gemma3nAttention(config, layerIdx: layerIdx)
        self.isSliding =
            (config.layerTypes
            ?? Array(repeating: "global_attention", count: config.numHiddenLayers))[layerIdx]
            == "sliding_attention"

        self._mlp.wrappedValue = MLP(config, layerIdx: layerIdx)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize,
            eps: config.rmsNormEps
        )

        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize,
            eps: config.rmsNormEps
        )
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize,
            eps: config.rmsNormEps
        )
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize,
            eps: config.rmsNormEps
        )

        self._altup.wrappedValue = Gemma3nAltUp(config: config)
        self._laurel.wrappedValue = Gemma3nTextLaurelBlock(config)

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
        self._postPerLayerInputNorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize,
            eps: config.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        caches: [KVCache?]? = nil,
        cachePosition: MLXArray? = nil
    ) -> MLXArray {
        var x = x
        if x.ndim == 1 {
            x = expandedDimensions(x, axis: 0)
        }

        var finalMask = mask
        if isSliding, case .array(let maskArray) = mask {
            let effectiveSeqLen = max(cachePosition?.shape[0] ?? 0, slidingWindow)
            let minDtype = MLXArray(Float.leastNormalMagnitude, dtype: maskArray.dtype)

            let slidingWindowMask = tril(
                MLXArray.ones(maskArray.shape, dtype: .bool),
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

        let attn = selfAttn(
            activePredictionNormed,
            mask: finalMask,
            cache: cache
        )

        let attnNormed = postAttentionLayernorm(attn)
        let attnGated = activePrediction + attnNormed
        let attnLaurel =
            (attnGated + laurelOutput) * rsqrt(MLXArray(2.0, dtype: activePrediction.dtype))

        let attnNormFf = preFeedforwardLayernorm(attnLaurel)
        let attnFfw = mlp(attnNormFf)
        let attnFfwNorm = postFeedforwardLayernorm(attnFfw)
        let attnFfwLaurelGated = attnLaurel + attnFfwNorm

        let correctedPredictions = altup.correct(
            predictions: predictions, activated: attnFfwLaurelGated)

        var firstPrediction = correctedPredictions[config.altupActiveIdx]
        if config.altupCorrectScale {
            firstPrediction = firstPrediction * altup.correctOutputScale
        }

        firstPrediction = perLayerInputGate(firstPrediction)
        firstPrediction = geluApproximate(firstPrediction)

        guard let perLayerInput = perLayerInput else {
            fatalError(
                "per_layer_input is required but was nil. This should never happen in normal operation."
            )
        }
        firstPrediction = firstPrediction * perLayerInput

        firstPrediction = perLayerProjection(firstPrediction)
        firstPrediction = postPerLayerInputNorm(firstPrediction)

        let result = correctedPredictions
        result[1...] = result[1...] + firstPrediction

        return result
    }
}

private class LanguageModel: Module {
    let config: Gemma3nTextConfiguration
    let hiddenSize: Int
    let vocabSize: Int
    let vocabSizePerLayerInput: Int
    let numHiddenLayers: Int
    let firstKvSharedLayerIdx: Int
    let firstSlidingIdx: Int
    let firstFullIdx: Int
    let layerIdxToCacheIdx: [Int]
    let finalLogitSoftcapping: Float?
    private let _perLayerProjectionScale: MLXArray
    private let _perLayerInputScale: MLXArray
    private let _embedTokensScale: Float
    private let _embedTokensPerLayerScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma3nDecoderLayer]
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm

    @ModuleInfo(key: "altup_projections") var altupProjections: [Linear]
    @ModuleInfo(key: "altup_unembed_projections") var altupUnembedProjections: [Linear]

    @ModuleInfo var norm: RMSNorm

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        var caches: [any KVCache] = []
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        let layerTypes =
            config.layerTypes ?? Array(repeating: "global_attention", count: config.numHiddenLayers)

        for i in 0 ..< firstKvSharedLayerIdx {
            let layerType = layerTypes[i]
            if layerType == "full_attention" {
                caches.append(StandardKVCache())
            } else if layerType == "sliding_attention" {
                caches.append(RotatingKVCache(maxSize: slidingWindow, keep: 0))
            } else {
                fatalError("Unknown layer type: \(layerType) for layer \(i)")
            }
        }
        return caches
    }

    init(_ config: Gemma3nTextConfiguration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.vocabSize = config.vocabSize
        self.vocabSizePerLayerInput = config.vocabSizePerLayerInput
        self.numHiddenLayers = config.numHiddenLayers
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self.firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers

        let layerTypes =
            config.layerTypes ?? Array(repeating: "global_attention", count: config.numHiddenLayers)

        guard let firstSlidingIdx = layerTypes.firstIndex(of: "sliding_attention") else {
            fatalError("Layer type 'sliding_attention' not found in layer_types")
        }
        guard let firstFullIdx = layerTypes.firstIndex(of: "full_attention") else {
            fatalError("Layer type 'full_attention' not found in layer_types")
        }
        self.firstSlidingIdx = firstSlidingIdx
        self.firstFullIdx = firstFullIdx

        var layerIdxToCacheIdx: [Int] = []
        let concreteLayerTypes = Array(layerTypes[..<firstKvSharedLayerIdx])
        let sharedFullIdx = concreteLayerTypes.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0

        for (i, layerType) in layerTypes.enumerated() {
            if i < firstKvSharedLayerIdx {
                layerIdxToCacheIdx.append(i)
            } else {
                if layerType == "full_attention" {
                    layerIdxToCacheIdx.append(sharedFullIdx)
                } else if layerType == "sliding_attention" {
                    layerIdxToCacheIdx.append(sharedSlidingIdx)
                } else {
                    fatalError("Unknown layer type: \(layerType)")
                }
            }
        }
        self.layerIdxToCacheIdx = layerIdxToCacheIdx

        assert(vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._embedTokensScale = pow(Float(config.hiddenSize), 0.5)

        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { layerIdx in
            Gemma3nDecoderLayer(config, layerIdx: layerIdx)
        }

        self._embedTokensPerLayer.wrappedValue = Embedding(
            embeddingCount: config.vocabSizePerLayerInput,
            dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput
        )
        self._embedTokensPerLayerScale = pow(Float(config.hiddenSizePerLayerInput), 0.5)

        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numHiddenLayers * config.hiddenSizePerLayerInput,
            bias: false
        )

        self._perLayerProjectionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSizePerLayerInput,
            eps: config.rmsNormEps
        )

        self._altupProjections.wrappedValue = (0 ..< (config.altupNumInputs - 1)).map { _ in
            Linear(config.hiddenSize, config.hiddenSize, bias: false)
        }
        self._altupUnembedProjections.wrappedValue = (0 ..< (config.altupNumInputs - 1)).map { _ in
            Linear(config.hiddenSize, config.hiddenSize, bias: false)
        }

        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )

        self._perLayerProjectionScale = MLXArray(pow(Float(hiddenSize), -0.5))
        self._perLayerInputScale = rsqrt(MLXArray(2.0))

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
        let firstKvSharedLayerIdx = self.firstKvSharedLayerIdx
        let maxCacheIdx = layerIdxToCacheIdx.max() ?? 0
        let requiredCacheSize = max(firstKvSharedLayerIdx, maxCacheIdx + 1)
        let cacheArray = cache ?? Array(repeating: nil as KVCache?, count: requiredCacheSize)

        let pastSeenTokens = cacheArray.first??.offset ?? 0
        let cachePosition = MLXArray(pastSeenTokens ..< (pastSeenTokens + h.shape[1]))

        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = .none

        if mask == nil {
            let fullCacheSlice = Array(cacheArray[firstFullIdx...]).compactMap { $0 }
            fullMask = createAttentionMask(h: h, cache: fullCacheSlice, returnArray: true)

            let slidingCacheSlice = Array(cacheArray[firstSlidingIdx...]).compactMap { $0 }
            slidingWindowMask = createAttentionMask(
                h: h, cache: slidingCacheSlice, returnArray: true)
        }

        let h0 = h

        let targetMagnitude = pow(mean(h0.square(), axis: -1, keepDims: true), 0.5)
        let epsilonTensor = MLXArray(Float.leastNormalMagnitude, dtype: h0.dtype)

        var hList = Array(repeating: h0, count: config.altupNumInputs)

        for i in 1 ..< config.altupNumInputs {
            let altupProj = altupProjections[i - 1](hList[i])
            hList[i] = altupProj.asType(h0.dtype)
        }

        h = stacked(hList, axis: 0)

        if config.altupNumInputs > 1 {
            let mags = pow(mean(h[1...].square(), axis: -1, keepDims: true), 0.5)
            h[1...] = h[1...] * (targetMagnitude / maximum(mags, epsilonTensor))
        }

        for (i, layer) in layers.enumerated() {
            let perLayerInput = finalPerLayerInputs[0..., 0..., i, 0...]

            let layerTypes =
                config.layerTypes
                ?? Array(repeating: "global_attention", count: config.numHiddenLayers)
            let isGlobal = layerTypes[i] == "full_attention"

            let localMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                localMask = mask
            } else if isGlobal {
                localMask = fullMask
            } else {
                localMask = slidingWindowMask
            }

            let cacheIdx = layerIdxToCacheIdx[i]
            let layerCache = cacheIdx < cacheArray.count ? cacheArray[cacheIdx] : nil

            h = layer(
                h,
                mask: localMask,
                cache: layerCache,
                perLayerInput: perLayerInput,
                caches: cacheArray,
                cachePosition: cachePosition
            )
        }

        let targetMagnitudeFinal = pow(mean(h[0].square(), axis: -1, keepDims: true), 0.5)

        for i in 1 ..< config.altupNumInputs {
            let altupUnembProj = altupUnembedProjections[i - 1](h[i])
            h[i] = altupUnembProj.asType(h0.dtype)
        }

        if config.altupNumInputs > 1 {
            let mags = pow(mean(h[1...].square(), axis: -1, keepDims: true), 0.5)
            h[1...] = h[1...] * (targetMagnitudeFinal / maximum(mags, epsilonTensor))
        }

        h = mean(h, axis: 0)
        var out = norm(h)

        out = embedTokens.asLinear(out)

        if let softcap = finalLogitSoftcapping {
            out = tanh(out / softcap) * softcap
        }

        return out
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
            * MLXArray(pow(2.0, -0.5), dtype: inputsEmbeds.dtype)
    }
}

public class Gemma3nTextModel: Module, LLMModel {
    @ModuleInfo(key: "language_model") private var languageModel: LanguageModel

    let config: Gemma3nTextConfiguration
    let modelType: String
    let textVocabSize: Int

    var kvHeads: [Int]

    public init(config: Gemma3nTextConfiguration) {
        self.config = config
        self.modelType = config.modelType
        self.textVocabSize = config.vocabSizePerLayerInput

        self._languageModel.wrappedValue = LanguageModel(config)

        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)

        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        return languageModel.newCache(parameters: parameters)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        return callAsFunction(inputs, inputsEmbeds: nil, mask: nil, cache: cache)
    }

    public func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let cacheArray: [KVCache?]? = cache?.map { $0 as KVCache? }
        return languageModel(
            inputs: inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cacheArray)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasPrefix("model.language_model.") {
                let newKey = key.replacingOccurrences(
                    of: "model.language_model.", with: "language_model.")
                processedWeights[newKey] = value
            }
        }

        return processedWeights
    }

    /// Handles prompt processing for sequences
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        let promptCount = promptTokens.shape[0]

        guard promptCount > 0 else {
            print("Warning: Preparing with empty prompt tokens.")
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }

        return .tokens(input.text)
    }
}

extension Gemma3nTextModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.layers
    }
}
