//
//  Gemma3Text.swift
//  mlx-swift-examples
//
//  Created by Anthony DePasquale on 14.03.2025.
//

// Based on https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gemma3_text.py

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN

public struct Gemma3TextConfiguration: Codable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let ropeGlobalBaseFreq: Float
    let ropeLocalBaseFreq: Float
    let ropeTraditional: Bool
    let queryPreAttnScalar: Float
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let finalLogitSoftcapping: Float?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeGlobalBaseFreq = "rope_global_base_freq"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case finalLogitSoftcapping = "final_logit_softcapping"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)

        // Default values with optional decoding
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262208
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        ropeGlobalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeGlobalBaseFreq) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256  // 0.0625 ?
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 4096
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
        finalLogitSoftcapping = try container.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
    }
}

private class Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

    @ModuleInfo var rope: RoPE

    init(_ config: Gemma3TextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.repeats = nHeads / nKVHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow

        self.scale = pow(config.queryPreAttnScalar, -0.5)

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

        let baseFreq = isSliding ? config.ropeLocalBaseFreq : config.ropeGlobalBaseFreq
        self._rope.wrappedValue = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            base: baseFreq
        )

        super.init()
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

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        // Apply normalization before RoPE
        queries = queryNorm(queries)
        keys = keyNorm(keys)

        var localMask = mask

        if let cache {
            // Apply RoPE with offset
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)

            // Handle sliding window for cached generation
            if isSliding && cache.offset > slidingWindow && L == 1 {
                // Create a sliding window mask for generation
                let size = cache.offset + L
                let windowStart = max(0, cache.offset - slidingWindow)

                // Create a mask where everything is invalid (large negative value)
                var slidingMaskData = Array(repeating: Float32(-1e9), count: size)

                // Set the sliding window positions to valid (0)
                for i in windowStart ..< min(windowStart + slidingWindow + 1, size) {
                    slidingMaskData[i] = 0
                }

                // Create the MLXArray from the data
                let slidingMask = MLXArray(slidingMaskData).reshaped(1, 1, 1, size)
                localMask = slidingMask
            }
        } else {
            // Apply RoPE without offset
            queries = rope(queries)
            keys = rope(keys)
        }

        // Scale queries by the pre-attention scalar
        queries = queries * MLXArray(scale).asType(queries.dtype)

        // Adjust mask for sliding window if needed
        if isSliding && localMask != nil && localMask!.dim(-1) != keys.dim(-2) {
            let keyLen = keys.dim(-2)
            localMask = localMask![0..., 0..., 0..., (localMask!.dim(-1) - keyLen)...]
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,  // We already scaled the queries
            mask: localMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Attention
    @ModuleInfo var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    let numAttentionHeads: Int
    let hiddenSize: Int

    init(_ config: Gemma3TextConfiguration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize

        self._selfAttention.wrappedValue = Attention(config, layerIdx: layerIdx)
        self.mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
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

private class Gemma3Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    let config: Gemma3TextConfiguration

    init(_ config: Gemma3TextConfiguration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    private func createAdditiveCausalMask(n: Int, offset: Int) -> MLXArray {
        let rinds = MLXArray(Int32(0) ..< Int32(offset + n))
        let linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
        let mask = linds[0..., .newAxis] .< rinds[.newAxis]
        // Make sure the mask has shape [1, 1, n, offset+n]
        return (mask * Float32(-1e9)).reshaped(1, 1, n, offset + n)
    }

    // Create attention mask with sliding window support
    private func createAttentionMask(h: MLXArray, cache: [KVCache]?, isSliding: Bool = false)
        -> MLXArray?
    {
        let t = h.dim(1)

        var offset = 0
        if let cache = cache, !cache.isEmpty, let firstCache = cache.first(where: { $0 != nil }) {
            offset = firstCache.offset
        }

        // For single token generation with history
        if t == 1 && offset > 0 {
            return nil  // No mask needed for single token generation
        } else if t <= 1 && offset == 0 {
            return nil
        }

        // Create basic causal mask
        var mask = createAdditiveCausalMask(n: t, offset: offset).asType(h.dtype)

        // Apply sliding window constraint if needed
        if isSliding && config.slidingWindow > 0 && (t + offset) > config.slidingWindow {
            let windowSize = config.slidingWindow

            // Create a mask that limits attention to the sliding window
            for i in 0 ..< t {
                let row = i + offset
                let minCol = max(0, row - windowSize)

                // Set values outside the window to large negative
                if minCol > 0 {
                    let maskSlice = mask[0, 0, i, 0 ..< minCol]
                    let shape = maskSlice.shape
                    mask[0, 0, i, 0 ..< minCol] = MLXArray(
                        Array(repeating: Float(-1e9), count: minCol))
                }
            }
        }

        return mask
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        // Apply embedding with scaling
        let scale = sqrt(Float(config.hiddenSize))
        var h = embedTokens(inputs) * MLXArray(scale).asType(inputs.dtype)

        // Create masks if needed
        var localMasks: [MLXArray?] = Array(repeating: nil, count: config.hiddenLayers)

        for i in 0 ..< config.hiddenLayers {
            let isGlobal = (i + 1) % config.slidingWindowPattern == 0
            let isSliding = !isGlobal

            if isSliding && inputs.dim(1) > 1 {
                localMasks[i] = createAttentionMask(h: h, cache: cache, isSliding: true)
            } else {
                localMasks[i] = createAttentionMask(h: h, cache: cache, isSliding: false)
            }
        }

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            h = layer(h, mask: localMasks[i], cache: layerCache)
        }

        return norm(h)
    }
}

public class Gemma3TextModel: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo private var model: Gemma3Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma3TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int]

    public init(_ config: Gemma3TextConfiguration) {
        self.config = config
        self.model = Gemma3Model(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)

        // Set up KV heads array based on sliding window pattern
        var heads: [Int] = []
        for i in 0 ..< config.hiddenLayers {
            heads.append(config.kvHeads)
        }
        self.kvHeads = heads

        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        var finalLogits = lmHead(out)
        if let softcap = config.finalLogitSoftcapping {
            finalLogits = tanh(finalLogits / MLXArray(softcap)) * MLXArray(softcap)
        }
        return finalLogits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        // Copy embedding weights to lm_head if not present
        if sanitizedWeights["lm_head.weight"] == nil {
            if let embedWeight = sanitizedWeights["model.embed_tokens.weight"] {
                sanitizedWeights["lm_head.weight"] = embedWeight
            }
        }
        // Remove RoPE frequency weights as they're computed on the fly
        return sanitizedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

extension Gemma3TextModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}

extension MLXArray {
    public static func arange(_ size: Int) -> MLXArray {
        return MLXArray(Array(0 ..< size))
    }
}
