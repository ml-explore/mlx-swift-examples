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
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)

        // Default values with optional decoding
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        ropeGlobalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeGlobalBaseFreq) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
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

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: GemmaUtils.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: GemmaUtils.RMSNorm

    @ModuleInfo var rope: RoPE

    init(_ config: Gemma3TextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.repeats = nHeads / nKVHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx

        self.scale = pow(config.queryPreAttnScalar, -0.5)

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

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

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        var localMask = mask

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)

            // For generation with cache, create a mask if none was provided
            if localMask == nil && L == 1 && cache.offset > 0 {
                // Create a mask that allows attending to all previous tokens
                let size = cache.offset + L
                localMask = MLXArray.zeros([1, 1, L, size])
            }
        } else {
            queries = rope(queries)
            keys = rope(keys)

            // For training or initial generation, create a causal mask if none was provided
            if localMask == nil && L > 1 {
                // Create a causal mask using MLXArray operations instead of loops
                let rows = MLXArray.arange(L).reshaped(L, 1)
                let cols = MLXArray.arange(L).reshaped(1, L)
                let causalMask = rows .>= cols  // This creates a boolean mask where rows >= cols (lower triangular)

                // Convert to attention mask format
                let attentionMask = causalMask.asType(.float32)  // Convert to float
                let negativeAttentionMask = (1 - attentionMask) * Float32(-1e9)
                localMask = negativeAttentionMask.reshaped(1, 1, L, L)
            }
        }

        // Sliding window mask adjustment
        if localMask != nil && localMask!.dim(-1) != keys.dim(-2) {
            let keyLen = keys.dim(-2)
            localMask = localMask![0..., 0..., 0..., (localMask!.dim(-1) - keyLen)...]
        }

        // Print debug info
        print("Attention input shape:", x.shape)
        print("Queries shape:", queries.shape)
        print("Keys shape:", keys.shape)
        print("Mask shape:", localMask?.shape ?? "nil")

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
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
    func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
        let t = h.dim(1)

        var offset = 0
        if let cache = cache, !cache.isEmpty, let firstCache = cache.first(where: { $0 != nil }) {
            offset = firstCache.offset
        }

        // Even for t=1, we need a mask during generation to attend to previous tokens
        if t == 1 && offset > 0 {
            // For single token generation with history, create a mask that allows
            // attending to all previous tokens
            let size = offset + t
            var mask = MLXArray.zeros([1, 1, t, size])

            // Set all positions to valid (0) since we want to attend to all previous tokens
            // No need to modify the mask as causal attention is implicit here
            return mask.asType(h.dtype)
        } else if t <= 1 && offset == 0 {
            // No mask needed for the very first token
            return nil
        }

        // Create basic causal mask for multi-token processing
        var mask = createAdditiveCausalMask(n: t, offset: offset).asType(h.dtype)

        // For sliding window attention, we need to limit the attention span
        if config.slidingWindow > 0 && (t + offset) > config.slidingWindow {
            // Create position indices
            let posRows = MLXArray(Int32(offset) ..< Int32(offset + t))
            let posCols = MLXArray(Int32(0) ..< Int32(offset + t))

            // Calculate position differences
            let posRowsExpanded = posRows[0..., .newAxis]
            let posColsExpanded = posCols[.newAxis]
            let posDiff = posRowsExpanded - posColsExpanded

            // Create sliding window mask: True where distance > slidingWindow
            let slidingWindowValue = MLXArray(Int32(config.slidingWindow))
            let slidingWindowMask = posDiff .> slidingWindowValue

            // Convert to attention mask values (-1e9 where True, 0 where False)
            let slidingWindowValues = slidingWindowMask * Float32(-1e9)

            // Reshape to [1, 1, t, offset+t] for attention
            let slidingWindowMaskReshaped = slidingWindowValues.reshaped(1, 1, t, offset + t)

            // Add to the causal mask
            mask = mask + slidingWindowMaskReshaped
        }

        return mask
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil, cache: [KVCache]? = nil)
        -> MLXArray
    {
        var h = embedTokens(inputs)
        let scaleFactor = MLXArray(sqrt(Float(config.hiddenSize))).asType(h.dtype)
        h = h * scaleFactor

        var localCache = cache

        // If no mask is provided, create appropriate masks
        var fullMask: MLXArray? = nil
        var slidingWindowMask: MLXArray? = nil

        if mask == nil {
            // Create the standard causal mask
            fullMask = createAttentionMask(h: h, cache: localCache)

            // For sliding window layers, we need a different mask
            if config.slidingWindow > 0 {
                // Create a sliding window mask by modifying the full mask
                slidingWindowMask = createAttentionMask(h: h, cache: localCache)
                // The sliding window logic is already handled in createAttentionMask
            }
        }

        for (i, layer) in layers.enumerated() {
            let isSliding = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)

            var layerMask = mask
            if mask == nil {
                layerMask = isSliding ? slidingWindowMask : fullMask
            }

            // Debug print for first few layers
            if i < 3 {
                print("Layer \(i) mask shape:", layerMask?.shape ?? "nil")
            }

            let layerCache = localCache?[i]
            h = layer(h, mask: layerMask, cache: layerCache)
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
        // Explicitly create a mask for autoregressive generation
        var mask: MLXArray? = nil

        if inputs.dim(1) == 1 && cache != nil && !cache!.isEmpty {
            // For single token generation with cache, create a mask that allows
            // attending to all previous tokens
            if let firstCache = cache!.first(where: { $0 != nil }) {
                let offset = firstCache.offset
                if offset > 0 {
                    // Create a mask that allows the current token to attend to all previous tokens
                    let size = offset + 1  // +1 for the current token
                    mask = MLXArray.zeros([1, 1, 1, size])
                }
            }
        }

        let out = model(inputs, mask: mask, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        if sanitizedWeights["lm_head.weight"] == nil {
            sanitizedWeights["lm_head.weight"] = sanitizedWeights["model.embed_tokens.weight"]
        }

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
