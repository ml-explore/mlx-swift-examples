// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/phi.py

private class LayerNorm: MLXNN.LayerNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x.asType(Float.self)).asType(x.dtype)
    }
}

private class PhiAttention: Module {

    let args: PhiConfiguration
    let heads: Int
    let headDim: Int
    let repeats: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "dense") var dense: Linear

    let rope: RoPE

    public init(_ args: PhiConfiguration) {
        self.args = args

        let hiddenSize = args.hiddenSize
        self.heads = args.attentionHeads
        self.headDim = args.hiddenSize / heads
        let kvHeads = args.kvHeads
        self.repeats = heads / kvHeads

        if headDim * heads != hiddenSize {
            fatalError("hidden_size must be divisible by num_heads")
        }

        self._wq.wrappedValue = Linear(hiddenSize, heads * headDim, bias: true)
        self._wk.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: true)
        self._dense.wrappedValue = Linear(heads * headDim, hiddenSize, bias: true)

        self.rope = RoPE(
            dimensions: Int(args.partialRotaryFactor * Float(headDim)), traditional: false,
            base: args.ropeTheta)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, headDim).transposed(0, 2, 1, 3)

        if repeats > 1 {
            keys = MLXArray.repeat(keys, count: repeats, axis: 1)
            values = MLXArray.repeat(values, count: repeats, axis: 1)
        }

        // Add RoPE to the queries and keys and combine them with the cache
        if let (keyCache, valueCache) = cache {
            queries = rope(queries, offset: keyCache.dim(2))
            keys = rope(keys, offset: keyCache.dim(2))
            keys = concatenated([keyCache, keys], axis: 2)
            values = concatenated([valueCache, values], axis: 2)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        queries = queries.asType(Float.self)
        keys = keys.asType(Float.self)

        // Finally perform the attention computation
        let scale = sqrt(1 / Float(queries.dim(-1)))
        var scores = (queries * scale).matmul(keys.transposed(0, 1, 3, 2))
        if let mask {
            scores = scores + mask
        }

        scores = softMax(scores, axis: -1).asType(values.dtype)
        let valuesHat = matmul(scores, values).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return (dense(valuesHat), (keys, values))
    }
}

private class PhiMLP: Module, UnaryLayer {

    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var act: GELU

    public init(_ config: PhiConfiguration) {
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize)
        self.act = GELU(approximation: .precise)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(act(fc1(x)))
    }
}

private class PhiDecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttention: PhiAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    var mlp: PhiMLP

    public init(_ config: PhiConfiguration) {
        self._selfAttention.wrappedValue = PhiAttention(config)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = PhiMLP(config)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let h = inputLayerNorm(x)
        let (attentionH, cache) = selfAttention(h, mask: mask, cache: cache)
        let ffH = mlp(h)
        return (attentionH + ffH + x, cache)
    }
}

private class PhiModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    @ModuleInfo var layers: [PhiDecoderLayer]
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: LayerNorm

    public init(_ args: PhiConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                PhiDecoderLayer(args)
            }
        self._finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: [(MLXArray, MLXArray)]? = nil
    ) -> (
        MLXArray, [(MLXArray, MLXArray)]
    ) {
        var x = embedTokens(x)

        var newCache = [(MLXArray, MLXArray)]()

        for (i, layer) in layers.enumerated() {
            var cacheUpdate: (MLXArray, MLXArray)
            (x, cacheUpdate) = layer(x, mask: mask, cache: cache?[i])
            newCache.append(cacheUpdate)
        }

        return (finalLayerNorm(x), newCache)
    }
}

public class PhiModel: Module, LLMModel {

    public let vocabularySize: Int

    fileprivate let model: PhiModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: PhiConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.model = PhiModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: true)
    }

    public func callAsFunction(_ x: MLXArray, cache: [(MLXArray, MLXArray)]?) -> (
        MLXArray, [(MLXArray, MLXArray)]
    ) {
        var mask: MLXArray? = nil
        if x.dim(1) > 1 {
            mask = MultiHeadAttention.createAdditiveCausalMask(x.dim(1))
            mask = mask?.asType(x.dtype)
        }

        let (y, cache) = model(x, mask: mask, cache: cache)
        return (lmHead(y), cache)
    }
}

public struct PhiConfiguration: Codable {
    var maxPositionalEmbeddings = 2048
    var vocabularySize = 51200
    var hiddenSize = 2560
    var attentionHeads = 32
    var hiddenLayers = 32
    var kvHeads = 32
    var partialRotaryFactor: Float = 0.4
    var intermediateSize = 10240
    var layerNormEps: Float = 1e-5
    var ropeTheta: Float = 10_000

    enum CodingKeys: String, CodingKey {
        case maxPositionalEmbeddings = "max_position_embeddings"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case partialRotaryFactor = "partial_rotary_factor"
        case intermediateSize = "intermediate_size"
        case layerNormEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<PhiConfiguration.CodingKeys> = try decoder.container(
            keyedBy: PhiConfiguration.CodingKeys.self)

        self.maxPositionalEmbeddings = try container.decode(
            Int.self, forKey: PhiConfiguration.CodingKeys.maxPositionalEmbeddings)
        self.vocabularySize = try container.decode(
            Int.self, forKey: PhiConfiguration.CodingKeys.vocabularySize)
        self.hiddenSize = try container.decode(
            Int.self, forKey: PhiConfiguration.CodingKeys.hiddenSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: PhiConfiguration.CodingKeys.attentionHeads)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: PhiConfiguration.CodingKeys.hiddenLayers)
        self.kvHeads =
            try container.decodeIfPresent(Int.self, forKey: PhiConfiguration.CodingKeys.kvHeads)
            ?? attentionHeads
        self.partialRotaryFactor = try container.decode(
            Float.self, forKey: PhiConfiguration.CodingKeys.partialRotaryFactor)
        self.intermediateSize = try container.decode(
            Int.self, forKey: PhiConfiguration.CodingKeys.intermediateSize)
        self.layerNormEps = try container.decode(
            Float.self, forKey: PhiConfiguration.CodingKeys.layerNormEps)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: PhiConfiguration.CodingKeys.ropeTheta)
            ?? 10_000

    }
}
