// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXFast
import MLXNN

private class Attention: Module {

    let args: Phi3Configuration
    let scale: Float

    let heads: Int
    let kvHeads: Int
    let headDim: Int

    @ModuleInfo(key: "qkv_proj") var wqkv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    enum PositionalEncoding {
        case rope(RoPE)
        case suScaledRotaryEmbedding(SuScaledRotaryEmbedding)

        func applyEncoding(_ x: MLXArray, offset: Int = 0) -> MLXArray {
            switch self {
            case .rope(let rope):
                return rope.callAsFunction(x, offset: offset)
            case .suScaledRotaryEmbedding(let suScaledRotaryEmbedding):
                return suScaledRotaryEmbedding.callAsFunction(x, offset: offset)
            }
        }
    }

    let rope: PositionalEncoding

    public init(_ args: Phi3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads

        self.headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        self._wqkv.wrappedValue = Linear(dim, (heads + 2 * kvHeads) * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        let ropeScale: Float

        if let ropeScaling = args.ropeScaling, ropeScaling.type == "linear",
            let factor = ropeScaling.factor
        {
            ropeScale = 1 / factor
        } else {
            ropeScale = 1
        }

        if let ropeScaling = args.ropeScaling,
            ropeScaling.type == "su" || ropeScaling.type == "longrope",
            let shortFactor = ropeScaling.shortFactor, let longFactor = ropeScaling.longFactor
        {
            self.rope = .suScaledRotaryEmbedding(
                SuScaledRotaryEmbedding(
                    dimensions: headDim, base: args.ropeTheta,
                    maxPositionEmbeddings: args.maxPositionEmbeddings,
                    originalMaxPositionEmbeddings: args.originalMaxPositionEmbeddings,
                    longFactor: longFactor))

        } else {
            self.rope = .rope(
                RoPE(
                    dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta,
                    scale: ropeScale))
        }
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let queryPos = heads * headDim
        let qkv = split(wqkv(x), indices: [queryPos, queryPos + kvHeads * headDim], axis: -1)
        var queries = qkv[0]
        var keys = qkv[1]
        var values = qkv[2]

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope.applyEncoding(queries, offset: cache.offset)
            keys = rope.applyEncoding(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope.applyEncoding(queries)
            keys = rope.applyEncoding(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_up_proj") var gate_up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate_up.wrappedValue = Linear(dimensions, 2 * hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gu = split(gate_up(x), parts: 2, axis: -1)
        return down(silu(gu[0]) * gu[1])
    }
}

private class TransformerBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Phi3Configuration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class Phi3ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Phi3Configuration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Phi3Model: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let headDim: IntOrPair

    let model: Phi3ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: Phi3Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.headDim = .init(args.hiddenSize / args.attentionHeads)
        self.model = Phi3ModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }
}

public struct Phi3Configuration: Codable, Sendable {
    struct RopeScaling: Codable {
        let longFactor: [Float]?
        let shortFactor: [Float]?
        let factor: Float?
        let type: String?

        enum CodingKeys: String, CodingKey {
            case type
            case factor
            case longFactor = "long_factor"
            case shortFactor = "short_factor"
        }
    }

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: RopeScaling?
    var maxPositionEmbeddings: Int
    var originalMaxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Phi3Configuration.CodingKeys> = try decoder.container(
            keyedBy: Phi3Configuration.CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: Phi3Configuration.CodingKeys.hiddenSize)
        hiddenLayers = try container.decode(
            Int.self, forKey: Phi3Configuration.CodingKeys.hiddenLayers)
        intermediateSize = try container.decode(
            Int.self, forKey: Phi3Configuration.CodingKeys.intermediateSize)
        attentionHeads = try container.decode(
            Int.self, forKey: Phi3Configuration.CodingKeys.attentionHeads)
        rmsNormEps = try container.decode(
            Float.self, forKey: Phi3Configuration.CodingKeys.rmsNormEps)
        vocabularySize = try container.decode(
            Int.self, forKey: Phi3Configuration.CodingKeys.vocabularySize)
        kvHeads = try container.decode(Int.self, forKey: Phi3Configuration.CodingKeys.kvHeads)
        ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Phi3Configuration.CodingKeys.ropeTheta) ?? 10_000
        ropeTraditional =
            try container.decodeIfPresent(
                Bool.self, forKey: Phi3Configuration.CodingKeys.ropeTraditional) ?? false
        ropeScaling = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        originalMaxPositionEmbeddings = try container.decode(
            Int.self, forKey: .originalMaxPositionEmbeddings)

    }
}

// MARK: - LoRA

extension Phi3Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["qkv_proj"]) }
    }
}
