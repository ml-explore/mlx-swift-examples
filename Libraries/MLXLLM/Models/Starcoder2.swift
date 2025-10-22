//
//  Starcoder2.swift
//  LLM
//
//  Created by John Mai on 2024/3/7.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/starcoder2.py

private class Attention: Module {
    let args: Starcoder2Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: Starcoder2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: true)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta)
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

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") var cFc: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _cFc.wrappedValue = Linear(dimensions, hiddenDimensions, bias: true)
        _cProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        cProj(gelu(cFc(x)))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    public init(_ args: Starcoder2Configuration) {
        _attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.normEpsilon)
        _postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.normEpsilon)
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

private class Starcoder2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: LayerNorm

    public init(_ args: Starcoder2Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                TransformerBlock(args)
            }
        self.norm = LayerNorm(dimensions: args.hiddenSize, eps: args.normEpsilon)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Starcoder2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let tieWordEmbeddings: Bool
    private let model: Starcoder2ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: Starcoder2Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Starcoder2ModelInner(args)
        self.tieWordEmbeddings = args.tieWordEmbeddings
        if !self.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)

        if !tieWordEmbeddings {
            return lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
            return out
        }
    }
}

public struct Starcoder2Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int = 16384
    var normEpsilon: Float = 1e-5
    var normType: String = "layer_norm"
    var vocabularySize: Int = 49152
    var ropeTheta: Float = 100000
    var tieWordEmbeddings: Bool = true

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEpsilon = "norm_epsilon"
        case normType = "norm_type"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Starcoder2Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Starcoder2Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Starcoder2Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Starcoder2Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Starcoder2Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Starcoder2Configuration.CodingKeys.attentionHeads)
        self.kvHeads = try container.decode(
            Int.self, forKey: Starcoder2Configuration.CodingKeys.kvHeads)
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(
                Int.self, forKey: Starcoder2Configuration.CodingKeys.maxPositionEmbeddings) ?? 16384
        self.normEpsilon =
            try container.decodeIfPresent(
                Float.self, forKey: Starcoder2Configuration.CodingKeys.normEpsilon) ?? 1e-5
        self.normType =
            try container.decodeIfPresent(
                String.self, forKey: Starcoder2Configuration.CodingKeys.normType) ?? "layer_norm"
        self.vocabularySize =
            try container.decodeIfPresent(
                Int.self, forKey: Starcoder2Configuration.CodingKeys.vocabularySize) ?? 49152
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Starcoder2Configuration.CodingKeys.ropeTheta)
            ?? 100000
        self.tieWordEmbeddings =
            try container.decodeIfPresent(
                Bool.self, forKey: Starcoder2Configuration.CodingKeys.tieWordEmbeddings)
            ?? true
    }
}

// MARK: - LoRA

extension Starcoder2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
