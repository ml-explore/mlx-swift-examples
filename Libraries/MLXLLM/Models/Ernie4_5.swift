//
//  Ernie4_5.swift
//  mlx-swift-examples
//
//  Created by Sachin Desai on 7/3/25.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ernie4_5.py

public struct Ernie45Configuration: Codable {
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int?
    var numHiddenLayers: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float
    var useBias: Bool
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numHiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case useBias = "use_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<Ernie45Configuration.CodingKeys> =
            try decoder.container(keyedBy: Ernie45Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.useBias = try container.decode(Bool.self, forKey: .useBias)
        self.tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
    }
}

private class Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    public init(_ args: Ernie45Configuration) {
        let dim = args.hiddenSize
        self.nHeads = args.numAttentionHeads
        self.nKVHeads = args.numKeyValueHeads
        self.headDim = args.headDim ?? (dim / args.numAttentionHeads)
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: args.useBias)
        self._kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: args.useBias)
        self._vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: args.useBias)
        self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: args.useBias)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: true,
            base: args.ropeTheta
        )
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

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

        return oProj(output)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    public init(dim: Int, hiddenDim: Int, useBias: Bool = false) {
        self._gateProj.wrappedValue = Linear(dim, hiddenDim, bias: useBias)
        self._downProj.wrappedValue = Linear(hiddenDim, dim, bias: useBias)
        self._upProj.wrappedValue = Linear(dim, hiddenDim, bias: useBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private class DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    public init(_ args: Ernie45Configuration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(
            dim: args.hiddenSize, hiddenDim: args.intermediateSize, useBias: args.useBias)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayernorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayernorm(h))
        return h + r
    }
}

private class Ernie45ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [DecoderLayer]
    let norm: RMSNorm

    public init(_ args: Ernie45Configuration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize
        )
        self.layers = (0 ..< args.numHiddenLayers).map { _ in
            DecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

public class Ernie45Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: Ernie45ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Ernie45Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.numHiddenLayers)
        self.model = Ernie45ModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)

        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }
}

// MARK: - LoRA

extension Ernie45Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
