//
//  GLM4.swift
//  LLM
//
//  Created by John Mai on 2025/5/1.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm4.py

private class Attention: Module {
    let args: GLM4Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: GLM4Configuration) {
        self.args = args

        let headDim = args.headDim > 0 ? args.headDim : args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(args.attentionHeads * headDim, args.hiddenSize, bias: false)

        self.rope = RoPE(
            dimensions: Int(Float(headDim) * args.partialRotaryFactor),
            traditional: args.ropeTraditional, base: args.ropeTheta)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

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
    @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(_ args: GLM4Configuration) {
        _gateUp.wrappedValue = Linear(args.hiddenSize, 2 * args.intermediateSize, bias: false)
        _down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = gateUp(x)
        let chunks = split(x, parts: 2, axis: -1)
        return down(silu(chunks[0]) * chunks[1])
    }
}

private class GLM4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttnLayerNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm

    public init(_ args: GLM4Configuration) {
        _attention.wrappedValue = Attention(args)
        self.mlp = MLP(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postSelfAttnLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postMlpLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var x =
            x
            + postSelfAttnLayerNorm(
                attention(inputLayerNorm(x), mask: mask, cache: cache)
            )
        let residual = x
        x = postMlpLayerNorm(mlp(postAttentionLayerNorm(x))) + residual
        return x
    }
}

private class GLM4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [GLM4DecoderLayer]
    let norm: RMSNorm

    public init(_ args: GLM4Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                GLM4DecoderLayer(args)
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

public class GLM4Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: GLM4ModelInner
    let configuration: GLM4Configuration
    let modelType: String

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: GLM4Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.modelType = args.modelType
        self.model = GLM4ModelInner(args)

        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

public struct GLM4Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var attentionBias: Bool
    var headDim: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var partialRotaryFactor: Float
    var ropeTheta: Float = 10000.0
    var ropeTraditional: Bool = true
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768
    var modelType: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case modelType = "model_type"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<GLM4Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: GLM4Configuration.CodingKeys.self)

        self.modelType = try container.decode(
            String.self, forKey: GLM4Configuration.CodingKeys.modelType)
        self.hiddenSize = try container.decode(
            Int.self, forKey: GLM4Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: GLM4Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: GLM4Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: GLM4Configuration.CodingKeys.attentionHeads)
        self.attentionBias = try container.decode(
            Bool.self, forKey: GLM4Configuration.CodingKeys.attentionBias)
        self.headDim = try container.decode(
            Int.self, forKey: GLM4Configuration.CodingKeys.headDim)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: GLM4Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: GLM4Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: GLM4Configuration.CodingKeys.kvHeads)
        self.partialRotaryFactor = try container.decode(
            Float.self, forKey: GLM4Configuration.CodingKeys.partialRotaryFactor)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: GLM4Configuration.CodingKeys.ropeTheta)
            ?? 10000.0
        self.ropeTraditional =
            try container.decodeIfPresent(
                Bool.self, forKey: GLM4Configuration.CodingKeys.ropeTraditional)
            ?? true
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}

// MARK: - LoRA

extension GLM4Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
