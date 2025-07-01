// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable

private class Attention: Module {

    let args: Phi3Configuration
    let scale: Float

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let ropeDim: Int

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
        self.ropeDim = Int(Float(headDim) * args.partialRotaryFactor)
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
                    dimensions: ropeDim, base: args.ropeTheta,
                    maxPositionEmbeddings: args.maxPositionEmbeddings,
                    originalMaxPositionEmbeddings: args.originalMaxPositionEmbeddings,
                    longFactor: longFactor))

        } else {
            self.rope = .rope(
                RoPE(
                    dimensions: ropeDim, traditional: args.ropeTraditional, base: args.ropeTheta,
                    scale: ropeScale))
        }
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
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
        } else {
            queries = rope.applyEncoding(queries)
            keys = rope.applyEncoding(keys)
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

private class Phi3ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm
    let args: Phi3Configuration

    public init(_ args: Phi3Configuration) {
        precondition(args.vocabularySize > 0)
        self.args = args

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

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Phi3Model: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: Phi3ModelInner
    private let args: Phi3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Phi3Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Phi3ModelInner(args)
        self.args = args

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if args.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        } else if let lmHead {
            return lmHead(out)
        } else {
            fatalError(
                "Model configuration error: Neither tied embeddings nor lm_head is available")
        }
    }
}

@Codable
public struct RopeScalingWithFactorArrays: Sendable {
    @CodingKey("long_factor") public var longFactor: [Float]?
    @CodingKey("short_factor") public var shortFactor: [Float]?
    @CodingKey("long_mscale") public var longMScale: Float?
    @CodingKey("short_mscale") public var shortMScale: Float?
    public var factor: Float?
    public var type: String?

    enum CodingKeys: String, CodingKey {
        case type
        case factor
        case longFactor = "long_factor"
        case shortFactor = "short_factor"
        case longMScale = "long_mscale"
        case shortMScale = "short_mscale"
    }
}

@Codable
public struct Phi3Configuration: Sendable {
    @CodingKey("hidden_size") public var hiddenSize: Int
    @CodingKey("num_hidden_layers") public var hiddenLayers: Int
    @CodingKey("intermediate_size") public var intermediateSize: Int
    @CodingKey("num_attention_heads") public var attentionHeads: Int
    @CodingKey("rms_norm_eps") public var rmsNormEps: Float
    @CodingKey("vocab_size") public var vocabularySize: Int
    @CodingKey("num_key_value_heads") public var kvHeads: Int
    @CodingKey("rope_theta") public var ropeTheta: Float = 10_000
    @CodingKey("rope_traditional") public var ropeTraditional: Bool = false
    @CodingKey("rope_scaling") public var ropeScaling: RopeScalingWithFactorArrays?
    @CodingKey("partial_rotary_factor") public var partialRotaryFactor: Float = 1.0
    @CodingKey("max_position_embeddings") public var maxPositionEmbeddings: Int
    @CodingKey("original_max_position_embeddings") public var originalMaxPositionEmbeddings: Int
    @CodingKey("tie_word_embeddings") public var tieWordEmbeddings: Bool = false
}

// MARK: - LoRA

extension Phi3Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["qkv_proj"]) }
    }
}
