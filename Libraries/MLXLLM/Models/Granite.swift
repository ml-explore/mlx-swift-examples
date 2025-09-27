//
//  Granite.swift
//  mlx-swift-examples
//
//  Created by Sachin Desai on 4/25/25.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/granite.py

private class Attention: Module {
    let args: GraniteConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: GraniteConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let nHeads = args.attentionHeads
        let nKvHeads = args.kvHeads
        let headDim = dim / nHeads

        self.scale = args.attentionMultiplier
        let attentionBias = args.attentionBias

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: attentionBias)
        self._wk.wrappedValue = Linear(dim, nKvHeads * headDim, bias: attentionBias)
        self._wv.wrappedValue = Linear(dim, nKvHeads * headDim, bias: attentionBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: attentionBias)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }
        rope = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta, scale: ropeScale)
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
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(_ args: GraniteConfiguration) {
        let dim = args.hiddenSize
        let hiddenDim = args.intermediateSize
        let mlpBias = args.mlpBias

        self._gate.wrappedValue = Linear(dim, hiddenDim, bias: mlpBias)
        self._down.wrappedValue = Linear(hiddenDim, dim, bias: mlpBias)
        self._up.wrappedValue = Linear(dim, hiddenDim, bias: mlpBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let residualMultiplier: Float

    public init(_ args: GraniteConfiguration) {
        let attentionHeads = args.attentionHeads
        let hiddenSize = args.hiddenSize

        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(args)

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize, eps: args.rmsNormEps)

        self.residualMultiplier = args.residualMultiplier
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r * residualMultiplier
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r * residualMultiplier
        return out
    }
}

private class GraniteModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm
    let embeddingMultiplier: Float

    public init(_ args: GraniteConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                TransformerBlock(args)
            }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.embeddingMultiplier = args.embeddingMultiplier
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs) * embeddingMultiplier

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class GraniteModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let logitsScaling: Float

    private let model: GraniteModelInner
    let configuration: GraniteConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: GraniteConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }

        self.model = GraniteModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        self.logitsScaling = args.logitsScaling
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out / logitsScaling
    }
}

@Codable
public struct GraniteConfiguration: Sendable {
    @CodingKey("hidden_size") public var hiddenSize: Int
    @CodingKey("num_hidden_layers") public var hiddenLayers: Int
    @CodingKey("intermediate_size") public var intermediateSize: Int
    @CodingKey("num_attention_heads") public var attentionHeads: Int
    @CodingKey("rms_norm_eps") public var rmsNormEps: Float
    @CodingKey("vocab_size") public var vocabularySize: Int
    @CodingKey("logits_scaling") public var logitsScaling: Float
    @CodingKey("attention_multiplier") public var attentionMultiplier: Float
    @CodingKey("embedding_multiplier") public var embeddingMultiplier: Float
    @CodingKey("residual_multiplier") public var residualMultiplier: Float
    @CodingKey("max_position_embeddings") public var maxPositionEmbeddings: Int
    @CodingKey("num_key_value_heads") public var kvHeads: Int
    @CodingKey("attention_bias") public var attentionBias: Bool
    @CodingKey("mlp_bias") public var mlpBias: Bool
    @CodingKey("rope_theta") public var ropeTheta: Float
    @CodingKey("rope_scaling") public var ropeScaling: [String: StringOrNumber]? = nil
    @CodingKey("tie_word_embeddings") public var tieWordEmbeddings: Bool = true
}

// MARK: - LoRA

extension GraniteModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
