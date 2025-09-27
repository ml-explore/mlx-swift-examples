//
//  Exaone4.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/7/15.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/exaone4.py

private class Attention: Module {
    let args: Exaone4Configuration
    let scale: Float
    let isLocal: Bool
    let useRope: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE?

    public init(_ args: Exaone4Configuration, isLocal: Bool?) {
        self.args = args
        self.isLocal = isLocal ?? false
        self.useRope = isLocal == nil || (isLocal ?? false)

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        if useRope {
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

            self.rope = RoPE(
                dimensions: headDim, traditional: false, base: args.ropeTheta,
                scale: ropeScale)
        } else {
            self.rope = nil
        }
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache, useRope, let rope {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else if useRope, let rope {
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
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    public init(_ args: Exaone4Configuration, isLocal: Bool?) {
        _attention.wrappedValue = Attention(args, isLocal: isLocal)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(x, mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        r = mlp(h)
        let out = h + postFeedforwardLayerNorm(r)
        return out
    }
}

private class ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Exaone4Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { i in
                let isLocal: Bool?
                if let pattern = args.slidingWindowPattern {
                    let patternIndex = i % pattern.count
                    let character = pattern[
                        pattern.index(pattern.startIndex, offsetBy: patternIndex)]
                    isLocal = character == "L"
                } else {
                    isLocal = nil
                }
                return TransformerBlock(args, isLocal: isLocal)
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

public class Exaone4Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: ModelInner
    let configuration: Exaone4Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Exaone4Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        return model.layers.map { layer in
            if layer.attention.isLocal, let slidingWindow = configuration.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow, keep: 0)
            } else {
                return StandardKVCache()
            }
        }
    }
}

@Codable
public struct Exaone4Configuration: Sendable {
    @CodingKey("hidden_size") public var hiddenSize: Int
    @CodingKey("num_hidden_layers") public var hiddenLayers: Int
    @CodingKey("intermediate_size") public var intermediateSize: Int
    @CodingKey("num_attention_heads") public var attentionHeads: Int
    @CodingKey("rms_norm_eps") public var rmsNormEps: Float
    @CodingKey("vocab_size") public var vocabularySize: Int
    @CodingKey("num_key_value_heads") public var kvHeads: Int
    @CodingKey("max_position_embeddings") public var maxPositionEmbeddings: Int
    @CodingKey("rope_theta") public var ropeTheta: Float
    @CodingKey("head_dim") public var headDim: Int
    @CodingKey("tie_word_embeddings") public var tieWordEmbeddings: Bool
    @CodingKey("rope_scaling") public var ropeScaling: [String: StringOrNumber]?
    @CodingKey("sliding_window") public var slidingWindow: Int?
    @CodingKey("sliding_window_pattern") public var slidingWindowPattern: String?
}

// MARK: - LoRA

extension Exaone4Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
