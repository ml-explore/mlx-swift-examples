//
//  SmolLM3.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/7/4.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable

private protocol PositionEmbedding {
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

extension RoPE: PositionEmbedding {}

// MARK: - NoPE

private final class NoPE: Module, PositionEmbedding {
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        return x
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, offset: 0)
    }
}

// MARK: - Attention

private class Attention: Module {
    let args: SmolLM3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    var rope: PositionEmbedding

    init(_ args: SmolLM3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: 1.0
        )

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)
    }

    func callAsFunction(
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

// MARK: - MLP

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: SmolLM3Configuration) {
        _gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        _down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        _up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(gate(x))
        return down(activation * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: SmolLM3Configuration) {
        _attention.wrappedValue = Attention(args)
        _mlp.wrappedValue = MLP(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

// MARK: - Model

private class SmolLM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    init(_ args: SmolLM3Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in TransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class SmolLM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: SmolLM3ModelInner
    let configuration: SmolLM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SmolLM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }

        self.model = SmolLM3ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        super.init()

        let identityRope = NoPE()
        for (idx, useRope) in args.noRopeLayers.enumerated() {
            if useRope == 0 && idx < model.layers.count {
                model.layers[idx].attention.rope = identityRope
            }
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

        weights = weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

// MARK: - Configuration

@Codable
public struct SmolLM3Configuration: Sendable {
    @CodingKey("model_type") public var modelType: String = "smollm3"
    @CodingKey("hidden_size") public var hiddenSize: Int
    @CodingKey("num_hidden_layers") public var hiddenLayers: Int
    @CodingKey("intermediate_size") public var intermediateSize: Int
    @CodingKey("num_attention_heads") public var attentionHeads: Int
    @CodingKey("head_dim") public var headDimensions: Int?
    @CodingKey("rms_norm_eps") public var rmsNormEps: Float
    @CodingKey("vocab_size") public var vocabularySize: Int
    @CodingKey("num_key_value_heads") public var kvHeads: Int = 0  // Will be set in didDecode
    @CodingKey("max_position_embeddings") public var maxPositionEmbeddings: Int?
    @CodingKey("rope_theta") public var ropeTheta: Float = 10_000
    @CodingKey("rope_traditional") public var ropeTraditional: Bool = false
    @CodingKey("rope_scaling") public var ropeScaling: [String: StringOrNumber]?
    @CodingKey("tie_word_embeddings") public var tieWordEmbeddings: Bool = true
    @CodingKey("attention_bias") public var attentionBias: Bool = false
    @CodingKey("mlp_bias") public var mlpBias: Bool = false
    @CodingKey("no_rope_layer_interval") public var noRopeLayerInterval: Int = 4
    @CodingKey("no_rope_layers") public var noRopeLayers: [Int] = []

    public var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    public mutating func didDecode(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: AnyCodingKey.self)

        // Set kvHeads to attentionHeads if not provided in JSON
        if kvHeads == 0 {
            kvHeads = attentionHeads
        }

        // Compute noRopeLayers if not provided in JSON
        if noRopeLayers.isEmpty,
            (try? container.decode(Int.self, forKey: AnyCodingKey("num_hidden_layers"))) != nil
        {
            let providedNoRopeLayers = try? container.decode(
                [Int].self, forKey: AnyCodingKey("no_rope_layers"))
            if providedNoRopeLayers == nil {
                noRopeLayers = (0 ..< hiddenLayers).map { i in
                    (i + 1) % noRopeLayerInterval != 0 ? 1 : 0
                }
            }
        }
    }
}

// MARK: - LoRA

extension SmolLM3Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
