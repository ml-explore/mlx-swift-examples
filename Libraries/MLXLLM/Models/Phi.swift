// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable

// https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models/phi.py

private class PhiAttention: Module {

    let args: PhiConfiguration
    let heads: Int
    let headDim: Int

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
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, headDim).transposed(0, 2, 1, 3)

        // Add RoPE to the queries and keys and combine them with the cache
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Finally perform the attention computation
        let scale = sqrt(1 / Float(queries.dim(-1)))
        let output = attentionWithCacheUpdate(
            queries: queries.asType(.float32),
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .asType(values.dtype)
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return dense(output)
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
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = inputLayerNorm(x)
        let attentionH = selfAttention(h, mask: mask, cache: cache)
        let ffH = mlp(h)
        return attentionH + ffH + x
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
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: [KVCache]? = nil
    ) -> MLXArray {
        var x = embedTokens(x)

        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: cache?[i])
        }

        return finalLayerNorm(x)
    }
}

public class PhiModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: PhiModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: PhiConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = PhiModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: true)
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        let mask = createAttentionMask(h: x, cache: cache)

        let y = model(x, mask: mask, cache: cache)
        return lmHead(y)
    }
}

@Codable
public struct PhiConfiguration: Sendable {
    @CodingKey("max_position_embeddings") public var maxPositionEmbeddings = 2048
    @CodingKey("vocab_size") public var vocabularySize = 51200
    @CodingKey("hidden_size") public var hiddenSize = 2560
    @CodingKey("num_attention_heads") public var attentionHeads = 32
    @CodingKey("num_hidden_layers") public var hiddenLayers = 32
    @CodingKey("num_key_value_heads") public var kvHeads = 32
    @CodingKey("partial_rotary_factor") public var partialRotaryFactor: Float = 0.4
    @CodingKey("intermediate_size") public var intermediateSize = 10240
    @CodingKey("layer_norm_eps") public var layerNormEps: Float = 1e-5
    @CodingKey("rope_theta") public var ropeTheta: Float = 10_000
}

// MARK: - LoRA

extension PhiModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
