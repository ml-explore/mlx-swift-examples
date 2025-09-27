//
//  Lille130m.swift
//  mlx-swift-examples
//
//  Created by Sachin Desai on 9/10/25.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable

// MARK: - Attention

private final class Lille130mAttention: Module {
    let args: Lille130mConfiguration
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm

    var rope: RoPE

    init(_ args: Lille130mConfiguration) {
        self.args = args
        self.headDim = args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDim), -0.5)

        let qkvOut = (args.attentionHeads + 2 * args.kvHeads) * headDim
        _qkvProj.wrappedValue = Linear(args.hiddenSize, qkvOut, bias: false)
        _outProj.wrappedValue = Linear(args.attentionHeads * headDim, args.hiddenSize, bias: false)

        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: true,
            base: args.ropeTheta,
            scale: 1.0
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qkv = qkvProj(norm(x))

        let qSize = args.attentionHeads * headDim
        let kvSize = args.kvHeads * headDim

        var queries = qkv[.ellipsis, 0 ..< qSize]
        let kv = qkv[.ellipsis, qSize...]
        var keys = kv[.ellipsis, 0 ..< kvSize]
        var values = kv[.ellipsis, kvSize...]

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE with cache-aware offset if available
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

        return outProj(output)
    }
}

// MARK: - MLP

private final class Lille130mMLP: Module, UnaryLayer {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ args: Lille130mConfiguration) {
        let numerator = (8 * args.hiddenSize) / 3
        let rounded = Int(round(Float(numerator) / 256.0))
        let hiddenDim = max(256 * rounded, 1)

        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        _gate.wrappedValue = Linear(args.hiddenSize, hiddenDim, bias: false)
        _up.wrappedValue = Linear(args.hiddenSize, hiddenDim, bias: false)
        _down.wrappedValue = Linear(hiddenDim, args.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm(x)
        return down(silu(gate(h)) * up(h))
    }
}

// MARK: - Block

private final class Lille130mBlock: Module {
    @ModuleInfo(key: "attention") var attention: Lille130mAttention
    @ModuleInfo(key: "feed_forward") var feedForward: Lille130mMLP

    init(_ args: Lille130mConfiguration) {
        _attention.wrappedValue = Lille130mAttention(args)
        _feedForward.wrappedValue = Lille130mMLP(args)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(x, mask: mask, cache: cache)
        return h + feedForward(h)
    }
}

// MARK: - Model (inner)

private final class Lille130mModelInner: Module {
    @ModuleInfo(key: "tok_embeddings") var embedTokens: Embedding

    let layers: [Lille130mBlock]
    let norm: RMSNorm

    init(_ args: Lille130mConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in Lille130mBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
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

// MARK: - Public Model

public final class Lille130mModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "transformer") fileprivate var transformer: Lille130mModelInner
    private let configuration: Lille130mConfiguration

    public init(_ args: Lille130mConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }

        _transformer.wrappedValue = Lille130mModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = transformer(inputs, cache: cache)
        return transformer.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let w = weights.filter { key, _ in !key.contains("rotary_emb") }
        return w
    }
}

// MARK: - Configuration

@Codable
public struct Lille130mConfiguration: Sendable {
    @CodingKey("model_type") public var modelType: String
    @CodingKey("block_size") public var blockSize: Int
    @CodingKey("layer_norm_eps") public var layerNormEps: Float
    @CodingKey("n_embd") public var hiddenSize: Int
    @CodingKey("n_head") public var attentionHeads: Int
    @CodingKey("n_kv_heads") public var kvHeads: Int
    @CodingKey("n_layer") public var hiddenLayers: Int
    @CodingKey("rope_theta") public var ropeTheta: Float
    @CodingKey("vocab_size") public var vocabularySize: Int
    @CodingKey("tie_word_embeddings") public var tieWordEmbeddings: Bool = true
}

// MARK: - LoRA

extension Lille130mModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        transformer.layers.map { ($0.attention, ["qkv_proj"]) }
    }
}
