//
//  BaichuanM1.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/6/16.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXRandom

public struct BaichuanM1Configuration: Codable, Sendable {
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var ropeTheta: Float
    var slidingWindow: Int
    var slidingWindowLayers: [Int]
    var convWindow: Int
    var rmsNormEps: Float
    var swaAttentionHeads: Int?
    var swaKvHeads: Int?
    var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case slidingWindowLayers = "sliding_window_layers"
        case convWindow = "conv_window"
        case rmsNormEps = "rms_norm_eps"
        case swaAttentionHeads = "num_swa_attention_heads"
        case swaKvHeads = "num_swa_key_value_heads"
        case tieWordEmbeddings = "tie_word_embeddings"
    }
}

private class Attention: Module {
    let config: BaichuanM1Configuration
    let layerIdx: Int
    let isSWA: Bool
    let numHeads: Int
    let numKVHeads: Int
    let hiddenSize: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "W_pack") var wPack: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPE

    @ParameterInfo(key: "conv_k") var convK: MLXArray
    @ParameterInfo(key: "conv_v") var convV: MLXArray

    init(_ config: BaichuanM1Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx

        self.isSWA = config.slidingWindowLayers.contains(layerIdx)
        self.numHeads =
            isSWA && config.swaAttentionHeads != nil
            ? config.swaAttentionHeads! : config.attentionHeads
        self.numKVHeads = isSWA && config.swaKvHeads != nil ? config.swaKvHeads! : config.kvHeads

        self.hiddenSize = config.hiddenSize
        self.headDim = hiddenSize / numHeads
        self.scale = pow(Float(headDim), -0.5)

        _wPack.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize + 2 * numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)

        _convK.wrappedValue = MLXArray.zeros([1, 1, numKVHeads, 1, config.convWindow])
        _convV.wrappedValue = MLXArray.zeros([1, 1, numKVHeads, 1, config.convWindow])
    }

    func customConvolution(_ u: MLXArray, _ weights: MLXArray, state: MLXArray? = nil) -> MLXArray {
        let (B, H, L, D) = (u.dim(0), u.dim(1), u.dim(2), u.dim(3))
        let reshapedWeights = weights.reshaped(1, H, config.convWindow, 1, 1)
        let w0 = reshapedWeights[0..., 0..., 0]
        let w1 = reshapedWeights[0..., 0..., 1]

        let state = state ?? MLXArray.zeros([B, H, 1, D], dtype: u.dtype)

        let uPrev: MLXArray =
            L > 1 ? concatenated([state, u[0..., 0..., ..<(L - 1), 0...]], axis: 2) : state

        return uPrev * w0 + u * w1
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))

        let proj = wPack(x)
        let qkv = split(proj, indices: [D, D + self.numKVHeads * self.headDim], axis: -1)

        var queries = qkv[0].reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = qkv[1].reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = qkv[2].reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        var offset = 0
        var lastK: MLXArray? = nil
        var lastV: MLXArray? = nil

        if let cacheList = cache as? CacheList {
            offset = cacheList[1].offset
            if let mambaCache = cacheList[0] as? MambaCache {
                lastK = mambaCache[0]
                lastV = mambaCache[1]
            }
        }

        let kInit = keys
        let vInit = values

        keys = customConvolution(keys, convK, state: lastK)
        values = customConvolution(values, convV, state: lastV)

        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        if let cache = cache as? CacheList {
            let kvCache = cache[1]
            let (cachedKeys, cachedValues) = kvCache.update(keys: keys, values: values)
            keys = cachedKeys
            values = cachedValues

            if L > 0 {
                let convCache = cache[0] as! MambaCache
                convCache[0] = kInit[0..., 0..., (L - 1)..., 0...]
                convCache[1] = vInit[0..., 0..., (L - 1)..., 0...]
            }
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(out)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: BaichuanM1Configuration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

private class DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: BaichuanM1Configuration, layerIdx: Int) {
        _attention.wrappedValue = Attention(config, layerIdx: layerIdx)
        self.mlp = MLP(config)
        _inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayernorm(x), mask: mask, cache: cache)
        let x = x + r
        r = mlp(postAttentionLayernorm(x))
        return x + r
    }
}

private class BaichuanM1ModelInner: Module {
    let args: BaichuanM1Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [DecoderLayer]
    let norm: RMSNorm

    init(_ config: BaichuanM1Configuration) {
        self.args = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { DecoderLayer(config, layerIdx: $0) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        var x = embedTokens(inputs)

        let mask = mask ?? createAttentionMask(h: x, cache: cache)

        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: cache?[i])
        }

        return norm(x)
    }
}

public class BaichuanM1Model: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: BaichuanM1ModelInner
    let configuration: BaichuanM1Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: BaichuanM1Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = BaichuanM1ModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var outputs = model(inputs, cache: cache)

        if let lmHead {
            outputs = lmHead(outputs)
        }

        return outputs
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.enumerated().map { (i, _) in
            let isSWA = configuration.slidingWindowLayers.contains(i)
            let convCache = MambaCache()
            let kvCache: KVCache =
                isSWA ? RotatingKVCache(maxSize: configuration.slidingWindow) : KVCacheSimple()
            return CacheList(convCache, kvCache)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        let isQuantized = weights["lm_head.scales"] != nil

        if !isQuantized, let w = weights["lm_head.weight"] {
            var w = w
            if w.dtype != .float32 {
                w = w.asType(.float32)
            }

            let norm = sqrt(sum(w * w, axes: [-1], keepDims: true))
            w = (w / (norm + 1e-7)).asType(w.dtype)
            weights["lm_head.weight"] = w
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

extension BaichuanM1Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
