//
//  LFM2.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/7/12.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct LFM2Configuration: Codable, Sendable {
    let modelType: String
    let vocabularySize: Int
    let hiddenSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let maxPositionEmbeddings: Int?
    let normEps: Float
    let convBias: Bool
    let convLCache: Int
    private let _blockDim: Int?
    var blockDim: Int { _blockDim ?? hiddenSize }
    private let _blockFFDim: Int?
    var blockFFDim: Int { _blockFFDim ?? hiddenSize }
    let blockMultipleOf: Int
    let blockFFNDimMultiplier: Float
    let blockAutoAdjustFFDim: Bool
    private let _fullAttnIdxs: [Int]?
    private let layerTypes: [String]?
    var fullAttnIdxs: [Int] {
        if let fullAttnIdxs = _fullAttnIdxs {
            return fullAttnIdxs
        }

        if let layerTypes {
            return layerTypes.enumerated().compactMap { index, layerType in
                layerType == "full_attention" ? index : nil
            }
        }

        return Array(0 ..< hiddenLayers)
    }
    let ropeTheta: Float
    var headDimensions: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case _blockDim = "block_dim"
        case _blockFFDim = "block_ff_dim"
        case blockMultipleOf = "block_multiple_of"
        case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
        case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
        case _fullAttnIdxs = "full_attn_idxs"
        case layerTypes = "layer_types"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm2"
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 65536
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        self.normEps = try container.decode(Float.self, forKey: .normEps)
        self.convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        self.convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
        self._blockDim = try container.decodeIfPresent(Int.self, forKey: ._blockDim)
        self._blockFFDim = try container.decodeIfPresent(Int.self, forKey: ._blockFFDim)
        self.blockMultipleOf =
            try container.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
        self.blockFFNDimMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .blockFFNDimMultiplier) ?? 1.0
        self.blockAutoAdjustFFDim =
            try container.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDim) ?? true
        self._fullAttnIdxs = try container.decodeIfPresent([Int].self, forKey: ._fullAttnIdxs)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
    }
}

private class Attention: Module {
    let args: LFM2Configuration
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

    let rope: RoPE

    public init(_ args: LFM2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        self.headDim = args.headDimensions

        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
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

        queries = qLayerNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kLayerNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
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

        return outProj(output)
    }
}

private class ShortConv: Module {
    let args: LFM2Configuration
    let layerIdx: Int
    let lCache: Int
    let bias: Bool

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(_ args: LFM2Configuration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx
        self.lCache = args.convLCache
        self.bias = args.convBias

        _conv.wrappedValue = Conv1d(
            inputChannels: args.hiddenSize,
            outputChannels: args.hiddenSize,
            kernelSize: lCache,
            groups: args.hiddenSize,
            bias: bias
        )

        _inProj.wrappedValue = Linear(args.hiddenSize, 3 * args.hiddenSize, bias: bias)
        _outProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let BCx = inProj(x)
        let BCxSplit = BCx.split(parts: 3, axis: -1)
        let B = BCxSplit[0]
        let C = BCxSplit[1]
        let x = BCxSplit[2]
        var Bx = B * x

        var state: MLXArray? = nil
        if let cache {
            state = cache[0]
        }
        if state == nil {
            state = MLXArray.zeros([Bx.dim(0), lCache - 1, args.hiddenSize], dtype: Bx.dtype)
        }

        Bx = concatenated([state!, Bx], axis: -2)
        if let cache {
            cache[0] = Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...]
        }

        let convOut = conv(Bx)
        let y = C * convOut
        return outProj(y)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    public init(
        dim: Int,
        ffDim: Int,
        multipleOf: Int,
        autoAdjustFFDim: Bool,
        ffnDimMultiplier: Float?
    ) {
        var adjustedFFDim = ffDim

        if autoAdjustFFDim {
            adjustedFFDim = Int(Float(2 * ffDim) / 3.0)
            if let multiplier = ffnDimMultiplier {
                adjustedFFDim = Int(multiplier * Float(adjustedFFDim))
            }
            adjustedFFDim = multipleOf * ((adjustedFFDim + multipleOf - 1) / multipleOf)
        }

        _w1.wrappedValue = Linear(dim, adjustedFFDim, bias: false)
        _w2.wrappedValue = Linear(adjustedFFDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, adjustedFFDim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

private class DecoderLayer: Module {
    let isAttentionLayer: Bool

    @ModuleInfo(key: "self_attn") var attention: Attention?
    @ModuleInfo(key: "conv") var conv: ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    public init(_ args: LFM2Configuration, layerIdx: Int) {
        self.isAttentionLayer = args.fullAttnIdxs.contains(layerIdx)

        if isAttentionLayer {
            _attention.wrappedValue = Attention(args)
        } else {
            _conv.wrappedValue = ShortConv(args, layerIdx: layerIdx)
        }

        _feedForward.wrappedValue = MLP(
            dim: args.blockDim,
            ffDim: args.blockFFDim,
            multipleOf: args.blockMultipleOf,
            autoAdjustFFDim: args.blockAutoAdjustFFDim,
            ffnDimMultiplier: args.blockFFNDimMultiplier
        )
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r: MLXArray
        if isAttentionLayer {
            r = attention!(operatorNorm(x), mask: mask, cache: cache)
        } else {
            r = conv!(operatorNorm(x), cache: cache as? MambaCache)
        }
        let h = x + r
        let out = h + feedForward(ffnNorm(h))
        return out
    }
}

private class LFM2ModelInner: Module {
    let args: LFM2Configuration
    let vocabularySize: Int
    let numHiddenLayers: Int

    fileprivate let layers: [DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    public init(_ args: LFM2Configuration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.numHiddenLayers = args.hiddenLayers

        precondition(vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< numHiddenLayers).map { i in
            DecoderLayer(args, layerIdx: i)
        }

        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var h = inputEmbeddings ?? embedTokens(inputs)

        let mask =
            mask
            ?? {
                let firstAttnIdx = args.fullAttnIdxs.first ?? 0
                let c = cache != nil && firstAttnIdx < cache!.count ? [cache![firstAttnIdx]] : nil
                return createAttentionMask(h: h, cache: c)
            }()

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return embeddingNorm(h)
    }
}

public class LFM2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: LFM2ModelInner
    let configuration: LFM2Configuration

    public init(_ args: LFM2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize

        self.kvHeads = (0 ..< args.hiddenLayers).map { layerIdx in
            args.fullAttnIdxs.contains(layerIdx) ? args.kvHeads : 0
        }

        self.model = LFM2ModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]

        for (name, param) in weights {
            var sanitizedParam = param

            if name.contains("conv.weight") {
                if param.shape[param.shape.count - 1] > param.shape[1] {
                    sanitizedParam = param.transposed(0, 2, 1)
                }
            }

            sanitizedWeights[name] = sanitizedParam
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { layerIdx in
            if configuration.fullAttnIdxs.contains(layerIdx) {
                KVCacheSimple()
            } else {
                MambaCache()
            }
        }
    }
}

extension LFM2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
