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
    var modelType: String = "lfm2"
    var headDim: Int?
    var blockFFDim: Int?
    var vocabularySize: Int = 65536
    var hiddenSize: Int
    var intermediateSize: Int?
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var normEps: Float
    var padTokenId: Int?
    var bosTokenId: Int = 1
    var eosTokenId: Int = 2
    var tieWordEmbeddings: Bool = true
    var convBias: Bool = false
    var convLCache: Int = 3
    var blockMultipleOf: Int = 256
    var blockFFNDimMultiplier: Float = 1.0
    var blockAutoAdjustFFDim: Bool = true
    var fullAttnIdxs: [Int]?
    var layerTypes: [String]?
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var ropeTheta: Float = 1000000.0

    var resolvedIntermediateSize: Int {
        blockFFDim ?? intermediateSize ?? hiddenSize
    }

    var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / attentionHeads)
    }

    var resolvedLayerTypes: [String] {
        if let layerTypes {
            return layerTypes
        }

        let fullAttnIdxs = fullAttnIdxs ?? Array(0 ..< hiddenLayers)
        return (0 ..< hiddenLayers).map { i in
            fullAttnIdxs.contains(i) ? "full_attention" : "conv"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case headDim = "head_dim"
        case blockFFDim = "block_ff_dim"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case blockMultipleOf = "block_multiple_of"
        case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
        case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
        case fullAttnIdxs = "full_attn_idxs"
        case layerTypes = "layer_types"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm2"
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.blockFFDim = try container.decodeIfPresent(Int.self, forKey: .blockFFDim)
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 65536
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        self.normEps = try container.decode(Float.self, forKey: .normEps)
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
        self.bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        self.eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        self.convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
        self.blockMultipleOf =
            try container.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
        self.blockFFNDimMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .blockFFNDimMultiplier) ?? 1.0
        self.blockAutoAdjustFFDim =
            try container.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDim) ?? true
        self.fullAttnIdxs = try container.decodeIfPresent([Int].self, forKey: .fullAttnIdxs)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
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
        self.headDim = args.resolvedHeadDim

        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

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
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: ropeScale
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

private class LFM2ShortConv: Module {
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
            padding: lCache - 1,
            groups: args.hiddenSize,
            bias: bias
        )

        _inProj.wrappedValue = Linear(args.hiddenSize, 3 * args.hiddenSize, bias: bias)
        _outProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let seqlen = x.dim(1)
        let bcx = inProj(x)
        let bcxSplit = bcx.split(parts: 3, axis: -1)
        let b = bcxSplit[0]
        let c = bcxSplit[1]
        let x_proj = bcxSplit[2]
        let bx = b * x_proj

        var convOut: MLXArray

        if let cache, x.dim(1) == 1 {
            var convState = cache[0] ?? MLXArray.zeros([bx.dim(0), lCache, args.hiddenSize])

            convState = roll(convState, shift: -2, axis: -2)
            convState[0..., -1, 0...] = bx[0..., 0, 0...]
            cache[0] = convState

            convOut = (convState.transposed(0, 2, 1) * conv.weight[0..., 0..., 0]).sum(
                axis: -1, keepDims: true)
            if bias {
                convOut = convOut + conv.bias!.expandedDimensions(axes: [-1])
            }
            convOut = convOut.reshaped(bx.dim(0), 1, -1)
        } else {
            if let cache {
                cache[0] = bx[0..., (-lCache)..., 0...]
            }
            convOut = conv(bx)[0..., ..<seqlen, 0...]
        }

        let y = c * convOut
        return outProj(y)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    public init(_ args: LFM2Configuration) {
        var intermediateSize = args.resolvedIntermediateSize

        if args.blockAutoAdjustFFDim {
            intermediateSize = Int(Float(2 * intermediateSize) / 3.0)
            intermediateSize = Int(args.blockFFNDimMultiplier * Float(intermediateSize))
            intermediateSize =
                args.blockMultipleOf
                * ((intermediateSize + args.blockMultipleOf - 1) / args.blockMultipleOf)
        }

        _w1.wrappedValue = Linear(args.hiddenSize, intermediateSize, bias: false)
        _w2.wrappedValue = Linear(intermediateSize, args.hiddenSize, bias: false)
        _w3.wrappedValue = Linear(args.hiddenSize, intermediateSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

private class DecoderLayer: Module {
    let isAttentionLayer: Bool

    @ModuleInfo(key: "self_attn") var attention: Attention?
    @ModuleInfo(key: "conv") var conv: LFM2ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    public init(_ args: LFM2Configuration, layerIdx: Int) {
        self.isAttentionLayer = args.resolvedLayerTypes[layerIdx] == "full_attention"

        if isAttentionLayer {
            _attention.wrappedValue = Attention(args)
        } else {
            _conv.wrappedValue = LFM2ShortConv(args, layerIdx: layerIdx)
        }

        _feedForward.wrappedValue = MLP(args)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let residual = x
        var x = x

        if isAttentionLayer {
            x = attention!(operatorNorm(x), mask: mask, cache: cache)
        } else {
            x = conv!(operatorNorm(x), cache: cache as? MambaCache)
        }

        x = x + residual
        x = x + feedForward(ffnNorm(x))

        return x
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
                let firstAttnIdx = args.resolvedLayerTypes.firstIndex(of: "full_attention") ?? 0
                let c = cache != nil ? [cache![firstAttnIdx]] : nil
                return createAttentionMask(h: h, cache: c)
            }()

        let resolvedCache: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)

        for (layer, c) in zip(layers, resolvedCache) {
            h = layer(h, mask: mask, cache: c)
        }

        return embeddingNorm(h)
    }
}

public class LFM2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: LFM2ModelInner
    let configuration: LFM2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LFM2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize

        self.kvHeads = args.resolvedLayerTypes.map { layerType in
            layerType == "full_attention" ? args.kvHeads : 0
        }

        self.model = LFM2ModelInner(args)

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
        var sanitizedWeights: [String: MLXArray] = [:]

        for (name, param) in weights {
            var sanitizedParam = param

            if name.contains("conv.weight") {
                if param.shape.count > 2, param.shape[2] > param.shape[1] {
                    sanitizedParam = param.transposed(0, 2, 1)
                }
            }

            sanitizedWeights[name] = sanitizedParam
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.resolvedLayerTypes.map { layerType in
            if layerType == "full_attention" {
                KVCacheSimple()
            } else {
                MambaCache()
            }
        }
    }
}

extension LFM2Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.compactMap { layer in
            if layer.isAttentionLayer, let attention = layer.attention {
                return (attention, ["q_proj", "v_proj"])
            }
            return nil
        }
    }
}
