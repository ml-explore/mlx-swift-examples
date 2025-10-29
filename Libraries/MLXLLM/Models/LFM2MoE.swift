//
//  LFM2MoE.swift
//  mlx-swift-examples
//
//  Created by Sachin Desai on 10/08/25.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2_moe.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct LFM2MoEConfiguration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let hiddenLayers: Int
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let normTopkProb: Bool
    public let attentionHeads: Int
    public let kvHeads: Int
    public let maxPositionEmbeddings: Int
    public let useExpertBias: Bool
    public let numDenseLayers: Int
    public let normEps: Float
    public let convBias: Bool
    public let convLCache: Int
    public let ropeTheta: Float

    private let _fullAttnIdxs: [Int]?
    private let layerTypes: [String]?

    public var fullAttnIdxs: [Int] {
        if let explicit = _fullAttnIdxs {
            return explicit
        }
        guard let layerTypes else { return [] }
        return layerTypes.enumerated().compactMap { index, type in
            type == "full_attention" ? index : nil
        }
    }

    public var headDimensions: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case useExpertBias = "use_expert_bias"
        case numDenseLayers = "num_dense_layers"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case ropeTheta = "rope_theta"
        case _fullAttnIdxs = "full_attn_idxs"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.useExpertBias = try container.decode(Bool.self, forKey: .useExpertBias)
        self.numDenseLayers = try container.decode(Int.self, forKey: .numDenseLayers)
        self.normEps = try container.decode(Float.self, forKey: .normEps)
        self.convBias = try container.decode(Bool.self, forKey: .convBias)
        self.convLCache = try container.decode(Int.self, forKey: .convLCache)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self._fullAttnIdxs = try container.decodeIfPresent([Int].self, forKey: ._fullAttnIdxs)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
    }
}

private class Attention: Module {
    let args: LFM2MoEConfiguration
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

    let rope: RoPE

    init(_ args: LFM2MoEConfiguration) {
        self.args = args
        self.headDim = args.headDimensions
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, args.attentionHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, args.kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, args.kvHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(args.attentionHeads * headDim, dim, bias: false)

        _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

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
    let args: LFM2MoEConfiguration
    let layerIdx: Int
    let lCache: Int
    let bias: Bool

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM2MoEConfiguration, layerIdx: Int) {
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

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let BCx = inProj(x)
        let parts = BCx.split(parts: 3, axis: -1)
        var B = parts[0]
        let C = parts[1]
        let xComp = parts[2]
        var Bx = B * xComp

        if let mask {
            let expandedMask = mask[.ellipsis, .newAxis]
            let zeros = MLXArray.zeros(Bx.shape, dtype: Bx.dtype)
            Bx = MLX.where(expandedMask, Bx, zeros)
        }

        var state = cache?[0]
        if state == nil {
            state = MLXArray.zeros([Bx.dim(0), lCache - 1, args.hiddenSize], dtype: Bx.dtype)
        }

        Bx = concatenated([state!, Bx], axis: -2)
        if let cache {
            let start = Bx.dim(1) - (lCache - 1)
            cache[0] = Bx[0..., start..., 0...]
        }

        let convOut = conv(Bx)
        let y = C * convOut
        return outProj(y)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: LFM2MoEConfiguration, intermediateSize: Int? = nil) {
        let hidden = args.hiddenSize
        let inter = intermediateSize ?? args.intermediateSize

        _gateProj.wrappedValue = Linear(hidden, inter, bias: false)
        _upProj.wrappedValue = Linear(hidden, inter, bias: false)
        _downProj.wrappedValue = Linear(inter, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private class Lfm2MoeSparseMoeBlock: Module, UnaryLayer {
    let args: LFM2MoEConfiguration
    let numExperts: Int
    let topK: Int
    let normTopKProb: Bool
    let useExpertBias: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "expert_bias") var expertBias: MLXArray?

    init(_ args: LFM2MoEConfiguration) {
        self.args = args
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopKProb = args.normTopkProb
        self.useExpertBias = args.useExpertBias

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: numExperts,
            bias: false
        )

        if useExpertBias {
            _expertBias.wrappedValue = MLXArray.zeros([numExperts])
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x).asType(.float32)
        gates = MLX.softmax(gates, axis: -1)

        if useExpertBias, let expertBias {
            gates = gates + expertBias
        }

        let k = topK
        var indices = argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = takeAlong(gates, indices, axis: -1)
        if normTopKProb {
            let denom = scores.sum(axis: -1, keepDims: true) + 1e-20
            scores = scores / denom
        }
        scores = scores.asType(x.dtype)

        let expertOutputs = switchMLP(x, indices)
        let weighted = expertOutputs * scores[.ellipsis, .newAxis]
        return weighted.sum(axis: -2)
    }
}

private class DecoderLayer: Module {
    let isAttentionLayer: Bool
    let usesDenseFeedForward: Bool

    @ModuleInfo(key: "self_attn") var attention: Attention?
    @ModuleInfo(key: "conv") var conv: ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ args: LFM2MoEConfiguration, layerIdx: Int) {
        self.isAttentionLayer = args.fullAttnIdxs.contains(layerIdx)
        self.usesDenseFeedForward = layerIdx < args.numDenseLayers

        if isAttentionLayer {
            _attention.wrappedValue = Attention(args)
        } else {
            _conv.wrappedValue = ShortConv(args, layerIdx: layerIdx)
        }

        if usesDenseFeedForward {
            _feedForward.wrappedValue = MLP(args)
        } else {
            _feedForward.wrappedValue = Lfm2MoeSparseMoeBlock(args)
        }

        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let residual = operatorNorm(x)
        let r: MLXArray
        if isAttentionLayer {
            r = attention!(residual, mask: attentionMask, cache: cache)
        } else {
            r = conv!(residual, mask: ssmMask, cache: cache as? MambaCache)
        }

        let h = x + r
        let out = feedForward(ffnNorm(h))
        return h + out
    }
}

private class LFM2MoEModelInner: Module {
    let args: LFM2MoEConfiguration
    let layers: [DecoderLayer]
    let firstAttentionIndex: Int?
    let firstConvIndex: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ args: LFM2MoEConfiguration) {
        self.args = args
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { DecoderLayer(args, layerIdx: $0) }
        self.firstAttentionIndex = args.fullAttnIdxs.first
        self.firstConvIndex = layers.firstIndex(where: { !$0.isAttentionLayer })

        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var hidden = inputEmbeddings ?? embedTokens(inputs)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let index = firstAttentionIndex,
                let cache,
                index < cache.count
            else { return .none }
            return createAttentionMask(h: hidden, cache: [cache[index]])
        }()

        let ssmMask: MLXArray? = {
            guard let index = firstConvIndex,
                let cache,
                index < cache.count
            else { return nil }
            return createSSMMask(h: hidden, cache: cache[index] as? MambaCache)
        }()

        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache?[i])
        }

        return embeddingNorm(hidden)
    }
}

public class LFM2MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: LFM2MoEConfiguration

    private let model: LFM2MoEModelInner

    public init(_ args: LFM2MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { layerIdx in
            args.fullAttnIdxs.contains(layerIdx) ? args.kvHeads : 0
        }
        self.model = LFM2MoEModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (name, param) in weights {
            var tensor = param
            if name.contains("conv.weight") {
                if tensor.shape.last! > tensor.shape[1] {
                    tensor = tensor.transposed(0, 2, 1)
                }
            }

            let replacements = [
                "w1.weight": "gate_proj.weight",
                "w2.weight": "down_proj.weight",
                "w3.weight": "up_proj.weight",
            ]
            var updatedName = name
            for (old, new) in replacements where updatedName.contains(old) {
                updatedName = updatedName.replacingOccurrences(of: old, with: new)
            }

            sanitized[updatedName] = tensor
        }

        for layerIdx in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIdx)"
            let expertPrefix = "\(prefix).feed_forward.experts"

            if sanitized["\(expertPrefix).0.gate_proj.weight"] == nil {
                continue
            }

            for name in ["gate_proj", "down_proj", "up_proj"] {
                let key = "\(expertPrefix).0.\(name).weight"
                guard sanitized[key] != nil else { continue }
                let stacked = (0 ..< configuration.numExperts).map { expert -> MLXArray in
                    sanitized.removeValue(forKey: "\(expertPrefix).\(expert).\(name).weight")!
                }
                sanitized["\(prefix).feed_forward.switch_mlp.\(name).weight"] = MLX.stacked(stacked)
            }
        }

        return sanitized
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

extension LFM2MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
