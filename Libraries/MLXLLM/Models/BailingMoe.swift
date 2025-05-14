//
//  BailingMoe.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/bailing_moe.py
//  This architecture is used by the Ling-family models (e.g., Ling Mini).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable

@Codable
public struct BailingMoeConfiguration: Sendable {
    @CodingKey("model_type") public var modelType: String
    @CodingKey("hidden_size") public var hiddenSize: Int
    @CodingKey("intermediate_size") public var intermediateSize: Int
    @CodingKey("max_position_embeddings") public var maxPositionEmbeddings: Int?
    @CodingKey("moe_intermediate_size") public var moeIntermediateSize: Int
    @CodingKey("num_experts") public var numExperts: Int
    @CodingKey("num_shared_experts") public var numSharedExperts: Int
    @CodingKey("norm_topk_prob") public var normTopkProb: Bool
    @CodingKey("num_attention_heads") public var attentionHeads: Int
    @CodingKey("num_experts_per_tok") public var numExpertsPerToken: Int
    @CodingKey("num_hidden_layers") public var hiddenLayers: Int
    @CodingKey("num_key_value_heads") public var kvHeads: Int
    @CodingKey("rms_norm_eps") public var rmsNormEps: Float
    @CodingKey("rope_theta") public var ropeTheta: Float
    @CodingKey("vocab_size") public var vocabularySize: Int
    @CodingKey("first_k_dense_replace") public var firstKDenseReplace: Int

    // Optional features
    @CodingKey("rope_scaling") public var ropeScaling: [String: StringOrNumber]? = nil
    @CodingKey("use_bias") public var useBias: Bool = false
    @CodingKey("use_qkv_bias") public var useQKVBias: Bool = false
    @CodingKey("use_qk_norm") public var useQKNorm: Bool = false
    @CodingKey("tie_word_embeddings") public var tieWordEmbeddings: Bool = false
    @CodingKey("partial_rotary_factor") public var partialRotaryFactor: Float = 1.0
    @CodingKey("moe_router_enable_expert_bias") public var moeRouterEnableExpertBias: Bool = false
    @CodingKey("routed_scaling_factor") public var routedScalingFactor: Float = 1.0
    @CodingKey("score_function") public var scoreFunction: String = "softmax"
    @CodingKey("n_group") public var nGroup: Int = 1
    @CodingKey("topk_group") public var topkGroup: Int = 4
    @CodingKey("moe_shared_expert_intermediate_size") public var moeSharedExpertIntermediateSize: Int? = nil
}

private class Attention: Module {
    let args: BailingMoeConfiguration
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let ropeDim: Int
    let scale: Float

    @ModuleInfo(key: "query_key_value") var qkv: Linear
    @ModuleInfo(key: "dense") var wo: Linear

    @ModuleInfo(key: "query_layernorm") var qNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ args: BailingMoeConfiguration) {
        self.args = args
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.hiddenSize / heads
        self.ropeDim = Int(Float(headDim) * args.partialRotaryFactor)
        self.scale = pow(Float(headDim), -0.5)

        _qkv.wrappedValue = Linear(
            args.hiddenSize,
            (heads + 2 * kvHeads) * headDim,
            bias: args.useQKVBias
        )
        _wo.wrappedValue = Linear(heads * headDim, args.hiddenSize, bias: args.useBias)

        if args.useQKNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        } else {
            _qNorm.wrappedValue = nil
            _kNorm.wrappedValue = nil
        }

        self.rope = RoPE(
            dimensions: ropeDim, traditional: false, base: args.ropeTheta,
            scale: 1.0)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qSize = heads * headDim
        let kSize = kvHeads * headDim
        let qkvOut = qkv(x)
        let splits = split(qkvOut, indices: [qSize, qSize + kSize], axis: -1)
        var queries = splits[0]
        var keys = splits[1]
        var values = splits[2]

        // reshape to (B, L, H, Hd), apply optional per-head norms, then transpose to (B, H, L, Hd)
        queries = queries.reshaped(B, L, heads, -1)
        keys = keys.reshaped(B, L, kvHeads, -1)

        if let qNorm { queries = qNorm(queries) }
        if let kNorm { keys = kNorm(keys) }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

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

private class BailingMoeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: BailingMoeConfiguration, hiddenDims: Int? = nil) {
        let inter = hiddenDims ?? args.intermediateSize
        _gate.wrappedValue = Linear(args.hiddenSize, inter, bias: args.useBias)
        _down.wrappedValue = Linear(inter, args.hiddenSize, bias: args.useBias)
        _up.wrappedValue = Linear(args.hiddenSize, inter, bias: args.useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

private class BailingMoeGate: Module, UnaryLayer {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let numExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoreFunction: String

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ args: BailingMoeConfiguration) {
        self.topK = args.numExpertsPerToken
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb
        self.scoreFunction = args.scoreFunction
        self.numExperts = args.numExperts

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _expertBias.wrappedValue = zeros([args.numExperts])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // This returns a packed result not directly used; callers use groupSelect to get inds and scores.
        gate(x)
    }

    func groupSelect(_ x: MLXArray) -> (inds: MLXArray, scores: MLXArray) {
        let (bsz, seqLen, h) = (x.dim(0), x.dim(1), x.dim(2))

        let logits = gate(x)
        var scores = sigmoid(logits.asType(.float32))
        let scoresForChoice = scores + expertBias
        let groupScores = scoresForChoice.reshaped(bsz, seqLen, self.nGroup, -1)

        let topKGroup = top(groupScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        var k = nGroup - topkGroup
        var groupIdx = argPartition(topKGroup, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0), axis: -2)
        scores = flattened(scores, start: -2, end: -1)

        k = topK
        let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        scores = takeAlong(scores, inds, axis: -1)
        if topK ?? 1 > 1, normTopkProb {
            let denominator = scores.sum(axis: -1, keepDims: true) + 1e-20
            scores = scores / denominator
        }
        scores = scores * routedScalingFactor
        return (inds, scores.asType(logits.dtype))
    }
}

private class BailingMoeSparseMoeBlock: Module, UnaryLayer {
    let args: BailingMoeConfiguration
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: BailingMoeGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingMoeMLP?

    init(_ args: BailingMoeConfiguration) {
        self.args = args
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts,
            bias: args.useBias
        )
        _gate.wrappedValue = BailingMoeGate(args)

        if args.numSharedExperts > 0 {
            let sharedDim =
                (args.moeSharedExpertIntermediateSize ?? args.moeIntermediateSize)
                * args.numSharedExperts
            _sharedExperts.wrappedValue = BailingMoeMLP(args, hiddenDims: sharedDim)
        } else {
            _sharedExperts.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = gate.groupSelect(x)
        var out = switchMLP(x, inds)
        out = (out * weights[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts {
            out = out + shared(x)
        }
        return out
    }
}

private class TransformerBlock: Module {
    let args: BailingMoeConfiguration
    let layerIdx: Int

    @ModuleInfo(key: "attention") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: BailingMoeConfiguration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx

        _attention.wrappedValue = Attention(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if args.numExperts > 0 && layerIdx >= args.firstKDenseReplace {
            _mlp.wrappedValue = BailingMoeSparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = BailingMoeMLP(args)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

private class BailingMoeModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var embedTokens: Embedding
    let layers: [TransformerBlock]
    let norm: RMSNorm

    init(_ args: BailingMoeConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { TransformerBlock(args, layerIdx: $0) }
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

public class BailingMoeModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    fileprivate let model: BailingMoeModelInner
    let configuration: BailingMoeConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: BailingMoeConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = BailingMoeModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }
}

extension BailingMoeModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["query_key_value"]) }
    }
}
