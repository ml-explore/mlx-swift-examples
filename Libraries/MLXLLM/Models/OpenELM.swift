//
//  OpenELM.swift
//  LLM
//
//  Created by Sachin Desai on 2024/4/27.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

func computeHeads(modelDim: Int, headDim: Int) -> Int {
    assert(modelDim % headDim == 0, "modelDim must be divisible by headDim")
    return modelDim / headDim
}

func makeDivisible(_ v: Float, divisor: Int = 8, minValue: Float? = nil) -> Int {
    let minVal = minValue ?? Float(divisor)
    var roundDown = max(minVal, Float(Int((v + Float(divisor) / 2) / Float(divisor)) * divisor))

    if roundDown < 0.9 * v {
        roundDown += Float(divisor)
    }
    return Int(roundDown)
}

private class MultiHeadCausalAttention: Module {
    let scale: Float
    let heads: Int
    let headDim: Int
    let kvHeads: Int

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    public init(_ args: OpenElmConfiguration, layerId: Int) {
        self.headDim = args.headDimensions
        let modelDim = args.modelDim

        self.heads = args.numQueryHeads[layerId]
        self.kvHeads = args.kvHeads[layerId]
        self.scale = pow(Float(headDim), -0.5)

        let opSize = (heads + (kvHeads * 2)) * headDim
        self._qkvProj.wrappedValue = Linear(modelDim, opSize, bias: false)
        self._outProj.wrappedValue = Linear(heads * headDim, modelDim, bias: false)

        if args.normalizeQkProjections {
            self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let qkv = qkvProj(x).reshaped(B, L, heads + (kvHeads * 2), headDim).transposed(0, 2, 1, 3)

        let qkvSplit = split(qkv, indices: [heads, heads + kvHeads], axis: 1)
        var queries = qkvSplit[0]
        var keys = qkvSplit[1]
        var values = qkvSplit[2]

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

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
        .reshaped(B, L, heads * headDim)

        return outProj(output)
    }
}

private class FeedForwardNetwork: Module, UnaryLayer {
    @ModuleInfo var proj_1: Linear
    @ModuleInfo var proj_2: Linear

    public init(_ args: OpenElmConfiguration, layedId: Int) {
        let dim = args.modelDim
        let ffnMultiplier = args.ffnMultipliers[layedId]
        let intermediateDim = Int(
            makeDivisible(Float(ffnMultiplier) * Float(dim), divisor: args.ffnDimDivisor))

        self.proj_1 = Linear(dim, 2 * intermediateDim, bias: false)
        self.proj_2 = Linear(intermediateDim, dim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = proj_1(x)
        let b = split(a, parts: 2, axis: -1)
        let gate = b[0]
        let x = b[1]
        return proj_2(silu(gate) * x)
    }
}

private class TransformerDecoderLayer: Module {
    @ModuleInfo(key: "attn") var attn: MultiHeadCausalAttention
    let ffn: FeedForwardNetwork

    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm

    public init(_ args: OpenElmConfiguration, layerId: Int) {
        let dim = args.modelDim
        self._attn.wrappedValue = MultiHeadCausalAttention(args, layerId: layerId)
        self.ffn = FeedForwardNetwork(args, layedId: layerId)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: args.rmsNormEps)
        self._attnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attn(attnNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = ffn(ffnNorm(h))
        let out = h + r
        return out
    }
}

class OpenELMModelInner: Module {
    @ModuleInfo(key: "token_embeddings") var embedTokens: Embedding

    fileprivate let layers: [TransformerDecoderLayer]
    fileprivate let norm: RMSNorm

    public init(_ args: OpenElmConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.modelDim)

        self.layers = (0 ..< args.numTransformerLayers)
            .map { layerId in
                TransformerDecoderLayer(args, layerId: layerId)
            }

        self.norm = RMSNorm(dimensions: args.modelDim, eps: args.rmsNormEps)
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

public class OpenELMModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let transformer: OpenELMModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: OpenElmConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.kvHeads

        self.transformer = OpenELMModelInner(args)
        if !args.shareInputOutputLayers {
            self._lmHead.wrappedValue = Linear(
                args.numTransformerLayers, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = transformer(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = transformer.embedTokens.asLinear(out)
        }

        return out
    }
}

public struct OpenElmConfiguration: Codable, Sendable {
    var modelType: String
    var headDimensions: Int
    var numTransformerLayers: Int
    var modelDim: Int
    var vocabularySize: Int
    var ffnDimDivisor: Int
    var numQueryHeads: [Int] = []
    var kvHeads: [Int] = []
    var ffnWithGlu: Bool = true
    var normalizeQkProjections: Bool = true
    var shareInputOutputLayers: Bool = true
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var numGqaGroups: Int = 4
    var ffnMultipliers: [Float] = [0.5, 4.0]
    var qkvMultiplier: [Float] = [0.5, 1.0]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case headDimensions = "head_dim"
        case numTransformerLayers = "num_transformer_layers"
        case modelDim = "model_dim"
        case vocabularySize = "vocab_size"
        case ffnDimDivisor = "ffn_dim_divisor"
        case ffnMultipliers = "ffn_multipliers"
        case ffnWithGlu = "ffn_with_glu"
        case normalizeQkProjections = "normalize_qk_projections"
        case shareInputOutputLayers = "share_input_output_layers"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<OpenElmConfiguration.CodingKeys> =
            try decoder.container(
                keyedBy: OpenElmConfiguration.CodingKeys.self)

        self.modelType = try container.decode(
            String.self, forKey: OpenElmConfiguration.CodingKeys.modelType)
        self.headDimensions = try container.decode(
            Int.self, forKey: OpenElmConfiguration.CodingKeys.headDimensions)
        self.numTransformerLayers = try container.decode(
            Int.self, forKey: OpenElmConfiguration.CodingKeys.numTransformerLayers)

        self.modelDim = try container.decode(
            Int.self, forKey: OpenElmConfiguration.CodingKeys.modelDim)
        self.vocabularySize = try container.decode(
            Int.self, forKey: OpenElmConfiguration.CodingKeys.vocabularySize)
        self.ffnDimDivisor = try container.decode(
            Int.self, forKey: OpenElmConfiguration.CodingKeys.ffnDimDivisor)

        let qkvMultipliers = stride(
            from: qkvMultiplier[0], through: qkvMultiplier[1],
            by: (qkvMultiplier[1] - qkvMultiplier[0]) / Float(numTransformerLayers - 1)
        )
        .map { round($0 * 100) / 100 }

        let headMultipleOf = numGqaGroups
        let queryDims = qkvMultipliers.map { a in
            makeDivisible(Float(self.modelDim) * a, divisor: self.headDimensions * headMultipleOf)
        }

        self.numQueryHeads = queryDims.map { qDim in
            Int(computeHeads(modelDim: qDim, headDim: self.headDimensions))
        }

        self.kvHeads = self.numQueryHeads.map { qHeads in
            qHeads / numGqaGroups
        }

        self.ffnMultipliers = stride(
            from: ffnMultipliers[0], through: ffnMultipliers[1],
            by: (ffnMultipliers[1] - ffnMultipliers[0]) / Float(numTransformerLayers - 1)
        )
        .map { round($0 * 100) / 100 }

        self.ffnWithGlu =
            try container.decodeIfPresent(
                Bool.self, forKey: OpenElmConfiguration.CodingKeys.ffnWithGlu) ?? true
        self.normalizeQkProjections =
            try container.decodeIfPresent(
                Bool.self, forKey: OpenElmConfiguration.CodingKeys.normalizeQkProjections) ?? true
        self.shareInputOutputLayers =
            try container.decodeIfPresent(
                Bool.self, forKey: OpenElmConfiguration.CodingKeys.shareInputOutputLayers) ?? true
    }
}

// MARK: - LoRA

extension OpenELMModel: LoRAModel {
    public var loraLayers: [Module] {
        transformer.layers
    }
}
