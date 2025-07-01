// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ReerCodable

// Port of https://github.com/maiqingqiang/mlx-examples/blob/main/llms/mlx_lm/models/internlm2.py

private class DynamicNTKScalingRoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    let originalBase: Float
    var scale: Float

    init(
        dims: Int, maxPositionEmbeddings: Int = 2048, traditional: Bool = false,
        base: Float = 10000, scale: Float = 1.0
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional
        self.originalBase = base
        self.scale = scale
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.dim(1) + offset
        var base = originalBase
        if seqLen > maxPositionEmbeddings {
            base *= pow(
                (scale * Float(seqLen) / Float(maxPositionEmbeddings)) - (scale - 1),
                Float(dims) / Float(dims - 2))
        }
        return MLXFast.RoPE(
            x, dimensions: dims, traditional: traditional, base: base, scale: scale, offset: offset)
    }
}

private class Attention: Module {
    let args: InternLM2Configuration
    let scale: Float

    let heads: Int
    let kvHeads: Int
    let kvGroups: Int
    let headDim: Int

    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    let rope: DynamicNTKScalingRoPE

    init(_ args: InternLM2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.kvGroups = args.kvGroups

        self.headDim = args.hiddenSize / self.heads
        self.scale = pow(Float(headDim), -0.5)

        self._wqkv.wrappedValue = Linear(
            dim, (self.heads + 2 * self.kvHeads) * self.headDim, bias: args.bias)
        self._wo.wrappedValue = Linear(self.heads * self.headDim, dim, bias: args.bias)

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

        self.rope = DynamicNTKScalingRoPE(
            dims: self.headDim,
            maxPositionEmbeddings: args.maxPositionEmbeddings,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: ropeScale
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var qkvStates = wqkv(x)
        qkvStates = qkvStates.reshaped(B, L, -1, 2 + self.kvGroups, self.headDim)
        var queries = qkvStates[.ellipsis, ..<self.kvGroups, 0...]
        queries = queries.reshaped(B, L, -1, self.headDim)
        var keys = qkvStates[.ellipsis, -2, 0...]
        var values = qkvStates[.ellipsis, -1, 0...]

        // prepare the queries, keys and values for the attention computation
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

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return w2(silu(w1(x)) * w3(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: Attention
    @ModuleInfo(key: "feed_forward") var feedForward: MLP

    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ args: InternLM2Configuration) {
        self._attention.wrappedValue = Attention(args)
        self._feedForward.wrappedValue = MLP(dim: args.hiddenSize, hiddenDim: args.intermediateSize)
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(attentionNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = feedForward(ffnNorm(h))
        let out = h + r
        return out
    }
}

private class InternLM2ModelInner: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding

    let layers: [TransformerBlock]
    let norm: RMSNorm

    init(_ args: InternLM2Configuration) {
        precondition(args.vocabularySize > 0)

        self._tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in TransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = tokEmbeddings(inputs)

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class InternLM2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: InternLM2ModelInner

    @ModuleInfo(key: "output") var output: Linear?

    public init(_ args: InternLM2Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = InternLM2ModelInner(args)
        if !args.tieWordEmbeddings {
            self._output.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let output {
            return output(out)
        } else {
            return model.tokEmbeddings.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        weights.filter {
            !$0.key.contains("attention.rope.inv_freq")
        }
    }
}

extension InternLM2Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}

@Codable
public struct InternLM2Configuration: Sendable {
    @CodingKey("hidden_size") public var hiddenSize: Int
    @CodingKey("num_hidden_layers") public var hiddenLayers: Int
    @CodingKey("intermediate_size") public var intermediateSize: Int
    @CodingKey("num_attention_heads") public var attentionHeads: Int
    @CodingKey("rms_norm_eps") public var rmsNormEps: Float
    @CodingKey("vocab_size") public var vocabularySize: Int
    @CodingKey("num_key_value_heads") public var kvHeads: Int
    @CodingKey("max_position_embeddings") public var maxPositionEmbeddings: Int = 32768
    @CodingKey("rope_theta") public var ropeTheta: Float = 10000
    @CodingKey("rope_traditional") public var ropeTraditional: Bool = false
    @CodingKey("rope_scaling") public var ropeScaling: [String: StringOrNumber]?
    @CodingKey("tie_word_embeddings") public var tieWordEmbeddings: Bool = false
    @CodingKey("bias") public var bias: Bool = true

    var kvGroups: Int {
        attentionHeads / kvHeads
    }

    public func didDecode(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: AnyCodingKey.self)
        let codingKey = AnyCodingKey("rope_scaling")

        if let ropeScaling {
            let requiredKeys: Set<String> = ["factor", "type"]
            let keys = Set(ropeScaling.keys)
            if !requiredKeys.isSubset(of: keys) {
                throw DecodingError.dataCorruptedError(
                    forKey: codingKey, in: container,
                    debugDescription: "rope_scaling must contain keys \(requiredKeys)"
                )
            }
            if let type = ropeScaling["type"],
                type != .string("linear") && type != .string("dynamic")
            {
                throw DecodingError.dataCorruptedError(
                    forKey: codingKey, in: container,
                    debugDescription:
                        "rope_scaling 'type' currently only supports 'linear' or 'dynamic'"
                )
            }
        }
    }
}
